"""Retrieves biomedical knowledge and generates grounded answers with source citations."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from mao.core.state import MAOState, risk_level_of
from mao.core.web_search import web_search as _web_search
from mao.memory.mem0_handler import build_system_prompt
from mao.prompts import get_prompt
from mao.providers import gateway
from mao.trust.classes import TrustClass
from mao.trust.egress.policy import EgressPurpose
from mao.providers.registry import ModelRole
from mao.rag.retriever import retrieve
from mao.safety.policy import get_policy
from mao.schemas.evidence import from_chunk as evidence_from_chunk

try:
    from mao.data.ingest_pubmed import live_pubmed_search as _live_pubmed_search
except Exception:
    _live_pubmed_search = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Minimum score to treat RAG results as sufficient (no web fallback needed).
# Cosine similarity range is 0.0-1.0; 0.10 is a pragmatic floor that fires
# web search only when the collection truly has no relevant content.
# (bge-reranker scores are 0-1 after normalize=True, but when the reranker is
# disabled the raw cosine similarity is used instead — keep threshold low.)
RAG_CONFIDENCE_THRESHOLD: float = 0.10

# Answer budget. The bound model's analysis channel is paid for on top of this
# by the gateway — see mao/providers/registry.py.
_SYNTHESIS_MAX_TOKENS = 768


def rag_is_sufficient(ranked_chunks: list) -> bool:
    """Whether retrieval covered the question well enough to skip the fallback.

    The threshold is calibrated against the cross-encoder's normalised 0-1
    output. It is applied ONLY to a score the cross-encoder produced.

    A passthrough score — when the reranker is disabled, absent, or failed to
    load — is on an unknown scale. The BM25 leg emits unbounded Okapi scores
    routinely in the 20-30 range, so `27.17 >= 0.10` was true for every such
    query and the PubMed and web fallback could not fire on any of them. It read
    as "the corpus covers this beautifully" when it meant "we cannot tell".

    Unknown confidence resolves toward MORE evidence, not less: an uncalibrated
    score is insufficient, so the fallback runs. The cost of being wrong that way
    is one extra literature search; the other way it is an answer grounded in
    whatever BM25 happened to rank first.
    """
    from mao.rag.reranker import CROSS_ENCODER

    if not ranked_chunks:
        return False
    best = ranked_chunks[0]
    if getattr(best, "scorer", None) != CROSS_ENCODER:
        logger.info(
            "Retrieval confidence is uncalibrated (scorer=%s) — treating as "
            "insufficient so the literature fallback runs",
            getattr(best, "scorer", "unknown"),
        )
        return False
    return best.score >= RAG_CONFIDENCE_THRESHOLD

_CONTEXT_TEMPLATE = """\
--- Retrieved Knowledge ---
{chunks}
--------------------------
"""

_CHUNK_TEMPLATE = (
    "[RAG {idx}] source: {source} | doc_id: {doc_id} | chunk_id: {chunk_id} | "
    "score: {score:.3f}\n{text}"
)

_WEB_TEMPLATE = "[WEB {idx}] title: {title} | url: {url}\n{body}"


def graphrag_node(state: MAOState) -> MAOState:
    """
    LangGraph node: GraphRAG retrieval + optional web-search fallback + LLM synthesis.

    Memory is not this node's concern. The graph recalls it once before dispatch
    and persists the verified answer once after supervision; this node only
    reads `state["memory_context"]` to inject into the system prompt.
    """
    user_query: str = state["user_query"]
    memory_context: str = state.get("memory_context", "")


    # Step 2: GraphRAG retrieval pipeline
    # If the decomposer produced multiple sub-queries, retrieve in parallel.
    # `domain` comes from the classifier node and must reach retrieve(), or
    # every query is served from the default index.
    domain: str = state.get("domain") or "alzheimer"
    sub_queries: list[str] = state.get("sub_queries", [])
    try:
        if len(sub_queries) > 1:
            all_chunks: list = []
            with ThreadPoolExecutor(max_workers=min(3, len(sub_queries))) as pool:
                futures = {
                    pool.submit(retrieve, sq, top_k=5, domain=domain): sq
                    for sq in sub_queries
                }
                for future in as_completed(futures):
                    try:
                        chunks = future.result()
                        all_chunks.extend(chunks)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Parallel retrieve failed for sub-query %r: %s",
                            futures[future],
                            exc,
                        )
            # Deduplicate by chunk_id and re-sort by score descending
            seen: set[str] = set()
            unique_chunks = []
            for c in sorted(all_chunks, key=lambda x: x.score, reverse=True):
                cid = (c.metadata or {}).get("chunk_id", id(c))
                if cid not in seen:
                    seen.add(cid)
                    unique_chunks.append(c)
            ranked_chunks = unique_chunks
            logger.info(
                "Parallel retrieval: %d sub-queries → %d unique chunks",
                len(sub_queries),
                len(ranked_chunks),
            )
        else:
            ranked_chunks = retrieve(user_query, domain=domain)
    except Exception as exc:  # noqa: BLE001
        logger.error("GraphRAG retrieval failed: %s", exc)
        ranked_chunks = []

    # Step 3: Assess confidence — web search supplements when RAG score is low,
    # but we ALWAYS synthesize a response from whatever RAG chunks exist.
    top_score = ranked_chunks[0].score if ranked_chunks else 0.0
    rag_sufficient = rag_is_sufficient(ranked_chunks)
    # Web fallback fires when: no chunks OR top score below floor
    trigger_web = not rag_sufficient
    logger.info(
        "GraphRAG confidence: top_score=%.3f sufficient=%s chunks=%d web_fallback=%s",
        top_score, rag_sufficient, len(ranked_chunks), trigger_web,
    )

    # Step 4: Fallbacks when RAG score is insufficient
    web_results: list[dict[str, str]] = []
    pubmed_snippets: list[dict[str, str]] = []
    if trigger_web:
        # 4a: Live PubMed search — higher quality than web for clinical queries
        if _live_pubmed_search is not None:
            try:
                pubmed_snippets = _live_pubmed_search(user_query, max_results=5)
                if pubmed_snippets:
                    logger.info("PubMed live fallback: %d abstracts", len(pubmed_snippets))
            except Exception as exc:  # noqa: BLE001
                logger.debug("PubMed live fallback skipped: %s", exc)

        # 4b: DuckDuckGo/web search as additional supplementary source
        try:
            web_results = _web_search(user_query, num_results=5)
            logger.info("Web search returned %d results for fallback", len(web_results))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Web search fallback failed: %s", exc)

    # Step 5: Build merged context — RAG first (higher trust), web second
    context_parts: list[str] = []

    if ranked_chunks:
        rag_strs = [
            _CHUNK_TEMPLATE.format(
                idx=i + 1,
                source=(c.metadata or {}).get("source", "unknown"),
                doc_id=(c.metadata or {}).get("title", (c.metadata or {}).get("source", "unknown")),
                chunk_id=(c.metadata or {}).get("chunk_id", "n/a"),
                score=c.score,
                text=c.text[:600],
            )
            for i, c in enumerate(ranked_chunks)
        ]
        context_parts.append("=== RAG Knowledge Base ===\n" + "\n\n".join(rag_strs))
    else:
        context_parts.append("=== RAG Knowledge Base ===\nNo relevant documents found.")

    if pubmed_snippets:
        pm_strs = [
            f"[PUBMED {i+1}] title: {r.get('title', '')} | year: {r.get('year', '')} | "
            f"journal: {r.get('journal', '')} | source: {r.get('source', '')}\n"
            f"{r.get('abstract', '')[:500]}"
            for i, r in enumerate(pubmed_snippets)
        ]
        context_parts.append("=== PubMed Live Search (peer-reviewed supplement) ===\n" + "\n\n".join(pm_strs))

    if web_results:
        web_strs = [
            _WEB_TEMPLATE.format(
                idx=i + 1,
                title=r.get("title", ""),
                url=r.get("href", ""),
                body=r.get("body", "")[:400],
            )
            for i, r in enumerate(web_results)
        ]
        context_parts.append("=== Web Search Results (supplementary) ===\n" + "\n\n".join(web_strs))

    context_block = _CONTEXT_TEMPLATE.format(chunks="\n\n".join(context_parts))

    # Step 6: LLM generation with citations
    system_prompt = build_system_prompt(
        get_prompt("graphrag.synthesis").template, memory_context
    )
    user_prompt = f"{context_block}\n\nUser question: {user_query}"

    # Build messages once — reused by both sync and deferred paths.
    #
    # The history goes to the provider verbatim, and on the deferred path it
    # goes there through `_stream_messages` after the route has returned. Both
    # were unscrubbed egress until the route stopped putting the caller's raw
    # turns in state (ADV15-8); the turns on this key are minted by the
    # protected input boundary and are not transformed a second time here.
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(state.get("chat_history", [])[-4:])
    messages.append({"role": "user", "content": user_prompt})

    # P0-1: deferring generation to the streaming endpoint leaves
    # state["response"] empty, which short-circuits the entire supervision
    # chain (verification -> domain_supervisor -> council -> senior). That is
    # only acceptable where the policy does not require verification at all.
    # Transport must never be an input to a safety decision, so the risk level
    # — not `_want_stream` — decides. This trades true token streaming for
    # output safety on every verified route, deliberately.
    risk = risk_level_of(state)
    may_defer = not get_policy().requires_verification(risk)

    # Which capability this synthesis actually is.
    #
    # When the curated index was insufficient and the answer is being built from
    # live PubMed abstracts and web results, this is synthesis over primary
    # literature, not over the vetted corpus — which is exactly what
    # RESEARCH_SYNTHESIS names. The role previously had no call site at all, so
    # the gateway advertised a capability nothing could request.
    #
    # Both roles resolve to the same model today, so this changes no behaviour
    # now. What it changes is that literature synthesis becomes independently
    # rebindable and independently traceable, which is the point of addressing
    # capabilities rather than models.
    synthesis_role = (
        ModelRole.RESEARCH_SYNTHESIS
        if (pubmed_snippets or web_results)
        else ModelRole.GENERAL_SYNTHESIS
    )

    if state.get("_want_stream") and may_defer:
        state["_stream_messages"] = messages
        state["_stream_model"]    = gateway.model_id_for(synthesis_role)
        state["_stream_role"]     = synthesis_role.value
        response = ""
    else:
        if state.get("_want_stream"):
            logger.info(
                "GraphRAG: streaming deferral refused at risk=%s — verification required",
                risk.value,
            )
        state["_stream_messages"] = []
        state["_stream_model"]    = ""
        state["_stream_role"]     = ""
        response = _call_llm_from_messages(messages, role=synthesis_role)

    # Memory is persisted once by the graph's `remember` node, after the
    # supervision chain — so what gets remembered is the verified answer, not
    # this draft.
    state["response"]   = response
    state["agent_used"] = "graphrag"
    # Typed Evidence, not ad-hoc dicts. The council, the judge's premise and the
    # domain supervisor all consume this; when it was a dict they each reached
    # for `.text`, missed, and fell back to `str(doc)`, so the safety chain
    # judged clinical answers against Python repr syntax.
    state["retrieved_docs"] = [evidence_from_chunk(c) for c in ranked_chunks]
    state["web_results"] = [
        {"title": r.get("title", ""), "url": r.get("href", ""), "body": r.get("body", "")}
        for r in web_results
    ]
    state["metadata"] = {
        "chunks_retrieved":     len(ranked_chunks),
        "top_rag_score":        round(top_score, 4),
        "rag_sufficient":       rag_sufficient,
        "web_search_triggered": bool(web_results),
        "web_results_count":    len(web_results),
        "top_scores":           [round(c.score, 4) for c in ranked_chunks],
        # Which scorer produced those numbers. Without it a stored `top_score`
        # cannot be interpreted later, because its scale depends on whether the
        # cross-encoder was available on that request.
        "score_scorer":         (
            getattr(ranked_chunks[0], "scorer", "") if ranked_chunks else ""
        ),
        "sources": [
            {
                "chunk_id": (c.metadata or {}).get("chunk_id", "n/a"),
                "source":   (c.metadata or {}).get("source", "unknown"),
                "doc_id":   (c.metadata or {}).get("title", (c.metadata or {}).get("source", "unknown")),
                "score":    round(c.score, 4),
            }
            for c in ranked_chunks
        ],
        "web_sources": [
            {"title": r.get("title", ""), "url": r.get("href", "")}
            for r in web_results
        ],
    }
    return state


def _call_llm_from_messages(
    messages: list[dict[str, Any]],
    role: ModelRole = ModelRole.GENERAL_SYNTHESIS,
) -> str:
    try:
        return gateway.complete(
            role=role,
            messages=messages,
            temperature=0.1,
            purpose=EgressPurpose.GENERAL_SYNTHESIS,
            trust_class=TrustClass.SAFE_DERIVED_TEXT,
            max_tokens=_SYNTHESIS_MAX_TOKENS,
        ).text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("GraphRAG LLM call failed: %s", exc)
        # Deliberately does not carry `exc` to the user: this string becomes the
        # response body. The API layer makes the same distinction in
        # `client_safe_detail` — the cause belongs in the log.
        logger.exception("GraphRAG synthesis failed")
        return (
            "I could not generate an answer for this question. "
            "Please try again, or rephrase the question."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from mao.core.state import make_initial_state
    state = make_initial_state("What drugs are used to treat Alzheimer's disease?", "user-test")
    state = graphrag_node(state)
    print(state["response"])
    print("\nMetadata:", state["metadata"])
