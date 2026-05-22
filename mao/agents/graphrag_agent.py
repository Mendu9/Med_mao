"""
mao/agents/graphrag_agent.py
-----------------------------
GraphRAG retrieval agent — answers factual/knowledge questions.

Route trigger: intent == "graphrag" or intent == "fallback"

Pipeline:
  1. GraphRAG retrieval (9-step hybrid pipeline) — see rag/retriever.py
  2. Confidence check — if top chunk score < RAG_CONFIDENCE_THRESHOLD, trigger
     web-search fallback
  3. Optional web search — DuckDuckGo/Brave/SerpAPI via web_search()
  4. Merge RAG chunks + web results (RAG first = higher trust)
  5. LLM synthesis with full provenance citations (chunk_id, source doc, web URL)

Anti-hallucination grounding strategy:
  - RAG chunks from ingested PDFs are primary ground truth (cite chunk_id + doc)
  - Web results are supplementary — never override a RAG-grounded fact
  - Domain supervisor (downstream) re-checks for ungrounded claims

Integration points:
  - rag/retriever.py    full 9-step pipeline
  - core/web_search.py  web-search fallback
  - memory/mem0_handler search before / save after
  - core/state.py       MAOState contract
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from mao.core import llm as groq_llm
from mao.core.state import MAOState
from mao.core.web_search import web_search as _web_search
from mao.memory.mem0_handler import build_system_prompt, save_memory, search_memories
from mao.rag.retriever import retrieve

logger = logging.getLogger(__name__)

# Minimum score to treat RAG results as sufficient (no web fallback needed).
# Cosine similarity range is 0.0-1.0; 0.10 is a pragmatic floor that fires
# web search only when the collection truly has no relevant content.
# (bge-reranker scores are 0-1 after normalize=True, but when the reranker is
# disabled the raw cosine similarity is used instead — keep threshold low.)
RAG_CONFIDENCE_THRESHOLD: float = 0.10

_GRAPHRAG_SYSTEM = """\
You are a precise, factual medical and scientific assistant with access to a \
curated knowledge base of research papers and a supplementary web search.

GROUNDING RULES (strictly enforced):
  1. RAG chunks (marked [RAG #N]) come from peer-reviewed ingested documents.
     Treat them as the primary ground truth for specific clinical facts.
  2. Web results (marked [WEB #N]) are supplementary. Use them to fill gaps or
     provide recency, but NEVER use a web result to contradict a RAG chunk.
  3. If RAG chunks are present, your answer MUST cite them using their source
     and chunk_id: e.g. "According to [RAG 2: alzheimer_review.pdf / chunk a3b4c5]..."
  4. For web results, cite the URL: e.g. "A recent report ([WEB 1]: https://...)..."
  5. If neither source confirms a claim, say so explicitly — do not fabricate.
  6. Always end with a "Sources:" section listing every cited RAG chunk_id and web URL.
"""

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

    Follows the mandatory Mem0 pattern:
      1. search_memories BEFORE LLM
      2. inject into system prompt
      3. save_memory AFTER LLM responds
    """
    user_query: str = state["user_query"]
    user_id: str    = state["user_id"]
    memory_context: str = state.get("memory_context", "")

    # Step 1: Mem0 (refresh if empty)
    if not memory_context:
        memory_context = search_memories(user_query, user_id)
        state["memory_context"] = memory_context

    # Step 2: GraphRAG retrieval pipeline
    # If the decomposer produced multiple sub-queries, retrieve in parallel.
    sub_queries: list[str] = state.get("sub_queries", [])
    try:
        if len(sub_queries) > 1:
            all_chunks: list = []
            with ThreadPoolExecutor(max_workers=min(3, len(sub_queries))) as pool:
                futures = {pool.submit(retrieve, sq, top_k=5): sq for sq in sub_queries}
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
            ranked_chunks = retrieve(user_query)
    except Exception as exc:  # noqa: BLE001
        logger.error("GraphRAG retrieval failed: %s", exc)
        ranked_chunks = []

    # Step 3: Assess confidence — web search supplements when RAG score is low,
    # but we ALWAYS synthesize a response from whatever RAG chunks exist.
    top_score = ranked_chunks[0].score if ranked_chunks else 0.0
    rag_sufficient = bool(ranked_chunks) and top_score >= RAG_CONFIDENCE_THRESHOLD
    # Web fallback fires when: no chunks OR top score below floor
    trigger_web = not rag_sufficient
    logger.info(
        "GraphRAG confidence: top_score=%.3f sufficient=%s chunks=%d web_fallback=%s",
        top_score, rag_sufficient, len(ranked_chunks), trigger_web,
    )

    # Step 4: Web-search fallback (supplements RAG, never replaces it)
    web_results: list[dict[str, str]] = []
    if trigger_web:
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
    system_prompt = build_system_prompt(_GRAPHRAG_SYSTEM, memory_context)
    user_prompt = f"{context_block}\n\nUser question: {user_query}"
    response = _call_llm(system_prompt, user_prompt, state.get("chat_history", []))

    # Step 7: Mem0 post-hook
    save_memory(user_query, response, user_id)

    state["response"]   = response
    state["agent_used"] = "graphrag"
    state["retrieved_docs"] = [
        {"text": c.text, "source": (c.metadata or {}).get("source", ""), "score": c.score}
        for c in ranked_chunks
    ]
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


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    chat_history: list[dict[str, str]],
) -> str:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history[-4:])
    messages.append({"role": "user", "content": user_prompt})

    try:
        return groq_llm.chat(
            messages=messages,
            temperature=0.1,
            max_tokens=768,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("GraphRAG LLM call failed: %s", exc)
        return f"I encountered an error generating a response: {exc}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from mao.core.state import make_initial_state
    state = make_initial_state("What drugs are used to treat Alzheimer's disease?", "user-test")
    state = graphrag_node(state)
    print(state["response"])
    print("\nMetadata:", state["metadata"])
