"""
mao/agents/graphrag_agent.py
-----------------------------
GraphRAG retrieval agent — answers factual/knowledge questions.

Route trigger: intent == "graphrag" or intent == "fallback"

When to route here:
  - Factual questions: "Who invented the telephone?", "What is a transformer?"
  - Multi-hop questions: "What did Einstein's doctoral advisor work on?"
  - Knowledge-base questions about ingested Wikipedia/SQuAD content
  - Fallback for any unclassified intent

Pipeline (follows canonical 5-step pattern):
  1. Vector search → 20 candidates (ChromaDB)
  2. Entity extraction → NER on results
  3. Graph traversal → NetworkX expansion
  4. Merge + dedup
  5. Reranker → top 5
  Then: inject top-5 into llama3.1:8b prompt with structured context

Uses llama3.1:8b (not mistral) because multi-hop reasoning benefits from
the larger model's in-context reasoning capabilities.

Integration points:
  - rag/retriever.py    full 5-step pipeline
  - memory/mem0_handler search before / save after
  - core/state.py       MAOState contract
"""

from __future__ import annotations

import logging
from typing import Any

import requests
from mao.core import llm as groq_llm

from mao.core.config import cfg
from mao.core.state import MAOState
from mao.memory.mem0_handler import build_system_prompt, save_memory, search_memories
from mao.rag.retriever import retrieve

logger = logging.getLogger(__name__)

_GRAPHRAG_SYSTEM = """\
You are a precise, factual assistant with access to a curated knowledge base.

Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say "I don't have enough \
information in my knowledge base to answer that accurately."

Do NOT fabricate facts. Cite the source document when possible.
"""

_CONTEXT_TEMPLATE = """\
--- Retrieved Knowledge ---
{chunks}
--------------------------
"""

_CHUNK_TEMPLATE = "[{idx}] (source: {source}, score: {score:.3f})\n{text}"


def graphrag_node(state: MAOState) -> MAOState:
    """
    LangGraph node: GraphRAG retrieval + LLM generation.

    Follows the mandatory Mem0 pattern:
      1. search_memories BEFORE LLM
      2. inject into system prompt
      3. save_memory AFTER LLM responds
    """
    user_query: str = state["user_query"]
    user_id: str    = state["user_id"]
    memory_context: str = state.get("memory_context", "")

    # --- Step 1: Mem0 (already populated by router, but refresh if empty) ---
    if not memory_context:
        memory_context = search_memories(user_query, user_id)
        state["memory_context"] = memory_context

    # --- Step 2: GraphRAG retrieval pipeline ---
    try:
        ranked_chunks = retrieve(user_query)
    except Exception as exc:  # noqa: BLE001
        logger.error("GraphRAG retrieval failed: %s", exc)
        ranked_chunks = []

    # --- Step 3: Format context for LLM ---
    if ranked_chunks:
        chunk_strs = [
            _CHUNK_TEMPLATE.format(
                idx=i + 1,
                source=c.metadata.get("source", "unknown"),
                score=c.score,
                text=c.text[:600],
            )
            for i, c in enumerate(ranked_chunks)
        ]
        context_block = _CONTEXT_TEMPLATE.format(chunks="\n\n".join(chunk_strs))
    else:
        context_block = "No relevant context found in knowledge base."

    # --- Step 4: Build prompt with memory injection ---
    system_prompt = build_system_prompt(_GRAPHRAG_SYSTEM, memory_context)
    user_prompt = f"{context_block}\n\nUser question: {user_query}"

    # --- Step 5: LLM generation (llama3.1:8b for reasoning quality) ---
    response = _call_llm(system_prompt, user_prompt, state.get("chat_history", []))

    # --- Step 6: Mem0 post-hook (always after LLM responds) ---
    save_memory(user_query, response, user_id)

    state["response"]   = response
    state["agent_used"] = "graphrag"
    state["metadata"]   = {
        "chunks_retrieved": len(ranked_chunks),
        "top_scores": [round(c.score, 4) for c in ranked_chunks],
        "sources": list({c.metadata.get("source", "") for c in ranked_chunks}),
    }
    return state


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    chat_history: list[dict[str, str]],
) -> str:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    # Include last 4 history turns for coherence
    messages.extend(chat_history[-4:])
    messages.append({"role": "user", "content": user_prompt})

    try:
        return groq_llm.chat(
            messages=messages,
            temperature=0.1,
            max_tokens=512,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("GraphRAG LLM call failed: %s", exc)
        return f"I encountered an error generating a response: {exc}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from mao.core.state import make_initial_state
    state = make_initial_state("Who invented the telephone?", "user-test")
    state = graphrag_node(state)
    print(state["response"])
