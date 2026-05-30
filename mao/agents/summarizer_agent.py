"""Summarizer agent: condenses inline text or KB-retrieved content using map-reduce for long inputs."""

from __future__ import annotations

import logging
from typing import Any

from mao.core import llm as groq_llm

from mao.core.config import cfg
from mao.core.state import MAOState
from mao.memory.mem0_handler import build_system_prompt, save_memory, search_memories
from mao.rag.retriever import retrieve

logger = logging.getLogger(__name__)

# ~3000 chars ≈ ~750 tokens — safe limit per chunk for mistral context window
_CHUNK_SIZE   = 3000
_CHUNK_OVERLAP = 200

_SUMMARIZE_SYSTEM = """\
You are an expert summarizer. Produce a clear, concise summary.

Guidelines:
  - Lead with a 1-sentence TL;DR
  - Follow with 3-7 bullet points covering key facts/arguments
  - Preserve important numbers, names, and dates
  - Do NOT add information not present in the source
  - Use plain language; avoid jargon unless the source uses it
"""

_MAP_SYSTEM = """\
Summarize the following passage in 2-4 bullet points. Be concise.
"""

_REDUCE_SYSTEM = """\
You are given several partial summaries of sections of a longer document.
Combine them into a single coherent summary:
  - 1-sentence TL;DR at the top
  - 5-10 bullet points covering the most important points across all sections
  - Remove redundancy; keep the most specific facts
"""


def summarizer_node(state: MAOState) -> MAOState:
    """
    LangGraph node: summarize inline text or retrieved knowledge-base content.
    """
    user_query: str = state["user_query"]
    user_id: str    = state["user_id"]
    memory_context: str = state.get("memory_context", "")

    if not memory_context:
        memory_context = search_memories(user_query, user_id)
        state["memory_context"] = memory_context

    # --- Detect inline text vs. KB-retrieval mode ---
    inline_text = _extract_inline_text(user_query)

    if inline_text:
        source_text = inline_text
        mode = "inline"
    else:
        # Fall back to GraphRAG retrieval
        try:
            chunks = retrieve(user_query, top_k=5)
            source_text = "\n\n".join(c.text for c in chunks)
            mode = "kb_retrieval"
        except Exception as exc:  # noqa: BLE001
            logger.error("KB retrieval for summarizer failed: %s", exc)
            source_text = ""
            mode = "error"

    if not source_text:
        response = "I couldn't find any text to summarize. Please provide text directly or ask a more specific question."
        save_memory(user_query, response, user_id)
        state.update({"response": response, "agent_used": "summarizer", "metadata": {"mode": mode}})
        return state

    # --- Map-reduce for long texts ---
    system_prompt = build_system_prompt(_SUMMARIZE_SYSTEM, memory_context)

    if len(source_text) > _CHUNK_SIZE * 2:
        response = _map_reduce_summarize(source_text, system_prompt)
    else:
        response = _single_pass_summarize(source_text, user_query, system_prompt)

    save_memory(user_query, response, user_id)

    state["response"]   = response
    state["agent_used"] = "summarizer"
    state["metadata"]   = {
        "mode": mode,
        "source_length": len(source_text),
    }
    return state


def _extract_inline_text(query: str) -> str:
    """
    Heuristic: if the query is long (>300 chars after removing instruction words),
    treat the body as inline text to summarize.

    A production system would use a dedicated extraction prompt.
    """
    # Common summarise preambles to strip
    prefixes = [
        "summarize this:", "summarise this:", "summarize:", "summarise:",
        "give me a tldr of:", "tl;dr:", "tldr:", "condense:", "what are the key points of:",
        "summarize the following:", "summarise the following:",
    ]
    q_lower = query.lower().strip()
    for p in prefixes:
        if q_lower.startswith(p):
            candidate = query[len(p):].strip()
            if len(candidate) > 100:
                return candidate

    # If the whole query is >400 chars and has no question mark near the start,
    # assume the user pasted a document
    if len(query) > 400 and "?" not in query[:80]:
        return query

    return ""


def _single_pass_summarize(text: str, query: str, system_prompt: str) -> str:
    user_prompt = f"Summarize the following text:\n\n{text[:4000]}\n\nOriginal request: {query}"
    return _call_llm(system_prompt, user_prompt)


def _map_reduce_summarize(text: str, system_prompt: str) -> str:
    """
    Split text into chunks → summarize each (map) → combine (reduce).
    Used when the source text exceeds a safe single-pass context length.
    """
    chunks = _split_text(text, _CHUNK_SIZE, _CHUNK_OVERLAP)
    logger.debug("Map-reduce: %d chunks", len(chunks))

    # Map phase
    partial_summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        partial = _call_llm(
            _MAP_SYSTEM,
            f"Section {i + 1}:\n{chunk}",
        )
        partial_summaries.append(f"Section {i + 1} summary:\n{partial}")

    # Reduce phase
    combined = "\n\n".join(partial_summaries)
    return _call_llm(_REDUCE_SYSTEM, combined)


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple character-based sliding window splitter."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return groq_llm.chat(
            messages=messages,
            temperature=0.2,
            max_tokens=768,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Summarizer LLM call failed: %s", exc)
        return f"Summarization failed: {exc}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from mao.core.state import make_initial_state
    state = make_initial_state(
        "Summarize this: The transformer architecture, introduced in 'Attention is All You Need' "
        "(Vaswani et al., 2017), revolutionized NLP. It relies entirely on self-attention mechanisms "
        "to draw global dependencies between input and output, dispensing with recurrence and "
        "convolutions entirely. The model consists of an encoder and decoder, each composed of "
        "stacked identical layers containing multi-head self-attention and feed-forward sub-layers.",
        "user-test",
    )
    state = summarizer_node(state)
    print(state["response"])
