"""
mao/memory/mem0_handler.py
--------------------------
Mem0 wrapper that every agent node calls before and after its LLM call.

Pattern (MANDATORY in every agent):
  1. BEFORE LLM: memories = await search_memories(query, user_id)
  2. Inject into system prompt under "What you know about this user:"
  3. AFTER LLM:  await save_memory(user_query, llm_response, user_id)

Why Mem0 instead of custom memory:
  - Automatic compression and deduplication — no bespoke logic needed
  - Production-tested open-source library
  - Pluggable vector store (we use ChromaDB to keep infra homogeneous)

Integration points:
  - Called by EVERY agent node in agents/
  - Config from mao/core/config.py
"""

from __future__ import annotations

import logging
from typing import Any

from mem0 import Memory

from mao.core.config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mem0 client — initialised once, shared across all agents
# ---------------------------------------------------------------------------

def _build_mem0_config() -> dict[str, Any]:
    """Mem0 backed by ChromaDB (vectors) + Groq (LLM) + sentence-transformers (embed)."""
    return {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "host": cfg.chroma_host,
                "port": cfg.chroma_port,
                "collection_name": "mao_memory",
            },
        },
        # PINECONE: "vector_store": {
        # PINECONE:     "provider": "pinecone",
        # PINECONE:     "config": {
        # PINECONE:         "api_key": cfg.pinecone_api_key,
        # PINECONE:         "collection_name": cfg.pinecone_mem0_index,
        # PINECONE:         "embedding_model_dims": cfg.embed_dim,
        # PINECONE:         "metric": "cosine",
        # PINECONE:         "serverless_config": {
        # PINECONE:             "cloud": "aws",
        # PINECONE:             "region": cfg.pinecone_region,
        # PINECONE:         },
        # PINECONE:     },
        # PINECONE: },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": cfg.embed_model,
            },
        },
        "llm": {
            "provider": "groq",
            "config": {
                "api_key": cfg.groq_api_key,
                "model": cfg.groq_model,
            },
        },
        # OLLAMA: "llm": {
        # OLLAMA:     "provider": "ollama",
        # OLLAMA:     "config": {
        # OLLAMA:         "model": cfg.groq_model,
        # OLLAMA:         "ollama_base_url": "http://localhost:11434",
        # OLLAMA:     },
        # OLLAMA: },
    }


# Lazy singleton — avoids network calls at import time
_mem0_client: Memory | None = None


def get_mem0_client() -> Memory:
    """Return the shared Mem0 Memory instance, initialising on first call."""
    import os
    global _mem0_client
    if os.getenv("MAO_DISABLE_MEM0", "").lower() in ("1", "true", "yes"):
        raise RuntimeError("Mem0 disabled via MAO_DISABLE_MEM0 env var")
    if _mem0_client is None:
        logger.info("Initialising Mem0 client (chroma collection=mao_memory)")
        _mem0_client = Memory.from_config(_build_mem0_config())
    return _mem0_client


# ---------------------------------------------------------------------------
# Public API used by all agent nodes
# ---------------------------------------------------------------------------

def search_memories(query: str, user_id: str, limit: int = 5) -> str:
    """
    Search Mem0 for memories relevant to *query* for *user_id*.

    Returns a formatted string ready to be injected into a system prompt:

        What you know about this user:
        - Prefers concise answers
        - Works in Python, dislikes Java examples
        ...

    Returns an empty string if no memories exist or on error.
    """
    try:
        client = get_mem0_client()
        results: list[dict[str, Any]] = client.search(
            query=query,
            user_id=user_id,
            limit=limit,
        )
        # mem0 may return a dict with a "results" key in newer versions
        if isinstance(results, dict):
            results = results.get("results", [])
        if not results:
            return ""

        lines = ["What you know about this user:"]
        for mem in results:
            if isinstance(mem, str):
                text = mem
            else:
                text = mem.get("memory") or mem.get("text", "")
            if text:
                lines.append(f"- {text}")

        return "\n".join(lines) if len(lines) > 1 else ""

    except Exception as exc:  # noqa: BLE001
        logger.warning("Mem0 search failed (user=%s): %s", user_id, exc)
        return ""


def save_memory(
    user_query: str,
    assistant_response: str,
    user_id: str,
) -> None:
    """
    Persist the exchange to Mem0 after the agent responds.

    Mem0 handles compression and deduplication automatically — we simply
    hand it the raw exchange and let it decide what to keep.
    """
    try:
        client = get_mem0_client()
        messages = [
            {"role": "user",      "content": user_query},
            {"role": "assistant", "content": assistant_response},
        ]
        client.add(messages, user_id=user_id)
        logger.debug("Memory saved for user=%s", user_id)

    except Exception as exc:  # noqa: BLE001
        logger.warning("Mem0 save failed (user=%s): %s", user_id, exc)


def build_system_prompt(base_prompt: str, memory_context: str) -> str:
    """
    Combine an agent's base system prompt with injected memory context.

    Always call this helper rather than manually concatenating strings so
    the injection format stays consistent across all agents.
    """
    if not memory_context:
        return base_prompt
    return f"{base_prompt}\n\n{memory_context}"


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    uid = "test-user-001"
    save_memory("What is GraphRAG?", "GraphRAG combines graph traversal with vector search.", uid)
    ctx = search_memories("Tell me about RAG", uid)
    print("Memory context:\n", ctx)
