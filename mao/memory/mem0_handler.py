"""Per-user persistent memory — stores and retrieves conversation history."""

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
    """Mem0 vector store: Qdrant when VECTOR_BACKEND=qdrant, ChromaDB otherwise."""
    if cfg.vector_backend == "qdrant" and cfg.qdrant_url and cfg.qdrant_api_key:
        vector_store: dict[str, Any] = {
            "provider": "qdrant",
            "config": {
                "url": cfg.qdrant_url,
                "api_key": cfg.qdrant_api_key,
                "collection_name": "mao_memory",
                "embedding_model_dims": cfg.embed_dim,
            },
        }
    else:
        vector_store = {
            "provider": "chroma",
            "config": {
                "host": cfg.chroma_host,
                "port": cfg.chroma_port,
                "collection_name": "mao_memory",
            },
        }
    return {
        "vector_store": vector_store,
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
_mem0_disabled: bool = False  # latched True after first failure to stop per-request timeouts


def get_mem0_client() -> Memory:
    """Return shared Mem0 Memory instance, initialising on first call.

    After any connection failure the flag is latched so subsequent requests
    skip the 8-second ChromaDB timeout and degrade to empty memory context.
    """
    import os
    global _mem0_client, _mem0_disabled
    if _mem0_disabled or os.getenv("MAO_DISABLE_MEM0", "").lower() in ("1", "true", "yes"):
        raise RuntimeError("Mem0 disabled (ChromaDB unavailable or MAO_DISABLE_MEM0=1)")
    if _mem0_client is None:
        logger.info("Initialising Mem0 client (chroma collection=mao_memory)")
        try:
            _mem0_client = Memory.from_config(_build_mem0_config())
        except Exception as exc:
            _mem0_disabled = True
            logger.warning(
                "Mem0 init failed — disabling for this process to avoid repeated timeouts: %s", exc
            )
            raise RuntimeError(f"Mem0 init failed: {exc}") from exc
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
        # mem0 >= 0.1.40 moved user_id to filters=; try new API, fall back to old
        try:
            results: list[dict[str, Any]] = client.search(
                query=query,
                filters={"user_id": user_id},
                limit=limit,
            )
        except TypeError:
            results = client.search(query=query, user_id=user_id, limit=limit)  # type: ignore[call-arg]
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
        # Truncate to stay within Groq free-tier TPM limits (llama-3.1-8b: 6000 TPM).
        # Mem0 only needs a summary of the exchange — full RAG responses are too long.
        truncated_response = assistant_response[:800] + ("…" if len(assistant_response) > 800 else "")
        messages = [
            {"role": "user",      "content": user_query[:400]},
            {"role": "assistant", "content": truncated_response},
        ]
        try:
            client.add(messages, user_id=user_id)
        except TypeError:
            client.add(messages, filters={"user_id": user_id})  # type: ignore[call-arg]
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
