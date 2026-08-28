"""The one memory interface.

Memory is a pipeline concern, not an agent concern. Recall happens once before
dispatch and persistence happens once after supervision, so:

  - agents no longer each carry the same four-line recall block and their own
    save call (the architecture forbids repeating this boilerplate); and
  - what gets remembered is the answer that survived verification, not the
    draft the agent produced. Previously every agent saved its own pre-council
    text, so memory accumulated unverified answers — including ones the council
    subsequently blocked.

Failures here are always non-fatal: a memory outage degrades personalisation,
it must never fail a clinical request.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class MemoryStore(Protocol):
    """Per-user conversational memory."""

    def recall(self, query: str, user_id: str) -> str: ...

    def remember(self, query: str, response: str, user_id: str) -> None: ...


class Mem0MemoryStore:
    """MemoryStore backed by Mem0."""

    def recall(self, query: str, user_id: str) -> str:
        from mao.memory.mem0_handler import search_memories

        return search_memories(query, user_id)

    def remember(self, query: str, response: str, user_id: str) -> None:
        from mao.memory.mem0_handler import save_memory

        save_memory(query, response, user_id)


_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    """The process-wide memory store."""
    global _store
    if _store is None:
        _store = Mem0MemoryStore()
    return _store


def set_memory_store(store: MemoryStore) -> None:
    """Bind a store. Test and deployment seam."""
    global _store
    _store = store


def reset_memory_store() -> None:
    """Restore the default store."""
    global _store
    _store = None


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def recall_node(state: dict) -> dict:
    """Populate `state["memory_context"]` once, before any agent runs."""
    if state.get("memory_context"):
        return state
    try:
        context = get_memory_store().recall(
            state.get("user_query", ""), state.get("user_id", "")
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Memory recall failed (non-fatal): %s", exc)
        context = ""
    return {**state, "memory_context": context}


def remember_node(state: dict) -> dict:
    """Persist the verified exchange, once, after the supervision chain.

    A response the council blocked is deliberately not remembered: replaying it
    as context on a later turn would reintroduce exactly the content the safety
    chain rejected.
    """
    response = state.get("response") or ""
    if not response.strip():
        return state

    verdict = state.get("council_verdict") or {}
    if isinstance(verdict, dict) and verdict.get("passed") is False:
        logger.debug("Not remembering a blocked response (blocked_by=%s)", verdict.get("blocked_by"))
        return state

    try:
        get_memory_store().remember(
            state.get("user_query", ""), response, state.get("user_id", "")
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Memory persist failed (non-fatal): %s", exc)
    return state
