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
    """Persist an exchange that cleared every safety control.

    This runs *after* the output guardrails, not inside the graph. The graph can
    only see the council's verdict; the NLI-ratio and judge-score blocks live in
    `apply_output_guardrails`, which runs in the API layer after `graph.invoke`
    returns. Persisting from inside the graph therefore wrote text the guardrails
    went on to withdraw, and memory is replayed as prompt context on later turns,
    so rejected clinical content came back around.

    Both guards fail closed. The verdict guard used to be `passed is False`,
    which meant a missing verdict, a non-dict verdict, or a falsy `passed` were
    all remembered — the opposite polarity to the graph's own router.
    """
    response = state.get("response") or ""
    if not response.strip():
        return state

    if state.get("output_blocked"):
        logger.debug(
            "Not remembering a guardrail-blocked response (blocked_by=%s)",
            state.get("output_blocked_by"),
        )
        return state

    verdict = state.get("council_verdict")
    passed = verdict.get("passed") if isinstance(verdict, dict) else None
    if passed is not True:
        logger.debug("Not remembering a response without an affirmative council verdict")
        return state

    try:
        get_memory_store().remember(
            state.get("user_query", ""), response, state.get("user_id", "")
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Memory persist failed (non-fatal): %s", exc)
    return state
