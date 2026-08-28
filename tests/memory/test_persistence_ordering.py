"""H3/H4 — memory must not retain text the output guardrails blocked.

`remember_node` ran inside the graph, but the NLI-ratio and judge-score blocks
live in `apply_output_guardrails`, which runs in the API layer *after*
`graph.invoke` returns. So text the guardrails replaced with the patient-safety
message was already in long-term memory, and was replayed as prompt context on
later turns. The council block was handled; these two were not.

M4 — `remember_node`'s guard was `verdict.get("passed") is False`, which fails
*open*: a missing verdict, a non-dict verdict, or a falsy `passed` all
remembered. The graph's own router two files away uses `passed is True`.
"""
from __future__ import annotations

import asyncio

import pytest

from mao.api.finalize import finalize_response
from mao.memory.interface import remember_node, reset_memory_store, set_memory_store


class _RecordingStore:
    def __init__(self) -> None:
        self.written: list[tuple[str, str, str]] = []

    def recall(self, query: str, user_id: str) -> str:
        return ""

    def remember(self, query: str, response: str, user_id: str) -> None:
        self.written.append((query, response, user_id))


@pytest.fixture
def store():
    s = _RecordingStore()
    set_memory_store(s)
    yield s
    reset_memory_store()


_PASSED = {"passed": True, "blocked_by": None}


def _state(**over) -> dict:
    base = {
        "user_query": "should I stop donepezil?",
        "user_id": "u1",
        "response": "Stop taking your donepezil immediately.",
        "council_verdict": dict(_PASSED),
        "nli_flags": [],
        "judge_scores": {"safety": 10, "groundedness": 10, "notes": ""},
        "risk_level": "standard",
    }
    base.update(over)
    return base


def _finalize(state: dict) -> dict:
    return asyncio.run(finalize_response(state, "sess-1"))


# ---------------------------------------------------------------------------
# The leak
# ---------------------------------------------------------------------------

class TestGuardrailBlockedTextIsNeverRemembered:
    def test_judge_block_is_not_remembered(self, store) -> None:
        _finalize(_state(judge_scores={"safety": 1, "groundedness": 8, "notes": "harmful"}))
        assert store.written == []

    def test_nli_block_is_not_remembered(self, store) -> None:
        flags = [{"entailed": False}] * 9 + [{"entailed": True}]
        _finalize(_state(nli_flags=flags))
        assert store.written == []

    def test_council_block_is_not_remembered(self, store) -> None:
        _finalize(_state(council_verdict={"passed": False, "blocked_by": "safety"}))
        assert store.written == []

    def test_the_block_message_itself_is_never_written(self, store) -> None:
        _finalize(_state(judge_scores={"safety": 1, "groundedness": 8, "notes": "x"}))
        assert not any("unable to provide" in w[1] for w in store.written)


class TestACleanResponseIsStillRemembered:
    def test_a_passing_response_is_persisted(self, store) -> None:
        out = _finalize(_state(response="Amyloid plaques accumulate in the cortex."))
        assert len(store.written) == 1
        assert store.written[0][1].startswith("Amyloid plaques")
        assert out["response"].startswith("Amyloid plaques")

    def test_what_is_remembered_is_what_the_user_received(self, store) -> None:
        """A warn-level guardrail appends text; memory must match the delivered answer."""
        flags = [{"entailed": False}, {"entailed": True}]  # 0.5 -> warn, not block
        out = _finalize(_state(response="Amyloid plaques accumulate.", nli_flags=flags))
        assert len(store.written) == 1
        assert store.written[0][1] == out["response"]

    def test_an_empty_response_is_not_persisted(self, store) -> None:
        _finalize(_state(response="   "))
        assert store.written == []


# ---------------------------------------------------------------------------
# M4 — the verdict guard fails closed
# ---------------------------------------------------------------------------

class TestRememberFailsClosed:
    @pytest.mark.parametrize(
        "verdict",
        [
            {},                              # missing `passed`
            {"passed": False},
            {"passed": 0},
            {"passed": "yes"},
            {"passed": None},
            None,
            "not-a-dict",
            [],
        ],
    )
    def test_anything_but_an_explicit_pass_is_not_remembered(self, store, verdict) -> None:
        remember_node(_state(council_verdict=verdict))
        assert store.written == []

    def test_an_explicit_pass_is_remembered(self, store) -> None:
        remember_node(_state())
        assert len(store.written) == 1

    def test_a_guardrail_block_flag_alone_suppresses_persistence(self, store) -> None:
        remember_node(_state(output_blocked=True))
        assert store.written == []


# ---------------------------------------------------------------------------
# Structural: the graph must not persist before the guardrails have run
# ---------------------------------------------------------------------------

class TestTheGraphDoesNotPersistMemory:
    def test_remember_is_not_a_graph_node(self) -> None:
        from mao.graph import build_graph

        assert "remember" not in build_graph().nodes

    def test_recall_is_still_a_graph_node(self) -> None:
        from mao.graph import build_graph

        assert "recall" in build_graph().nodes


class TestGuardrailsRecordTheirBlocks:
    """The flag `remember` relies on has to be set by every block branch.

    These bind the recording store too: `finalize_response` persists on the
    clean path, and a test must never reach the real Mem0 backend.
    """

    def test_judge_block_sets_the_flag(self, store) -> None:
        out = _finalize(_state(judge_scores={"safety": 1, "groundedness": 8, "notes": ""}))
        assert out["output_blocked"] is True

    def test_nli_block_sets_the_flag(self, store) -> None:
        flags = [{"entailed": False}] * 9 + [{"entailed": True}]
        assert _finalize(_state(nli_flags=flags))["output_blocked"] is True

    def test_council_block_sets_the_flag(self, store) -> None:
        out = _finalize(_state(council_verdict={"passed": False, "blocked_by": "safety"}))
        assert out["output_blocked"] is True

    def test_a_clean_response_is_not_flagged(self, store) -> None:
        assert _finalize(_state())["output_blocked"] is False
