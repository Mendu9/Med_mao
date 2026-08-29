"""Memory must sit behind one interface, exercised once per request.

Architecture: "Memory must be behind one interface. Workflows should not
manually repeat search/save memory boilerplate."

Five agents each carried the same four-line recall block and their own
`save_memory` call. Besides the duplication, every agent saved its OWN draft —
the text produced *before* the supervision chain ran — so memory accumulated
unverified answers, including ones the council went on to block.
"""
from __future__ import annotations

from typing import ClassVar
import pytest

from mao.memory.interface import MemoryStore, get_memory_store, recall_node, remember_node


class RecordingStore:
    """In-memory stand-in; records what the graph asked it to do."""

    def __init__(self, recalled: str = "") -> None:
        self.recalled = recalled
        self.searches: list[tuple[str, str]] = []
        self.saved: list[tuple[str, str, str]] = []

    def recall(self, query: str, user_id: str) -> str:
        self.searches.append((query, user_id))
        return self.recalled

    def remember(self, query: str, response: str, user_id: str) -> None:
        self.saved.append((query, response, user_id))


@pytest.fixture
def store() -> RecordingStore:
    from mao.memory import interface

    recording = RecordingStore()
    interface.set_memory_store(recording)
    yield recording
    interface.reset_memory_store()


def _state(**kw) -> dict:
    base = {"user_query": "what is tau?", "user_id": "u1", "memory_context": "", "response": ""}
    base.update(kw)
    return base


class TestInterface:
    def test_recording_store_satisfies_the_protocol(self, store: RecordingStore) -> None:
        assert isinstance(store, MemoryStore)

    def test_default_store_is_available(self) -> None:
        assert get_memory_store() is not None


class TestRecallHappensOnce:
    def test_recall_populates_memory_context(self, store: RecordingStore) -> None:
        store.recalled = "User prefers concise answers"
        assert recall_node(_state())["memory_context"] == "User prefers concise answers"

    def test_recall_queries_the_store_exactly_once(self, store: RecordingStore) -> None:
        recall_node(_state())
        assert len(store.searches) == 1

    def test_recall_is_skipped_when_context_is_already_present(self, store: RecordingStore) -> None:
        recall_node(_state(memory_context="already here"))
        assert store.searches == []

    def test_recall_never_raises_when_the_store_fails(self) -> None:
        from mao.memory import interface

        class Broken:
            def recall(self, query, user_id):
                raise RuntimeError("memory down")

            def remember(self, query, response, user_id):
                pass

        interface.set_memory_store(Broken())
        try:
            assert recall_node(_state())["memory_context"] == ""
        finally:
            interface.reset_memory_store()


class TestRememberSavesTheVerifiedAnswer:
    def test_remember_persists_the_final_response(self, store: RecordingStore) -> None:
        """An affirmative council verdict is now required.

        The guard used to be `passed is False`, which failed *open*: a state with
        no verdict at all was remembered. It now matches the polarity the graph's
        own router uses (`passed is True`), so the verdict must be present.
        """
        remember_node(
            _state(
                response="Donepezil is first-line [1].",
                council_verdict={"passed": True, "blocked_by": None},
            )
        )
        assert store.saved == [("what is tau?", "Donepezil is first-line [1].", "u1")]

    def test_a_response_with_no_verdict_is_not_remembered(self, store: RecordingStore) -> None:
        remember_node(_state(response="Donepezil is first-line [1]."))
        assert store.saved == []

    def test_nothing_is_saved_when_there_is_no_response(self, store: RecordingStore) -> None:
        remember_node(_state(response=""))
        assert store.saved == []

    def test_a_blocked_response_is_not_remembered(self, store: RecordingStore) -> None:
        """The council blocked it; it must not become a remembered fact."""
        blocked = _state(
            response="I cannot provide this response.",
            council_verdict={"passed": False, "blocked_by": "safety"},
        )
        remember_node(blocked)
        assert store.saved == []

    def test_remember_never_raises_when_the_store_fails(self) -> None:
        from mao.memory import interface

        class Broken:
            def recall(self, query, user_id):
                return ""

            def remember(self, query, response, user_id):
                raise RuntimeError("memory down")

        interface.set_memory_store(Broken())
        try:
            remember_node(_state(response="text"))
        finally:
            interface.reset_memory_store()


class TestAgentsNoLongerRepeatTheBoilerplate:
    AGENTS: ClassVar[list[str]] = [
        "clinical_agent",
        "critic_agent",
        "graphrag_agent",
        "multimodal_agent",
        "summarizer_agent",
        "tool_agent",
    ]

    @pytest.mark.parametrize("name", AGENTS)
    def test_agent_does_not_call_search_memories(self, name: str) -> None:
        import importlib
        import inspect

        module = importlib.import_module(f"mao.agents.{name}")
        assert "search_memories(" not in inspect.getsource(module)

    @pytest.mark.parametrize("name", AGENTS)
    def test_agent_does_not_call_save_memory(self, name: str) -> None:
        import importlib
        import inspect

        module = importlib.import_module(f"mao.agents.{name}")
        assert "save_memory(" not in inspect.getsource(module)


class TestGraphWiring:
    def test_recall_is_a_graph_node(self) -> None:
        from mao.graph import build_graph

        assert "recall" in set(build_graph().nodes)

    def test_remember_is_deliberately_not_a_graph_node(self) -> None:
        """Persistence sits behind the output guardrails, not inside the graph.

        `remember` was a graph node running after `senior_supervisor`. That is
        after the *council*, but the NLI-ratio and judge-score blocks live in
        `apply_output_guardrails`, which runs in the API layer after
        `graph.invoke` returns — so guardrail-blocked clinical text was written
        to long-term memory and replayed as context on later turns. The ordering
        now lives in `mao/api/finalize.py`.
        """
        from mao.graph import build_graph

        assert "remember" not in set(build_graph().nodes)

    def test_no_agent_writes_memory_directly(self) -> None:
        from mao.graph import build_graph

        edges = build_graph().get_graph().edges
        assert not {e.target for e in edges if e.target == "remember"}

    def test_finalize_applies_the_guardrails_before_persisting(self) -> None:
        import inspect

        from mao.api import finalize

        source = inspect.getsource(finalize.finalize_response)
        assert source.index("apply_output_guardrails") < source.index("remember_node")
