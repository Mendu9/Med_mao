"""Contract tests for TraceSchema and the ExperienceStore.

Phase 1 must add the trace fields later RL work depends on, without training
anything now.
"""
from __future__ import annotations

import pytest

from mao.learning.experience_store import ExperienceStore, InMemoryExperienceStore
from mao.schemas.trace import RetrievalTrace, ToolCallTrace, TraceSchema


def _trace(**kw) -> TraceSchema:
    base = dict(
        trace_id="t-1",
        workflow="evidence_qa",
        risk_level="standard",
        provider="groq",
        model_id="llama-3.1-8b-instant",
        model_role="GENERAL_SYNTHESIS",
        prompt_ref="graphrag.synthesis@1.0.0",
        policy_version="2026.08-1",
        latency_ms=120.5,
    )
    base.update(kw)
    return TraceSchema(**base)


class TestTraceSchemaFields:
    ARCHITECTURE_FIELDS = (
        "trace_id",
        "workflow",
        "risk_level",
        "provider",
        "model_id",
        "prompt_ref",
        "tool_calls",
        "retrieval",
        "policy_version",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "quality_scores",
        "safety_flags",
    )

    @pytest.mark.parametrize("field_name", ARCHITECTURE_FIELDS)
    def test_trace_carries_architecture_field(self, field_name: str) -> None:
        assert hasattr(_trace(), field_name)

    def test_trace_round_trips_through_dict(self) -> None:
        trace = _trace()
        assert TraceSchema.from_dict(trace.to_dict()) == trace

    def test_tool_calls_default_to_empty_not_shared(self) -> None:
        a, b = _trace(trace_id="a"), _trace(trace_id="b")
        a.tool_calls.append(ToolCallTrace(tool_id="pubmed", latency_ms=1.0, ok=True))
        assert b.tool_calls == []

    def test_retrieval_trace_records_index_version(self) -> None:
        trace = _trace(retrieval=RetrievalTrace(index_version="v3", k=10, hits=7))
        assert trace.retrieval is not None
        assert trace.retrieval.index_version == "v3"

    def test_safety_flags_are_recorded(self) -> None:
        trace = _trace(safety_flags=["council_safety_veto"])
        assert "council_safety_veto" in trace.safety_flags


class TestExperienceStore:
    def test_in_memory_store_satisfies_the_protocol(self) -> None:
        assert isinstance(InMemoryExperienceStore(), ExperienceStore)

    def test_appended_trace_is_retrievable(self) -> None:
        store = InMemoryExperienceStore()
        store.append(_trace(trace_id="x-1"))
        assert [t.trace_id for t in store.recent(limit=10)] == ["x-1"]

    def test_recent_returns_newest_first(self) -> None:
        store = InMemoryExperienceStore()
        store.append(_trace(trace_id="old"))
        store.append(_trace(trace_id="new"))
        assert [t.trace_id for t in store.recent(limit=2)] == ["new", "old"]

    def test_recent_respects_limit(self) -> None:
        store = InMemoryExperienceStore()
        for i in range(5):
            store.append(_trace(trace_id=f"t{i}"))
        assert len(store.recent(limit=2)) == 2

    def test_append_never_raises_on_a_broken_sink(self) -> None:
        """Trace capture must never break a user-facing request."""

        class Broken(InMemoryExperienceStore):
            def _write(self, trace: TraceSchema) -> None:
                raise RuntimeError("sink down")

        Broken().append(_trace())  # must not raise

    def test_store_does_not_retain_raw_query_text_by_default(self) -> None:
        """Privacy: traces are for policy learning, not conversation retention."""
        assert not hasattr(_trace(), "user_query")
