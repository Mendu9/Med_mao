"""Every eligible request must emit a trace.

Phase 1 establishes capture only. Nothing here may feed back into serving, and
a failure to capture must never break a user-facing request.
"""
from __future__ import annotations

from mao.api.tracing import build_trace, emit_trace
from mao.learning.experience_store import InMemoryExperienceStore
from mao.schemas.trace import TraceSchema


def _result(**kw) -> dict:
    base = {
        "intent": "graphrag",
        "agent_used": "graphrag",
        "risk_level": "standard",
        "response": "an answer",
        "council_verdict": {"passed": True, "blocked_by": None},
        "nli_flags": [],
        "judge_scores": {"safety": 9, "groundedness": 8},
        "metadata": {"top_scores": [0.81], "chunks_retrieved": 5, "sources": [{"chunk_id": "c1"}]},
    }
    base.update(kw)
    return base


class TestTraceConstruction:
    def test_builds_a_trace_schema(self) -> None:
        assert isinstance(build_trace(trace_id="t1", result=_result(), latency_ms=42.0), TraceSchema)

    def test_records_the_identifiers(self) -> None:
        trace = build_trace(trace_id="t1", result=_result(), latency_ms=42.0)
        assert trace.trace_id == "t1"
        assert trace.workflow == "graphrag"
        assert trace.risk_level == "standard"
        assert trace.latency_ms == 42.0

    def test_records_policy_version(self) -> None:
        from mao.safety.policy import get_policy

        trace = build_trace(trace_id="t1", result=_result(), latency_ms=1.0)
        assert trace.policy_version == get_policy().policy_version

    def test_records_the_resolved_model_and_provider(self) -> None:
        trace = build_trace(trace_id="t1", result=_result(), latency_ms=1.0)
        assert trace.model_id
        assert trace.provider
        assert trace.model_role

    def test_records_retrieval_when_present(self) -> None:
        trace = build_trace(trace_id="t1", result=_result(), latency_ms=1.0)
        assert trace.retrieval is not None
        assert trace.retrieval.hits == 5
        assert trace.retrieval.top_score == 0.81

    def test_omits_retrieval_when_nothing_was_retrieved(self) -> None:
        trace = build_trace(trace_id="t1", result=_result(metadata={}), latency_ms=1.0)
        assert trace.retrieval is None

    def test_quality_scores_carry_judge_output(self) -> None:
        trace = build_trace(trace_id="t1", result=_result(), latency_ms=1.0)
        assert trace.quality_scores["safety"] == 9


class TestSafetyFlags:
    def test_council_veto_is_flagged(self) -> None:
        result = _result(council_verdict={"passed": False, "blocked_by": "safety"})
        assert "council_blocked_safety" in build_trace(
            trace_id="t", result=result, latency_ms=1.0
        ).safety_flags

    def test_unentailed_claims_are_flagged(self) -> None:
        result = _result(nli_flags=[{"entailed": False}, {"entailed": True}])
        assert "nli_unentailed" in build_trace(
            trace_id="t", result=result, latency_ms=1.0
        ).safety_flags

    def test_a_clean_response_carries_no_flags(self) -> None:
        assert build_trace(trace_id="t", result=_result(), latency_ms=1.0).safety_flags == []

    # `test_uncertainty_flag_is_recorded` was here. It asserted that
    # `metadata["uncertainty_flag"]` raised an `mri_low_confidence` safety flag.
    # That flag is gone with the MRI workflow (M-3): its only producer was the
    # clinical agent's confidence gate, so it named a cause that can no longer
    # occur. `uncertainty_flag` survives as a field for a later control to set,
    # and that control will want a flag named after what it actually measures.

    def test_trace_does_not_carry_attachment_payloads(self) -> None:
        result = _result(metadata={"image_b64": "PATIENTSCAN", "top_scores": [0.5]})
        trace = build_trace(trace_id="t", result=result, latency_ms=1.0)
        assert "PATIENTSCAN" not in str(trace.to_dict())


class TestEmission:
    def test_emitted_trace_reaches_the_store(self) -> None:
        store = InMemoryExperienceStore()
        emit_trace(trace_id="t1", result=_result(), latency_ms=1.0, store=store)
        assert [t.trace_id for t in store.recent()] == ["t1"]

    def test_emission_never_raises_on_a_malformed_result(self) -> None:
        store = InMemoryExperienceStore()
        emit_trace(trace_id="t1", result={"metadata": None}, latency_ms=1.0, store=store)

    def test_emission_never_raises_on_a_broken_store(self) -> None:
        class Broken(InMemoryExperienceStore):
            def _write(self, trace):
                raise RuntimeError("sink down")

        emit_trace(trace_id="t1", result=_result(), latency_ms=1.0, store=Broken())
