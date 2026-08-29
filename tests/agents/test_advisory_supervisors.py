"""The advisory supervisors must not compete with the safety controls for capacity.

Found during Wave 7 while measuring live. The safety chain makes six
SAFETY_JUDGE calls per clinical request. Provider limits are per model, so all
six draw on one quota — and two of them are `domain_supervisor` and
`senior_supervisor`, neither of which is a safety adjudicator:

  - `domain_supervisor_node` writes `ungrounded_claims`. Nothing reads it.
    P0-2 deliberately stopped it editing the response, so it observes only.
  - `senior_supervisor_node` writes `completeness_ok` and `missing_sub_queries`.
    Nothing reads those either.

Verified by grep across `mao/`: outside the two nodes that write them and the
state schema that declares them, there are no consumers. So a third of the
safety model's per-request token budget was being spent on signals that could
not change any decision — starving the council and the judge, which do.

Moving them off the safety model is not a loosened control. It cannot change a
safety decision, because these outputs gate nothing. It stops a third of that
model's per-request spend going on signals no code reads, and leaves that quota
to the council and the judge, which do gate what a clinician sees.

They keep running, and their results now reach the trace, so they become an
observable quality signal rather than a discarded one.
"""
from __future__ import annotations

import pytest

from mao.providers.registry import ModelRole


class _Recorder:
    name = "recorder"

    def __init__(self) -> None:
        self.roles: list[str] = []

    def complete(self, *, model_id, messages, temperature, max_tokens):
        from mao.providers.llm.base import ProviderResponse

        self.roles.append(model_id)
        return ProviderResponse(text='{"ungrounded_claims": [], "missing": []}')


@pytest.fixture
def recorder():
    from mao.providers import gateway

    rec = _Recorder()
    gateway.set_provider(rec)
    yield rec
    gateway.reset_provider()


class TestTheAdvisorySupervisorsAreNotOnTheSafetyModel:
    def test_domain_supervisor_does_not_use_the_safety_judge_role(self) -> None:
        from mao.agents.domain_supervisor import _ROLE

        assert _ROLE is not ModelRole.SAFETY_JUDGE

    def test_senior_supervisor_does_not_use_the_safety_judge_role(self) -> None:
        from mao.agents.senior_supervisor import _ROLE

        assert _ROLE is not ModelRole.SAFETY_JUDGE

    def test_the_council_still_uses_the_safety_judge_role(self) -> None:
        """The reallocation must not touch the controls that actually block."""
        import inspect

        from mao.agents import llm_council

        assert "ModelRole.SAFETY_JUDGE" in inspect.getsource(llm_council._async_call_agent)

    def test_the_judge_still_uses_the_safety_judge_role(self) -> None:
        import inspect

        from mao.safety import verification

        assert "ModelRole.SAFETY_JUDGE" in inspect.getsource(verification._run_judge)

    def test_they_resolve_to_a_different_model_than_the_judge(self) -> None:
        """Different model means a different rate-limit bucket."""
        from mao.agents.domain_supervisor import _ROLE as ds_role
        from mao.providers import gateway

        assert gateway.model_id_for(ds_role) != gateway.model_id_for(
            ModelRole.SAFETY_JUDGE
        )


class TestTheyStillRunAndStillReport:
    def test_domain_supervisor_still_produces_its_signal(self, recorder) -> None:
        from mao.agents.domain_supervisor import domain_supervisor_node

        out = domain_supervisor_node({"response": "An answer.", "retrieved_docs": []})
        assert "ungrounded_claims" in out
        assert recorder.roles, "the supervisor stopped calling the model entirely"

    def test_senior_supervisor_still_produces_its_signal(self, recorder) -> None:
        from mao.agents.senior_supervisor import senior_supervisor_node

        out = senior_supervisor_node(
            {"response": "An answer.", "sub_queries": ["q1"]}
        )
        assert "completeness_ok" in out


class TestTheirOutputReachesTheTrace:
    """They were written and read by nothing. A discarded signal is waste."""

    def test_ungrounded_claims_reach_the_trace(self) -> None:
        from mao.api.tracing import build_trace

        trace = build_trace(
            trace_id="t",
            result={"intent": "graphrag", "ungrounded_claims": ["a claim"]},
            latency_ms=1.0,
        )
        assert "ungrounded_claims" in trace.safety_flags

    def test_incompleteness_reaches_the_trace(self) -> None:
        from mao.api.tracing import build_trace

        trace = build_trace(
            trace_id="t",
            result={
                "intent": "graphrag",
                "completeness_ok": False,
                "missing_sub_queries": ["q2"],
            },
            latency_ms=1.0,
        )
        assert "incomplete_answer" in trace.safety_flags

    def test_a_clean_answer_carries_neither_flag(self) -> None:
        from mao.api.tracing import build_trace

        trace = build_trace(
            trace_id="t",
            result={
                "intent": "graphrag",
                "ungrounded_claims": [],
                "completeness_ok": True,
            },
            latency_ms=1.0,
        )
        assert "ungrounded_claims" not in trace.safety_flags
        assert "incomplete_answer" not in trace.safety_flags
