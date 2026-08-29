"""What `graph.invoke` actually RETURNS — the boundary Wave 6 blocker 2 lived at.

`verification_node` computed `judge_scores` correctly and returned it. `MAOState`
did not declare the channel, so LangGraph discarded it at the node boundary and
`apply_output_guardrails` always read its `safety` default of 10. The judge's
BLOCK (<5) and WARN (<7) branches were unreachable in production: a judge
returning `{"safety": 0, "notes": "would kill the patient"}` shipped the answer.

Every existing judge test called `verification_node` directly, which is exactly
why none of them could see it. The assertion that would have caught it is an
assertion on the *result of the graph*, so that is what this module makes.

The same schema gap silenced `verification_trace`, which is where `prompt_ref`
lives — so the trace field the architecture requires for prompt provenance was
empty on every trace ever emitted.
"""
from __future__ import annotations

import pytest

from mao.core.state import MAOState


# ---------------------------------------------------------------------------
# The schema itself
# ---------------------------------------------------------------------------

class TestEverySafetyChannelIsDeclared:
    """An undeclared channel is silently dropped, not an error. Declare them."""

    @pytest.mark.parametrize(
        "channel",
        ["judge_scores", "verification_trace", "llm_usage", "output_blocked"],
    )
    def test_channel_is_declared_on_the_state_schema(self, channel: str) -> None:
        assert channel in MAOState.__annotations__, (
            f"{channel} is written by a node but not declared on MAOState, "
            "so LangGraph will discard it at the node boundary"
        )


# ---------------------------------------------------------------------------
# Survival through the real graph
# ---------------------------------------------------------------------------

class TestJudgeScoresSurviveTheGraph:
    def test_judge_scores_are_present_in_the_result(self, graph_harness) -> None:
        result = graph_harness.invoke()
        assert "judge_scores" in result, (
            "verification_node returned judge_scores and the graph dropped it"
        )

    def test_the_scored_value_is_the_one_the_judge_returned(self, graph_harness) -> None:
        graph_harness.provider.replies["judge.safety"] = (
            '{"safety": 3, "groundedness": 9, "notes": "contraindicated"}'
        )
        result = graph_harness.invoke()
        assert result["judge_scores"]["safety"] == 3

    def test_verification_trace_survives_with_its_prompt_ref(self, graph_harness) -> None:
        result = graph_harness.invoke()
        trace = result.get("verification_trace") or {}
        assert trace.get("judge_ran") is True
        assert trace.get("prompt_ref"), "prompt_ref is empty on every trace"


# ---------------------------------------------------------------------------
# Reachability of the control, end to end
# ---------------------------------------------------------------------------

class TestTheJudgeCanActuallyBlock:
    """The BLOCK branch must be reachable from a real request, not just from a
    hand-built `judge_scores` dict."""

    def test_a_harmful_verdict_blocks_the_answer(self, graph_harness) -> None:
        graph_harness.provider.replies["judge.safety"] = (
            '{"safety": 0, "notes": "would kill the patient"}'
        )
        result = graph_harness.finalize(graph_harness.invoke())

        assert result["output_blocked"] is True
        assert result["output_blocked_by"] == "judge_safety"
        assert "unable to provide this response" in result["response"]

    def test_a_moderate_verdict_warns_rather_than_blocks(self, graph_harness) -> None:
        graph_harness.provider.replies["judge.safety"] = (
            '{"safety": 6, "groundedness": 9, "notes": "verify dosing"}'
        )
        result = graph_harness.finalize(graph_harness.invoke())

        assert result["output_blocked"] is False
        assert "moderate confidence" in result["response"]

    def test_an_unreadable_verdict_fails_closed_through_the_graph(
        self, graph_harness
    ) -> None:
        """H2's fail-closed rework, exercised where it was previously severed."""
        graph_harness.provider.replies["judge.safety"] = "I am unable to comply."
        result = graph_harness.finalize(graph_harness.invoke())

        assert result["output_blocked"] is True
        assert result["output_blocked_by"] == "judge_safety"

    def test_a_safe_verdict_still_delivers_the_answer(self, graph_harness) -> None:
        result = graph_harness.finalize(graph_harness.invoke())

        assert result["output_blocked"] is False
        assert "acetylcholinesterase" in result["response"]

    def test_blocked_text_never_reaches_memory(self, graph_harness) -> None:
        graph_harness.provider.replies["judge.safety"] = '{"safety": 0, "notes": "harm"}'
        graph_harness.finalize(graph_harness.invoke())

        assert getattr(graph_harness.memory, "written", []) == []
