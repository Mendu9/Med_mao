""""We could not review this" and "we reviewed this and it is unsafe" are
different facts, and the clinician was only ever told the second one.

Found during Wave 7 remediation while measuring live. `_async_call_agent` maps
every failure — provider 429, timeout, connection error — to the literal string
`"VERDICT: FAIL. Agent error."`. `parse_member_verdict` then reads it as a FAIL,
`run_council_async` sets `blocked_by` to the member's name, and
`blocked_response_node` renders:

    "I cannot provide this response. It was flagged by the safety review
     for patient safety. Please consult a licensed clinician directly."

So a rate limit produced a message asserting that a clinical answer had been
examined and found dangerous. It had not been examined at all.

Blocking is right and stays: an unreviewed clinical answer must not ship, and
`run_council` failing closed is one of the controls Wave 6 confirmed as
genuinely working. What is wrong is the *claim*. A clinician who is told the
content was judged unsafe will reason about the patient; one who is told the
review was unavailable will retry. Those lead to different actions, and only one
of them is supported by what actually happened.

This matters more than it looks on the current provider tier: the safety chain
makes six SAFETY_JUDGE calls per clinical request against an 8000 tokens/minute
allowance, so infrastructure refusals are not a rare edge — they were 4 of 5 in
a measured run.
"""
from __future__ import annotations

import asyncio

import pytest

from mao.agents import llm_council
from mao.agents.llm_council import REVIEW_UNAVAILABLE, run_council
from mao.graph import blocked_response_node


class _Boom:
    """A provider that fails the way a rate limit does."""

    name = "boom"

    def __init__(self, message: str = "Error code: 429 - rate_limit_exceeded") -> None:
        self.message = message

    def complete(self, **kwargs):
        raise RuntimeError(self.message)


@pytest.fixture
def failing_provider():
    from mao.providers import gateway

    gateway.set_provider(_Boom())
    yield
    gateway.reset_provider()


class TestAnInfrastructureFailureIsNotASafetyVerdict:
    def test_the_answer_is_still_withheld(self, failing_provider) -> None:
        """Fail-closed is correct and must not regress."""
        verdict = run_council(response="Donepezil raises acetylcholine.", context="ctx")
        assert verdict["passed"] is False

    def test_it_is_not_attributed_to_a_safety_finding(self, failing_provider) -> None:
        verdict = run_council(response="Donepezil raises acetylcholine.", context="ctx")
        assert verdict["blocked_by"] == REVIEW_UNAVAILABLE, (
            "a provider failure was reported as a safety veto"
        )

    def test_the_failing_members_are_named(self, failing_provider) -> None:
        verdict = run_council(response="Donepezil raises acetylcholine.", context="ctx")
        assert verdict.get("unavailable_members")

    def test_the_message_does_not_claim_the_content_was_judged(self) -> None:
        out = blocked_response_node(
            {"council_verdict": {"passed": False, "blocked_by": REVIEW_UNAVAILABLE}}
        )
        response = out["response"]
        assert "flagged" not in response.lower()
        assert "not safe" not in response.lower()

    def test_the_message_says_the_review_could_not_run(self) -> None:
        out = blocked_response_node(
            {"council_verdict": {"passed": False, "blocked_by": REVIEW_UNAVAILABLE}}
        )
        assert "could not be completed" in out["response"].lower()

    def test_the_message_tells_the_clinician_it_is_retryable(self) -> None:
        out = blocked_response_node(
            {"council_verdict": {"passed": False, "blocked_by": REVIEW_UNAVAILABLE}}
        )
        assert "again" in out["response"].lower()


class TestAGenuineSafetyVetoStillReadsAsOne:
    """The distinction is worthless if it blurs the real case."""

    def test_a_real_safety_fail_is_still_attributed_to_safety(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _reviewed(member, response, context):
            return member, "VERDICT: FAIL. Recommends an unsafe dose.", True

        monkeypatch.setattr(llm_council, "_async_call_agent", _reviewed)
        verdict = asyncio.run(llm_council.run_council_async("answer", "ctx"))

        assert verdict["passed"] is False
        assert verdict["blocked_by"] == "safety"

    def test_the_safety_message_still_names_the_review(self) -> None:
        out = blocked_response_node(
            {"council_verdict": {"passed": False, "blocked_by": "safety"}}
        )
        assert "flagged by the safety review" in out["response"]

    def test_a_unanimous_pass_still_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _passed(member, response, context):
            return member, "VERDICT: PASS. Fine.", True

        monkeypatch.setattr(llm_council, "_async_call_agent", _passed)
        verdict = asyncio.run(llm_council.run_council_async("answer", "ctx"))

        assert verdict["passed"] is True
        assert verdict["blocked_by"] is None

    def test_a_member_that_answered_outside_the_format_is_a_review_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It ANSWERED — we just could not read it. That is not infrastructure,
        and it must still fail closed as an unreadable review."""
        async def _garbled(member, response, context):
            return member, "I would rather not say.", True

        monkeypatch.setattr(llm_council, "_async_call_agent", _garbled)
        verdict = asyncio.run(llm_council.run_council_async("answer", "ctx"))

        assert verdict["passed"] is False
        assert verdict["blocked_by"] != REVIEW_UNAVAILABLE
