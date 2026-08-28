"""C2 — the council must require an affirmative PASS, not the absence of "FAIL".

The audited defect: `run_council_async` adjudicated with `"FAIL" in verdict`.
An empty completion, a provider refusal, hedging prose, or even
"VERDICT: UNSAFE" all passed. `mao/safety/verification.py` justifies its LLM
judge failing *open* on the grounds that the council is the fail-closed
backstop — so a degraded (not crashed) provider made both controls fail open at
once.

These tests drive the REAL parser. The pre-existing council test patched
`run_council_async` wholesale, which is precisely why 650 tests missed this.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mao.agents import llm_council
from mao.agents.llm_council import parse_member_verdict, run_council_async


class _StubCompletion:
    def __init__(self, text: str) -> None:
        self.text = text


@pytest.fixture
def council_says(monkeypatch: pytest.MonkeyPatch):
    """Bind every council member to a fixed provider reply. No network."""

    def _bind(reply: str) -> None:
        def _complete(**kwargs: Any) -> _StubCompletion:
            return _StubCompletion(reply)

        monkeypatch.setattr(llm_council.gateway, "complete", _complete)

    return _bind


# ---------------------------------------------------------------------------
# The parser itself
# ---------------------------------------------------------------------------

class TestOnlyAnAffirmativePassPasses:
    @pytest.mark.parametrize(
        "text",
        [
            "VERDICT: PASS",
            "VERDICT: PASS. The response is safe and appropriately hedged.",
            "verdict: pass — nothing harmful here",
            "VERDICT:PASS",
            "VERDICT - PASS. Fine.",
        ],
    )
    def test_explicit_pass_is_a_pass(self, text: str) -> None:
        assert parse_member_verdict(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "VERDICT: FAIL. Unsafe dosing advice.",
            "VERDICT: UNSAFE. Do not release this to a patient.",
            "I'm sorry, I can't help with that request.",
            "This response seems mostly reasonable to me.",
            "The model declined to answer.",
            "PASS",  # no VERDICT token — not the declared contract
            "I would pass this, but I am not certain.",
            "ERROR: rate limit exceeded",
        ],
    )
    def test_anything_else_is_a_fail(self, text: str) -> None:
        assert parse_member_verdict(text) is False

    @pytest.mark.parametrize("value", [None, 0, [], {}, 3.14])
    def test_non_text_is_a_fail(self, value: object) -> None:
        assert parse_member_verdict(value) is False

    def test_a_fail_anywhere_beats_a_pass(self) -> None:
        """A judge that says both must not be read as approving."""
        assert parse_member_verdict("VERDICT: FAIL. It would not VERDICT: PASS review.") is False


# ---------------------------------------------------------------------------
# End-to-end through the real council, with only the provider stubbed
# ---------------------------------------------------------------------------

def _run(response: str, context: str) -> dict:
    return asyncio.run(run_council_async(response=response, context=context))


class TestDegradedProviderFailsClosed:
    def test_empty_completion_blocks(self, council_says) -> None:
        council_says("")
        verdict = _run("Take 900mg of donepezil daily.", "ctx")
        assert verdict["passed"] is False
        assert verdict["blocked_by"] == "safety"

    def test_refusal_prose_blocks(self, council_says) -> None:
        council_says("I'm sorry, I can't help with that.")
        assert _run("Take 900mg of donepezil daily.", "ctx")["passed"] is False

    def test_unsafe_wording_without_the_fail_token_blocks(self, council_says) -> None:
        council_says("VERDICT: UNSAFE. Do not release this.")
        assert _run("Take 900mg of donepezil daily.", "ctx")["passed"] is False

    def test_hedging_blocks(self, council_says) -> None:
        council_says("It seems broadly fine, though I am not sure.")
        assert _run("Amyloid plaques accumulate.", "ctx")["passed"] is False


class TestAGenuinePassStillPasses:
    def test_unanimous_pass_proceeds(self, council_says) -> None:
        council_says("VERDICT: PASS. Accurate and safe.")
        verdict = _run("Amyloid plaques accumulate in the cortex.", "ctx")
        assert verdict["passed"] is True
        assert verdict["blocked_by"] is None

    def test_explicit_fail_blocks_and_names_the_member(self, council_says) -> None:
        council_says("VERDICT: FAIL. Unsafe dosing advice.")
        verdict = _run("Take 900mg of donepezil daily.", "ctx")
        assert verdict["passed"] is False
        assert verdict["blocked_by"] == "safety"


class TestNarrowingStillWorks:
    def test_empty_context_still_runs_the_safety_member(self, council_says) -> None:
        council_says("VERDICT: PASS. Safe.")
        verdict = _run("Amyloid plaques accumulate.", "")
        assert verdict["members"] == ["safety"]
        assert verdict["passed"] is True

    def test_empty_context_can_still_block(self, council_says) -> None:
        council_says("VERDICT: FAIL. Dangerous.")
        assert _run("Stop taking your medication.", "")["passed"] is False
