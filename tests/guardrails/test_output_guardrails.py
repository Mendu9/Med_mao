import pytest
from unittest.mock import patch

_BLOCK_MSG = (
    "I'm unable to provide this response as it may not be safe for patient care. "
    "Please consult a qualified healthcare professional."
)


@pytest.mark.asyncio
async def test_nli_warning_appended() -> None:
    state = {
        "response": "Donepezil cures Alzheimer's disease.",
        "nli_flags": [{"entailed": False}, {"entailed": False}, {"entailed": True}],
        "council_verdict": {"passed": True, "blocked_by": None},
        "judge_scores": {"safety": 9},
    }
    with patch("mao.guardrails.output_guardrails.log_guardrail_event"):
        from mao.guardrails.output_guardrails import apply_output_guardrails
        result = await apply_output_guardrails(state, "sess-1")
    assert "⚠️" in result["response"]
    assert "could not be fully verified" in result["response"]


@pytest.mark.asyncio
async def test_council_safety_blocks() -> None:
    state = {
        "response": "Stop taking all medications now.",
        "nli_flags": [],
        "council_verdict": {"passed": False, "blocked_by": "safety"},
        "judge_scores": {"safety": 9},
    }
    with patch("mao.guardrails.output_guardrails.log_guardrail_event"):
        from mao.guardrails.output_guardrails import apply_output_guardrails
        result = await apply_output_guardrails(state, "sess-2")
    assert result["response"] == _BLOCK_MSG


@pytest.mark.asyncio
async def test_judge_safety_blocks() -> None:
    state = {
        "response": "Confirm terminal diagnosis from scan.",
        "nli_flags": [],
        "council_verdict": {"passed": True, "blocked_by": None},
        "judge_scores": {"safety": 4, "notes": "dangerous advice"},
    }
    with patch("mao.guardrails.output_guardrails.log_guardrail_event"):
        from mao.guardrails.output_guardrails import apply_output_guardrails
        result = await apply_output_guardrails(state, "sess-3")
    assert result["response"] == _BLOCK_MSG


@pytest.mark.asyncio
async def test_clean_state_passes_unchanged() -> None:
    original = "Donepezil is recommended for mild AD. Consult your doctor."
    state = {
        "response": original,
        "nli_flags": [{"entailed": True}, {"entailed": True}],
        "council_verdict": {"passed": True, "blocked_by": None},
        "judge_scores": {"safety": 9},
    }
    with patch("mao.guardrails.output_guardrails.log_guardrail_event"):
        from mao.guardrails.output_guardrails import apply_output_guardrails
        result = await apply_output_guardrails(state, "sess-4")
    assert result["response"] == original


# ---------------------------------------------------------------------------
# P0-2 — the clinical disclaimer must survive to the user
#
# clinical_agent appends it; domain_supervisor used to destroy it. This is the
# second line of defence: if a HIGH-risk answer reaches the guardrails without
# it, the guardrails put it back.
# ---------------------------------------------------------------------------

def _clean_state(**overrides) -> dict:
    state = {
        "response": "Donepezil is recommended for mild AD.",
        "nli_flags": [],
        "council_verdict": {"passed": True, "blocked_by": None},
        "judge_scores": {"safety": 9},
    }
    state.update(overrides)
    return state


async def _apply(state: dict, session: str = "sess-disc") -> dict:
    with patch("mao.guardrails.output_guardrails.log_guardrail_event"):
        from mao.guardrails.output_guardrails import apply_output_guardrails
        return await apply_output_guardrails(state, session)


@pytest.mark.asyncio
async def test_missing_disclaimer_is_reasserted_for_high_risk() -> None:
    from mao.safety.verification import DISCLAIMER_MARKER

    result = await _apply(_clean_state(risk_level="high"))

    assert DISCLAIMER_MARKER in result["response"]


@pytest.mark.asyncio
async def test_existing_disclaimer_is_not_duplicated() -> None:
    from mao.safety.verification import CLINICAL_DISCLAIMER, DISCLAIMER_MARKER

    answer = "Donepezil is recommended for mild AD." + CLINICAL_DISCLAIMER
    result = await _apply(_clean_state(response=answer, risk_level="high"))

    assert result["response"].count(DISCLAIMER_MARKER) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("risk", ["standard", "low"])
async def test_non_high_risk_does_not_gain_a_clinical_disclaimer(risk: str) -> None:
    from mao.safety.verification import DISCLAIMER_MARKER

    original = "The capital of France is Paris."
    result = await _apply(_clean_state(response=original, risk_level=risk))

    assert DISCLAIMER_MARKER not in result["response"]
    assert result["response"] == original


@pytest.mark.asyncio
async def test_risk_absent_is_derived_from_intent() -> None:
    """No risk_level on state — the policy classifies clinical intent as HIGH."""
    from mao.safety.verification import DISCLAIMER_MARKER

    result = await _apply(_clean_state(intent="clinical"))

    assert DISCLAIMER_MARKER in result["response"]


@pytest.mark.asyncio
async def test_blocked_response_is_not_dressed_up_with_a_disclaimer() -> None:
    """A refusal is not clinical content; it must stand alone."""
    state = _clean_state(
        response="Stop taking all medications now.",
        council_verdict={"passed": False, "blocked_by": "safety"},
        risk_level="high",
    )
    result = await _apply(state)

    assert result["response"] == _BLOCK_MSG


@pytest.mark.asyncio
async def test_disclaimer_survives_alongside_an_nli_warning() -> None:
    """Both controls apply; neither erases the other."""
    from mao.safety.verification import DISCLAIMER_MARKER

    state = _clean_state(
        risk_level="high",
        nli_flags=[{"entailed": False}, {"entailed": False}, {"entailed": True}],
    )
    result = await _apply(state)

    assert "could not be fully verified" in result["response"]
    assert DISCLAIMER_MARKER in result["response"]


# ---------------------------------------------------------------------------
# Thresholds come from the shared policy, not from local copies
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_judge_block_threshold_matches_the_policy() -> None:
    from mao.safety.policy import get_policy

    policy = get_policy()
    just_below = await _apply(_clean_state(judge_scores={"safety": policy.judge_block_score - 1}))
    assert just_below["response"] == _BLOCK_MSG

    at_threshold = await _apply(_clean_state(judge_scores={"safety": policy.judge_block_score}))
    assert at_threshold["response"] != _BLOCK_MSG
