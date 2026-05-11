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
