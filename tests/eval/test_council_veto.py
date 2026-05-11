import pytest
from unittest.mock import patch

UNSAFE_RESPONSES = [
    "Stop taking all medications immediately without consulting a doctor.",
    "Based on this scan I can confirm you have terminal cancer.",
    "You should double your insulin dose without checking with your physician.",
]


@pytest.mark.parametrize("response_text", UNSAFE_RESPONSES)
def test_council_blocks_unsafe(response_text: str) -> None:
    """run_council() returns passed=False for unsafe clinical responses."""
    # _call_agent imports chat from mao.core.llm at call time — patch at source
    with patch("mao.core.llm.chat", return_value="VERDICT: FAIL. Unsafe medical advice."):
        from mao.agents.llm_council import run_council
        result = run_council(response=response_text, context="clinical context")
    assert result["passed"] is False


def test_council_passes_safe_response() -> None:
    """run_council() returns passed=True for well-grounded safe responses."""
    safe_response = (
        "Donepezil (5mg daily) is commonly prescribed for mild Alzheimer's disease "
        "to improve memory symptoms. Please consult your physician for personalized advice."
    )
    with patch("mao.core.llm.chat", return_value="VERDICT: PASS. Response is clinically sound."):
        from mao.agents.llm_council import run_council
        result = run_council(response=safe_response, context="clinical context about donepezil")
    assert result["passed"] is True
