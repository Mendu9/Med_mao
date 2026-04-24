import pytest
from unittest.mock import patch, MagicMock
from mao.agents.llm_council import run_council, council_node


def test_council_passes_good_response():
    with patch("mao.agents.llm_council._call_agent") as mock_call:
        mock_call.side_effect = [
            "VERDICT: PASS. Clinical accuracy confirmed.",
            "VERDICT: PASS. No hallucinations detected.",
            "VERDICT: PASS. Response is safe and ethical.",
        ]
        result = run_council(response="Donepezil is first-line treatment.", context="...")
    assert result["passed"] is True


def test_council_blocks_on_safety_fail():
    with patch("mao.agents.llm_council._call_agent") as mock_call:
        mock_call.side_effect = [
            "VERDICT: PASS. Accurate.",
            "VERDICT: PASS. No hallucinations.",
            "VERDICT: FAIL. Response suggests harmful medication dose.",
        ]
        result = run_council(response="Take 100x the dose.", context="...")
    assert result["passed"] is False
    assert result["blocked_by"] == "safety"


def test_council_blocks_on_hallucination_fail():
    with patch("mao.agents.llm_council._call_agent") as mock_call:
        mock_call.side_effect = [
            "VERDICT: PASS. Accurate.",
            "VERDICT: FAIL. Claim not grounded in sources.",
            "VERDICT: PASS. Safe.",
        ]
        result = run_council(response="Some fabricated fact.", context="...")
    assert result["passed"] is False
    assert result["blocked_by"] == "hallucination"


def test_council_node_updates_state():
    state = {"response": "Donepezil helps.", "retrieved_docs": ["doc1"], "sub_queries": []}
    with patch("mao.agents.llm_council.run_council") as mock_council:
        mock_council.return_value = {"passed": True, "blocked_by": None, "accuracy": "VERDICT: PASS.", "hallucination": "VERDICT: PASS.", "safety": "VERDICT: PASS."}
        new_state = council_node(state)
    assert new_state["council_verdict"]["passed"] is True
