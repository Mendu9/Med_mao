import asyncio
import pytest
from unittest.mock import patch, AsyncMock
from mao.agents.llm_council import run_council, council_node


def test_council_passes_good_response():
    async def _fake_council(response, context):
        return {
            "accuracy":      "VERDICT: PASS. Clinical accuracy confirmed.",
            "hallucination": "VERDICT: PASS. No hallucinations detected.",
            "safety":        "VERDICT: PASS. Response is safe and ethical.",
            "passed": True,
            "blocked_by": None,
        }
    with patch("mao.agents.llm_council.run_council_async", side_effect=_fake_council):
        result = run_council(response="Donepezil is first-line treatment.", context="...")
    assert result["passed"] is True


def test_council_blocks_on_safety_fail():
    async def _fake_council(response, context):
        return {
            "accuracy":      "VERDICT: PASS. Accurate.",
            "hallucination": "VERDICT: PASS. No hallucinations.",
            "safety":        "VERDICT: FAIL. Response suggests harmful medication dose.",
            "passed": False,
            "blocked_by": "safety",
        }
    with patch("mao.agents.llm_council.run_council_async", side_effect=_fake_council):
        result = run_council(response="Take 100x the dose.", context="...")
    assert result["passed"] is False
    assert result["blocked_by"] == "safety"


def test_council_blocks_on_hallucination_fail():
    async def _fake_council(response, context):
        return {
            "accuracy":      "VERDICT: PASS. Accurate.",
            "hallucination": "VERDICT: FAIL. Claim not grounded in sources.",
            "safety":        "VERDICT: PASS. Safe.",
            "passed": False,
            "blocked_by": "hallucination",
        }
    with patch("mao.agents.llm_council.run_council_async", side_effect=_fake_council):
        result = run_council(response="Some fabricated fact.", context="...")
    assert result["passed"] is False
    assert result["blocked_by"] == "hallucination"


def test_council_node_updates_state():
    state = {"response": "Donepezil helps.", "retrieved_docs": ["doc1"], "sub_queries": []}
    with patch("mao.agents.llm_council.run_council") as mock_council:
        mock_council.return_value = {
            "passed": True, "blocked_by": None,
            "accuracy": "VERDICT: PASS.",
            "hallucination": "VERDICT: PASS.",
            "safety": "VERDICT: PASS.",
        }
        new_state = council_node(state)
    assert new_state["council_verdict"]["passed"] is True
