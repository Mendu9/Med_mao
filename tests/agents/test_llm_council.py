from unittest.mock import patch
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


# ---------------------------------------------------------------------------
# P0-1 — supervision must not depend on the transport
# ---------------------------------------------------------------------------

def test_streaming_request_with_an_answer_is_still_reviewed():
    """A streamed request carries the same patient-safety risk as a buffered one.
    Once there is a response to review, the council must review it."""
    state = {
        "response": "Stop all medication immediately.",
        "retrieved_docs": ["doc1"],
        "_want_stream": True,
    }
    with patch("mao.agents.llm_council.run_council") as mock_council:
        mock_council.return_value = {"passed": False, "blocked_by": "safety"}
        new_state = council_node(state)

    mock_council.assert_called_once()
    assert new_state["council_verdict"]["blocked_by"] == "safety"


def test_empty_response_skips_on_its_own_merits_not_because_of_streaming():
    """An empty response is nothing to verify — but the reason recorded must be
    the empty response, not the transport."""
    with patch("mao.agents.llm_council.run_council") as mock_council:
        new_state = council_node({"response": "", "retrieved_docs": ["doc1"]})

    mock_council.assert_not_called()
    verdict = new_state["council_verdict"]
    assert verdict["passed"] is True
    assert verdict.get("skipped") == "empty_response"


def test_empty_response_without_stream_flag_behaves_identically():
    """Same input, no streaming hint — identical verdict. Transport is not an input."""
    with patch("mao.agents.llm_council.run_council"):
        streamed = council_node({"response": "", "retrieved_docs": [], "_want_stream": True})
        buffered = council_node({"response": "", "retrieved_docs": []})
    assert streamed["council_verdict"] == buffered["council_verdict"]


# ---------------------------------------------------------------------------
# P1-3 — empty retrieval must not auto-pass
# ---------------------------------------------------------------------------

def test_council_node_runs_council_even_without_retrieved_docs():
    """No retrieved context used to return passed=True without asking anyone."""
    state = {"response": "Stop all medication immediately.", "retrieved_docs": []}
    with patch("mao.agents.llm_council.run_council") as mock_council:
        mock_council.return_value = {"passed": False, "blocked_by": "safety"}
        new_state = council_node(state)

    mock_council.assert_called_once()
    assert new_state["council_verdict"]["passed"] is False
