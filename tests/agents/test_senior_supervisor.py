"""Tests for senior_supervisor_node — completeness check + streaming short-circuit."""
from __future__ import annotations

from unittest.mock import patch

from mao.agents.senior_supervisor import senior_supervisor_node


def test_no_sub_queries_short_circuits_without_llm():
    """With no sub_queries, the node returns complete without calling the LLM."""
    with patch("mao.core.llm.chat") as mock_chat:
        out = senior_supervisor_node({"sub_queries": [], "response": ""})
    mock_chat.assert_not_called()
    assert out["completeness_ok"] is True
    assert out["missing_sub_queries"] == []


def test_streaming_path_skips_llm_when_answer_empty():
    """On the streaming path the final answer is deferred (response==''), so the
    completeness LLM call must be skipped — otherwise every streamed request burns
    a 70B-model call comparing sub-questions against an empty answer."""
    state = {
        "sub_queries": ["What is CBF?", "What is CBV?"],
        "response": "",           # deferred by the streaming path
        "_want_stream": True,
    }
    with patch("mao.core.llm.chat") as mock_chat:
        out = senior_supervisor_node(state)
    mock_chat.assert_not_called()
    assert out["completeness_ok"] is True
    assert out["missing_sub_queries"] == []


def test_non_streaming_with_answer_runs_completeness_check():
    """Normal (non-streaming) path with a real answer still runs the LLM check."""
    state = {
        "sub_queries": ["What is CBF?"],
        "response": "CBF is cerebral blood flow.",
    }
    with patch("mao.core.llm.chat", return_value='{"answered": ["What is CBF?"], "missing": []}') as mock_chat:
        out = senior_supervisor_node(state)
    mock_chat.assert_called_once()
    assert out["completeness_ok"] is True
    assert out["missing_sub_queries"] == []
