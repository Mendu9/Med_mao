import pytest
from unittest.mock import patch, MagicMock
from mao.agents.query_decomposer import decompose_query, decomposer_node

def test_single_question_returns_one():
    with patch("mao.agents.query_decomposer._llm_decompose") as mock_llm:
        mock_llm.return_value = ["What is Alzheimer's disease?"]
        result = decompose_query("What is Alzheimer's disease?")
    assert result == ["What is Alzheimer's disease?"]

def test_multi_question_returns_list():
    with patch("mao.agents.query_decomposer._llm_decompose") as mock_llm:
        mock_llm.return_value = ["What is amyloid?", "How does tau relate to AD?"]
        result = decompose_query("What is amyloid and how does tau relate to AD?")
    assert len(result) == 2

def test_decomposer_node_updates_state():
    state = {"query": "What is amyloid?", "pii_scrubbed_query": "What is amyloid?", "sub_queries": []}
    with patch("mao.agents.query_decomposer._llm_decompose") as mock_llm:
        mock_llm.return_value = ["What is amyloid?"]
        new_state = decomposer_node(state)
    assert new_state["sub_queries"] == ["What is amyloid?"]
    assert new_state["missing_sub_queries"] == ["What is amyloid?"]

def test_decomposer_node_scrubs_pii():
    state = {"query": "Patient john.doe@example.com asks about amyloid", "pii_scrubbed_query": "", "sub_queries": []}
    with patch("mao.agents.query_decomposer._llm_decompose") as mock_llm:
        mock_llm.return_value = ["What is amyloid?"]
        new_state = decomposer_node(state)
    assert "john.doe@example.com" not in new_state["pii_scrubbed_query"]
