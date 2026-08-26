from unittest.mock import patch

import pytest

from mao.agents import query_decomposer as qd
from mao.agents.query_decomposer import decompose_query, decomposer_node, needs_decomposition
from mao.prompts import get_prompt


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
    state = {"user_query": "What is amyloid?", "pii_scrubbed_query": "What is amyloid?", "sub_queries": []}
    with patch("mao.agents.query_decomposer._llm_decompose") as mock_llm:
        mock_llm.return_value = ["What is amyloid?"]
        new_state = decomposer_node(state)
    assert new_state["sub_queries"] == ["What is amyloid?"]
    assert new_state["missing_sub_queries"] == ["What is amyloid?"]

def test_decomposer_node_scrubs_pii():
    state = {"user_query": "Patient john.doe@example.com asks about amyloid", "pii_scrubbed_query": "", "sub_queries": []}
    with patch("mao.agents.query_decomposer._llm_decompose") as mock_llm:
        mock_llm.return_value = ["What is amyloid?"]
        new_state = decomposer_node(state)
    assert "john.doe@example.com" not in new_state["pii_scrubbed_query"]


# ---------------------------------------------------------------------------
# P2-15 — decomposition is conditional; single-part queries cost no LLM call
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "What is amyloid?",
        "Explain the amyloid cascade hypothesis.",
        "hello",
        "What are the symptoms of ischemic stroke?",
        "",
    ],
)
def test_single_part_queries_are_not_multi_part(query: str) -> None:
    assert needs_decomposition(query) is False


@pytest.mark.parametrize(
    "query",
    [
        "What is amyloid and how does tau relate to AD?",
        "What is tau? What is amyloid?",
        "Compare donepezil and memantine.",
        "What is the difference between ischemic and haemorrhagic stroke?",
        "Explain APOE4; also what does it do to risk?",
    ],
)
def test_multi_part_queries_are_detected(query: str) -> None:
    assert needs_decomposition(query) is True


def test_single_part_query_skips_the_llm_entirely() -> None:
    with patch.object(qd, "_llm_decompose") as llm:
        result = decompose_query("What is amyloid?")
    llm.assert_not_called()
    assert result == ["What is amyloid?"]


def test_multi_part_query_spends_the_llm_call() -> None:
    with patch.object(qd, "_llm_decompose", return_value=["a", "b"]) as llm:
        result = decompose_query("What is amyloid and how does tau relate to AD?")
    assert llm.call_count == 1
    assert result == ["a", "b"]


def test_decomposer_node_skips_the_llm_for_a_single_part_query() -> None:
    with patch.object(qd, "_llm_decompose") as llm:
        state = decomposer_node({"user_query": "What is amyloid?"})
    llm.assert_not_called()
    assert state["sub_queries"] == ["What is amyloid?"]


# ---------------------------------------------------------------------------
# Registry-owned prompt and its JSON output contract
# ---------------------------------------------------------------------------

def test_prompt_comes_from_the_registry() -> None:
    payload = '{"sub_queries": ["What is amyloid?", "How does tau relate to AD?"]}'
    with patch("mao.core.llm.chat", return_value=payload) as chat:
        result = qd._llm_decompose("What is amyloid and how does tau relate to AD?")

    system = chat.call_args.kwargs["messages"][0]["content"]
    assert system == get_prompt("decomposer.split").template
    assert result == ["What is amyloid?", "How does tau relate to AD?"]


def test_bare_json_list_is_still_accepted() -> None:
    with patch("mao.core.llm.chat", return_value='["a?", "b?"]'):
        assert qd._llm_decompose("a? and b?") == ["a?", "b?"]


def test_unparseable_output_falls_back_to_the_original_query() -> None:
    with patch("mao.core.llm.chat", return_value="not json at all"):
        assert qd._llm_decompose("a and b?") == ["a and b?"]


def test_decomposer_uses_a_role_resolved_model() -> None:
    from mao.providers.gateway import model_id_for
    from mao.providers.registry import ModelRole

    with patch("mao.core.llm.chat", return_value='["a?"]') as chat:
        qd._llm_decompose("a? and b?")
    assert chat.call_args.kwargs["model"] == model_id_for(ModelRole.EXTRACTION_FAST)


def test_module_holds_no_inline_prompt_constant() -> None:
    assert not hasattr(qd, "_SYSTEM")
