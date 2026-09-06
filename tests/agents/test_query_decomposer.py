from unittest.mock import patch

from tests.agents.gateway_stub import _completion

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

def test_decomposer_node_does_not_scrub_a_second_time():
    """Inverted deliberately. The decomposer used to call `scrub_pii` again.

    `state["user_query"]` is already the protected input boundary's output, so
    the second call was a second SEMANTIC TRANSFORMATION of an
    already-transformed string, and it over-redacted a clinical line the first
    pass had correctly left alone:

        'Patient Name:\nMRN:\nHarold Nkemdirim\nRockwood Frailty\n'
          x1 -> 'Patient Name:\nMRN:\n[NAME]\nRockwood Frailty\n'
          x2 -> 'Patient Name:\nMRN:\n[NAME]\n[NAME]\n'

    Two mechanisms were tried to make a second pass a no-op, and both had to
    recognise the first pass's output from a marker or a position in
    caller-controlled text — so both were forgeable, and removing the forgeable
    one reopened the destruction. The property is not achievable while a second
    pass exists, so this test asserts there is not one.

    The de-identification itself is asserted where it now happens, at the
    boundary: see tests/trust/.
    """
    from unittest.mock import MagicMock

    protected = "Patient [EMAIL] asks about amyloid"
    state = {"user_query": protected, "pii_scrubbed_query": "", "sub_queries": []}
    scrubber = MagicMock(side_effect=AssertionError("the decomposer scrubbed again"))
    with patch("mao.agents.query_decomposer._llm_decompose") as mock_llm,          patch("mao.core.pii_scrubber.scrub_pii", scrubber):
        mock_llm.return_value = ["What is amyloid?"]
        new_state = decomposer_node(state)

    scrubber.assert_not_called()
    assert new_state["pii_scrubbed_query"] == protected, (
        "the decomposer must carry the boundary's result through unchanged"
    )


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
    with patch("mao.providers.gateway.complete", return_value=_completion(payload)) as chat:
        result = qd._llm_decompose("What is amyloid and how does tau relate to AD?")

    system = chat.call_args.kwargs["messages"][0]["content"]
    assert system == get_prompt("decomposer.split").template
    assert result == ["What is amyloid?", "How does tau relate to AD?"]


def test_bare_json_list_is_still_accepted() -> None:
    with patch("mao.providers.gateway.complete", return_value=_completion('["a?", "b?"]')):
        assert qd._llm_decompose("a? and b?") == ["a?", "b?"]


def test_unparseable_output_falls_back_to_the_original_query() -> None:
    with patch("mao.providers.gateway.complete", return_value=_completion("not json at all")):
        assert qd._llm_decompose("a and b?") == ["a and b?"]


def test_decomposer_asks_the_gateway_for_a_capability_role() -> None:
    from mao.providers.registry import ModelRole

    with patch("mao.providers.gateway.complete", return_value=_completion('["a?"]')) as chat:
        qd._llm_decompose("a? and b?")
    assert chat.call_args.kwargs["role"] is ModelRole.EXTRACTION_FAST
    assert "model" not in chat.call_args.kwargs


def test_module_holds_no_inline_prompt_constant() -> None:
    assert not hasattr(qd, "_SYSTEM")
