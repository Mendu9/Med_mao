from unittest.mock import patch

from tests.agents.gateway_stub import _completing, _completion

from mao.agents import domain_classifier as dc
from mao.agents.domain_classifier import classify_domain, classifier_node
from mao.prompts import get_prompt


def test_alzheimer_query():
    with patch("mao.agents.domain_classifier._llm_classify") as mock:
        mock.return_value = "alzheimer"
        assert classify_domain("What are early signs of Alzheimer's?") == "alzheimer"

def test_stroke_query():
    with patch("mao.agents.domain_classifier._llm_classify") as mock:
        mock.return_value = "stroke"
        assert classify_domain("What is ischemic stroke treatment?") == "stroke"

def test_general_query():
    with patch("mao.agents.domain_classifier._llm_classify") as mock:
        mock.return_value = "general"
        assert classify_domain("What is a healthy diet?") == "general"

def test_invalid_label_falls_back_to_general():
    with patch("mao.agents.domain_classifier._llm_classify") as mock:
        mock.return_value = "cardiology"
        assert classify_domain("Something unclear") == "general"

def test_classifier_node_updates_state():
    state = {"pii_scrubbed_query": "amyloid plaques", "sub_queries": ["amyloid plaques"]}
    with patch("mao.agents.domain_classifier._llm_classify") as mock:
        mock.return_value = "alzheimer"
        new_state = classifier_node(state)
    assert new_state["domain"] == "alzheimer"


# ---------------------------------------------------------------------------
# Registry-owned prompt and role-resolved model
# ---------------------------------------------------------------------------

def test_prompt_comes_from_the_registry() -> None:
    with patch("mao.providers.gateway.complete", return_value=_completion("stroke")) as chat:
        assert dc._llm_classify("what is ischemic stroke?") == "stroke"
    system = chat.call_args.kwargs["messages"][0]["content"]
    assert system == get_prompt("domain.classify").template


def test_classifier_asks_the_gateway_for_a_capability_role() -> None:
    """The agent names a role; only the registry turns that into a model id.

    Asserting the role rather than the resolved id is the stronger check — it
    holds even when the registry rebinds the role to a different model.
    """
    from mao.providers.registry import ModelRole

    with patch("mao.providers.gateway.complete", return_value=_completion("stroke")) as chat:
        dc._llm_classify("what is ischemic stroke?")
    assert chat.call_args.kwargs["role"] is ModelRole.EXTRACTION_FAST
    assert "model" not in chat.call_args.kwargs


def test_module_holds_no_inline_prompt_constant() -> None:
    assert not hasattr(dc, "_SYSTEM")
