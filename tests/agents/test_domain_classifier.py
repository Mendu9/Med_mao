from unittest.mock import patch

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
    with patch("mao.core.llm.chat", return_value="stroke") as chat:
        assert dc._llm_classify("what is ischemic stroke?") == "stroke"
    system = chat.call_args.kwargs["messages"][0]["content"]
    assert system == get_prompt("domain.classify").template


def test_classifier_uses_a_role_resolved_model() -> None:
    from mao.providers.gateway import model_id_for
    from mao.providers.registry import ModelRole

    with patch("mao.core.llm.chat", return_value="stroke") as chat:
        dc._llm_classify("what is ischemic stroke?")
    assert chat.call_args.kwargs["model"] == model_id_for(ModelRole.EXTRACTION_FAST)


def test_module_holds_no_inline_prompt_constant() -> None:
    assert not hasattr(dc, "_SYSTEM")
