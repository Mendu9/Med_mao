import pytest
from unittest.mock import patch
from mao.agents.domain_classifier import classify_domain, classifier_node

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
