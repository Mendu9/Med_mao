import pytest
from tests.eval.conftest import load_golden

ROUTING_ENTRIES = [e for e in load_golden() if e.get("expected_agent") and e.get("query")][:10]


@pytest.mark.parametrize("entry", ROUTING_ENTRIES, ids=[e["id"] for e in ROUTING_ENTRIES])
def test_routing_correct(entry, mock_retrieve, mock_llm):
    """Domain classifier routes clinical queries to clinical domain."""
    from mao.agents.domain_classifier import classify_domain
    domain = classify_domain(entry["query"])
    if entry["domain"] in ("alzheimer", "stroke"):
        assert domain in ("alzheimer", "stroke"), (
            f"Expected clinical domain for {entry['id']}, got {domain!r}"
        )
    else:
        assert domain not in ("alzheimer", "stroke") or domain == "general", (
            f"Expected non-clinical domain for {entry['id']}, got {domain!r}"
        )
