import pytest
from tests.eval.conftest import load_golden, fixture_chunks_for

KEYWORD_ENTRIES = [e for e in load_golden() if e.get("expected_keywords") and e.get("query")]


@pytest.mark.parametrize("entry", KEYWORD_ENTRIES, ids=[e["id"] for e in KEYWORD_ENTRIES])
def test_rag_keywords_present(entry):
    """Fixture chunks contain expected keywords for known queries."""
    chunks = fixture_chunks_for(entry["id"])
    if not entry["expected_keywords"]:
        pytest.skip("No keywords to assert for this entry")
    if not chunks:
        pytest.skip(f"No fixture chunks for {entry['id']}")
    all_text = " ".join(c["text"] for c in chunks).lower()
    for kw in entry["expected_keywords"]:
        assert kw.lower() in all_text, (
            f"Keyword '{kw}' not found in fixture chunks for {entry['id']}"
        )
