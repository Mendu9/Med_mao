"""Unit tests for pure retriever helper functions."""
from __future__ import annotations

from mao.rag.retriever import _apply_domain_boost


def test_domain_boost_raises_matching_domain_score():
    """Chunks whose stored domain matches the query domain get a score boost."""
    chunks = [
        {"chunk_id": "a", "domain": "stroke", "score": 0.50},
        {"chunk_id": "b", "domain": "alzheimer", "score": 0.50},
    ]
    boosted = _apply_domain_boost(chunks, domain="stroke", factor=1.2)
    by_id = {c["chunk_id"]: c for c in boosted}
    assert by_id["a"]["score"] == 0.60   # 0.50 * 1.2 — matching domain
    assert by_id["b"]["score"] == 0.50   # unchanged — different domain


def test_domain_boost_never_excludes_chunks():
    """No chunk is dropped, even ones with a non-matching or missing domain —
    this preserves recall for 'general'/'pubmed'-tagged data."""
    chunks = [
        {"chunk_id": "a", "domain": "pubmed", "score": 0.50},
        {"chunk_id": "b", "score": 0.50},  # no domain field at all
        {"chunk_id": "c", "domain": "stroke", "score": 0.50},
    ]
    boosted = _apply_domain_boost(chunks, domain="stroke", factor=1.2)
    assert len(boosted) == 3, "domain boost must not remove any chunk"
    assert {c["chunk_id"] for c in boosted} == {"a", "b", "c"}


def test_domain_boost_general_domain_is_noop():
    """A 'general' query domain matches no stored tag, so it must not alter scores
    (and must not wipe results)."""
    chunks = [
        {"chunk_id": "a", "domain": "alzheimer", "score": 0.50},
        {"chunk_id": "b", "domain": "stroke", "score": 0.50},
    ]
    boosted = _apply_domain_boost(chunks, domain="general", factor=1.2)
    assert all(c["score"] == 0.50 for c in boosted)
    assert len(boosted) == 2
