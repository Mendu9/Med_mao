"""retrieve() must build cache keys from stable digests only (P2-6, P1-19).

The retrieval algorithm itself is out of scope here — these tests only pin the
key construction, so gunicorn workers agree on keys and a reranker toggle or a
re-ingest cannot serve stale results.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

RETRIEVER_SRC = Path(__file__).resolve().parents[2] / "mao" / "rag" / "retriever.py"


def _retrieve_source() -> str:
    tree = ast.parse(RETRIEVER_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "retrieve":
            return ast.get_source_segment(RETRIEVER_SRC.read_text(encoding="utf-8"), node) or ""
    raise AssertionError("retrieve() not found in mao/rag/retriever.py")


def test_retrieve_does_not_use_builtin_hash_for_cache_keys() -> None:
    """builtin hash() is PYTHONHASHSEED-salted — workers would fragment the cache (P2-6)."""
    tree = ast.parse(_retrieve_source())
    offenders = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "hash"
    ]
    assert not offenders, "retrieve() must not call builtin hash() when building cache keys"


def test_retrieve_passes_top_n_and_domain_into_both_cache_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both the semantic and the vector key must carry domain, top_n and top_k (P1-19)."""
    import mao.rag.retriever as retriever

    seen: dict[str, dict] = {}

    class _SpyCache:
        def make_semantic_key(self, query, **kwargs):
            seen["semantic"] = kwargs
            return "sem:spy"

        def make_key(self, embedding, **kwargs):
            seen["vector"] = kwargs
            return "vec:spy"

        def get(self, key):
            return ["sentinel"] if key == "sem:spy" else None

        def set(self, key, value):
            return None

    monkeypatch.setattr(retriever, "_cache", _SpyCache())
    monkeypatch.setattr(retriever, "embed_query", lambda q: [0.1, 0.2, 0.3])

    result = retriever.retrieve("some query", top_n=37, top_k=7, domain="stroke")

    assert result == ["sentinel"]
    assert seen["semantic"]["domain"] == "stroke"
    assert seen["semantic"]["top_k"] == 7
    assert seen["semantic"]["top_n"] == 37


def test_retrieve_vector_key_receives_context_when_semantic_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mao.rag.retriever as retriever

    seen: dict[str, dict] = {}

    class _SpyCache:
        def make_semantic_key(self, query, **kwargs):
            seen["semantic"] = kwargs
            return "sem:spy"

        def make_key(self, embedding, **kwargs):
            seen["vector"] = kwargs
            return "vec:spy"

        def get(self, key):
            return ["sentinel"] if key == "vec:spy" else None

        def set(self, key, value):
            return None

    monkeypatch.setattr(retriever, "_cache", _SpyCache())
    monkeypatch.setattr(retriever, "embed_query", lambda q: [0.1, 0.2, 0.3])

    result = retriever.retrieve("some query", top_n=37, top_k=7, domain="stroke")

    assert result == ["sentinel"]
    assert seen["vector"]["domain"] == "stroke"
    assert seen["vector"]["top_n"] == 37
    assert seen["vector"]["top_k"] == 7
