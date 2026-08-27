"""Cache-key correctness for QueryCache (P1-19, P2-6, P2-7).

Every parameter that can change the answer must change the key, and identical
inputs must produce identical keys in *any* process (no builtin ``hash()``).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mao.core.query_cache import QueryCache

EMBEDDING = [0.1, 0.2, 0.3]


@pytest.fixture
def cache() -> QueryCache:
    return QueryCache(ttl=60)


# ---------------------------------------------------------------------------
# P1-19 — semantic key must cover every answer-changing parameter
# ---------------------------------------------------------------------------

def test_semantic_key_identical_inputs_are_stable(cache: QueryCache) -> None:
    """Case, punctuation and whitespace are normalised away; nothing else is."""
    a = cache.make_semantic_key("What is Alzheimer disease?", domain="alzheimer", top_k=10, top_n=80)
    b = cache.make_semantic_key("what   is alzheimer disease", domain="alzheimer", top_k=10, top_n=80)
    assert a == b


def test_semantic_key_does_not_collapse_distinct_terms(cache: QueryCache) -> None:
    """Possessive stripping leaves a token boundary — 'alzheimer s' != 'alzheimers'."""
    a = cache.make_semantic_key("What is Alzheimer's?", domain="alzheimer", top_k=10, top_n=80)
    b = cache.make_semantic_key("what is alzheimers", domain="alzheimer", top_k=10, top_n=80)
    assert a != b


def test_semantic_key_varies_with_top_k(cache: QueryCache) -> None:
    a = cache.make_semantic_key("q", domain="d", top_k=10, top_n=80)
    b = cache.make_semantic_key("q", domain="d", top_k=5, top_n=80)
    assert a != b


def test_semantic_key_varies_with_top_n(cache: QueryCache) -> None:
    """top_n changes the candidate pool, so it changes the answer (P1-19)."""
    a = cache.make_semantic_key("q", domain="d", top_k=10, top_n=80)
    b = cache.make_semantic_key("q", domain="d", top_k=10, top_n=40)
    assert a != b


def test_semantic_key_varies_with_domain(cache: QueryCache) -> None:
    a = cache.make_semantic_key("q", domain="alzheimer", top_k=10, top_n=80)
    b = cache.make_semantic_key("q", domain="stroke", top_k=10, top_n=80)
    assert a != b


def test_semantic_key_varies_with_index_version(
    cache: QueryCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-ingesting the corpus must invalidate every cached answer (P1-19)."""
    monkeypatch.setenv("MAO_INDEX_VERSION", "2024-01-01")
    a = cache.make_semantic_key("q", domain="d", top_k=10, top_n=80)
    monkeypatch.setenv("MAO_INDEX_VERSION", "2024-06-01")
    b = cache.make_semantic_key("q", domain="d", top_k=10, top_n=80)
    assert a != b


def test_semantic_key_varies_with_reranker_state(
    cache: QueryCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Toggling MAO_DISABLE_RERANKER must not serve the other mode's results (P1-19)."""
    monkeypatch.setenv("MAO_DISABLE_RERANKER", "0")
    on = cache.make_semantic_key("q", domain="d", top_k=10, top_n=80)
    monkeypatch.setenv("MAO_DISABLE_RERANKER", "1")
    off = cache.make_semantic_key("q", domain="d", top_k=10, top_n=80)
    assert on != off


# ---------------------------------------------------------------------------
# P1-19 — the vector key needs the same context
# ---------------------------------------------------------------------------

def test_vector_key_is_sha256_hex(cache: QueryCache) -> None:
    key = cache.make_key(EMBEDDING, domain="d", top_n=80, top_k=10)
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_vector_key_varies_with_domain(cache: QueryCache) -> None:
    a = cache.make_key(EMBEDDING, domain="alzheimer", top_n=80, top_k=10)
    b = cache.make_key(EMBEDDING, domain="stroke", top_n=80, top_k=10)
    assert a != b


def test_vector_key_varies_with_top_n_and_top_k(cache: QueryCache) -> None:
    base = cache.make_key(EMBEDDING, domain="d", top_n=80, top_k=10)
    assert base != cache.make_key(EMBEDDING, domain="d", top_n=40, top_k=10)
    assert base != cache.make_key(EMBEDDING, domain="d", top_n=80, top_k=5)


def test_vector_key_varies_with_index_version(
    cache: QueryCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAO_INDEX_VERSION", "a")
    a = cache.make_key(EMBEDDING, domain="d", top_n=80, top_k=10)
    monkeypatch.setenv("MAO_INDEX_VERSION", "b")
    b = cache.make_key(EMBEDDING, domain="d", top_n=80, top_k=10)
    assert a != b


def test_vector_key_varies_with_reranker_state(
    cache: QueryCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAO_DISABLE_RERANKER", "0")
    on = cache.make_key(EMBEDDING, domain="d", top_n=80, top_k=10)
    monkeypatch.setenv("MAO_DISABLE_RERANKER", "1")
    off = cache.make_key(EMBEDDING, domain="d", top_n=80, top_k=10)
    assert on != off


def test_semantic_and_vector_keys_never_collide(cache: QueryCache) -> None:
    sem = cache.make_semantic_key("q", domain="d", top_k=10, top_n=80)
    vec = cache.make_key(EMBEDDING, domain="d", top_n=80, top_k=10)
    assert sem != vec


# ---------------------------------------------------------------------------
# P2-6 — keys must be stable ACROSS processes (no PYTHONHASHSEED salting)
# ---------------------------------------------------------------------------

_KEY_SCRIPT = (
    "from mao.core.query_cache import QueryCache;"
    "c = QueryCache(ttl=60);"
    "print(c.make_semantic_key('what is alzheimers', domain='alzheimer', top_k=10, top_n=80));"
    "print(c.make_key([0.1, 0.2, 0.3], domain='alzheimer', top_n=80, top_k=10))"
)


def _keys_from_subprocess(repo_root: str, seed: str) -> list[str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    env["PYTHONHASHSEED"] = seed
    env["PYTHONPATH"] = repo_root
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", _KEY_SCRIPT],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()


def test_keys_are_identical_under_different_hash_seeds() -> None:
    """gunicorn workers each get a different PYTHONHASHSEED — keys must still match (P2-6)."""
    repo_root = str(Path(__file__).resolve().parents[2])
    first = _keys_from_subprocess(repo_root, "1")
    second = _keys_from_subprocess(repo_root, "424242")
    assert len(first) == 2
    assert first == second


# ---------------------------------------------------------------------------
# P2-7 — Redis must be bound lazily, with retry
# ---------------------------------------------------------------------------

def test_construction_does_not_connect_to_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    import mao.core.redis_client as rc

    monkeypatch.setattr(rc, "get_redis", lambda: calls.append(1))
    QueryCache(ttl=60)
    assert calls == [], "QueryCache must not bind Redis at construction/import time (P2-7)"


def test_redis_is_retried_after_the_retry_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    import mao.core.redis_client as rc

    def _fail() -> None:
        calls.append(1)
        raise ConnectionError("redis down")

    monkeypatch.setattr(rc, "get_redis", _fail)
    cache = QueryCache(ttl=60, redis_retry_interval=0.0)
    assert cache.get("missing") is None
    assert cache.get("missing") is None
    assert len(calls) >= 2, "a failed Redis bind must be retried, not latched for the process life"


def test_redis_failure_does_not_break_get_or_set(monkeypatch: pytest.MonkeyPatch) -> None:
    import mao.core.redis_client as rc

    monkeypatch.setattr(rc, "get_redis", lambda: (_ for _ in ()).throw(ConnectionError("down")))
    cache = QueryCache(ttl=60, redis_retry_interval=0.0)
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
