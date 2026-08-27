from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")

# Bumped by the ingestion pipeline (or by hand) whenever the corpus behind the
# vector store changes. Without it a re-ingest silently keeps serving answers
# built from the previous corpus for the whole cache TTL (P1-19).
_DEFAULT_INDEX_VERSION = "v1"
_DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def index_version() -> str:
    """Corpus/index generation marker; set ``MAO_INDEX_VERSION`` after re-ingesting."""
    return os.getenv("MAO_INDEX_VERSION", "").strip() or _DEFAULT_INDEX_VERSION


def reranker_fingerprint() -> str:
    """Identity *and* on/off state of the reranker.

    Results with the reranker disabled are ordered by raw cosine similarity and
    are not interchangeable with reranked results, so the two must never share
    a cache entry (P1-19).
    """
    if os.getenv("MAO_DISABLE_RERANKER", "0") == "1":
        return "off"
    try:
        from mao.core.config import cfg

        model = cfg.reranker_model
    except Exception:  # pragma: no cover - config always imports in practice
        model = _DEFAULT_RERANKER_MODEL
    return f"on:{model}"


def _context_fingerprint(domain: str, top_n: int, top_k: int) -> str:
    """Every non-query input that can change the retrieved set, as stable JSON.

    Deliberately built from ``json.dumps`` + ``hashlib`` rather than the builtin
    ``hash()``, which is salted per process by PYTHONHASHSEED and would give
    each gunicorn worker its own private cache (P2-6).
    """
    return json.dumps(
        {
            "domain": domain or "",
            "top_n": int(top_n or 0),
            "top_k": int(top_k or 0),
            "index_version": index_version(),
            "reranker": reranker_fingerprint(),
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalize_query(query: str) -> str:
    """Normalize for semantic cache keying: lowercase, strip punctuation, collapse whitespace."""
    q = query.lower()
    q = _PUNCT_RE.sub(" ", q)
    q = _WHITESPACE_RE.sub(" ", q).strip()
    return q


def _to_json(value: Any) -> Any:
    """Recursively convert dataclasses (e.g. RankedChunk) to JSON-safe dicts."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, list):
        return [_to_json(v) for v in value]
    return value


def _from_json_ranked(data: Any) -> Any:
    """Reconstruct list[RankedChunk] from cached JSON dicts."""
    if not isinstance(data, list):
        return data
    try:
        from mao.rag.reranker import RankedChunk
        return [RankedChunk(**item) for item in data]
    except Exception:
        return data


class QueryCache:
    """RAG query result cache with semantic normalization.

    Uses Redis when available (TTL-based), falls back to in-process dict.
    The public API (get/set/make_key/make_semantic_key/clear) is identical in both modes.
    """

    def __init__(self, ttl: int = 300, redis_retry_interval: float = 30.0):
        self._ttl = ttl
        self._store: dict[str, tuple[Any, float]] = {}
        # Redis is bound lazily. Binding at construction (i.e. at import time,
        # since retriever.py builds the module-level singleton) meant a Redis
        # that was down during startup was never retried again for the whole
        # process lifetime (P2-7).
        self._redis: Any | None = None
        self._redis_retry_interval = redis_retry_interval
        self._redis_last_attempt: float | None = None

    # -- Redis binding ------------------------------------------------------

    def _get_redis(self) -> Any | None:
        """Bind Redis on first use, retrying at most once per retry interval."""
        if self._redis is not None:
            return self._redis
        now = time.monotonic()
        if (
            self._redis_last_attempt is not None
            and now - self._redis_last_attempt < self._redis_retry_interval
        ):
            return None
        self._redis_last_attempt = now
        try:
            from mao.core.redis_client import get_redis

            self._redis = get_redis()
        except Exception as exc:
            logger.debug("Redis bind failed, will retry in %ss: %s", self._redis_retry_interval, exc)
            self._redis = None
        return self._redis

    # -- key construction ---------------------------------------------------

    def make_key(
        self,
        embedding: list[float],
        *,
        domain: str = "",
        top_n: int = 0,
        top_k: int = 0,
    ) -> str:
        """Vector-based key — exact match on embedding plus retrieval context.

        The context carries domain, top_n, top_k, the index version and the
        reranker fingerprint, so a re-ingest or a reranker toggle can never
        serve a stale entry (P1-19).
        """
        raw = json.dumps(embedding, separators=(",", ":")) + "|" + _context_fingerprint(
            domain, top_n, top_k
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def make_semantic_key(
        self,
        query: str,
        domain: str = "",
        top_k: int = 0,
        top_n: int = 0,
    ) -> str:
        """Normalized text key — matches phrasings that differ only in surface form.

        Case, punctuation and whitespace are normalised away, so
        "What is Alzheimer disease?" and "what  is alzheimer disease" share a key.
        Normalisation is deliberately conservative: stripping the apostrophe from
        "Alzheimer's" leaves a token boundary, so it does NOT collide with
        "alzheimers". Everything else that changes the answer — domain, top_n, top_k, the
        index version and the reranker fingerprint — is part of the key.
        """
        normalized = _normalize_query(query)
        raw = normalized + "|" + _context_fingerprint(domain, top_n, top_k)
        return "sem:" + hashlib.sha256(raw.encode()).hexdigest()

    # -- storage ------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        redis = self._get_redis()
        if redis is not None:
            try:
                raw = redis.get(f"qcache:{key}")
                if raw is not None:
                    return _from_json_ranked(json.loads(raw))
            except Exception as exc:
                logger.debug("Redis cache GET failed: %s", exc)

        entry = self._store.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        redis = self._get_redis()
        if redis is not None:
            try:
                redis.set(f"qcache:{key}", json.dumps(_to_json(value)), ex=self._ttl)
                return
            except Exception as exc:
                logger.debug("Redis cache SET failed, using in-memory: %s", exc)

        self._store[key] = (value, time.monotonic())

    def clear(self) -> None:
        self._store.clear()
