from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


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

    def __init__(self, ttl: int = 300):
        self._ttl = ttl
        self._store: dict[str, tuple[Any, float]] = {}
        try:
            from mao.core.redis_client import get_redis
            self._redis = get_redis()
        except Exception:
            self._redis = None

    def make_key(self, embedding: list[float]) -> str:
        """Vector-based key — exact match on embedding."""
        raw = json.dumps(embedding, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def make_semantic_key(self, query: str, domain: str = "", top_k: int = 0) -> str:
        """Normalized text key — matches semantically equivalent phrasings.

        "What is Alzheimer's?" and "what is alzheimers" get the same key.
        Domain and top_k are included to prevent cross-domain cache pollution.
        """
        normalized = _normalize_query(query)
        raw = f"{normalized}|{domain}|{top_k}"
        return "sem:" + hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        if self._redis is not None:
            try:
                raw = self._redis.get(f"qcache:{key}")
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
        if self._redis is not None:
            try:
                self._redis.set(f"qcache:{key}", json.dumps(_to_json(value)), ex=self._ttl)
                return
            except Exception as exc:
                logger.debug("Redis cache SET failed, using in-memory: %s", exc)

        self._store[key] = (value, time.monotonic())

    def clear(self) -> None:
        self._store.clear()
