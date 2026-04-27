from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class QueryCache:
    """RAG query result cache.

    Uses Redis when available (TTL-based), falls back to in-process dict.
    The public API (get/set/make_key/clear) is identical in both modes.
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
        raw = json.dumps(embedding, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        if self._redis is not None:
            try:
                raw = self._redis.get(f"qcache:{key}")
                if raw is not None:
                    return json.loads(raw)
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
                self._redis.set(f"qcache:{key}", json.dumps(value), ex=self._ttl)
                return
            except Exception as exc:
                logger.debug("Redis cache SET failed, using in-memory: %s", exc)

        self._store[key] = (value, time.monotonic())

    def clear(self) -> None:
        self._store.clear()
