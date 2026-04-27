from __future__ import annotations

import logging
import os
from typing import Any

import redis

logger = logging.getLogger(__name__)

_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

_sync_client: redis.Redis | None = None


def get_redis() -> redis.Redis | None:
    global _sync_client
    if _sync_client is not None:
        return _sync_client
    try:
        client = redis.Redis.from_url(_REDIS_URL, decode_responses=True)
        client.ping()
        _sync_client = client
        return _sync_client
    except Exception as exc:
        logger.warning("Redis unavailable — caching disabled: %s", exc)
        return None


def safe_get(client: redis.Redis | None, key: str) -> str | None:
    if client is None:
        return None
    try:
        return client.get(key)
    except Exception as exc:
        logger.debug("Redis GET failed for key %r: %s", key, exc)
        return None


def safe_set(client: redis.Redis | None, key: str, value: Any, ex: int = 300) -> None:
    if client is None:
        return
    try:
        client.set(key, value, ex=ex)
    except Exception as exc:
        logger.debug("Redis SET failed for key %r: %s", key, exc)
