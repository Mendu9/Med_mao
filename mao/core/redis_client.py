from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

_sync_client: Any | None = None
_redis_unavailable: bool = False  # True after first failed connect or missing package


def get_redis() -> Any | None:
    global _sync_client, _redis_unavailable
    if _sync_client is not None:
        return _sync_client
    if _redis_unavailable:
        return None
    try:
        import redis as _redis
        client = _redis.Redis.from_url(_REDIS_URL, decode_responses=True)
        client.ping()
        _sync_client = client
        return _sync_client
    except ImportError:
        logger.warning("redis package not installed — caching disabled")
        _redis_unavailable = True
        return None
    except Exception as exc:
        logger.warning("Redis unavailable — caching disabled: %s", exc)
        _redis_unavailable = True
        return None


def safe_get(client: Any | None, key: str) -> str | None:
    if client is None:
        return None
    try:
        return client.get(key)
    except Exception as exc:
        logger.debug("Redis GET failed for key %r: %s", key, exc)
        return None


def safe_set(client: Any | None, key: str, value: Any, ex: int = 300) -> None:
    if client is None:
        return
    try:
        client.set(key, value, ex=ex)
    except Exception as exc:
        logger.debug("Redis SET failed for key %r: %s", key, exc)
