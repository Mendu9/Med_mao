from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 60
_MAX_REQUESTS = 20


def check_rate_limit(user_id: str) -> bool:
    """Return True if the request is allowed, False if rate limit exceeded.

    Uses a Redis counter with a 60-second sliding window (20 req/min).
    Fails open — returns True when Redis is unavailable so the API keeps working.
    """
    try:
        from mao.core.redis_client import get_redis
        client = get_redis()
        if client is None:
            return True

        key = f"ratelimit:{user_id}"
        pipe = client.pipeline()
        pipe.incr(key)
        pipe.expire(key, _WINDOW_SECONDS)
        results = pipe.execute()
        count = results[0]

        if count > _MAX_REQUESTS:
            logger.warning("Rate limit exceeded for user_id=%s count=%d", user_id, count)
            return False
        return True
    except Exception as exc:
        logger.debug("Rate limiter error (failing open): %s", exc)
        return True
