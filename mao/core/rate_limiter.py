from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 60
_MAX_PER_USER   = 20   # requests per minute per user_id
_MAX_PER_IP     = 60   # requests per minute per IP (prevents anonymous burst)


_LUA_INCR_WITH_EXPIRE = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return current
"""


def _check_key(client, key: str, limit: int, window: int) -> bool:
    """Atomically increment counter and set TTL on first creation; return True if under limit.

    Uses a Lua script so INCR+EXPIRE is atomic — prevents the TTL-never-set race
    where INCR succeeds but EXPIRE fails, creating a permanent key that bans the user.
    """
    try:
        current = client.eval(_LUA_INCR_WITH_EXPIRE, 1, key, window)
        return int(current) <= limit
    except Exception:
        # Fallback to pipeline if eval not available (Redis < 2.6 or ACL restriction)
        pipe = client.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)
        results = pipe.execute()
        return int(results[0]) <= limit


def check_rate_limit(user_id: str, client_ip: str | None = None) -> bool:
    """Return True if the request is allowed, False if rate limit exceeded.

    Checks two independent Redis buckets:
      1. Per user_id — 20 req/min  (user-level fairness)
      2. Per IP      — 60 req/min  (prevents anonymous burst from same host)

    Fails open — returns True when Redis is unavailable so the API keeps working.
    """
    try:
        from mao.core.redis_client import get_redis
        client = get_redis()
        if client is None:
            return True

        # Bucket 1: per-user
        if not _check_key(client, f"ratelimit:user:{user_id}", _MAX_PER_USER, _WINDOW_SECONDS):
            logger.warning("Rate limit exceeded for user_id=%s", user_id)
            return False

        # Bucket 2: per-IP (skip if IP unknown — e.g. unit tests or load-balancer setups)
        if client_ip:
            if not _check_key(client, f"ratelimit:ip:{client_ip}", _MAX_PER_IP, _WINDOW_SECONDS):
                logger.warning("Rate limit exceeded for ip=%s", client_ip)
                return False

        return True
    except Exception as exc:
        logger.warning("Rate limiter error (failing open — all requests allowed): %s", exc)
        return True
