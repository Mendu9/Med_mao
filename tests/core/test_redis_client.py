"""Redis client behaviour.

These tests previously patched `mao.core.redis_client.redis`, but commit 2b28619
made that import function-local, so the module attribute no longer exists and
both tests errored. They also never reset `_redis_unavailable`, which latches
True after the first failure — so ordering alone could make them pass or fail.
Both problems are fixed here: patch the real `redis` package, and reset module
state through an explicit seam.
"""
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_redis_state():
    """Every test starts from an unbound, un-latched client."""
    import mao.core.redis_client as rc

    rc.reset_redis()
    yield
    rc.reset_redis()


def test_get_redis_returns_singleton():
    import mao.core.redis_client as rc

    with patch("redis.Redis") as mock_cls:
        inst = MagicMock()
        inst.ping.return_value = True
        mock_cls.from_url.return_value = inst
        c1 = rc.get_redis()
        c2 = rc.get_redis()
    assert c1 is inst
    assert c1 is c2
    assert mock_cls.from_url.call_count == 1


def test_get_redis_returns_none_on_connection_error():
    import mao.core.redis_client as rc

    with patch("redis.Redis") as mock_cls:
        inst = MagicMock()
        inst.ping.side_effect = Exception("refused")
        mock_cls.from_url.return_value = inst
        result = rc.get_redis()
    assert result is None


def test_connection_failure_latches_so_we_do_not_retry_every_request():
    import mao.core.redis_client as rc

    with patch("redis.Redis") as mock_cls:
        inst = MagicMock()
        inst.ping.side_effect = Exception("refused")
        mock_cls.from_url.return_value = inst
        rc.get_redis()
        rc.get_redis()
        assert mock_cls.from_url.call_count == 1


def test_reset_redis_clears_the_latch():
    import mao.core.redis_client as rc

    with patch("redis.Redis") as mock_cls:
        inst = MagicMock()
        inst.ping.side_effect = Exception("refused")
        mock_cls.from_url.return_value = inst
        rc.get_redis()

    rc.reset_redis()

    with patch("redis.Redis") as mock_cls:
        good = MagicMock()
        good.ping.return_value = True
        mock_cls.from_url.return_value = good
        assert rc.get_redis() is good


def test_safe_get_returns_none_on_error():
    from mao.core.redis_client import safe_get

    client = MagicMock()
    client.get.side_effect = Exception("down")
    assert safe_get(client, "key") is None


def test_safe_get_returns_none_for_none_client():
    from mao.core.redis_client import safe_get

    assert safe_get(None, "key") is None


def test_safe_set_swallows_error():
    from mao.core.redis_client import safe_set

    client = MagicMock()
    client.set.side_effect = Exception("down")
    safe_set(client, "key", "value", ex=300)
