from unittest.mock import MagicMock, patch

def test_get_redis_returns_singleton():
    import mao.core.redis_client as rc
    rc._sync_client = None
    with patch("mao.core.redis_client.redis.Redis") as mock_cls:
        inst = MagicMock()
        inst.ping.return_value = True
        mock_cls.from_url.return_value = inst
        c1 = rc.get_redis()
        c2 = rc.get_redis()
    assert c1 is c2

def test_get_redis_returns_none_on_connection_error():
    import mao.core.redis_client as rc
    rc._sync_client = None
    with patch("mao.core.redis_client.redis.Redis") as mock_cls:
        inst = MagicMock()
        inst.ping.side_effect = Exception("refused")
        mock_cls.from_url.return_value = inst
        result = rc.get_redis()
    assert result is None

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
