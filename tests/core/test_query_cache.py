import time
import pytest
from mao.core.query_cache import QueryCache

def test_miss_returns_none():
    cache = QueryCache(ttl=60)
    assert cache.get("abc123") is None

def test_hit_returns_value():
    cache = QueryCache(ttl=60)
    cache.set("abc123", {"docs": ["a", "b"]})
    result = cache.get("abc123")
    assert result == {"docs": ["a", "b"]}

def test_ttl_expiry():
    cache = QueryCache(ttl=1)
    cache.set("key1", "value")
    time.sleep(1.1)
    assert cache.get("key1") is None

def test_cache_key_from_embedding():
    cache = QueryCache(ttl=60)
    embedding = [0.1, 0.2, 0.3]
    key = cache.make_key(embedding)
    assert isinstance(key, str)
    assert len(key) == 64  # sha256 hex

def test_cache_key_deterministic():
    cache = QueryCache(ttl=60)
    embedding = [0.1, 0.2, 0.3]
    assert cache.make_key(embedding) == cache.make_key(embedding)
