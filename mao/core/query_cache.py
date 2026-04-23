import hashlib
import json
import time
from typing import Any

class QueryCache:
    def __init__(self, ttl: int = 300):
        self._ttl = ttl
        self._store: dict[str, tuple[Any, float]] = {}

    def make_key(self, embedding: list[float]) -> str:
        raw = json.dumps(embedding, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (value, time.monotonic())

    def clear(self) -> None:
        self._store.clear()
