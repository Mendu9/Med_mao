"""ExperienceStore — append-only sink for request traces.

Capture must never break a user-facing request, so `append()` swallows sink
errors and logs them. Traces are write-only from the request path: nothing here
feeds back into serving. Offline training reads this store, never the reverse.
"""
from __future__ import annotations

import logging
import threading
from typing import Protocol, runtime_checkable

from mao.schemas.trace import TraceSchema

logger = logging.getLogger(__name__)

_MAX_RETAINED = 1000


@runtime_checkable
class ExperienceStore(Protocol):
    """Sink for completed request traces."""

    def append(self, trace: TraceSchema) -> None: ...

    def recent(self, limit: int = 100) -> list[TraceSchema]: ...


class InMemoryExperienceStore:
    """Bounded in-process store.

    The default for Phase 1: enough to prove the capture path works and to
    inspect traces locally, without introducing a new persistence dependency.
    """

    def __init__(self, max_retained: int = _MAX_RETAINED) -> None:
        self._traces: list[TraceSchema] = []
        self._max = max_retained
        self._lock = threading.Lock()

    def append(self, trace: TraceSchema) -> None:
        """Record a trace. Never raises — capture must not break a request."""
        try:
            self._write(trace)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Experience capture failed (non-fatal): %s", exc)

    def recent(self, limit: int = 100) -> list[TraceSchema]:
        """Most recent traces, newest first."""
        with self._lock:
            return list(reversed(self._traces[-limit:]))

    # -- override point for real sinks --------------------------------------

    def _write(self, trace: TraceSchema) -> None:
        with self._lock:
            self._traces.append(trace)
            if len(self._traces) > self._max:
                del self._traces[: len(self._traces) - self._max]


_store: ExperienceStore | None = None


def get_experience_store() -> ExperienceStore:
    """The process-wide experience store."""
    global _store
    if _store is None:
        _store = InMemoryExperienceStore()
    return _store
