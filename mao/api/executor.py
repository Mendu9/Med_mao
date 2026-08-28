"""The API's shared thread pool.

LangGraph's `invoke` and the SQLAlchemy session are both synchronous, so the
async endpoints hand that work to a pool rather than blocking the event loop.
It lives here, not in `main`, so a router module can use it without importing
`main` and creating an import cycle.

Recreated on lifespan startup: a process-level restart (the HF Spaces daemon
thread) must get a fresh executor rather than a shut-down one.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

_MAX_WORKERS = 8

_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=_MAX_WORKERS)


def get_executor() -> ThreadPoolExecutor:
    """The process-wide pool for synchronous work."""
    return _executor


def restart_executor() -> ThreadPoolExecutor:
    """Replace the pool with a fresh one. Called on lifespan startup."""
    global _executor
    _executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS)
    return _executor


def shutdown_executor(*, wait: bool = False) -> None:
    """Tear the pool down on lifespan shutdown."""
    _executor.shutdown(wait=wait)
