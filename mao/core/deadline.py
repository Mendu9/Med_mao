"""A per-request time budget that bounded waits can consult.

The rate-limit policy in `mao/core/retry.py` is right to wait: a burst limit
clears in seconds, and a slower correct answer beats a spurious refusal for
clinical decision support. What was missing is a ceiling on the *total*.

A clinical `/chat` makes seven sequential gateway calls. Each could wait up to
`MAX_RATE_LIMIT_WAIT_SECONDS` per attempt, and only the council leg was bounded,
so a single request could hold one of eight executor threads for roughly half an
hour. Eight such requests take the pool. `asyncio.to_thread` cannot cancel a
running thread, so the council's own 120s timeout did not stop a sleeping worker
either, and the containing rate limiter fails open when Redis is down.

A `ContextVar` rather than a parameter, because the waits happen four or five
frames below the request handler — through the gateway, the provider, and
`mao.core.llm` — and threading a deadline argument through every one of those
signatures would mean any call site that forgot it was unbounded again. This way
the budget is a property of the request, and code that never heard of deadlines
still respects it.

`contextvars` propagate into `loop.run_in_executor` and `asyncio.to_thread`
because both copy the current context, which is what makes this reach the
threads that actually do the sleeping.

Absent a deadline, everything behaves exactly as before: a CLI run, an
ingestion job or a test has no request to bound.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

#: Monotonic timestamp by which the current request must be done, if bounded.
_deadline: ContextVar[float | None] = ContextVar("mao_request_deadline", default=None)


@contextmanager
def request_deadline(seconds: float) -> Iterator[None]:
    """Bound everything inside this block to *seconds* from now.

    Nesting never *extends* an existing budget. A sub-task must not be able to
    buy itself more time than the request it belongs to has left, or the outer
    bound is advisory.
    """
    now = time.monotonic()
    proposed = now + max(seconds, 0.0)
    current = _deadline.get()
    effective = proposed if current is None else min(current, proposed)

    token = _deadline.set(effective)
    try:
        yield
    finally:
        _deadline.reset(token)


def remaining_seconds() -> float | None:
    """Seconds left in the current request's budget, or None if unbounded.

    Never negative: an exhausted budget reads as 0.0, which callers treat as
    "no time for another wait" rather than as a nonsensical negative delay.
    """
    deadline = _deadline.get()
    if deadline is None:
        return None
    return max(deadline - time.monotonic(), 0.0)


def exceeded() -> bool:
    """Whether the current request has run out of time."""
    left = remaining_seconds()
    return left is not None and left <= 0.0
