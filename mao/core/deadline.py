"""A per-request time budget that bounded waits can consult.

The rate-limit policy in `mao/core/retry.py` is right to wait: a burst limit
clears in seconds, and a slower correct answer beats a spurious refusal for
clinical decision support. What was missing is a ceiling on the total time a
request may spend WAITING TO RETRY.

## What this does and does not bound (NB2 / ADV-3)

It bounds the SLEEPS. It does not bound total worker hold time, and this file
used to say it was "a ceiling on the total", which is not true and is corrected
here rather than left standing. Measured under a 1.0s budget, `run_graph`
returned after 6.02s in one probe and 3.00s in another, with
`remaining_seconds()` reading 0.0 INSIDE the worker: the budget was visibly
exhausted and the work ran to completion anyway. The deadline is consulted only
in `with_groq_retry`'s decision to sleep; `run_in_executor` is never wrapped in
`asyncio.wait_for`.

So `REQUEST_DEADLINE_SECONDS = 180` does not mean "a request finishes within
180 seconds". It means "a request does not SLEEP past 180 seconds". The residual
is bounded by the provider SDK's own timeouts rather than by anything here, and
one live clinical chain took 276s. Closing the gap needs the executor future
wrapped in `asyncio.wait_for` and an explicit `timeout=` on the provider client;
that is recorded as NB2 against a later phase.

What this DID close is real, and is the finding it was written for:

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
