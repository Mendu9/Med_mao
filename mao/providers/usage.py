"""Per-request LLM usage collection.

A trace is supposed to record what a request cost. It could not: the counts live
on individual `Completion`s, scattered across however many model calls a request
makes — a router classification, a synthesis, three council members, a judge —
and nothing added them up.

The obvious fix, having every agent accumulate onto graph state, is the same
per-agent boilerplate the architecture already rejected for memory. Instead the
gateway reports into whatever collector is active for the current request, and
agents stay unaware that accounting happens at all.

Context, not global state: two concurrent requests must not pool their tokens.
`ContextVar` gives each request its own collector, and `asyncio.to_thread`
(which the council uses) copies the context, so a council member's tokens land
in the right request's total. `ThreadPoolExecutor` does *not* copy context, so
the collector is bound inside the worker function rather than around it.
"""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class UsageTotals:
    """Running totals for one request. Safe to update from several threads."""

    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, *, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        with self._lock:
            self.input_tokens += int(input_tokens or 0)
            self.output_tokens += int(output_tokens or 0)
            self.estimated_cost_usd += float(cost_usd or 0.0)
            self.calls += 1

    def to_dict(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "estimated_cost_usd": round(self.estimated_cost_usd, 8),
                "calls": self.calls,
            }


_current: ContextVar[UsageTotals | None] = ContextVar("mao_llm_usage", default=None)


@contextmanager
def collecting() -> Iterator[UsageTotals]:
    """Collect every gateway completion made inside this block."""
    totals = UsageTotals()
    token = _current.set(totals)
    try:
        yield totals
    finally:
        _current.reset(token)


def record(*, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
    """Report one completion. A no-op when nothing is collecting."""
    totals = _current.get()
    if totals is None:
        return
    totals.add(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


def current() -> UsageTotals | None:
    """The active collector, if any."""
    return _current.get()
