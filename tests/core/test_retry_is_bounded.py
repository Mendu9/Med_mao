"""Wave 9 / B9 — the rate-limit wait is a DoS amplifier.

The waiting behaviour itself is right, and both Wave 8 reviewers said so: a
burst limit clears in seconds, and honouring the provider's stated wait turns a
spurious refusal into a slower correct answer, which for clinical decision
support is the right trade. Two things were missing.

(a) `with_groq_retry` slept after EVERY failure INCLUDING THE LAST, then raised.
    The final sleep buys nothing — no attempt follows it. With the 75s cap and
    four attempts that is up to 75 wasted seconds per call, and 244-300s per
    call was measured at f757375.

(b) Nothing bounded the REQUEST. A clinical `/chat` makes 7 sequential gateway
    calls and only the council leg is bounded, so ~28 minutes could elapse
    holding 1 of 8 executor threads. `asyncio.to_thread` cannot cancel a thread,
    so the council's own 120s timeout does not stop a sleeping worker either.
    The containing rate limiter fails OPEN when Redis is down — its state at the
    time — and `user_id` is caller-supplied in the request body.

The invariant: no request holds a worker indefinitely, because every wait is
bounded by that request's own remaining budget.
"""
from __future__ import annotations

import time

import pytest

from mao.core import retry as retry_module
from mao.core.deadline import remaining_seconds, request_deadline
from mao.core.retry import DeadlineExceeded, with_groq_retry


class _RateLimit(Exception):
    def __init__(self, wait: float = 5.0) -> None:
        super().__init__(f"rate_limit_exceeded. Please try again in {wait}s")


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record sleeps instead of performing them."""
    recorded: list[float] = []
    monkeypatch.setattr(retry_module.time, "sleep", recorded.append)
    return recorded


class TestNoSleepAfterTheFinalAttempt:
    def test_the_last_failure_does_not_sleep(self, slept: list[float]) -> None:
        attempts = []

        def _always_limited():
            attempts.append(1)
            raise _RateLimit()

        with pytest.raises(_RateLimit):
            with_groq_retry(_always_limited, max_retries=4)

        assert len(attempts) == 4, "every attempt must still be made"
        assert len(slept) == 3, (
            f"slept {len(slept)} times for 4 attempts — the wait after the final "
            "attempt buys nothing, because no attempt follows it"
        )

    @pytest.mark.parametrize("max_retries", [1, 2, 3, 5])
    def test_sleeps_are_always_one_fewer_than_attempts(
        self, slept: list[float], max_retries: int
    ) -> None:
        def _always_limited():
            raise _RateLimit()

        with pytest.raises(_RateLimit):
            with_groq_retry(_always_limited, max_retries=max_retries)
        assert len(slept) == max_retries - 1

    def test_a_single_attempt_never_sleeps(self, slept: list[float]) -> None:
        def _always_limited():
            raise _RateLimit()

        with pytest.raises(_RateLimit):
            with_groq_retry(_always_limited, max_retries=1)
        assert slept == []


class TestTheRequestDeadlineBoundsTheWait:
    def test_a_wait_past_the_deadline_is_refused_rather_than_slept(
        self, slept: list[float]
    ) -> None:
        def _always_limited():
            raise _RateLimit(wait=60.0)

        with request_deadline(2.0):
            with pytest.raises(DeadlineExceeded):
                with_groq_retry(_always_limited, max_retries=4)

        assert not any(s > 2.0 for s in slept), (
            f"slept past the request deadline: {slept}"
        )

    def test_an_exhausted_budget_stops_immediately(self, slept: list[float]) -> None:
        def _always_limited():
            raise _RateLimit(wait=5.0)

        with request_deadline(0.0):
            with pytest.raises(DeadlineExceeded):
                with_groq_retry(_always_limited, max_retries=4)
        assert slept == []

    def test_without_a_deadline_the_previous_behaviour_is_unchanged(
        self, slept: list[float]
    ) -> None:
        """A CLI or a background job has no request to bound."""
        def _always_limited():
            raise _RateLimit(wait=5.0)

        with pytest.raises(_RateLimit):
            with_groq_retry(_always_limited, max_retries=3)
        assert len(slept) == 2
        assert all(s > 0 for s in slept)

    def test_the_deadline_is_scoped_to_its_block(self) -> None:
        assert remaining_seconds() is None
        with request_deadline(30.0):
            left = remaining_seconds()
            assert left is not None and 0 < left <= 30.0
        assert remaining_seconds() is None

    def test_a_nested_deadline_never_extends_the_outer_one(self) -> None:
        """A sub-task must not be able to buy itself more time than the request
        it belongs to has left."""
        with request_deadline(5.0):
            with request_deadline(600.0):
                left = remaining_seconds()
                assert left is not None and left <= 5.0


class TestTheWaitingBehaviourItselfIsUnchanged:
    """The fix must bound the wait, not remove it — a burst limit that clears in
    seconds should still be waited out."""

    def test_a_successful_call_returns(self, slept: list[float]) -> None:
        assert with_groq_retry(lambda: "ok") == "ok"
        assert slept == []

    def test_a_recovered_rate_limit_returns_the_answer(self, slept: list[float]) -> None:
        calls = {"n": 0}

        def _limited_once():
            calls["n"] += 1
            if calls["n"] == 1:
                raise _RateLimit(wait=3.0)
            return "recovered"

        assert with_groq_retry(_limited_once, max_retries=4) == "recovered"
        assert len(slept) == 1

    def test_a_non_rate_limit_error_is_not_retried(self, slept: list[float]) -> None:
        def _bad_request():
            raise ValueError("model_not_found")

        with pytest.raises(ValueError):
            with_groq_retry(_bad_request, max_retries=4)
        assert slept == []

    def test_a_quota_exhaustion_still_fails_fast(self, slept: list[float]) -> None:
        """Beyond the cap is a daily quota, not a burst; waiting 11 minutes
        helps nobody."""
        def _quota():
            raise _RateLimit(wait=665.0)

        with pytest.raises(_RateLimit):
            with_groq_retry(_quota, max_retries=4)
        assert slept == []


class TestTheRequestPathSetsADeadline:
    def test_the_chat_route_bounds_its_own_lifetime(self) -> None:
        import inspect

        from mao.api import main

        source = inspect.getsource(main)
        assert "request_deadline" in source, (
            "no request-level deadline: a clinical /chat makes 7 sequential "
            "gateway calls and nothing bounds their total wait"
        )

    def test_the_deadline_is_shorter_than_the_worst_case_serial_wait(self) -> None:
        from mao.api.main import REQUEST_DEADLINE_SECONDS
        from mao.core.retry import MAX_RATE_LIMIT_WAIT_SECONDS

        assert REQUEST_DEADLINE_SECONDS < 7 * 4 * MAX_RATE_LIMIT_WAIT_SECONDS


class TestTheDeadlineSurvivesIntoWorkerThreads:
    """The graph — and therefore every sleep — runs in a ThreadPoolExecutor.

    `loop.run_in_executor` does NOT copy contextvars into the worker;
    `mao/api/invocation.py` documents this hazard directly, which is why the
    usage collector is bound inside the worker rather than around the call.

    A first version of this test copied the context by hand and asserted on the
    result. It passed while the production path was still unbounded — it was
    testing `contextvars`, not this system. So it exercises `run_graph` itself.
    """

    @pytest.mark.asyncio
    async def test_run_graph_carries_the_deadline_into_the_worker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mao.api.invocation as invocation

        seen: dict[str, float | None] = {}

        def _capture(graph, state):  # noqa: ANN001
            seen["left"] = remaining_seconds()
            return {"response": "ok"}

        monkeypatch.setattr(invocation, "invoke_with_usage", _capture)
        monkeypatch.setattr(invocation, "get_graph", lambda: object())

        with request_deadline(30.0):
            await invocation.run_graph({"user_query": "q"}, "req-1")

        assert seen["left"] is not None, (
            "the graph worker saw no deadline — every rate-limit sleep inside "
            "the graph is still unbounded, which is the whole of B9(b)"
        )
        assert seen["left"] <= 30.0

    @pytest.mark.asyncio
    async def test_no_deadline_outside_a_bounded_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mao.api.invocation as invocation

        seen: dict[str, float | None] = {}

        def _capture(graph, state):  # noqa: ANN001
            seen["left"] = remaining_seconds()
            return {}

        monkeypatch.setattr(invocation, "invoke_with_usage", _capture)
        monkeypatch.setattr(invocation, "get_graph", lambda: object())

        await invocation.run_graph({"user_query": "q"}, "req-1")
        assert seen["left"] is None

    def test_time_actually_elapses_against_the_budget(self) -> None:
        with request_deadline(1.0):
            first = remaining_seconds()
            time.sleep(0.05)
            second = remaining_seconds()
        assert first is not None and second is not None
        assert second < first
