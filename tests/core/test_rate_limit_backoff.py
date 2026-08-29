"""A rate limit must be waited out, not converted into a clinical refusal.

Measured live during Wave 7. The safety chain makes six SAFETY_JUDGE calls per
clinical request; the provider tier allows 8000 tokens per minute. A paced run
of five ordinary clinical questions produced:

    REFUSED 5/5  (rate-limited on 4)

`with_groq_retry` backed off 1s, 2s, 4s, 8s — fifteen seconds in total, against
a bucket that refills on a **one-minute** boundary. It could not clear the limit
by construction, so every retry was spent and the call still failed. The council
member then reported "VERDICT: FAIL. Agent error.", the answer was withheld, and
before the `review_unavailable` fix the clinician was told their answer had been
flagged for patient safety.

The provider says exactly how long to wait, in a `retry-after` header and in the
error text ("Please try again in 12m22.176s"). Honouring it converts a spurious
refusal into a slower, correct answer — which for clinical decision support is
the right trade.

Bounded deliberately: a wait longer than `MAX_RATE_LIMIT_WAIT_SECONDS` is a
daily-quota exhaustion, not a burst, and blocking a request path for twelve
minutes is not a kindness. Those fail fast and surface as unavailability.
"""
from __future__ import annotations

import pytest

from mao.core.retry import MAX_RATE_LIMIT_WAIT_SECONDS, _retry_after_seconds, with_groq_retry


class _RateLimited(Exception):
    """Shaped like the provider's 429, including its headers."""

    def __init__(self, message: str, retry_after: str | None = None) -> None:
        super().__init__(message)
        if retry_after is not None:
            self.response = type(
                "R", (), {"headers": {"retry-after": retry_after}}
            )()


class TestTheProvidersStatedWaitIsRead:
    def test_the_retry_after_header_is_used(self) -> None:
        exc = _RateLimited("Error code: 429 - rate_limit_exceeded", retry_after="30")
        assert _retry_after_seconds(exc) == pytest.approx(30.0)

    def test_a_wait_stated_only_in_the_message_is_parsed(self) -> None:
        exc = _RateLimited(
            "Error code: 429 - Rate limit reached. Please try again in 8.5s"
        )
        assert _retry_after_seconds(exc) == pytest.approx(8.5)

    def test_a_minutes_and_seconds_wait_is_parsed(self) -> None:
        exc = _RateLimited(
            "Error code: 429 - Rate limit reached. Please try again in 12m22.176s"
        )
        assert _retry_after_seconds(exc) == pytest.approx(742.176, abs=0.5)

    def test_no_stated_wait_returns_none(self) -> None:
        assert _retry_after_seconds(_RateLimited("Error code: 429 - rate limit")) is None


class TestAShortRateLimitIsWaitedOut:
    def test_the_call_succeeds_after_the_stated_wait(self, monkeypatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr("mao.core.retry.time.sleep", slept.append)

        attempts = {"n": 0}

        def _flaky():
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise _RateLimited("429 rate_limit_exceeded", retry_after="20")
            return "answer"

        assert with_groq_retry(_flaky) == "answer"
        assert slept, "did not wait at all"
        # A small margin over the stated wait, so we return just after the
        # window opens rather than just before it and 429 again.
        assert 20.0 <= slept[0] <= 22.0, (
            f"waited {slept} instead of the 20s the provider asked for"
        )

    def test_the_wait_is_long_enough_for_a_per_minute_bucket(
        self, monkeypatch
    ) -> None:
        """The whole defect: 1+2+4+8 can never clear a one-minute window."""
        assert MAX_RATE_LIMIT_WAIT_SECONDS >= 60

    def test_a_wait_beyond_the_cap_fails_fast(self, monkeypatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr("mao.core.retry.time.sleep", slept.append)

        def _daily_quota():
            raise _RateLimited(
                "429 - tokens per day (TPD). Please try again in 12m22.176s"
            )

        with pytest.raises(_RateLimited):
            with_groq_retry(_daily_quota)
        assert not slept, "blocked the request path on a daily-quota exhaustion"

    def test_an_unstated_wait_still_backs_off(self, monkeypatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr("mao.core.retry.time.sleep", slept.append)

        attempts = {"n": 0}

        def _flaky():
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise _RateLimited("429 rate limit")
            return "ok"

        assert with_groq_retry(_flaky) == "ok"
        assert slept


class TestNonRateLimitErrorsAreUnchanged:
    def test_a_normal_error_is_raised_immediately(self, monkeypatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr("mao.core.retry.time.sleep", slept.append)

        def _boom():
            raise RuntimeError("model not found")

        with pytest.raises(RuntimeError):
            with_groq_retry(_boom)
        assert not slept

    def test_a_successful_call_does_not_sleep(self, monkeypatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr("mao.core.retry.time.sleep", slept.append)
        assert with_groq_retry(lambda: "fine") == "fine"
        assert not slept
