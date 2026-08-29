"""Provider retry policy.

The rate-limit path is the interesting one. The safety chain makes several
SAFETY_JUDGE calls per clinical request, so 429s are an ordinary operating
condition here rather than an exotic failure — and every one that is not
recovered becomes a withheld clinical answer, because the council fails closed
on a member that cannot reply.

The provider raises two different things under one status code, and they need
opposite handling:

  - a BURST limit (tokens per minute) clears in seconds. The previous policy
    backed off 1s, 2s, 4s, 8s and gave up; honouring the provider's stated wait
    instead turns a spurious refusal into a slower, correct answer, which for
    clinical decision support is the right trade.

  - a QUOTA exhaustion (tokens per day) does not clear in any useful time.
    Observed live: "tokens per day (TPD): Limit 200000, Used 199967 ... try
    again in 11m5.712s". Holding a clinical request open for eleven minutes
    helps nobody.

The provider states the wait in a `retry-after` header and again in the error
text, so the two are told apart by how long it asks for. Waits within the cap
are honoured; waits beyond it fail fast and surface as unavailability — which
the pipeline now reports honestly rather than as a patient-safety finding.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Callable, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)

# Longest we will hold a request open waiting for a rate-limit window. Must
# exceed the provider's per-minute bucket, or a burst limit can never be cleared
# — which was the whole defect.
MAX_RATE_LIMIT_WAIT_SECONDS = 75.0

# "Please try again in 12m22.176s", "try again in 8.5s"
_WAIT_RE = re.compile(
    r"try again in\s+(?:(?P<minutes>\d+)m)?(?P<seconds>[\d.]+)s", re.IGNORECASE
)


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "rate_limit" in text or "rate limit" in text or "429" in text


def _retry_after_seconds(exc: Exception) -> float | None:
    """How long the provider asked us to wait, or None if it did not say.

    The header is authoritative; the message is the fallback, because not every
    error path populates headers and the wait is stated in both.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

    match = _WAIT_RE.search(str(exc))
    if match:
        minutes = float(match.group("minutes") or 0)
        return minutes * 60 + float(match.group("seconds"))
    return None


def with_groq_retry(fn: Callable[[], T], max_retries: int = 4, base_delay: float = 1.0) -> T:
    """Call `fn`, waiting out transient rate limits.

    Only rate limits are retried. Everything else raises immediately: a wrong
    model id or a malformed request will not become correct by being repeated.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if not _is_rate_limit(e):
                raise

            stated = _retry_after_seconds(e)
            if stated is not None and stated > MAX_RATE_LIMIT_WAIT_SECONDS:
                # A quota exhaustion, not a burst. Waiting will not help within
                # any reasonable request lifetime.
                logger.error(
                    "Provider asked for %.0fs, beyond the %.0fs cap — not waiting",
                    stated, MAX_RATE_LIMIT_WAIT_SECONDS,
                )
                raise

            # Honour the stated wait; fall back to exponential backoff when the
            # provider did not say. Add a small margin so we return just after
            # the window opens rather than just before it.
            delay = (
                min(stated + 1.0, MAX_RATE_LIMIT_WAIT_SECONDS)
                if stated is not None
                else base_delay * (2 ** attempt)
            )
            logger.warning(
                "Provider rate limit, waiting %.1fs (attempt %d/%d, stated=%s)",
                delay, attempt + 1, max_retries, stated,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]
