import time
import logging
from typing import Callable, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)

def with_groq_retry(fn: Callable[[], T], max_retries: int = 4, base_delay: float = 1.0) -> T:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if "rate_limit" not in str(e).lower() and "429" not in str(e):
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("Groq rate limit hit, retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, max_retries)
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]
