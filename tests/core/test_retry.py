import pytest
import time
from unittest.mock import MagicMock, patch
from mao.core.retry import with_groq_retry

def test_retry_succeeds_on_first_try():
    fn = MagicMock(return_value="ok")
    result = with_groq_retry(fn)
    assert result == "ok"
    fn.assert_called_once()

def test_retry_retries_on_rate_limit():
    fn = MagicMock(side_effect=[Exception("rate_limit_exceeded"), "ok"])
    with patch("time.sleep"):
        result = with_groq_retry(fn, max_retries=3)
    assert result == "ok"
    assert fn.call_count == 2

def test_retry_raises_after_max():
    fn = MagicMock(side_effect=Exception("rate_limit_exceeded"))
    with patch("time.sleep"):
        with pytest.raises(Exception, match="rate_limit_exceeded"):
            with_groq_retry(fn, max_retries=2)
    assert fn.call_count == 2

def test_non_rate_limit_raises_immediately():
    fn = MagicMock(side_effect=Exception("connection_refused"))
    with patch("time.sleep") as mock_sleep:
        with pytest.raises(Exception, match="connection_refused"):
            with_groq_retry(fn, max_retries=4)
    fn.assert_called_once()
    mock_sleep.assert_not_called()

def test_retry_on_429_string():
    fn = MagicMock(side_effect=[Exception("HTTP 429 Too Many Requests"), "ok"])
    with patch("time.sleep"):
        result = with_groq_retry(fn, max_retries=3)
    assert result == "ok"
    assert fn.call_count == 2
