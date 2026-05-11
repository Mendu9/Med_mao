"""
tests/core/test_web_search.py
------------------------------
Unit tests for mao.core.web_search -- all providers mocked so no network calls.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from mao.core.web_search import web_search


def test_web_search_brave_succeeds() -> None:
    """Uses Brave when BRAVE_API_KEY is set and returns results."""
    fake_results = [
        {"title": "AD Treatment", "href": "http://example.com", "body": "Donepezil is used..."}
    ]
    with patch.dict(os.environ, {"BRAVE_API_KEY": "test-key"}), \
         patch("mao.core.web_search._brave_search", return_value=fake_results):
        results = web_search("Alzheimer treatment", num_results=3)

    assert len(results) == 1
    assert results[0]["title"] == "AD Treatment"


def test_web_search_falls_back_to_ddg() -> None:
    """Falls back to DDG when no API keys are set."""
    env_without_keys = {k: v for k, v in os.environ.items()
                        if k not in ("BRAVE_API_KEY", "SERPAPI_KEY")}
    fake_ddg = [{"title": "DDG Result", "href": "http://ddg.com", "body": "..."}]
    with patch.dict(os.environ, env_without_keys, clear=True), \
         patch("mao.core.web_search._ddg_search", return_value=fake_ddg):
        results = web_search("test query")

    assert results[0]["title"] == "DDG Result"


def test_web_search_returns_empty_on_all_failure() -> None:
    """Returns [] when all providers fail -- never raises."""
    env_without_keys = {k: v for k, v in os.environ.items()
                        if k not in ("BRAVE_API_KEY", "SERPAPI_KEY")}
    with patch.dict(os.environ, env_without_keys, clear=True), \
         patch("mao.core.web_search._ddg_search", side_effect=Exception("Rate limited")):
        results = web_search("test query")

    assert results == []


def test_web_search_skips_failed_brave_falls_through_to_ddg() -> None:
    """Brave failing (e.g. bad key) silently falls through to DDG."""
    fake_ddg = [{"title": "DDG Fallback", "href": "http://ddg.com", "body": "..."}]
    with patch.dict(os.environ, {"BRAVE_API_KEY": "bad-key"}), \
         patch("mao.core.web_search._brave_search", side_effect=Exception("401 Unauthorized")), \
         patch("mao.core.web_search._ddg_search", return_value=fake_ddg):
        results = web_search("test query")

    assert results[0]["title"] == "DDG Fallback"


def test_web_search_brave_skipped_silently_when_no_key() -> None:
    """ValueError from missing key is swallowed, not logged as a warning."""
    env_without_keys = {k: v for k, v in os.environ.items()
                        if k not in ("BRAVE_API_KEY", "SERPAPI_KEY")}
    fake_ddg = [{"title": "DDG Only", "href": "http://ddg.com", "body": "fallback"}]
    with patch.dict(os.environ, env_without_keys, clear=True), \
         patch("mao.core.web_search._ddg_search", return_value=fake_ddg):
        results = web_search("no key test")

    assert results[0]["title"] == "DDG Only"


def test_web_search_serpapi_used_when_brave_missing() -> None:
    """SerpAPI is used when BRAVE_API_KEY is absent but SERPAPI_KEY is set."""
    env = {k: v for k, v in os.environ.items() if k != "BRAVE_API_KEY"}
    env["SERPAPI_KEY"] = "serp-test-key"
    fake_serp = [{"title": "SerpAPI Result", "href": "http://serp.com", "body": "serp body"}]
    with patch.dict(os.environ, env, clear=True), \
         patch("mao.core.web_search._serpapi_search", return_value=fake_serp):
        results = web_search("serp query", num_results=2)

    assert results[0]["title"] == "SerpAPI Result"
