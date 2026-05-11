"""
mao/core/web_search.py
----------------------
Resilient web search with provider fallback chain.

Priority order:
  1. Brave Search API  (if BRAVE_API_KEY env var set — free tier: 2000 queries/month)
  2. SerpAPI           (if SERPAPI_KEY env var set — free tier: 100 queries/month)
  3. DuckDuckGo        via duckduckgo-search library (rate-limited, but free fallback)
  4. Empty results     (graceful degradation)

The caller never sees an exception — failures log a warning and return [].

Return format (all providers normalised to this):
    [{"title": str, "href": str, "body": str}, ...]

Integration points:
  - mao/agents/clinical_agent.py  _web_search_clinical()
  - mao/agents/tool_agent.py      _web_search()
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _brave_search(query: str, num_results: int) -> list[dict[str, str]]:
    """Brave Search API — https://api.search.brave.com"""
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        raise ValueError("BRAVE_API_KEY not set")

    import httpx

    resp = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": min(num_results, 10)},
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    results: list[dict[str, str]] = []
    for item in data.get("web", {}).get("results", []):
        results.append({
            "title": item.get("title", ""),
            "href":  item.get("url", ""),
            "body":  item.get("description", ""),
        })
    return results


def _serpapi_search(query: str, num_results: int) -> list[dict[str, str]]:
    """SerpAPI (Google backend) — https://serpapi.com"""
    api_key = os.environ.get("SERPAPI_KEY", "")
    if not api_key:
        raise ValueError("SERPAPI_KEY not set")

    import httpx

    resp = httpx.get(
        "https://serpapi.com/search",
        params={
            "q":       query,
            "num":     min(num_results, 10),
            "api_key": api_key,
            "engine":  "google",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    results: list[dict[str, str]] = []
    for item in data.get("organic_results", []):
        results.append({
            "title": item.get("title", ""),
            "href":  item.get("link", ""),
            "body":  item.get("snippet", ""),
        })
    return results


def _ddg_search(query: str, num_results: int) -> list[dict[str, str]]:
    """DuckDuckGo via duckduckgo-search — free but rate-limited."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError(
            "duckduckgo-search not installed. Run: pip install duckduckgo-search"
        )
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))
    return results  # already has title/href/body keys


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PROVIDERS: list[tuple[str, Any]] = [
    ("brave",      _brave_search),
    ("serpapi",    _serpapi_search),
    ("duckduckgo", _ddg_search),
]


def web_search(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """
    Search the web using the best available provider.

    Tries providers in order: Brave -> SerpAPI -> DuckDuckGo.
    Skips providers whose API key is not configured (ValueError).
    Logs a warning for providers that fail due to network/rate errors.

    Args:
        query:       The search query string.
        num_results: Maximum number of results to return (provider caps may apply).

    Returns:
        List of dicts with keys ``title``, ``href``, ``body``.
        Returns ``[]`` if all providers fail -- never raises.
    """
    for name, fn in _PROVIDERS:
        try:
            results = fn(query, num_results)
            if results:
                logger.debug(
                    "web_search: provider=%s query=%r results=%d",
                    name,
                    query[:60],
                    len(results),
                )
                return results
        except ValueError:
            # API key not configured -- skip silently (expected path)
            pass
        except Exception as exc:
            logger.warning("web_search provider %s failed: %s", name, exc)

    logger.warning("web_search: all providers failed for query %r", query[:60])
    return []


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.DEBUG)

    hits = web_search("Alzheimer treatment latest research", num_results=3)
    for hit in hits:
        print(f"[{hit['title']}] {hit['href']}")
        print(f"  {hit['body'][:120]}")
        print()
