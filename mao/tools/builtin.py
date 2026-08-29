"""The tools the ReAct loop can call.

Each handler takes one string and returns one string, and reports its own
failures as text — the ReAct loop feeds tool output straight back to the model,
so an exception would end the turn instead of letting the model recover.
"""
from __future__ import annotations

import logging
import re

from mao.tools.registry import AuthScope, ReadOrWrite, ToolSpec, TrustTier, registry

logger = logging.getLogger(__name__)


def _web_search(query: str) -> str:
    """Top web results as formatted text, via the multi-provider fallback."""
    from mao.core.web_search import web_search as provider

    results = provider(query, num_results=5)
    if not results:
        return "No web results found."
    return "\n\n".join(
        f"- {r.get('title', '')}\n  {r.get('body', '')[:300]}\n  URL: {r.get('href', '')}"
        for r in results
    )


def _wikipedia(topic: str) -> str:
    """English Wikipedia summary for a topic."""
    import wikipediaapi

    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="MAO-Agent/1.0 (https://github.com/mao-project)",
    )
    page = wiki.page(topic)
    if not page.exists():
        return f"No Wikipedia page found for '{topic}'."
    return page.summary[:1500]


def _calculator(expression: str) -> str:
    """Evaluate arithmetic with numexpr.

    numexpr supports arithmetic, trig and log and nothing else — deliberately
    NOT `eval()`, which would be arbitrary code execution driven by model output.
    """
    import numexpr

    clean = re.sub(r"```.*?```", "", expression, flags=re.DOTALL).strip()
    return str(float(numexpr.evaluate(clean)))


_SPECS = (
    (
        ToolSpec(
            tool_id="web_search",
            capability="search",
            domains=("general", "alzheimer", "stroke"),
            read_or_write=ReadOrWrite.READ,
            trust_tier=TrustTier.OPEN_WEB,
            input_schema="query: str",
            output_schema="list of {title, snippet, url} rendered as text",
            typical_latency_ms=1500.0,
            cost_per_call_usd=0.0,
            auth_scope=AuthScope.PUBLIC_NETWORK,
            failure_modes=(
                "provider rate limit or outage returns no results",
                "results may be stale, promotional, or factually wrong",
                "query terms are sent to a third party",
            ),
            description="Search the web for current information.",
        ),
        _web_search,
    ),
    (
        ToolSpec(
            tool_id="wikipedia",
            capability="reference_lookup",
            domains=("general",),
            read_or_write=ReadOrWrite.READ,
            trust_tier=TrustTier.REFERENCE,
            input_schema="topic: str",
            output_schema="summary text, truncated to 1500 chars",
            typical_latency_ms=800.0,
            cost_per_call_usd=0.0,
            auth_scope=AuthScope.PUBLIC_NETWORK,
            failure_modes=(
                "no page exists for the topic",
                "summary is truncated at 1500 characters mid-sentence",
                "content is editable by anyone and may be vandalised",
            ),
            description="Get an encyclopaedic summary for a topic.",
        ),
        _wikipedia,
    ),
    (
        ToolSpec(
            tool_id="calculator",
            capability="computation",
            domains=("general",),
            read_or_write=ReadOrWrite.READ,
            trust_tier=TrustTier.DETERMINISTIC,
            input_schema='expression: str (e.g. "15 * 0.15")',
            output_schema="numeric result as a string",
            typical_latency_ms=5.0,
            cost_per_call_usd=0.0,
            auth_scope=AuthScope.NONE,
            failure_modes=(
                "numexpr rejects any non-arithmetic expression",
                "float result loses precision on very large integers",
            ),
            description="Evaluate a math expression.",
        ),
        _calculator,
    ),
)


def register_builtin_tools() -> None:
    """Register the built-ins once, idempotently."""
    existing = set(registry().names())
    for spec, handler in _SPECS:
        if spec.tool_id not in existing:
            registry().register(spec, handler)
