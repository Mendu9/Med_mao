"""Running the graph for a request, with usage accounting and error translation.

Both chat endpoints did this identically, including the same nine-line
provider-outage `except` clause. One copy, so the two paths cannot drift on how
a failure is reported — or on whether usage is recorded at all.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import HTTPException

from mao.api.executor import get_executor
from mao.graph import get_graph
from mao.providers import usage

logger = logging.getLogger(__name__)


def invoke_with_usage(graph: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Run the graph and record what its model calls cost.

    The collector is bound *inside* this function because it runs in a
    ThreadPoolExecutor worker, and `run_in_executor` does not copy contextvars
    into the worker. Binding here means every gateway call the graph makes —
    including the council's, which reaches further threads via
    `asyncio.to_thread`, and that does copy context — lands in this request's
    totals rather than another request's or nowhere.
    """
    with usage.collecting() as totals:
        result = graph.invoke(state)
    if isinstance(result, dict):
        result["llm_usage"] = totals.to_dict()
    return result


async def run_graph(state: dict[str, Any], request_id: str) -> dict[str, Any]:
    """Invoke the graph off the event loop, translating failures to HTTP."""
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_executor(), invoke_with_usage, get_graph(), state
        )
    except Exception as exc:
        logger.error("Graph invocation failed request_id=%s: %s", request_id, exc)
        if _looks_like_a_connectivity_failure(exc):
            raise HTTPException(
                status_code=503,
                detail=(
                    "The AI service is temporarily unreachable. "
                    "Check your internet connection or provider credentials and try again."
                ),
            ) from exc
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc


def _looks_like_a_connectivity_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    return "connection error" in text or "connect" in text
