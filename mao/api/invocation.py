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
from mao.core.state import MAOState
from mao.graph import get_graph
from mao.providers import usage

logger = logging.getLogger(__name__)


def invoke_with_usage(graph: Any, state: MAOState) -> dict[str, Any]:
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


def client_safe_detail(exc: Exception, request_id: str) -> str:
    """What the client may be told about a failure.

    The exception string used to be returned verbatim as
    `detail=f"Agent error: {exc}"`. This graph handles PHI, and an exception
    message can carry whatever was being processed when it raised — a prompt
    fragment, a row, a filename. That is a disclosure channel held open for the
    benefit of a message nobody outside the process can act on anyway.

    The request id is the useful half: it is already on the response header and
    in every log line for the request, so it is what turns a support call into a
    lookup. The cause stays in the log.
    """
    if _looks_like_a_connectivity_failure(exc):
        return (
            "The AI service is temporarily unreachable. Check your connection or "
            f"provider credentials and try again. (request {request_id})"
        )
    return (
        "The request could not be completed. Quote this reference when reporting "
        f"the problem: {request_id}"
    )


async def run_graph(state: MAOState, request_id: str) -> dict[str, Any]:
    """Invoke the graph off the event loop, translating failures to HTTP."""
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_executor(), invoke_with_usage, get_graph(), state
        )
    except Exception as exc:
        # Full cause here, where it is useful and stays inside the process.
        logger.exception("Graph invocation failed request_id=%s: %s", request_id, exc)
        status = 503 if _looks_like_a_connectivity_failure(exc) else 500
        raise HTTPException(
            status_code=status, detail=client_safe_detail(exc, request_id)
        ) from exc


def _looks_like_a_connectivity_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    return "connection error" in text or "connect" in text
