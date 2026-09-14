"""Running the graph for a request, with usage accounting and error translation.

Both chat endpoints did this identically, including the same nine-line
provider-outage `except` clause. One copy, so the two paths cannot drift on how
a failure is reported — or on whether usage is recorded at all.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import Any

from fastapi import HTTPException

from mao.core.deident.ambiguity import AmbiguousDocument
from mao.trust.handoff.compiler import HandoffRefused
from mao.trust.inputs import limits

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
    """Invoke the graph off the event loop, translating failures to HTTP.

    The caller's context is copied into the worker explicitly. `run_in_executor`
    does not do it — see `invoke_with_usage` — and the request deadline that
    bounds every rate-limit wait is a `ContextVar` set by the route. Without the
    copy the deadline stops at this line, while all the waiting happens on the
    other side of it, and the bound would be decorative.

    `invoke_with_usage` still binds the usage collector inside the worker rather
    than relying on this: a collector bound out here would be the same object
    across concurrent requests.
    """
    try:
        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        return await loop.run_in_executor(
            get_executor(), context.run, invoke_with_usage, get_graph(), state
        )
    except limits.InputTooLarge:
        # NOT a failure either — a deliberate refusal the routes turn into a
        # 413 naming the channel, the size and the limit. It is raised where the
        # bytes are read (`_extract_pdf_text`, `handle_audio`), which is inside
        # the graph, so without this clause the generic handler below turned
        # every oversized attachment into an opaque 500 and the caller was never
        # told what to send instead. Same reasoning as `AmbiguousDocument`
        # below, including staying out of `logger.exception`.
        raise
    except AmbiguousDocument:
        # NOT a failure — a deliberate refusal that the route turns into a 422
        # asking the caller for structured patient fields. Swallowing it here
        # made the whole refusal path unreachable: `main.py`'s
        # `except AmbiguousDocument` could never fire, every ambiguous upload
        # answered HTTP 500, and no caller was ever asked for the fields.
        #
        # It must also not reach `logger.exception` below, which writes the
        # exception's message to the application log — and that message named
        # the very text that could not be de-identified, putting the patient's
        # name in a log on the de-identification failure path.
        raise
    except HandoffRefused:
        # ADV20-1. NOT a failure — the third deliberate refusal raised from
        # inside the graph, and the third to need this clause. `compile_handoff`
        # raises it when no payload can be built that both excludes the
        # identifiers and carries enough of the case to answer; the route turns
        # it into a 422 naming the structured fields that would resolve it.
        #
        # Without this clause it fell to the generic handler below and the
        # caller got a 500 with none of those seven fields — so the product's
        # own refusal -> structured-resupply workflow could not complete, while
        # the refusal message went to the application log instead, where no
        # caller can act on it. Section P made that the common path rather than
        # a rare one.
        #
        # Staying out of `logger.exception` matters for the same reason as the
        # two clauses above, though not for the same risk: the message holds no
        # document text, but a deliberate refusal logged at ERROR with a
        # traceback trains operators to read this path as broken.
        raise
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
