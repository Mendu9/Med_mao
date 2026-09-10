r"""The chat routes' single entry into the protected input boundary.

## ADV15-8 — the channel that was never a call site

`apply_input_guardrails` was applied to `request.query` and to nothing else.
`request.chat_history` went into `make_initial_state` verbatim at all three call
sites, and from `state["chat_history"]` it reached the provider in three places
with no scrub anywhere between:

    mao/agents/router.py          _format_history(state["chat_history"][-3:])
    mao/agents/graphrag_agent.py  messages.extend(state["chat_history"][-4:])
    mao/agents/critic_agent.py    messages.extend(chat_history[-4:])

Measured over the real HTTP surface, a history turn containing
`Patient Name: Harold Nkemdirim, MRN: RGT/44219/B, NHS Number: 943 476 5919,
DOB: 12/03/1948` delivered all four identifiers to the provider.

That is not a de-identification defect — the de-identifier was never called. It
is ordinary traffic, too: a chat UI resends the turns it displayed, and what it
displayed is what the clinician typed, because the server never returns the
de-identified query to the client for it to send back instead.

So this module exists to make "which channels the route protects" a property of
ONE function rather than of what each route handler remembered. The route hands
over everything the caller sent; `protect()` decides what happens to each
channel; and a channel added later is a field on `RawSensitiveInput` or it does
not reach a model.

## ADV15-7 — bounding what the route accepts

`ChatRequest.query` was capped at 8,000 characters while `metadata` was an
unvalidated `dict[str, Any]`, so the cap sat on the one channel that did not
need it. The metadata dict is not free: `CacheKeyInputs.metadata_digest()`
serialises *all* of it with `json.dumps` before the graph is ever entered, so an
unbounded dict is CPU spent on the request thread by an unauthenticated caller —
and there is no authentication anywhere in `mao/api/`.

The extracted-text and attachment-byte caps are enforced where the bytes are
read, in `clinical_agent._extract_pdf_text`. This module bounds what the ROUTE
accepts before any of that, and `payload_too_large` is what turns every one of
those refusals into the same 413 on both `/chat` and `/chat/stream`.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import HTTPException

from mao.guardrails.db_helper import log_guardrail_event
from mao.guardrails.input_guardrails import GuardrailSeverity
from mao.trust.classes import RawSensitiveInput
from mao.trust.inputs import limits
from mao.trust.inputs.boundary import ProtectedInput, protect

logger = logging.getLogger(__name__)

#: Top-level metadata keys one request may carry.
#:
#: The real payloads are `image_b64`, `report_b64`, `audio_b64`, a handful of
#: scoping ids and `patient_fields`. Sixty-four leaves generous room for a
#: client that sends more than anyone anticipated, while refusing the dictionary
#: whose only purpose is to be large.
MAX_METADATA_KEYS = 64

#: Characters across the whole metadata structure, nested values included.
#:
#: Sized so one attachment at the shared decoded-byte cap still fits: base64 of
#: `MAX_DECODED_ATTACHMENT_BYTES` is about 5.34 million characters, and the
#: remainder is slack for a data-URI prefix and the scalar fields beside it.
#: Deliberately expressed FROM that constant rather than as an independent
#: number, so raising the attachment cap cannot leave this one silently below it.
MAX_METADATA_CHARS = limits.MAX_DECODED_ATTACHMENT_BYTES // 3 * 4 + 200_000

#: Containers and scalars visited while measuring. Bounds the walk itself, so a
#: deeply nested or very wide structure costs a fixed amount to refuse rather
#: than being measured in full first.
MAX_METADATA_NODES = 10_000

#: What a non-string scalar counts as. Charged at a flat rate rather than
#: measured with `str()`, because `str()` on a caller-supplied integer of a
#: million digits is itself the denial of service being refused.
_SCALAR_COST = 8


def bound_metadata(metadata: Mapping[str, Any]) -> None:
    """Refuse a metadata structure this endpoint will not process.

    Raises `limits.InputTooLarge`, which both routes translate into a 413.
    Never truncates: a request answered from a silently trimmed attachment is
    the same class of failure as a silently trimmed report.
    """
    if len(metadata) > MAX_METADATA_KEYS:
        raise limits.InputTooLarge(
            "request metadata", len(metadata), MAX_METADATA_KEYS
        )

    total = 0
    visited = 0
    stack: list[Any] = [metadata]
    while stack:
        item = stack.pop()
        visited += 1
        if visited > MAX_METADATA_NODES:
            raise limits.InputTooLarge(
                "request metadata", visited, MAX_METADATA_NODES
            )
        if isinstance(item, str):
            total += len(item)
        elif isinstance(item, Mapping):
            for key, value in item.items():
                total += len(str(key))
                stack.append(value)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            stack.extend(item)
        elif isinstance(item, (bytes, bytearray)):
            total += len(item)
        else:
            total += _SCALAR_COST
        if total > MAX_METADATA_CHARS:
            raise limits.InputTooLarge(
                "request metadata", total, MAX_METADATA_CHARS
            )


def protect_chat_request(
    *,
    query: str,
    chat_history: Sequence[Mapping[str, str]],
    metadata: Mapping[str, Any],
    trace_id: str,
) -> ProtectedInput:
    """Take every channel this request carries through the boundary, once.

    `refuse_ambiguity=False` is the chat posture the boundary documents: the
    clinician wrote this text and reads the answer, so an over-redaction is
    visible and recoverable while a leak is not. It is not a decision about
    documents — a report or a transcript is extracted deep in the graph, by the
    agent that received the attachment, and enters through `protect_channel`
    with the refusal on, against THIS request's protection record.
    """
    bound_metadata(metadata)

    protected = protect(
        RawSensitiveInput(
            query=query,
            chat_history=tuple(
                (str(turn.get("role", "user")), str(turn.get("content", "")))
                for turn in chat_history
            ),
        ),
        trace_id=trace_id,
        refuse_ambiguity=False,
    )

    # The `pii_detected` guardrail event moved here with the scrub itself.
    # `apply_input_guardrails` used to emit it as a side effect of scrubbing;
    # now that it no longer scrubs, the boundary is the only place that knows,
    # and dropping the event would have deleted an audit signal silently rather
    # than moving it.
    identifiers = protected.protection.identifiers
    if identifiers:
        log_guardrail_event(
            trace_id,
            "pii_detected",
            triggered=True,
            detail=f"{len(identifiers)} removed at the protected input boundary",
            severity=GuardrailSeverity.INFO,
        )
    return protected


def payload_too_large(exc: limits.InputTooLarge, request_id: str) -> HTTPException:
    """The 413 both chat routes answer with. One construction, so they cannot
    drift the way `/chat` and `/chat/stream` drifted over `AmbiguousDocument` —
    one answered 422 and the other a bare 500.

    The exception's own message names the channel, the size and the limit, and
    tells the caller to send a smaller document or split it. It never carries
    the content: this is raised on the path that handles patient data, and it
    reaches both the application log and an HTTP body.
    """
    logger.warning(
        "request_id=%s refusing oversized input: %s (channel=%s size=%d limit=%d)",
        request_id, exc, exc.channel, exc.size, exc.limit,
    )
    return HTTPException(
        status_code=413,
        detail=f"{exc} Quote this reference if you need help: {request_id}",
    )
