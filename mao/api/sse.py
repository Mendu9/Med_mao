"""Server-sent-event framing for the streaming chat endpoint.

Two generators, and the difference between them is a safety decision, not a
performance one:

`raw_token_stream` forwards provider tokens the moment they arrive. Those tokens
have passed through neither the council, the NLI gate, the judge, nor the output
guardrails, so it is only ever reachable when the safety policy exempts the
request from verification — `may_stream_raw_tokens` decides that, and it fails
closed.

`chunked_text_stream` is the default. It streams the text that came *out* of the
verification chain. Slower to first token, deliberately: streaming and
non-streaming must carry equivalent safety guarantees.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

logger = logging.getLogger(__name__)

DONE = "data: [DONE]\n\n"


def sse_encode(token: str) -> str:
    """Encode a token as a safe SSE data line.

    JSON-encoded so newlines, colons and other SSE-special characters cannot
    corrupt the frame or be misread by EventSource. The client must JSON.parse
    each data payload.
    """
    return f"data: {json.dumps(token)}\n\n"


def meta_payload(result: dict[str, Any], agent_used: str, latency_ms: float) -> str:
    """The single metadata frame sent after the last token."""
    metadata = result.get("metadata") or {}
    payload = {
        "intent": result.get("intent", ""),
        "agent_used": agent_used,
        "latency_ms": round(latency_ms, 1),
        "sources": metadata.get("sources", []),
        "web_sources": metadata.get("web_sources", []),
        "uncertainty_flag": metadata.get("uncertainty_flag", False),
        # A-2. The buffered route returns this as a ChatResponse field; the
        # streamed one has only this frame, and the two must not differ about
        # what the boundary removed.
        "redaction_notice": metadata.get("redaction_notice", []),
    }
    return f"data: __meta__:{json.dumps(payload)}\n\n"


async def raw_token_stream(
    *,
    produce_tokens: Callable[[], Iterator[str]],
    submit: Callable[[Callable[[], None]], Any],
    trailer: Callable[[], str],
    request_id: str,
) -> AsyncIterator[str]:
    """Bridge a synchronous token generator onto the event loop.

    The queue is unbounded, but the producer blocks on each `put` via
    `run_coroutine_threadsafe(...).result()`, so a slow consumer back-pressures
    the producer thread rather than growing the queue without limit.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def _produce() -> None:
        try:
            for token in produce_tokens():
                if token:
                    asyncio.run_coroutine_threadsafe(queue.put(token), loop).result()
        except Exception as exc:  # noqa: BLE001
            logger.error("Token stream failed request_id=%s: %s", request_id, exc)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    submit(_produce)

    while True:
        token = await queue.get()
        if token is None:
            break
        yield sse_encode(token)
        await asyncio.sleep(0)

    yield trailer()
    yield DONE


async def chunked_text_stream(
    *,
    text: str,
    trailer: Callable[[], str],
) -> AsyncIterator[str]:
    """Chunk already-verified text to the client, word by word."""
    for word in text.split(" "):
        if word:
            yield sse_encode(word + " ")
            await asyncio.sleep(0)
    yield trailer()
    yield DONE


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
