"""
mao/core/llm.py — LLM chat helper used by all agents.
Primary backend: Groq. Ollama calls are preserved but commented out.
LangSmith tracing: enabled automatically when LANGSMITH_API_KEY is set.
"""
from __future__ import annotations
import json
import logging
import os
from collections.abc import Iterator
from typing import TypeVar

from pydantic import BaseModel
from groq import Groq
from mao.core.config import cfg, TOKEN_BUDGET
from mao.core.groq_usage import tracker as _usage_tracker

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangSmith tracing — enabled when LANGSMITH_API_KEY is present in env
# ---------------------------------------------------------------------------

def _configure_langsmith() -> bool:
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    if not api_key:
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "MAO"))
    os.environ.setdefault("LANGCHAIN_ENDPOINT", os.getenv("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com"))
    return True

_LANGSMITH_ENABLED = _configure_langsmith()

def _traceable(name: str = "", run_type: str = "llm"):
    """Decorator factory: wraps with LangSmith tracing when available, no-op otherwise."""
    def decorator(fn):
        if not _LANGSMITH_ENABLED:
            return fn
        try:
            from langsmith import traceable
            return traceable(name=name or fn.__name__, run_type=run_type)(fn)
        except ImportError:
            return fn
    return decorator

if _LANGSMITH_ENABLED:
    logger.info("LangSmith tracing enabled — project=%s", os.getenv("LANGCHAIN_PROJECT"))


# ---------------------------------------------------------------------------
# Groq singleton — lazy, initialised on first use
# ---------------------------------------------------------------------------

_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    """Return the shared Groq client, initialising it on first call."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=cfg.groq_api_key)
    return _groq_client


@_traceable(name="groq_chat", run_type="llm")
def chat(
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    model: str | None = None,
) -> str:
    """Call Groq chat. Returns content string."""
    from mao.core.retry import with_groq_retry
    try:
        client = _get_groq_client()
        resp = with_groq_retry(lambda: client.chat.completions.create(
            model=model or cfg.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ))
        usage = resp.usage
        if usage:
            _usage_tracker.record(
                model=model or cfg.groq_model,
                input_tokens=usage.prompt_tokens or 0,
                output_tokens=usage.completion_tokens or 0,
            )
        return resp.choices[0].message.content or ""
        # OLLAMA: resp = ollama.chat(
        # OLLAMA:     model=model or cfg.groq_model,
        # OLLAMA:     messages=messages,
        # OLLAMA:     options={"temperature": temperature, "num_predict": max_tokens},
        # OLLAMA: )
        # OLLAMA: return resp["message"]["content"] or ""
    except Exception as exc:
        logger.error("Groq LLM call failed: %s", exc)
        raise


async def achat(
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    model: str | None = None,
) -> str:
    """Async Groq chat — avoids thread executor overhead for async callers."""
    from groq import AsyncGroq
    _model = model or cfg.groq_model
    try:
        client = AsyncGroq(api_key=cfg.groq_api_key)
        resp = await client.chat.completions.create(
            model=_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = resp.usage
        if usage:
            _usage_tracker.record(
                model=_model,
                input_tokens=usage.prompt_tokens or 0,
                output_tokens=usage.completion_tokens or 0,
            )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.error("Async Groq LLM call failed: %s", exc)
        raise


def chat_stream(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Iterator[str]:
    """Streaming version — yields token strings one at a time.

    Uses Groq's stream=True API. Each yielded value is a non-empty delta
    string. Callers should iterate and buffer as needed.
    """
    client = _get_groq_client()
    stream = client.chat.completions.create(
        model=model or cfg.groq_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def chat_structured(
    messages: list[dict],
    schema: type[T],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> T:
    """Call Groq with JSON mode and validate response against a Pydantic model.

    Uses response_format={"type": "json_object"} — Groq enforces valid JSON output.
    Parses and validates against the provided Pydantic schema.
    Falls back to schema.model_validate({}) on parse failure.
    """
    from mao.core.retry import with_groq_retry

    try:
        client = _get_groq_client()
        resp = with_groq_retry(lambda: client.chat.completions.create(
            model=model or cfg.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        ))
        usage = resp.usage
        if usage:
            _usage_tracker.record(
                model=model or cfg.groq_model,
                input_tokens=usage.prompt_tokens or 0,
                output_tokens=usage.completion_tokens or 0,
            )
        raw = resp.choices[0].message.content or "{}"
        return schema.model_validate(json.loads(raw))
    except Exception as exc:
        logger.warning("chat_structured parse/validate failed (%s); returning empty model", exc)
        try:
            return schema.model_validate({})
        except Exception:
            raise exc


def chat_with_budget(
    model: str,
    system: str,
    user_message: str,
    chunks: list[str],
    web: list[str],
    history: list[str],
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """Call LLM after truncating context to stay within TOKEN_BUDGET."""
    from mao.core.token_counter import truncate_to_budget

    budgeted = truncate_to_budget(chunks=chunks, web=web, history=history, budget=TOKEN_BUDGET)

    context_parts = ["RETRIEVED CONTEXT:"] + budgeted["chunks"]
    if budgeted["web"]:
        context_parts += ["WEB RESULTS:"] + budgeted["web"]
    if budgeted["history"]:
        context_parts += ["CONVERSATION HISTORY:"] + budgeted["history"]

    full_user = "\n".join(context_parts) + "\n\nQUESTION: " + user_message

    from mao.core.retry import with_groq_retry
    client = _get_groq_client()
    resp = with_groq_retry(lambda: client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": full_user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,  # OLLAMA: was num_predict inside options={}
    ))
    usage = resp.usage
    if usage:
        _usage_tracker.record(
            model=model,
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
        )
    return (resp.choices[0].message.content or "").strip()
    # OLLAMA: resp = ollama.chat(
    # OLLAMA:     model=model,
    # OLLAMA:     messages=[
    # OLLAMA:         {"role": "system", "content": system},
    # OLLAMA:         {"role": "user", "content": full_user},
    # OLLAMA:     ],
    # OLLAMA:     options={"temperature": temperature, "num_predict": max_tokens},
    # OLLAMA: )
    # OLLAMA: return resp["message"]["content"].strip()
