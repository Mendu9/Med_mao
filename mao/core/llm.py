"""
mao/core/llm.py — single Groq call helper used by all agents.
"""
from __future__ import annotations
import logging
from groq import Groq
from mao.core.config import cfg, TOKEN_BUDGET

logger = logging.getLogger(__name__)
_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=cfg.groq_api_key)
    return _client


def chat(
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    model: str | None = None,
) -> str:
    """Call Groq chat completions. Returns content string."""
    try:
        resp = get_client().chat.completions.create(
            model=model or cfg.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.error("Groq LLM call failed: %s", exc)
        raise


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
    from mao.core.retry import with_groq_retry

    budgeted = truncate_to_budget(chunks=chunks, web=web, history=history, budget=TOKEN_BUDGET)

    context_parts = ["RETRIEVED CONTEXT:"] + budgeted["chunks"]
    if budgeted["web"]:
        context_parts += ["WEB RESULTS:"] + budgeted["web"]
    if budgeted["history"]:
        context_parts += ["CONVERSATION HISTORY:"] + budgeted["history"]

    full_user = "\n".join(context_parts) + "\n\nQUESTION: " + user_message

    resp = with_groq_retry(lambda: get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": full_user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    ))
    return resp.choices[0].message.content.strip()
