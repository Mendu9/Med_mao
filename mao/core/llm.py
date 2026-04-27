"""
mao/core/llm.py — Ollama chat helper used by all agents.
"""
from __future__ import annotations
import logging
import ollama
from mao.core.config import cfg, TOKEN_BUDGET

logger = logging.getLogger(__name__)


def chat(
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    model: str | None = None,
) -> str:
    """Call Ollama chat. Returns content string."""
    try:
        resp = ollama.chat(
            model=model or cfg.groq_model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return resp["message"]["content"] or ""
    except Exception as exc:
        logger.error("Ollama LLM call failed: %s", exc)
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

    budgeted = truncate_to_budget(chunks=chunks, web=web, history=history, budget=TOKEN_BUDGET)

    context_parts = ["RETRIEVED CONTEXT:"] + budgeted["chunks"]
    if budgeted["web"]:
        context_parts += ["WEB RESULTS:"] + budgeted["web"]
    if budgeted["history"]:
        context_parts += ["CONVERSATION HISTORY:"] + budgeted["history"]

    full_user = "\n".join(context_parts) + "\n\nQUESTION: " + user_message

    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": full_user},
        ],
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    return resp["message"]["content"].strip()
