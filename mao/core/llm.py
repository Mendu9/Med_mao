"""
mao/core/llm.py — single Groq call helper used by all agents.
"""
from __future__ import annotations
import logging
from groq import Groq
from mao.core.config import cfg

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
