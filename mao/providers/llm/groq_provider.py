"""Groq implementation of the ChatProvider interface.

Wraps the existing `mao.core.llm` helpers so the retry, usage-tracking, and
tracing behaviour the system already relies on is preserved unchanged.
"""
from __future__ import annotations

from mao.providers.llm.base import ProviderResponse


class GroqChatProvider:
    """ChatProvider backed by Groq."""

    name = "groq"

    def complete(
        self,
        *,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        from mao.core.llm import chat

        text = chat(
            messages=messages,
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return ProviderResponse(text=text)
