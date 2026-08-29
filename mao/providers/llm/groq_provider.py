"""Groq implementation of the ChatProvider interface.

Wraps the existing `mao.core.llm` helpers so the retry, usage-tracking, and
tracing behaviour the system already relies on is preserved unchanged.
"""
from __future__ import annotations

from collections.abc import Iterator

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
        from mao.core.llm import chat_with_usage

        text, input_tokens, output_tokens, truncated = chat_with_usage(
            messages=messages,
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return ProviderResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            truncated=truncated,
        )

    def stream(
        self,
        *,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        from mao.core.llm import chat_stream

        return chat_stream(
            messages,
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
