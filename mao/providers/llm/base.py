"""Provider interface for chat completion.

Business logic depends on this Protocol, never on a vendor SDK. Provider
implementations must not own business policy — no routing, no risk decisions,
no prompt selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ProviderResponse:
    """A raw completion plus the usage a trace needs."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


@runtime_checkable
class ChatProvider(Protocol):
    """Minimal chat-completion provider."""

    name: str

    def complete(
        self,
        *,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse: ...
