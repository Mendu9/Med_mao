"""Provider interface for chat completion.

Business logic depends on this Protocol, never on a vendor SDK. Provider
implementations must not own business policy — no routing, no risk decisions,
no prompt selection.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ProviderResponse:
    """A raw completion plus the usage a trace needs."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    # The provider stopped because the token ceiling was reached, not because
    # the model finished. On a reasoning model this is how an under-budgeted
    # call presents: a perfectly successful HTTP 200 carrying an empty string.
    truncated: bool = False


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

    def stream(
        self,
        *,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        """Yield answer deltas as they arrive.

        Declared here so the API layer does not have to import `mao.core.llm`
        to stream. That import was one of three non-agent gateway bypasses the
        architecture review found, and it is the one on the request path.
        """
        ...
