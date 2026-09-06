r"""Recorders bound at the REAL sinks, not at an upstream abstraction.

`00_RULES.md`: "The egress layer must be tested at the real sink or SDK/network
seam, not only at an upstream abstraction." Two findings are the reason.

The first: a PHI assertion was bound to the gateway's `complete` function, which
is upstream of the egress authorisation — so patching it did not merely observe
the call, it REPLACED the control being tested. A test written that way passes
whether or not the control exists.

The second: a memory-sink assertion watched the upload path, which structurally
cannot carry a report to `remember()`. Disabling de-identification entirely left
it green. A probe bound to a seam that cannot see the defect proves nothing, so
every recorder here carries a non-vacuity assertion — `assert recorder.calls`
— and every test that uses one is expected to make it.

`RecordingProvider` binds at `gateway.set_provider`, which is the last thing
before the vendor SDK. Everything the process would have put on the wire passes
through it, including `synthesise_clinical`, which does not go through
`complete()` at all and which an upstream patch would therefore have missed
entirely.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from mao.providers.llm.base import ProviderResponse


@dataclass
class RecordingProvider:
    """Everything that would have gone over the wire, verbatim.

    Bound with `gateway.set_provider(recorder)`; restore with
    `gateway.reset_provider()`.
    """

    name: str = "recording"
    reply: str = "Recorded stub answer."
    calls: list[dict] = field(default_factory=list)

    def complete(
        self,
        *,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        self.calls.append(
            {
                "model_id": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return ProviderResponse(text=self.reply, input_tokens=0, output_tokens=0)

    def stream(
        self,
        *,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        self.complete(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        yield self.reply

    # -- what a test asks it -------------------------------------------------

    def text(self) -> str:
        """Every string that reached the provider, joined.

        Multipart content is flattened too. Reading only `str` contents would
        skip the text half of a vision request, which is the request most likely
        to carry a patient's details beside the image.
        """
        parts: list[str] = []
        for call in self.calls:
            for message in call["messages"]:
                content = message.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for piece in content:
                        if isinstance(piece, dict):
                            for value in piece.values():
                                if isinstance(value, str):
                                    parts.append(value)
                                elif isinstance(value, dict):
                                    parts.extend(
                                        v for v in value.values() if isinstance(v, str)
                                    )
        return "\n".join(parts)

    def leaked(self, identifiers: list[str]) -> list[str]:
        """Which of `identifiers` reached the provider. Empty is the assertion."""
        blob = self.text()
        return [identifier for identifier in identifiers if identifier in blob]


@dataclass
class RecordingMemory:
    """What reached the external memory store.

    The interface `mao.memory.interface.set_memory_store` expects, kept here so
    a memory assertion binds at the sink rather than at the node that calls it.
    """

    calls: list[tuple[str, str, str]] = field(default_factory=list)

    def remember(self, query: str, response: str, user_id: str) -> None:
        self.calls.append((query, response, user_id))

    def recall(self, query: str, user_id: str) -> str:  # pragma: no cover - unused
        return ""

    def text(self) -> str:
        return "\n".join(f"{q}\n{r}" for q, r, _ in self.calls)

    def leaked(self, identifiers: list[str]) -> list[str]:
        blob = self.text()
        return [identifier for identifier in identifiers if identifier in blob]
