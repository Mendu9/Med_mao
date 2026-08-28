"""Gateway stubs for agent tests.

Agents no longer call `mao.core.llm.chat`; every LLM call goes through
`mao.providers.gateway.complete(role=...)`, which returns a `Completion` object
rather than a bare string. These helpers keep the tests asserting behaviour
rather than plumbing — and let them assert the capability *role* an agent asked
for, which is the real contract, instead of whichever concrete model id the
registry resolved it to today.
"""
from __future__ import annotations

from collections.abc import Callable

from mao.providers.gateway import Completion
from mao.providers.registry import ModelRole


def _completion(text: str, role: ModelRole = ModelRole.GENERAL_SYNTHESIS) -> Completion:
    """A Completion carrying `text`, for `return_value=`."""
    return Completion(
        text=str(text),
        model_id="stub-model",
        provider="stub",
        role=role,
        input_tokens=0,
        output_tokens=0,
    )


def _completing(fn: Callable[..., str]) -> Callable[..., Completion]:
    """Wrap a text-returning fake so it can be used as `side_effect=`."""

    def _side_effect(**kwargs: object) -> Completion:
        return _completion(fn(**kwargs))

    return _side_effect
