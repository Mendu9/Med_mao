"""Model gateway — the single entry point for role-addressed LLM calls.

Agents call `complete(role=..., messages=...)`. They never name a model id and
never import a provider SDK. Provider-specific code stays behind
`mao.providers.llm`.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from mao.providers import usage
from mao.providers.llm.base import ChatProvider
from mao.providers.registry import ModelRecord, ModelRegistry, ModelRole

logger = logging.getLogger(__name__)

_registry: ModelRegistry | None = None
_provider: ChatProvider | None = None

# How much extra reasoning room a retry gets after a truncated, empty reply.
# Multiplies the declared overhead rather than the answer budget: it is the
# analysis channel that overran, not the answer.
_TRUNCATION_RETRY_FACTOR = 3


def registry() -> ModelRegistry:
    """Process-wide ModelRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry.default()
    return _registry


def reset_registry() -> None:
    """Drop the cached registry so env changes take effect. Test/CLI use only."""
    global _registry
    _registry = None


def resolve(role: ModelRole) -> ModelRecord:
    return registry().resolve(role)


def model_id_for(role: ModelRole) -> str:
    """The concrete, active model id bound to `role`."""
    return registry().model_id_for(role)


@dataclass(frozen=True)
class Completion:
    """A gateway response plus the provenance a trace needs."""

    text: str
    model_id: str
    provider: str
    role: ModelRole
    input_tokens: int = 0
    output_tokens: int = 0
    # The model hit its ceiling before finishing. An empty `text` then means
    # "cut off mid-thought", which is an infrastructure failure — not "the model
    # declined to answer", which is a verdict. Wave 6 blocker 3 was invisible
    # precisely because those two were the same empty string.
    truncated: bool = False

    @property
    def estimated_cost_usd(self) -> float:
        record = registry().get(self.model_id)
        if record is None:
            return 0.0
        return (
            self.input_tokens * record.cost_per_1m_input_usd
            + self.output_tokens * record.cost_per_1m_output_usd
        ) / 1_000_000


# ---------------------------------------------------------------------------
# Provider binding
# ---------------------------------------------------------------------------


def provider() -> ChatProvider:
    """The active chat provider, defaulting to Groq."""
    global _provider
    if _provider is None:
        from mao.providers.llm.groq_provider import GroqChatProvider

        _provider = GroqChatProvider()
    return _provider


def set_provider(new_provider: ChatProvider) -> None:
    """Bind a provider. Used by tests and by deployment wiring."""
    global _provider
    _provider = new_provider


def reset_provider() -> None:
    """Restore the default provider."""
    global _provider
    _provider = None


def complete(
    *,
    role: ModelRole,
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Completion:
    """Run a completion for a capability role.

    `max_tokens` is the budget for the **answer**. The bound model's analysis
    channel is paid for on top of it, from the record's declared
    `reasoning_overhead_tokens`, because the provider counts both against one
    ceiling. A call site therefore states what it needs to read back, and
    rebinding a role to a model that thinks harder re-sizes every one of that
    role's call sites at once.

    Getting this wrong is silent: the model spends the ceiling reasoning and
    returns an empty `content`, which every parser downstream reads as "said
    nothing". That is Wave 6 blocker 3, and it refused 5 of 5 ordinary clinical
    questions while the test suite stayed green.

    The declared overhead is sized from measurement, not from worst-case
    paranoia, because the provider's rate limiter charges the *requested*
    ceiling and not the tokens actually produced — a 429 observed during
    remediation reported "Limit 200000, Used 199385, Requested 2333". An
    over-generous ceiling therefore costs real quota on every call, including
    the overwhelming majority that never approach it.

    So the tail is handled by retrying rather than by pre-paying for it: if the
    model is cut off mid-thought and returns nothing readable, the call is made
    once more with a materially larger allowance. The common case stays cheap,
    the rare case self-heals, and neither one returns the silent empty string
    that was Wave 6 blocker 3.

    There is deliberately no `model_id` parameter: business logic must not be
    able to bypass role resolution and hand a raw (possibly retired) id to a
    provider.
    """
    record = resolve(role)
    active = provider()
    raw = active.complete(
        model_id=record.model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens + record.reasoning_overhead_tokens,
    )

    # Cut off before it said anything. Not a verdict — a budget failure.
    if raw.truncated and not raw.text.strip():
        retry_ceiling = max_tokens + record.reasoning_overhead_tokens * _TRUNCATION_RETRY_FACTOR
        logger.warning(
            "role=%s model=%s produced no content before its ceiling (answer %d "
            "+ overhead %d); retrying once at %d",
            role.name, record.model_id, max_tokens,
            record.reasoning_overhead_tokens, retry_ceiling,
        )
        raw = active.complete(
            model_id=record.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=retry_ceiling,
        )
        if raw.truncated and not raw.text.strip():
            # Report it loudly. Every parser downstream fails closed on an empty
            # reply, so this becomes a refused clinical answer, and the declared
            # overhead for this model is the thing to fix.
            logger.error(
                "role=%s model=%s returned no content even at %d tokens — raise "
                "reasoning_overhead_tokens for this model",
                role.name, record.model_id, retry_ceiling,
            )
    elif raw.truncated:
        logger.warning(
            "role=%s model=%s hit its ceiling (answer budget %d + reasoning "
            "overhead %d) — the answer may be cut short",
            role.name, record.model_id, max_tokens, record.reasoning_overhead_tokens,
        )

    completion = Completion(
        text=raw.text,
        model_id=record.model_id,
        provider=active.name,
        role=role,
        input_tokens=raw.input_tokens,
        output_tokens=raw.output_tokens,
        truncated=raw.truncated,
    )
    # Accounting happens here so no agent has to carry it. A no-op unless the
    # request bound a collector.
    usage.record(
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cost_usd=completion.estimated_cost_usd,
    )
    return completion


def stream(
    *,
    role: ModelRole,
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Iterator[str]:
    """Stream a completion for a capability role.

    Same contract as `complete`, including the reasoning-overhead budgeting: a
    caller asks for the answer it needs and the analysis channel is paid for on
    top. Streaming exists on the gateway so the API layer never has to reach
    past it into `mao.core.llm` to get incremental delivery — which it did, and
    which also meant that path resolved its model from a config alias rather
    than from the registry.
    """
    record = resolve(role)
    return provider().stream(
        model_id=record.model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens + record.reasoning_overhead_tokens,
    )
