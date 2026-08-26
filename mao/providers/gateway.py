"""Model gateway — the single entry point for role-addressed LLM calls.

Agents call `complete(role=..., messages=...)`. They never name a model id and
never import a provider SDK. Provider-specific code stays behind
`mao.providers.llm`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from mao.providers.registry import ModelRecord, ModelRegistry, ModelRole

logger = logging.getLogger(__name__)

_registry: ModelRegistry | None = None


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

    @property
    def estimated_cost_usd(self) -> float:
        record = registry().get(self.model_id)
        if record is None:
            return 0.0
        return (
            self.input_tokens * record.cost_per_1m_input_usd
            + self.output_tokens * record.cost_per_1m_output_usd
        ) / 1_000_000
