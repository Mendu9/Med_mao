"""Modular prompt package.

Prompts are grouped by capability, one module each — deliberately not one giant
prompt file. `get_prompt("router.classify")` is the only supported lookup;
agents must not inline production prompt text.
"""
from __future__ import annotations

from mao.prompts import clinical, evidence_qa, routing, verification
from mao.prompts.registry import PromptRegistry, PromptSpec

_registry = PromptRegistry()

for _module in (routing, verification, evidence_qa, clinical):
    for _spec in _module.PROMPTS:
        _registry.register(_spec)


def registry() -> PromptRegistry:
    """The process-wide prompt registry."""
    return _registry


def get_prompt(name: str) -> PromptSpec:
    """Look up a versioned prompt by name."""
    return _registry.get(name)


__all__ = ["PromptRegistry", "PromptSpec", "get_prompt", "registry"]
