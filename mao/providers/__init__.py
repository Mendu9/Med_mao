"""Provider-neutral model access.

`mao.providers.registry` owns model metadata; `mao.providers.gateway` is the
call surface business logic uses. Provider SDK code lives under
`mao.providers.llm` and must not own business policy.
"""
from __future__ import annotations

from mao.providers.gateway import model_id_for, registry, resolve
from mao.providers.registry import (
    Modality,
    ModelRecord,
    ModelRegistry,
    ModelRole,
    ModelStatus,
)

__all__ = [
    "Modality",
    "ModelRecord",
    "ModelRegistry",
    "ModelRole",
    "ModelStatus",
    "model_id_for",
    "registry",
    "resolve",
]
