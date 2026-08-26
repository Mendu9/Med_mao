"""PromptSpec / PromptRegistry — versioned, traceable production prompts.

Every production prompt declares a name, a version, its required variables, and
the output contract callers may rely on. Responses record `spec.trace_ref` so a
trace can say exactly which prompt produced it.

Rendering substitutes only *declared* variables. Prompts that legitimately
contain literal braces (JSON schemas, citation templates) are therefore safe —
unlike `str.format`, which would treat them as fields.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    """One versioned prompt."""

    name: str
    version: str
    template: str
    required_variables: tuple[str, ...]
    output_contract: str
    description: str = ""

    @property
    def trace_ref(self) -> str:
        """Stable identifier recorded on traces: '<name>@<version>'."""
        return f"{self.name}@{self.version}"

    def render(self, **values: object) -> str:
        """Fill declared variables. Raises ValueError if any are missing."""
        missing = [v for v in self.required_variables if v not in values]
        if missing:
            raise ValueError(
                f"prompt {self.trace_ref} missing required variable(s): {', '.join(missing)}"
            )
        rendered = self.template
        for name in self.required_variables:
            rendered = rendered.replace("{" + name + "}", str(values[name]))
        return rendered


class PromptRegistry:
    """Name -> PromptSpec lookup."""

    def __init__(self) -> None:
        self._specs: dict[str, PromptSpec] = {}

    def register(self, spec: PromptSpec) -> PromptSpec:
        if spec.name in self._specs:
            raise ValueError(f"prompt {spec.name!r} is already registered")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> PromptSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(f"unknown prompt {name!r}") from None

    def all(self) -> list[PromptSpec]:
        return list(self._specs.values())

    def names(self) -> list[str]:
        return sorted(self._specs)
