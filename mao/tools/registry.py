"""ToolRegistry — typed, declared capabilities rather than a dict literal.

`01_ARCHITECTURE.md` requires a ToolRegistry and states the fields every tool
must declare; `00_RULES.md` requires tools to be centralised through it. Before
this, the only dispatch table lived inside the function that used it, so nothing
else in the system could discover what tools existed, what they cost, whether
they read or write, or how much to trust their output.

The field set matters for what comes later, not just for tidiness. A trace
records `tool_id`; a risk policy needs `read_or_write` and `trust_tier` to
decide what a tool's output may be used for; a budget policy needs cost and
latency. Declaring them now is what makes those decisions possible without
another migration.

Handlers take a single string and return a string. That is the ReAct loop's
contract: whatever a tool produces is fed straight back to the model as text, so
a handler that raises would break the loop rather than inform it. Every handler
therefore reports its own failure as its return value.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

ToolHandler = Callable[[str], str]


class ReadOrWrite(str, Enum):
    READ = "read"
    WRITE = "write"


class TrustTier(str, Enum):
    """How far a tool's output may be trusted without corroboration."""

    CURATED = "curated"      # peer-reviewed / vetted corpus
    REFERENCE = "reference"  # encyclopaedic, edited, but not peer-reviewed
    OPEN_WEB = "open_web"    # anyone can publish it
    DETERMINISTIC = "deterministic"  # computed locally, no external claim


@dataclass(frozen=True)
class ToolSpec:
    """One declared capability."""

    tool_id: str
    capability: str
    domains: tuple[str, ...]
    read_or_write: ReadOrWrite
    trust_tier: TrustTier
    input_schema: str
    output_schema: str
    typical_latency_ms: float
    cost_per_call_usd: float
    description: str = ""
    handler: ToolHandler | None = field(default=None, compare=False, repr=False)


class ToolRegistry:
    """tool_id -> ToolSpec, with dispatch."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> ToolSpec:
        if spec.tool_id in self._specs:
            raise ValueError(f"tool {spec.tool_id!r} is already registered")
        bound = ToolSpec(**{**spec.__dict__, "handler": handler})
        self._specs[spec.tool_id] = bound
        return bound

    def get(self, tool_id: str) -> ToolSpec:
        try:
            return self._specs[tool_id]
        except KeyError:
            raise KeyError(f"unknown tool {tool_id!r}") from None

    def names(self) -> list[str]:
        return sorted(self._specs)

    def all(self) -> list[ToolSpec]:
        return [self._specs[n] for n in self.names()]

    def invoke(self, tool_id: str, tool_input: str) -> str:
        """Run a tool. Always returns text, never raises.

        An unknown tool is reported back to the model rather than raised: the
        model chose the name, and telling it the name was wrong is more useful
        than terminating the request.
        """
        spec = self._specs.get(tool_id)
        if spec is None or spec.handler is None:
            logger.warning("Model requested unknown tool %r", tool_id)
            return f"Unknown tool: {tool_id}. Available: {', '.join(self.names())}"
        try:
            return spec.handler(tool_input)
        except Exception as exc:  # noqa: BLE001 - tool output is model input, not control flow
            logger.error("Tool %s failed on %r: %s", tool_id, tool_input, exc)
            return f"{tool_id} error: {exc}"

    def describe_for_prompt(self) -> str:
        """The tool list as the ReAct prompt presents it.

        Derived, so a newly registered tool is offered to the model without a
        second hand-maintained list drifting out of step with this one.
        """
        lines = [
            f"  {s.tool_id}({s.input_schema}) - {s.description} Returns: {s.output_schema}."
            for s in self.all()
        ]
        return "\n".join(lines)


_registry = ToolRegistry()


def registry() -> ToolRegistry:
    """The process-wide tool registry."""
    return _registry


def get_tool(tool_id: str) -> ToolSpec:
    return _registry.get(tool_id)
