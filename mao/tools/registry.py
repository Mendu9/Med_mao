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


class AuthScope(str, Enum):
    """What a tool needs to be allowed to reach.

    `01_ARCHITECTURE.md` mandates this field and it was missing, so the registry
    could describe what a tool *costs* but not what it can *touch* — which is
    the half a risk policy actually needs. 00_RULES forbids unrestricted SQL,
    shell, filesystem, secrets and operational DB access; naming the scope is
    what lets that be checked rather than assumed.
    """

    NONE = "none"                    # pure computation, no I/O
    PUBLIC_NETWORK = "public_network"  # unauthenticated outbound HTTP
    PROVIDER_API = "provider_api"    # authenticated third-party API
    INTERNAL_READ = "internal_read"  # read-only internal data
    INTERNAL_WRITE = "internal_write"  # mutates internal state


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
    # Architecture-mandated, and previously absent. PROJECT_STATE claimed this
    # dataclass carried "every architecture-mandated field"; the Wave 6 review
    # found that claim was wrong, and these are the two it was wrong about.
    auth_scope: AuthScope = AuthScope.NONE
    # How this tool is known to fail, so a caller can reason about degraded
    # output instead of discovering the modes in production. `invoke` returns
    # failures as text by contract, which makes them easy to miss.
    failure_modes: tuple[str, ...] = ()
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

    # -- capability lookup --------------------------------------------------
    #
    # `01_ARCHITECTURE.md` lists a CapabilityRegistry alongside the
    # ToolRegistry. It is not built as a second registry, because a second
    # registry over the same objects is a second thing to keep in step, and
    # every tool already *declares* its capability — the missing part was that
    # nothing could query by it. Indexing what is already declared is the whole
    # of what a CapabilityRegistry would have done here; a separate one would
    # add a synchronisation problem and no answer.

    def capabilities(self) -> list[str]:
        """Every capability the registered tools provide."""
        return sorted({s.capability for s in self._specs.values()})

    def by_capability(self, capability: str) -> list[ToolSpec]:
        """Tools providing a capability, cheapest first.

        Ordered by cost then latency so a caller asking for a capability rather
        than a tool gets the cheapest way to obtain it, which is the point of
        addressing capabilities instead of names.
        """
        return sorted(
            (s for s in self._specs.values() if s.capability == capability),
            key=lambda s: (s.cost_per_call_usd, s.typical_latency_ms, s.tool_id),
        )

    def for_domain(self, domain: str) -> list[ToolSpec]:
        """Tools declared usable in a domain."""
        return [s for s in self.all() if domain in s.domains]

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
