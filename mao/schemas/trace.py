"""TraceSchema — the per-request record later policy learning is trained from.

Phase 1 only *captures* these fields. No training, no policy selection, no
feedback loop is wired up here; that is Phase 5 work.

Privacy: a trace deliberately carries no raw user query and no response text.
It records what was done and how well it went, not what was said.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCallTrace:
    """One tool invocation inside a request."""

    tool_id: str
    latency_ms: float
    ok: bool
    error: str = ""


@dataclass
class RetrievalTrace:
    """Retrieval work performed for a request."""

    index_version: str
    k: int
    hits: int
    reranker_id: str = ""
    top_score: float = 0.0


@dataclass
class TraceSchema:
    """Canonical per-request trace.

    Field set is fixed by `01_ARCHITECTURE.md` § "Trace / Experience Metadata".
    """

    trace_id: str
    workflow: str
    risk_level: str

    # Model / provider resolution
    provider: str
    model_id: str
    model_role: str

    # Prompt provenance — "<name>@<version>"
    prompt_ref: str

    # Policy provenance
    policy_version: str

    latency_ms: float

    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    retrieval: RetrievalTrace | None = None

    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    quality_scores: dict[str, float] = field(default_factory=dict)
    safety_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TraceSchema:
        data = dict(payload)
        data["tool_calls"] = [ToolCallTrace(**t) for t in data.get("tool_calls") or []]
        retrieval = data.get("retrieval")
        data["retrieval"] = RetrievalTrace(**retrieval) if retrieval else None
        return cls(**data)
