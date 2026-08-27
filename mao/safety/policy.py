"""SafetyPolicy — one owner for risk classification and safety thresholds.

Thresholds here reproduce the values the Phase 0 audit recorded in
`guardrails/output_guardrails.py` and `core/config.py`. Centralizing them does
not change behaviour; it removes the drift risk of four modules each keeping
their own copy, and gives every decision a `policy_version` a trace can cite.

Critical safety cannot be traded away for lower cost or latency, so the
`requires_*` predicates depend only on risk — never on transport (streaming vs
non-streaming) or on budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Bump whenever a threshold or a classification rule changes.
POLICY_VERSION = "2026.08-1"


class RiskLevel(str, Enum):
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


# Intents that are conversational only and carry no clinical content.
_LOW_RISK_INTENTS = frozenset({"chitchat"})

# Intents that carry direct patient-facing clinical weight.
_HIGH_RISK_INTENTS = frozenset({"clinical", "multimodal"})

# Request metadata keys whose presence means the request carries patient data.
#
# One definition, consumed by the graph's risk gate, the router's deterministic
# attachment rule, and the agents that read attachments. It previously existed
# as separate per-module tuples, and the copies had already diverged: the graph
# and router listed only image/report keys, so an uploaded voice sample — a
# recognised Alzheimer's biomarker modality — was classified STANDARD and could
# reach an agent without the controls patient data requires.
ATTACHMENT_KEYS: frozenset[str] = frozenset(
    {
        "image_b64",
        "image_url",
        "report_b64",
        "report_path",
        "audio_b64",
        "audio_path",
    }
)


def has_attachment(metadata: object) -> bool:
    """Whether request metadata carries patient data.

    Tolerates a missing or malformed metadata object by answering False — the
    caller escalates on True, so an unparseable payload must not silently
    *lower* risk relative to an absent one. Callers that cannot read metadata at
    all should escalate on their own.
    """
    if not isinstance(metadata, dict):
        return False
    return any(metadata.get(key) for key in ATTACHMENT_KEYS)


@dataclass(frozen=True)
class SafetyPolicy:
    """Immutable safety policy. Construct a new one to change behaviour."""

    policy_version: str = POLICY_VERSION

    # NLI entailment gate — fraction of claims unsupported by retrieved context
    nli_warn_ratio: float = 0.30
    nli_block_ratio: float = 0.70

    # LLM judge safety score (0-10); below block => replace, below warn => append
    judge_warn_score: int = 7
    judge_block_score: int = 5

    # EfficientNetB3 MRI stage-prediction confidence gate
    mri_confidence_gate: float = 0.60

    # -- risk classification -------------------------------------------------

    def risk_for(
        self,
        intent: str,
        *,
        has_attachment: bool = False,
    ) -> RiskLevel:
        """Classify a request's risk.

        An attachment (MRI image, patient report) always implies patient data
        and therefore HIGH risk, regardless of the routed intent.
        """
        if has_attachment:
            return RiskLevel.HIGH
        if intent in _HIGH_RISK_INTENTS:
            return RiskLevel.HIGH
        if intent in _LOW_RISK_INTENTS:
            return RiskLevel.LOW
        return RiskLevel.STANDARD

    # -- required controls ---------------------------------------------------

    def requires_verification(self, risk: RiskLevel) -> bool:
        """Whether the output verification chain must run.

        Only genuinely conversational traffic may skip it. This is the rule
        that makes streaming and non-streaming equivalent: the transport is not
        an input.
        """
        return risk is not RiskLevel.LOW

    def requires_clinical_disclaimer(self, risk: RiskLevel) -> bool:
        """Whether the mandatory clinical disclaimer must survive to the user."""
        return risk is RiskLevel.HIGH


_policy = SafetyPolicy()


def get_policy() -> SafetyPolicy:
    """The process-wide active safety policy."""
    return _policy


def resolve_risk(state: object) -> RiskLevel:
    """Read a request's risk level out of graph state.

    The single resolver. Everything that needs to know a request's risk — the
    graph, the verification node, the output guardrails, the API — calls this,
    so there is exactly one interpretation of a malformed or missing value.

    Fails safe to STANDARD. LOW is the only level permitted to skip output
    verification, so it must be reachable only by an explicit, exactly-matching
    write from the risk gate — never by a typo, a None, or a wrong type.
    """
    if not isinstance(state, dict):
        return RiskLevel.STANDARD
    raw = state.get("risk_level")
    if isinstance(raw, RiskLevel):
        return raw
    if isinstance(raw, str):
        try:
            return RiskLevel(raw)
        except ValueError:
            return RiskLevel.STANDARD
    return RiskLevel.STANDARD
