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

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

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

# Metadata keys that name a resource for the SERVER to open, rather than
# carrying content the caller already holds.
#
# These are refused at the API boundary. `report_path` reached
# `PdfReader(path)` and `audio_path` reached whisper, so either one was an
# unauthenticated arbitrary server-side file read whose text was then summarised
# back into the response — read and exfiltrate in one request. `00_RULES`
# forbids exposing unrestricted filesystem access.
#
# They stay in `ATTACHMENT_KEYS` because internal and CLI callers still use
# them, and if one ever reappears in request metadata it must still classify
# HIGH rather than slipping through as unrecognised.
SERVER_LOCATOR_KEYS: frozenset[str] = frozenset({"report_path", "audio_path"})

# Request metadata keys that do NOT imply patient data.
#
# The default is inverted, for the reason `mao/api/cache_key.py` already gives
# about cache keys: a fixed list of attachment keys can only ever enumerate the
# inputs someone thought of. `ATTACHMENT_KEYS` had six, so `{"dicom_b64": <scan>}`
# with a short query classified LOW — skipping verification, the council, the
# judge and the disclaimer — and the upload was silently discarded with nothing
# telling the clinician. That is the Wave 5 audio defect, one key away.
#
# So an unrecognised key now implies an attachment. Being wrong in this
# direction costs a needless escalation to HIGH; being wrong in the other
# direction is what the two paragraphs above describe.
NON_ATTACHMENT_KEYS: frozenset[str] = frozenset(
    {
        "domain",          # retrieval domain hint
        "workflow",        # which workflow to route to
        "index_version",   # pinning a retrieval index
        "reranker_id",     # pinning a reranker
        "locale",
        "client",
        "client_version",
    }
)


def has_attachment(metadata: object) -> bool:
    """Whether request metadata may carry patient data.

    Three cases, and the middle one used to be collapsed into the first:

    - *absent* (None, or no metadata at all) — nothing was attached. False.
    - *unreadable* (present, but not a dict) — we cannot tell what was
      attached. True, because "we cannot tell" and "nothing was attached" are
      not the same answer on the path to a clinical response, and every caller
      here escalates on True. This previously returned False, so a malformed
      payload resolved exactly like an empty one (adversarial L-2).
    - *readable* — any key that is not known to be harmless counts. The docstring
      of `risk_for` states the rule without qualification — "an attachment ALWAYS
      implies patient data and therefore HIGH risk" — but this answered from a
      fixed six-key list, so `{"dicom_b64": <scan>}` classified LOW and the
      claimed invariant was simply false. See `NON_ATTACHMENT_KEYS`.

    The answer comes from the KEY alone, never from the value's truthiness.
    `{"dicom_b64": ""}` and `{"scan_b64": 0}` used to classify LOW, which decides
    a safety question from a serialisation accident: the key is the caller's
    declaration that a scan is attached, and an empty value is evidence about the
    payload, not about intent. This is the same falsy-elision defect as N2, fixed
    in `cache_key.py` in Wave 9 and left standing here until Wave 11.
    """
    if metadata is None:
        return False
    if not isinstance(metadata, dict):
        logger.warning(
            "Unreadable request metadata of type %s — treating as an attachment",
            type(metadata).__name__,
        )
        return True
    return any(key not in NON_ATTACHMENT_KEYS for key in metadata)


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


class PolicyRegistry:
    """Version -> SafetyPolicy, plus which version is active.

    `01_ARCHITECTURE.md` lists a PolicyRegistry among the required registries.
    The version metadata already existed and was threaded into the cache key and
    into every trace, but nothing could resolve a version *back* to the policy it
    named — so a stored trace saying `policy_version: 2026.08-1` recorded a
    string, not a decision anyone could later reconstruct.

    That matters for this phase specifically. Phase 1 is "Architecture
    Stabilization and Learning Data Plane", and a trace store whose policy
    references cannot be dereferenced is not a learning substrate: you cannot ask
    "did the threshold change explain the change in block rate" if the thresholds
    behind each version are unrecoverable.

    Deliberately small. It resolves and it records; it does not hot-swap policy
    at runtime, because a safety policy that can change mid-flight is a safety
    policy no trace can be trusted to describe.
    """

    def __init__(self, policies: dict[str, SafetyPolicy], active: str) -> None:
        if active not in policies:
            raise ValueError(f"active policy {active!r} is not registered")
        self._policies = dict(policies)
        self._active = active

    @classmethod
    def default(cls) -> PolicyRegistry:
        policy = SafetyPolicy()
        return cls({policy.policy_version: policy}, active=policy.policy_version)

    def active(self) -> SafetyPolicy:
        return self._policies[self._active]

    def get(self, version: str) -> SafetyPolicy:
        """The policy a recorded `policy_version` refers to."""
        try:
            return self._policies[version]
        except KeyError:
            raise KeyError(
                f"unknown policy version {version!r}; known: {sorted(self._policies)}"
            ) from None

    def versions(self) -> list[str]:
        return sorted(self._policies)

    def register(self, policy: SafetyPolicy) -> None:
        """Record a policy version so historical traces stay dereferenceable.

        A version is immutable once registered: re-registering a *different*
        policy under a version already in the store would silently rewrite what
        every trace citing it means.
        """
        existing = self._policies.get(policy.policy_version)
        if existing is not None and existing != policy:
            raise ValueError(
                f"policy version {policy.policy_version!r} is already registered "
                "with different values; bump POLICY_VERSION instead"
            )
        self._policies[policy.policy_version] = policy


_registry = PolicyRegistry.default()


def registry() -> PolicyRegistry:
    """The process-wide policy registry."""
    return _registry


def get_policy() -> SafetyPolicy:
    """The process-wide active safety policy."""
    return _registry.active()


def resolve_risk(state: object) -> RiskLevel:
    """Read a request's risk level out of graph state.

    The single resolver. Everything that needs to know a request's risk — the
    graph, the verification node, the output guardrails, the API, the streaming
    gate — calls this, so there is exactly one interpretation of a malformed or
    missing value. A second implementation used to live in
    `mao/safety/verification.py`, and the two disagreed about what LOW means.

    Three rules, in this order:

    0. An attachment always implies HIGH, and it is checked FIRST — before any
       explicit level is consulted. `risk_for` states this rule without
       qualification ("regardless of the routed intent"), but the resolver used
       to honour an explicit level before looking at attachments, so
       `{"risk_level": "low", "metadata": {"image_b64": ...}}` resolved to LOW.
       LOW is the only level allowed to skip output verification and the
       clinical disclaimer, so that single state defeated both streaming gates
       at once. It was unreachable only because `risk_gate_node` happens to be
       the sole writer of `risk_level` — a property of today's call graph, not
       of this function, which every safety control consults.

    1. Otherwise an explicit `risk_level` is honoured only on an *exact* match.
       LOW must never be reachable by a typo, a stray space, a None, or a wrong
       type. `"Low"` and `" LOW "` are not LOW.

    2. With no usable `risk_level`, classify from intent and attachments rather
       than assuming a floor. Defaulting a clinical request to STANDARD silently
       drops its mandatory disclaimer; `risk_for` escalates it to HIGH instead.
    """
    if not isinstance(state, dict):
        return RiskLevel.STANDARD

    # Rule 0. Nothing a caller or an upstream node can write may declassify a
    # request that carries patient data.
    if has_attachment(state.get("metadata")):
        return RiskLevel.HIGH

    raw = state.get("risk_level")
    if isinstance(raw, RiskLevel):
        return raw
    if isinstance(raw, str):
        try:
            return RiskLevel(raw)
        except ValueError:
            pass

    return get_policy().risk_for(str(state.get("intent") or ""), has_attachment=False)
