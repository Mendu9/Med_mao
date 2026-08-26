"""ModelRegistry — provider-neutral model records addressed by role.

Business logic asks for a *role* (CLINICAL_SYNTHESIS, SAFETY_JUDGE, ...), never
for a literal model id. Two audited defects are structurally prevented here:

P1-17  Each role reads exactly one env var (`MAO_MODEL_<ROLE>`) with a single
       literal default. There is no nested `getenv(getenv(...))` chain, so a
       generic `GROQ_MODEL` override can no longer silently downgrade the
       safety council to an 8B model.

P1-18  Retired model ids are recorded with `ModelStatus.RETIRED` and can never
       be resolved; `resolve()` walks the fallback chain and raises if no
       active model exists. A retired id cannot reach a provider.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class ModelStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class Modality(str, Enum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"


class ModelRole(str, Enum):
    """Capability roles business logic may request."""

    ROUTER_FAST = "router_fast"
    EXTRACTION_FAST = "extraction_fast"
    GENERAL_SYNTHESIS = "general_synthesis"
    CLINICAL_SYNTHESIS = "clinical_synthesis"
    RESEARCH_SYNTHESIS = "research_synthesis"
    VISION = "vision"
    SAFETY_JUDGE = "safety_judge"

    @property
    def env_var(self) -> str:
        """Per-role override, e.g. ROUTER_FAST -> MAO_MODEL_ROUTER_FAST."""
        return f"MAO_MODEL_{self.name}"


@dataclass(frozen=True)
class ModelRecord:
    """One concrete model, with the metadata the architecture mandates."""

    provider: str
    model_id: str
    modality: Modality
    context_limit: int
    status: ModelStatus = ModelStatus.ACTIVE
    revision: str = ""
    supports_structured_output: bool = False
    supports_tools: bool = False
    fallbacks: tuple[str, ...] = ()
    cost_per_1m_input_usd: float = 0.0
    cost_per_1m_output_usd: float = 0.0
    typical_latency_ms: float = 0.0
    last_verified_at: str = ""


# ---------------------------------------------------------------------------
# Catalogue
#
# `last_verified_at` records when a human last checked the id against the
# provider's live model list. Anything older than the provider's deprecation
# cadence should be re-verified before being trusted.
# ---------------------------------------------------------------------------

_GROQ_TEXT_8B = ModelRecord(
    provider="groq",
    model_id="llama-3.1-8b-instant",
    modality=Modality.TEXT,
    context_limit=131072,
    supports_structured_output=True,
    supports_tools=True,
    fallbacks=("llama-3.3-70b-versatile",),
    cost_per_1m_input_usd=0.05,
    cost_per_1m_output_usd=0.08,
    last_verified_at="2026-08-26",
)

_GROQ_TEXT_70B = ModelRecord(
    provider="groq",
    model_id="llama-3.3-70b-versatile",
    modality=Modality.TEXT,
    context_limit=131072,
    supports_structured_output=True,
    supports_tools=True,
    fallbacks=("llama-3.1-8b-instant",),
    cost_per_1m_input_usd=0.59,
    cost_per_1m_output_usd=0.79,
    last_verified_at="2026-08-26",
)

_GROQ_VISION = ModelRecord(
    provider="groq",
    model_id="meta-llama/llama-4-scout-17b-16e-instruct",
    modality=Modality.VISION,
    context_limit=131072,
    supports_structured_output=True,
    supports_tools=True,
    cost_per_1m_input_usd=0.11,
    cost_per_1m_output_usd=0.34,
    last_verified_at="2026-08-26",
)

# --- Retired: recorded so they can be *refused*, never resolved (P1-18) ---

_RETIRED = (
    ModelRecord(
        provider="groq",
        model_id="llama-3.2-11b-vision-preview",
        modality=Modality.VISION,
        context_limit=131072,
        status=ModelStatus.RETIRED,
        fallbacks=("meta-llama/llama-4-scout-17b-16e-instruct",),
        last_verified_at="2026-08-26",
    ),
    ModelRecord(
        provider="groq",
        model_id="llama-3.1-70b-versatile",
        modality=Modality.TEXT,
        context_limit=131072,
        status=ModelStatus.RETIRED,
        fallbacks=("llama-3.3-70b-versatile",),
        last_verified_at="2026-08-26",
    ),
    ModelRecord(
        provider="groq",
        model_id="mixtral-8x7b-32768",
        modality=Modality.TEXT,
        context_limit=32768,
        status=ModelStatus.RETIRED,
        fallbacks=("llama-3.3-70b-versatile",),
        last_verified_at="2026-08-26",
    ),
)

# Role -> default model id. Safety-critical roles default to the 70B model and
# are never derived from a generic env var (P1-17).
_DEFAULT_BINDINGS: dict[ModelRole, str] = {
    ModelRole.ROUTER_FAST: _GROQ_TEXT_8B.model_id,
    ModelRole.EXTRACTION_FAST: _GROQ_TEXT_8B.model_id,
    ModelRole.GENERAL_SYNTHESIS: _GROQ_TEXT_8B.model_id,
    ModelRole.CLINICAL_SYNTHESIS: _GROQ_TEXT_70B.model_id,
    ModelRole.RESEARCH_SYNTHESIS: _GROQ_TEXT_70B.model_id,
    ModelRole.VISION: _GROQ_VISION.model_id,
    ModelRole.SAFETY_JUDGE: _GROQ_TEXT_70B.model_id,
}


@dataclass
class ModelRegistry:
    """Resolves roles to concrete, active model records."""

    records: dict[str, ModelRecord] = field(default_factory=dict)
    role_bindings: dict[ModelRole, str] = field(default_factory=dict)

    # -- construction -------------------------------------------------------

    @classmethod
    def default(cls) -> ModelRegistry:
        """Build the catalogue, applying per-role env overrides."""
        records: dict[str, ModelRecord] = {
            r.model_id: r for r in (_GROQ_TEXT_8B, _GROQ_TEXT_70B, _GROQ_VISION, *_RETIRED)
        }
        bindings: dict[ModelRole, str] = {}
        for role, default_id in _DEFAULT_BINDINGS.items():
            chosen = os.getenv(role.env_var, "").strip() or default_id
            if chosen not in records:
                # An operator-supplied id we have no metadata for. Trust it, but
                # record it so traces and cost accounting still resolve.
                records[chosen] = ModelRecord(
                    provider=os.getenv("MAO_MODEL_PROVIDER", "groq"),
                    model_id=chosen,
                    modality=_GROQ_VISION.modality if role is ModelRole.VISION else Modality.TEXT,
                    context_limit=0,
                    fallbacks=(default_id,),
                )
            bindings[role] = chosen
        return cls(records=records, role_bindings=bindings)

    # -- lookup -------------------------------------------------------------

    def get(self, model_id: str) -> ModelRecord | None:
        return self.records.get(model_id)

    def binding_for(self, role: ModelRole) -> str:
        """The id bound to a role *after* retired-model substitution."""
        return self._first_active(role).model_id

    def resolve(self, role: ModelRole) -> ModelRecord:
        """Return the active record for a role, following fallbacks if needed."""
        return self._first_active(role)

    def model_id_for(self, role: ModelRole) -> str:
        return self.resolve(role).model_id

    # -- internals ----------------------------------------------------------

    def _first_active(self, role: ModelRole) -> ModelRecord:
        try:
            start = self.role_bindings[role]
        except KeyError:
            raise LookupError(f"no model bound to role {role.value}") from None

        seen: set[str] = set()
        queue = [start]
        while queue:
            model_id = queue.pop(0)
            if model_id in seen:
                continue
            seen.add(model_id)
            record = self.records.get(model_id)
            if record is None:
                continue
            if record.status is ModelStatus.RETIRED:
                queue.extend(record.fallbacks)
                continue
            return record

        raise LookupError(
            f"no active model for role {role.value}: "
            f"'{start}' is retired and no fallback is active"
        )
