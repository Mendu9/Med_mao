"""Acquisition policy: how Corpus V1 was built and how it scales.

The P2-0 audit found the legacy corpus recorded no acquisition attribution —
``pmc_manifest.json`` carried an empty ``query`` field for all 900 documents, so
the selection could not be reproduced or extended without manual curation.

This module makes the policy an explicit, versioned, fingerprinted artifact.
Expanding the corpus is then a change of *target counts* against a fixed query
set, not a paper-picking exercise.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

POLICY_PATH = Path(__file__).parent / "acquisition_policy.json"


@dataclass(frozen=True)
class ScalingTier:
    tier: str
    target_documents: int
    per_category: int
    status: str = ""


@dataclass(frozen=True)
class AcquisitionPolicy:
    policy_version: str
    categories: dict[str, tuple[str, ...]]
    license_allow_list: tuple[str, ...]
    exclude_retracted: bool
    scaling_tiers: tuple[ScalingTier, ...]
    dedupe_order: tuple[str, ...]

    @property
    def query_fingerprint(self) -> str:
        """Stable hash of the full query set — changes iff the queries change."""
        canonical = json.dumps(
            {name: list(queries) for name, queries in sorted(self.categories.items())},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def query_count(self) -> int:
        return sum(len(q) for q in self.categories.values())

    def with_categories(self, categories: dict[str, tuple[str, ...]]) -> AcquisitionPolicy:
        return replace(self, categories=categories)

    def tier(self, name: str) -> ScalingTier:
        for entry in self.scaling_tiers:
            if entry.tier == name:
                return entry
        raise ValueError(f"unknown tier: {name!r}")


@dataclass(frozen=True)
class ExpansionPlan:
    """A deterministic instruction for growing the corpus to a target tier."""

    from_documents: int
    target_documents: int
    additional_documents: int
    per_category: dict[str, int]
    query_fingerprint: str
    policy_version: str


def load_policy(path: Path | None = None) -> AcquisitionPolicy:
    raw = json.loads((path or POLICY_PATH).read_text(encoding="utf-8"))
    filters = raw.get("admission_filters", {})
    return AcquisitionPolicy(
        policy_version=raw["policy_version"],
        categories={
            name: tuple(spec["queries"]) for name, spec in raw["categories"].items()
        },
        license_allow_list=tuple(filters.get("license_allow_list", ())),
        exclude_retracted=bool(filters.get("exclude_retracted", True)),
        scaling_tiers=tuple(
            ScalingTier(
                tier=t["tier"],
                target_documents=int(t["target_documents"]),
                per_category=int(t["per_category"]),
                status=t.get("status", ""),
            )
            for t in raw.get("scaling_tiers", ())
        ),
        dedupe_order=tuple(raw.get("dedupe", {}).get("order", ())),
    )


def plan_expansion(
    policy: AcquisitionPolicy, *, to_tier: str, already_have: int
) -> ExpansionPlan:
    """Compute what must be fetched to reach ``to_tier``.

    Fully determined by the policy and the target tier, so two callers with the
    same inputs always produce the same plan.
    """
    tier = policy.tier(to_tier)
    additional = max(0, tier.target_documents - already_have)
    names = sorted(policy.categories)
    base, remainder = divmod(tier.target_documents, len(names)) if names else (0, 0)
    per_category = {
        name: base + (1 if index < remainder else 0) for index, name in enumerate(names)
    }
    return ExpansionPlan(
        from_documents=already_have,
        target_documents=tier.target_documents,
        additional_documents=additional,
        per_category=per_category,
        query_fingerprint=policy.query_fingerprint,
        policy_version=policy.policy_version,
    )
