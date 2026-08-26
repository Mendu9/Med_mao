"""Canonical Evidence and Claim objects.

Phase 1 declares the types so that safety, verification, and trace code can be
written against one shape. Migrating the retrieval pipeline to emit `Evidence`
end-to-end is Phase 2 scope ("canonical Evidence object") and is deliberately
not done here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class Evidence:
    """One retrieved unit of evidence.

    Field set is fixed by `01_ARCHITECTURE.md` § "Canonical Evidence Object".
    """

    evidence_id: str
    text: str
    source_id: str
    source_type: str = ""
    title: str = ""
    url: str = ""
    doi: str = ""
    pmid: str = ""
    section: str = ""
    page: int | None = None
    published_at: str = ""
    retrieved_at: str = ""
    score: float = 0.0
    evidence_level: str = ""
    trust_tier: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Claim:
    """One assertion extracted from a generated answer.

    Field set is fixed by `01_ARCHITECTURE.md` § "Canonical Claim Object".
    """

    claim_id: str
    text: str
    evidence_ids: tuple[str, ...] = ()
    support_score: float = 0.0
    contradiction_score: float = 0.0
    certainty: float = 0.0
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
