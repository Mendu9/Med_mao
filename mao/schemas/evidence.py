"""Canonical Evidence and Claim objects.

These types were declared in Phase 1 and consumed by nothing, with migration
deferred to Phase 2. Wave 7 found what that deferral actually cost, by tracing
what the safety chain is handed at a live `graph.invoke`:

  - `graphrag_node` published `retrieved_docs` as `list[dict]`, and all three
    consumers — the council, the judge's premise, the domain supervisor —
    stringified them with `str(doc)`. The judge's premise was four truncated
    Python dict reprs concatenated, each starting `{'text': '` and cut off
    before its `source` key. It scored groundedness against that.
  - `clinical_node` published nothing at all, keeping its chunks under a
    `_ranked_chunks` key that the response filter strips. On the highest-risk
    route the NLI gate had no premise, and an empty context narrows the council
    to its safety member, so the accuracy and hallucination vetoes never ran.

Both are the same absence: no agreed shape between what retrieval produces and
what verification consumes, so each side improvised. `as_text` and `from_chunk`
are that shape, and they are what makes this module a contract rather than a
declaration.
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


def as_text(doc: object) -> str:
    """The prose of one retrieved unit, whatever shape it arrived in.

    Every consumer of `retrieved_docs` needs the document's *text* and nothing
    else. Each used to reach for it differently — `getattr(doc, "text", None)`
    with a `str(doc)` fallback, or `str(doc)` outright — so a producer that
    emitted dicts silently fed Python repr syntax into an LLM premise.

    Order matters: dict-shaped documents are checked before the generic
    fallback, because that is the shape that used to fail, and the fallback is
    the thing that made the failure invisible.
    """
    text = getattr(doc, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(doc, dict):
        # The three text-bearing keys this codebase produces: retrieval chunks
        # use `text`, web results use `body`, and cached source records use
        # `snippet`. A web result reaching the domain supervisor as `str(dict)`
        # is the same defect as a RAG chunk doing so.
        for key in ("text", "body", "snippet"):
            value = doc.get(key)
            if value is not None:
                return str(value)
        return ""
    if isinstance(doc, str):
        return doc
    return str(doc)


def from_chunk(chunk: object) -> Evidence:
    """Build an `Evidence` from a retriever chunk.

    Retriever chunks carry their provenance in a loose `metadata` dict. Pulling
    it into the declared field set here means the safety chain, the trace and
    the report card all read the same names.
    """
    metadata = getattr(chunk, "metadata", None) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    source_id = str(metadata.get("source", "") or "")
    chunk_id = str(metadata.get("chunk_id", "") or "")
    return Evidence(
        evidence_id=f"{source_id}#{chunk_id}" if chunk_id else source_id,
        text=as_text(chunk),
        source_id=source_id,
        title=str(metadata.get("title", source_id) or ""),
        score=float(getattr(chunk, "score", 0.0) or 0.0),
        metadata=metadata,
    )


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
