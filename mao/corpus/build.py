"""Assemble Corpus V1 from verified provenance plus document text.

Corpus V1 is **document-level**. The legacy 154,250-chunk layout is a baseline
to benchmark against in Phase 2, not a contract to inherit, so no chunking
happens here.

Admission is fail-closed and fully accounted: every input document is either
admitted or recorded with an explicit exclusion reason. Nothing vanishes.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from mao.corpus.licensing import decide
from mao.corpus.provenance import DocumentProvenance

TextLookup = Callable[[str], str | None]

REASON_LICENSE = "license"
REASON_RETRACTED = "retracted"
REASON_TEXT_MISSING = "text_missing"


@dataclass(frozen=True)
class CorpusDocument:
    """One admitted document, with provenance and a content hash."""

    pmcid: str
    title: str
    journal: str
    year: str
    doi: str
    pmid: str
    authors: tuple[str, ...]
    article_type: str
    license_id: str
    license_url: str
    copyright_statement: str
    pmc_url: str
    text: str
    sha256: str
    word_count: int
    char_count: int


@dataclass(frozen=True)
class Exclusion:
    """A document that did not enter the corpus, and why."""

    pmcid: str
    reason: str
    detail: str


@dataclass(frozen=True)
class BuildResult:
    documents: tuple[CorpusDocument, ...] = ()
    excluded: tuple[Exclusion, ...] = ()

    @property
    def admitted(self) -> int:
        return len(self.documents)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _admit(provenance: DocumentProvenance, text: str) -> CorpusDocument:
    return CorpusDocument(
        pmcid=provenance.pmcid,
        title=provenance.title,
        journal=provenance.journal,
        year=provenance.year,
        doi=provenance.doi,
        pmid=provenance.pmid,
        authors=provenance.authors,
        article_type=provenance.article_type,
        license_id=provenance.license_id,
        license_url=provenance.license_url,
        copyright_statement=provenance.copyright_statement,
        pmc_url=provenance.pmc_url,
        text=text,
        sha256=content_hash(text),
        word_count=len(text.split()),
        char_count=len(text),
    )


def build_corpus(
    provenance: Mapping[str, DocumentProvenance],
    *,
    text_for: TextLookup,
) -> BuildResult:
    """Admit every document whose license permits redistribution and whose text exists.

    Order of checks is deliberate: retraction is evaluated before licensing so a
    retracted paper is reported as retracted rather than as a license problem.
    """
    documents: list[CorpusDocument] = []
    excluded: list[Exclusion] = []

    for pmcid in sorted(provenance):
        record = provenance[pmcid]

        if record.is_retracted:
            excluded.append(
                Exclusion(pmcid, REASON_RETRACTED, "article is retracted; not admissible as evidence")
            )
            continue

        decision = decide(record.license_id)
        if not decision.redistributable:
            excluded.append(Exclusion(pmcid, REASON_LICENSE, decision.reason))
            continue

        text = text_for(pmcid)
        if not text or not text.strip():
            excluded.append(
                Exclusion(pmcid, REASON_TEXT_MISSING, "no document text available on disk")
            )
            continue

        documents.append(_admit(record, text))

    return BuildResult(documents=tuple(documents), excluded=tuple(excluded))
