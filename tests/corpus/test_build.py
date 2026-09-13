"""Corpus V1 assembly: which documents enter the canonical corpus, and why.

Corpus V1 is document-level by design. Chunking is a downstream, benchmarked
decision in Phase 2 and is deliberately absent here.
"""
from __future__ import annotations


from mao.corpus.build import build_corpus
from mao.corpus.provenance import DocumentProvenance


def _doc(pmcid: str, license_id: str = "CC-BY-4.0", retracted: bool = False) -> DocumentProvenance:
    return DocumentProvenance(
        pmcid=pmcid,
        title=f"Title {pmcid}",
        journal="Test Journal",
        year="2024",
        doi=f"10.0000/{pmcid}",
        license_id=license_id,
        is_retracted=retracted,
    )


def _texts(mapping: dict[str, str]):
    return lambda pmcid: mapping.get(pmcid)


class TestLicenceGating:
    def test_includes_redistributable_documents(self) -> None:
        result = build_corpus(
            {"PMC1": _doc("PMC1")}, text_for=_texts({"PMC1": "body text here"})
        )
        assert [d.pmcid for d in result.documents] == ["PMC1"]

    def test_excludes_unknown_licence(self) -> None:
        result = build_corpus(
            {"PMC1": _doc("PMC1", license_id="UNKNOWN")},
            text_for=_texts({"PMC1": "body"}),
        )
        assert result.documents == ()
        assert result.excluded[0].reason == "license"

    def test_excludes_no_derivatives_licence(self) -> None:
        result = build_corpus(
            {"PMC1": _doc("PMC1", license_id="CC-BY-NC-ND-4.0")},
            text_for=_texts({"PMC1": "body"}),
        )
        assert result.documents == ()
        assert result.excluded[0].reason == "license"


class TestRetraction:
    def test_excludes_retracted_documents(self) -> None:
        """A retracted paper must never be served as clinical evidence."""
        result = build_corpus(
            {"PMC1": _doc("PMC1", retracted=True)}, text_for=_texts({"PMC1": "body"})
        )
        assert result.documents == ()
        assert result.excluded[0].reason == "retracted"

    def test_retraction_is_checked_even_when_licence_permits(self) -> None:
        result = build_corpus(
            {"PMC1": _doc("PMC1", license_id="CC-BY-4.0", retracted=True)},
            text_for=_texts({"PMC1": "body"}),
        )
        assert [e.reason for e in result.excluded] == ["retracted"]


class TestMissingText:
    def test_excludes_document_with_no_text_on_disk(self) -> None:
        result = build_corpus({"PMC1": _doc("PMC1")}, text_for=_texts({}))
        assert result.documents == ()
        assert result.excluded[0].reason == "text_missing"

    def test_excludes_document_with_blank_text(self) -> None:
        result = build_corpus({"PMC1": _doc("PMC1")}, text_for=_texts({"PMC1": "   "}))
        assert result.excluded[0].reason == "text_missing"


class TestDocumentRecord:
    def test_records_content_hash_and_counts(self) -> None:
        result = build_corpus(
            {"PMC1": _doc("PMC1")}, text_for=_texts({"PMC1": "alpha beta gamma"})
        )
        doc = result.documents[0]
        assert len(doc.sha256) == 64
        assert doc.word_count == 3
        assert doc.char_count == 16

    def test_content_hash_is_deterministic(self) -> None:
        first = build_corpus({"PMC1": _doc("PMC1")}, text_for=_texts({"PMC1": "same"}))
        second = build_corpus({"PMC1": _doc("PMC1")}, text_for=_texts({"PMC1": "same"}))
        assert first.documents[0].sha256 == second.documents[0].sha256

    def test_carries_provenance_onto_the_document(self) -> None:
        result = build_corpus({"PMC1": _doc("PMC1")}, text_for=_texts({"PMC1": "t"}))
        doc = result.documents[0]
        assert doc.doi == "10.0000/PMC1"
        assert doc.journal == "Test Journal"
        assert doc.year == "2024"
        assert doc.license_id == "CC-BY-4.0"

    def test_documents_are_sorted_by_pmcid(self) -> None:
        provenance = {p: _doc(p) for p in ("PMC3", "PMC1", "PMC2")}
        result = build_corpus(
            provenance, text_for=_texts({p: "body" for p in provenance})
        )
        assert [d.pmcid for d in result.documents] == ["PMC1", "PMC2", "PMC3"]


class TestAccounting:
    def test_every_input_is_either_included_or_excluded(self) -> None:
        """No document may vanish silently between input and output."""
        provenance = {
            "PMC1": _doc("PMC1"),
            "PMC2": _doc("PMC2", license_id="UNKNOWN"),
            "PMC3": _doc("PMC3", retracted=True),
            "PMC4": _doc("PMC4"),
        }
        result = build_corpus(
            provenance, text_for=_texts({"PMC1": "a", "PMC2": "b", "PMC3": "c"})
        )
        accounted = {d.pmcid for d in result.documents} | {e.pmcid for e in result.excluded}
        assert accounted == set(provenance)

    def test_exclusions_carry_a_human_readable_detail(self) -> None:
        result = build_corpus(
            {"PMC1": _doc("PMC1", license_id="UNKNOWN")}, text_for=_texts({"PMC1": "b"})
        )
        assert result.excluded[0].detail.strip()
