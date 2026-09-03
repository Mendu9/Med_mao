"""Wave 11 — the two de-identification invariants, on generated layouts.

The layouts are not chosen by the implementer. `pdf_layout_generator` renders
the cartesian product of separator x label/value order x column arrangement x
orphan labels x clinical placement x field set with reportlab and reads it back
with pypdf, which is the pair `clinical_agent._extract_pdf_text` uses. Every
layout carries ground truth measured against the extracted text.

Two invariants, asserted on every generated document:

  (a) NO RAW IDENTIFIER SURVIVES to any external or persistent sink. Asserted
      here at the scrubber, and end to end at the four sinks in
      tests/api/test_phi_never_reaches_the_provider.py.

  (b) NO CLINICAL CONTENT IS DELETED OR ALTERED, and NO PLACEHOLDER IS EMITTED
      unless a correctly typed identifier was actually removed there.

(b) is four separate properties, and Wave 9 held only the first of them:

  b1  a clinical line survives verbatim;
  b2  the line structure survives — nothing is joined away, so nothing can be
      deleted by a mis-join (the Wave 10 CRITICAL: every orphan label deleted
      the clinical line after it, 90/90);
  b3  a line that carried no identifier comes back byte-identical, so no
      placeholder is invented on it;
  b4  where an identifier WAS removed, the placeholder that replaced it names
      that identifier's own type — a name may not be reported as `[MRN]`.
"""
from __future__ import annotations

import re

import pytest

from mao.core.pii_scrubber import scrub_pii

from .pdf_layout_generator import (
    ALLOWED_PLACEHOLDERS,
    CLINICAL_CORPUS,
    VOCABULARY,
    Document,
    generate,
    split_lines,
)

#: Rendered once for the whole module: nothing about the documents varies per
#: test, and generating them dominates the cost.
DOCUMENTS: list[Document] = generate()

_PLACEHOLDER = re.compile(r"\[([A-Z_]+)\]")


def _ids(documents: list[Document]) -> list[str]:
    return [document.id for document in documents]


def _lines(text: str) -> list[str]:
    """Split on ANY line terminator.

    `text.split("\\n")` is what the first round used, and CRLF turned out to
    disable the whole labelled path — a test that cannot see `\\r` cannot see
    that defect either.
    """
    return split_lines(text)


class TestTheGeneratorItself:
    """A fuzzer that generates nothing proves nothing."""

    def test_it_produces_a_broad_population(self) -> None:
        assert len(DOCUMENTS) > 2000, f"only {len(DOCUMENTS)} layouts generated"

    def test_every_axis_is_actually_exercised(self) -> None:
        for axis in (
            "separator",
            "order",
            "columns",
            "orphans",
            "clinical",
            "field_set",
            "transport",
        ):
            values = {getattr(document.layout, axis) for document in DOCUMENTS}
            assert len(values) > 1, f"axis {axis} collapsed to {values}"

    def test_orphan_labels_and_clinical_lines_both_occur(self) -> None:
        assert any(document.orphan_at for document in DOCUMENTS)
        assert any(document.clinical_at for document in DOCUMENTS)

    def test_every_document_carries_a_live_identifier(self) -> None:
        for document in DOCUMENTS:
            assert document.identifiers, document.id

    def test_every_name_in_the_corpus_actually_renders(self) -> None:
        """Round 2. Non-ASCII names were added because the matcher was
        ASCII-only — but Helvetica cannot draw them, so an unrenderable name
        would silently drop out of the ground truth and the coverage it was
        added for would be fake instead of failing."""
        placed = {field.value for document in DOCUMENTS for field, _ in document.placed}
        missing = [value for _, value in VOCABULARY["NAME"] if value not in placed]
        assert not missing, f"these names never survived extraction: {missing}"

    def test_the_clinical_corpus_includes_titlecase_noun_phrases(self) -> None:
        """A Titlecase clinical noun phrase is the same SHAPE as a person's
        name, and is the case a stop-list of clinical words loses against."""
        titlecase = [
            line
            for line in CLINICAL_CORPUS
            if len(line.split()) >= 2
            and all(word[:1].isupper() for word in line.split()[:2])
        ]
        assert len(titlecase) >= 10, f"only {len(titlecase)} such phrases"


class TestNoRawIdentifierSurvives:
    """Invariant (a)."""

    @pytest.mark.parametrize("document", DOCUMENTS, ids=_ids(DOCUMENTS))
    def test_every_identifier_is_removed(self, document: Document) -> None:
        scrubbed = scrub_pii(document.extracted)
        survivors = [
            identifier
            for identifier in document.identifiers
            if identifier in scrubbed
        ]
        assert not survivors, (
            f"{document.id}: {survivors} reached the scrubbed text\n"
            f"--- extracted ---\n{document.extracted}\n"
            f"--- scrubbed ---\n{scrubbed}"
        )


class TestNoClinicalContentIsDestroyed:
    """Invariant (b1) and (b2)."""

    @pytest.mark.parametrize("document", DOCUMENTS, ids=_ids(DOCUMENTS))
    def test_every_clinical_line_survives_verbatim(self, document: Document) -> None:
        scrubbed = scrub_pii(document.extracted)
        lost = [line for line in document.clinical_lines if line not in scrubbed]
        assert not lost, (
            f"{document.id}: clinical content DELETED: {lost}\n"
            f"--- extracted ---\n{document.extracted}\n"
            f"--- scrubbed ---\n{scrubbed}"
        )

    @pytest.mark.parametrize("document", DOCUMENTS, ids=_ids(DOCUMENTS))
    def test_the_line_structure_is_preserved(self, document: Document) -> None:
        """Nothing may be joined away.

        A scrubber that rewrites `label \\n value` into `label: value` removes a
        line, and when it pairs the wrong two lines that removal is a silent
        deletion of clinical content. Holding the line count makes the whole
        class unreachable rather than merely untested.
        """
        scrubbed = scrub_pii(document.extracted)
        assert len(_lines(scrubbed)) == len(_lines(document.extracted)), (
            f"{document.id}: line count changed "
            f"{len(_lines(document.extracted))} -> {len(_lines(scrubbed))}\n"
            f"--- extracted ---\n{document.extracted}\n"
            f"--- scrubbed ---\n{scrubbed}"
        )


class TestNoPlaceholderWithoutARemoval:
    """Invariant (b3) — a line holding no identifier comes back unchanged."""

    @pytest.mark.parametrize("document", DOCUMENTS, ids=_ids(DOCUMENTS))
    def test_a_clinical_line_is_returned_byte_identical(
        self, document: Document
    ) -> None:
        scrubbed = _lines(scrub_pii(document.extracted))
        extracted = _lines(document.extracted)
        for line, index in document.clinical_at:
            if index >= len(scrubbed):
                pytest.fail(f"{document.id}: line {index} ({line!r}) no longer exists")
            assert scrubbed[index] == extracted[index], (
                f"{document.id}: clinical line {index} was altered\n"
                f"  was: {extracted[index]!r}\n"
                f"  now: {scrubbed[index]!r}"
            )

    @pytest.mark.parametrize("document", DOCUMENTS, ids=_ids(DOCUMENTS))
    def test_an_unpaired_label_gets_no_placeholder(self, document: Document) -> None:
        """`MRN:` with no record number anywhere identifies nobody.

        Emitting `MRN: [MRN]` there asserts a removal that never happened — and
        the registered `clinical.extraction` prompt then tells the model the
        report 'has already been de-identified'.
        """
        scrubbed = _lines(scrub_pii(document.extracted))
        extracted = _lines(document.extracted)
        for label, index in document.orphan_at:
            if index >= len(scrubbed):
                pytest.fail(f"{document.id}: orphan label line {index} no longer exists")
            assert scrubbed[index] == extracted[index], (
                f"{document.id}: a placeholder was invented for unpaired {label!r}\n"
                f"  was: {extracted[index]!r}\n"
                f"  now: {scrubbed[index]!r}"
            )


class TestThePlaceholderNamesTheRightType:
    """Invariant (b4) — the type of the placeholder matches what was removed."""

    @pytest.mark.parametrize("document", DOCUMENTS, ids=_ids(DOCUMENTS))
    def test_each_removed_value_is_replaced_by_its_own_type(
        self, document: Document
    ) -> None:
        scrubbed = _lines(scrub_pii(document.extracted))
        for field, index in document.placed:
            assert index < len(scrubbed), f"{document.id}: line {index} is gone"
            found = set(_PLACEHOLDER.findall(scrubbed[index]))
            allowed = ALLOWED_PLACEHOLDERS[field.kind]
            assert found & allowed, (
                f"{document.id}: {field.kind} value {field.value!r} on line "
                f"{index} was replaced by {found or 'nothing'}, not by one of "
                f"{sorted(allowed)}\n"
                f"--- scrubbed ---\n" + "\n".join(scrubbed)
            )


class TestScrubbingIsStable:
    @pytest.mark.parametrize("document", DOCUMENTS, ids=_ids(DOCUMENTS))
    def test_a_second_pass_changes_nothing(self, document: Document) -> None:
        once = scrub_pii(document.extracted)
        assert scrub_pii(once) == once, document.id
