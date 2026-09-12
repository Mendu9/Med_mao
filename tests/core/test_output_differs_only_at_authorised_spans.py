r"""O1 — the scrubbed output differs from the source ONLY at recorded spans.

## The property

    scrub_with_report(text).text
        == text with each recorded Removal span replaced by "[<kind>]"

Every other byte — every line terminator, the leading BOM, the soft hyphen that
pypdf emitted in the middle of a record number, the zero-width joiner, the
punctuation around a redacted value — must survive unchanged.

Before `41cfabd` this could not hold even in principle: `normalise()` ran NFD,
deleted marks, folded marks and ran NFKC, and its OUTPUT was the shipped string.
The source is now immutable outside authorised spans and the folding lives in a
match view that never ships, so the property is now a statement about the code
rather than an aspiration.

## How this oracle is independent

`_reconstruct` below is eleven lines of `str` slicing. It calls nothing from
`mao.core.deident`, imports no helper from any other test module, and does not
know how the scrubber decides anything. It takes two things the implementation
hands back — the source string it was given, and the `(start, end, kind)` triple
of each `Removal` — and performs the ONE edit those triples authorise.

So the comparison is between "what the transformation says it did" and "what the
transformation actually produced". Neither side is computed by asking the
scrubber a second question, which is the failure mode this project has recorded
three times: a second detector pass grading the first one and agreeing with it
for the same reason both were wrong.

The placeholder spelling `"[<kind>]"` is derived from `Removal.kind` and from
nothing else. That is deliberate: `Removal` is the record downstream reads, and
if the record is not sufficient to reconstruct the output then the record is
incomplete. See `TestTheRecordIsSufficientToReconstructTheOutput` for the family
where it is not, which is a reported product defect and not a relaxation here.
"""
from __future__ import annotations

import pytest

from mao.core.deident.report import Removal
from mao.core.pii_scrubber import scrub_with_report

# --------------------------------------------------------------------------
# The oracle
# --------------------------------------------------------------------------


def _reconstruct(source: str, removals: tuple[Removal, ...]) -> str:
    """`source` with exactly the recorded spans replaced by their placeholders.

    Independent by construction: pure slicing over `(start, end, kind)`. It has
    no view of any grammar, any layout rule or any normalisation stage.
    """
    ordered = sorted(removals, key=lambda removal: removal.start)
    out: list[str] = []
    cursor = 0
    for removal in ordered:
        assert removal.start >= cursor, (
            f"recorded spans overlap or are unsorted: {removal.start} < {cursor}"
        )
        out.append(source[cursor : removal.start])
        out.append(f"[{removal.kind}]")
        cursor = removal.end
    out.append(source[cursor:])
    return "".join(out)


def _assert_only_authorised_edits(source: str) -> None:
    result = scrub_with_report(source)
    expected = _reconstruct(source, result.removals)
    assert result.text == expected, (
        "output differs from the source somewhere OTHER than a recorded "
        f"removal span.\n  source   : {source!r}\n"
        f"  produced : {result.text!r}\n  authorised: {expected!r}\n"
        f"  removals : {[(r.kind, r.start, r.end) for r in result.removals]}"
    )
    for removal in result.removals:
        assert source[removal.start : removal.end] == removal.value, (
            f"a removal's recorded value is not what stood at its span: "
            f"{removal.value!r} vs {source[removal.start:removal.end]!r}"
        )


# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------

BOM = "﻿"
SHY = "­"          # SOFT HYPHEN — pypdf emits this mid-identifier
ZWSP = "​"         # ZERO WIDTH SPACE
ZWJ = "‍"          # ZERO WIDTH JOINER
WJ = "⁠"           # WORD JOINER
LS = " "           # LINE SEPARATOR
PS = " "           # PARAGRAPH SEPARATOR
NEL = ""          # NEXT LINE

_LETTERHEAD = (
    "Patient Name: Harold Nkemdirim{t}"
    "MRN: RGT/44219/B{t}"
    "NHS Number: 943 476 5919{t}"
    "DOB: 12/03/1948{t}"
    "Postcode: SW1A 1AA{t}"
    "Telephone: 020 7946 0958{t}"
    "Complete heart block with a permanent pacemaker in situ.{t}"
)

ASCII_LETTERHEADS = {
    "plain_lf": _LETTERHEAD.format(t="\n"),
    "no_final_terminator": _LETTERHEAD.format(t="\n").rstrip("\n"),
    "inline_banner": (
        "Patient Name: Harold Nkemdirim   MRN: RGT/44219/B\n"
        "Complete heart block confirmed on telemetry.\n"
    ),
    "pipe_separated": (
        "Patient Name: Harold Nkemdirim | MRN: RGT/44219/B | DOB: 12/03/1948\n"
        "Permanent pacemaker implanted, device dependent.\n"
    ),
    "wide_two_column": (
        "Patient Name:            Harold Nkemdirim\n"
        "MRN:                     RGT/44219/B\n"
        "Complete heart block.\n"
    ),
    "punctuation_hugging_value": (
        "Contact (Telephone: 020 7946 0958), and the MRN: RGT/44219/B.\n"
    ),
}

LINE_TERMINATORS = {
    f"terminator_{name}": _LETTERHEAD.format(t=term)
    for name, term in (
        ("crlf", "\r\n"),
        ("lone_cr", "\r"),
        ("lf", "\n"),
        ("line_separator_u2028", LS),
        ("paragraph_separator_u2029", PS),
        ("next_line_u0085", NEL),
        ("vertical_tab", "\v"),
        ("form_feed", "\f"),
    )
}

MIXED_TERMINATORS = {
    "mixed_crlf_cr_lf": (
        "Patient Name: Harold Nkemdirim\r\n"
        "MRN: RGT/44219/B\r"
        "DOB: 12/03/1948\n"
        f"Postcode: SW1A 1AA{LS}"
        "Complete heart block.\n"
    ),
}

BOM_CASES = {
    "leading_bom": BOM + _LETTERHEAD.format(t="\n"),
    "leading_bom_crlf": BOM + _LETTERHEAD.format(t="\r\n"),
}

#: BOM cases that carry no identifier. Kept apart from `BOM_CASES` so the
#: non-vacuity sweep below, which demands a removal from every identifier
#: document, is not handed a document that correctly produces none.
BOM_WITHOUT_IDENTIFIERS = {
    "bom_only": BOM,
    "bom_then_clinical_prose": BOM + "The patient reports increasing confusion.\n",
}

INVISIBLE_INSIDE_IDENTIFIERS = {
    "soft_hyphen_in_mrn": f"MRN: RGT/44219{SHY}/B\nComplete heart block.\n",
    "soft_hyphen_in_postcode": f"Postcode: SW1A{SHY} 1AA\nPacemaker in situ.\n",
    "soft_hyphen_in_name": f"Patient Name: Harold Nkem{SHY}dirim\nBradycardia noted.\n",
    "zwsp_in_mrn": f"MRN: A123{ZWSP}4567\nComplete heart block.\n",
    "zwj_in_phone": f"Telephone: 020 7946{ZWJ} 0958\nPacemaker in situ.\n",
    "word_joiner_in_nhs": f"NHS Number: 943 476{WJ} 5919\nBradycardia noted.\n",
    "several_invisibles": (
        f"Patient Name: Har{ZWSP}old Nkem{SHY}dirim{WJ}\n"
        f"MRN: RGT{ZWJ}/44219/B\n"
        "Complete heart block with a permanent pacemaker.\n"
    ),
}

NO_IDENTIFIER_AT_ALL = {
    "clinical_prose": (
        "The patient reports increasing confusion over six months.\n"
        "Complete heart block was confirmed on telemetry.\n"
        "A permanent pacemaker is in situ and the patient is device dependent.\n"
    ),
    "empty": "",
    "whitespace_only": "   \n\t\n  \r\n",
    "bare_numbers_column": "138\n102\n2024\n4.8\n21\n",
    "invisibles_but_no_identifier": (
        f"The patient{ZWSP} reports confusion{SHY} and falls.{WJ}\n"
    ),
    "bom_and_terminators_only": BOM + f"\r\n\r{LS}\n",
    **BOM_WITHOUT_IDENTIFIERS,
}

CORPUS: dict[str, str] = {
    **ASCII_LETTERHEADS,
    **LINE_TERMINATORS,
    **MIXED_TERMINATORS,
    **BOM_CASES,
    **INVISIBLE_INSIDE_IDENTIFIERS,
    **NO_IDENTIFIER_AT_ALL,
}

#: See `TestLineTerminatorsAreStructureNotContent` — U+0085 NEL is deleted by the
#: match view, which collapses the document to one line and leaks every labelled
#: identifier after the first. A reported product defect, isolated here so the
#: terminator sweep states it once instead of failing everywhere.
NEL_IS_BROKEN = "terminator_next_line_u0085"


class TestOutputDiffersOnlyAtAuthorisedSpans:
    """The whole corpus, byte for byte, against the reconstruction."""

    @pytest.mark.parametrize("name", sorted(CORPUS))
    def test_every_byte_outside_a_recorded_span_survives(self, name: str) -> None:
        _assert_only_authorised_edits(CORPUS[name])

    @pytest.mark.parametrize("name", sorted(NO_IDENTIFIER_AT_ALL))
    def test_text_with_no_identifier_is_byte_identical(self, name: str) -> None:
        """The strongest form: not merely 'no clinical loss', but NO CHANGE."""
        source = NO_IDENTIFIER_AT_ALL[name]
        result = scrub_with_report(source)
        assert result.removals == (), (
            f"{name}: a removal was recorded in text containing no identifier: "
            f"{[(r.kind, r.value) for r in result.removals]}"
        )
        assert result.text == source, (
            f"{name}: text with no identifier was altered.\n"
            f"  in  : {source!r}\n  out : {result.text!r}"
        )

    def test_a_leading_bom_is_still_leading(self) -> None:
        for name, source in {**BOM_CASES, **BOM_WITHOUT_IDENTIFIERS}.items():
            produced = scrub_with_report(source).text
            assert produced.startswith(BOM), (
                f"{name}: the leading BOM did not survive: {produced[:8]!r}"
            )

    def test_the_scrubber_actually_fired_on_the_identifier_corpus(self) -> None:
        """Non-vacuity for the corpus itself.

        Every assertion above would pass on a scrubber that did nothing at all.
        This is what says the corpus is exercising redaction and not just
        exercising the identity function.
        """
        redacting = {
            name: scrub_with_report(text).removals
            for name, text in CORPUS.items()
            if name not in NO_IDENTIFIER_AT_ALL
        }
        silent = [name for name, removals in redacting.items() if not removals]
        assert not silent, (
            "these documents contain labelled identifiers and produced no "
            f"removal at all, so O1 is vacuous on them: {silent}"
        )
        assert sum(len(r) for r in redacting.values()) >= 40, (
            "the identifier corpus is too thin to be measuring anything"
        )


class TestLineTerminatorsAreStructureNotContent:
    r"""A terminator is structure. Deleting one joins two lines.

    `mao/core/deident/text.py` states this as the module's reason for existing:
    "no rule can see across a line boundary at all", and names U+0085 NEL
    explicitly among the terminators whose loss "disabled the whole labelled
    path exactly as CRLF had disabled it".

    ## REPORTED PRODUCT DEFECT — U+0085 NEL is still deleted, and it leaks

    `text._TERMINATOR` includes U+0085. `text._carries_no_visible_content` does
    not exempt it: U+0085 is general category `Cc`, and the exemption is the
    literal set `"\t\n\r\v\f"`. So `_INVISIBLE_RE` matches NEL and
    `build_match_view` strips it, the whole document becomes ONE line in the
    match view, and the labelled path mis-associates from there.

    U+2028 and U+2029 survive because they are `Zl`/`Zp` rather than `Cc`. NEL
    is the one terminator that is simultaneously "a line break" to `split_lines`
    and "invisible content" to the strip stage, and the two halves of the module
    disagree about it.

    The consequence is not cosmetic. Measured on the corpus below, the first
    field's redaction swallows the terminator and the NEXT line's label, and
    every labelled identifier after the first is left RAW in the scrubbed text —
    MRN, NHS number and telephone — which is what then travels to the provider,
    the Redis cache and the trace.

    Reported, not patched: the fix belongs in `mao/core/deident/text.py`.
    """

    WELL_BEHAVED = sorted(set(LINE_TERMINATORS) - {NEL_IS_BROKEN})

    @pytest.mark.parametrize("name", WELL_BEHAVED)
    def test_the_terminator_sequence_is_preserved_exactly(self, name: str) -> None:
        source = LINE_TERMINATORS[name]
        produced = scrub_with_report(source).text
        for terminator in ("\r\n", "\r", "\n", LS, PS, NEL, "\v", "\f"):
            assert produced.count(terminator) == source.count(terminator), (
                f"{name}: count of {terminator!r} changed from "
                f"{source.count(terminator)} to {produced.count(terminator)}"
            )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "PRODUCT DEFECT: U+0085 NEL is category Cc and is not exempted by "
            "text._carries_no_visible_content, so the match view deletes it and "
            "the document collapses to one line."
        ),
    )
    def test_a_nel_terminated_document_keeps_its_terminators(self) -> None:
        source = LINE_TERMINATORS[NEL_IS_BROKEN]
        produced = scrub_with_report(source).text
        assert produced.count(NEL) == source.count(NEL), (
            f"NEL count changed from {source.count(NEL)} to {produced.count(NEL)}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "PRODUCT DEFECT: with U+0085 as the terminator every labelled "
            "identifier after the first survives the scrubber in full."
        ),
    )
    def test_a_nel_terminated_document_redacts_every_labelled_identifier(self) -> None:
        source = (
            f"Patient Name: Harold Nkemdirim{NEL}"
            f"MRN: RGT/44219/B{NEL}"
            f"NHS Number: 943 476 5919{NEL}"
            f"Telephone: 020 7946 0958{NEL}"
            f"Complete heart block.{NEL}"
        )
        produced = scrub_with_report(source).text
        leaked = [
            value
            for value in ("Harold Nkemdirim", "RGT/44219/B", "943 476 5919",
                          "020 7946 0958")
            if value in produced
        ]
        assert leaked == [], f"raw identifiers survived the scrubber: {leaked}"

    def test_the_nel_defect_is_still_present_as_described(self) -> None:
        """Pin the defect, so the two xfails above are checkable claims.

        Asserts what IS true today. If this ever fails, U+0085 has been fixed
        and the two `xfail(strict=True)` markers above will say so loudly.
        """
        source = LINE_TERMINATORS[NEL_IS_BROKEN]
        produced = scrub_with_report(source).text
        assert produced.count(NEL) < source.count(NEL), (
            "NEL terminators are no longer being deleted — delete this test and "
            "the xfails above"
        )
        assert "RGT/44219/B" in produced, (
            "the MRN no longer leaks on the NEL path — the defect may be fixed"
        )


class TestTheOracleCanSeeADefect:
    """Non-vacuity control for `_reconstruct` itself.

    `_assert_only_authorised_edits` returning green is only meaningful if it
    would go red on an output that changed a byte it was not authorised to
    change. These three mutations are the three shapes the pre-`41cfabd`
    pipeline actually produced, applied by hand to a real result.
    """

    SOURCE = f"Patient Name: Harold Nkemdirim{SHY}\nCafé au lait spots noted.\n"

    def _reconstructed(self) -> tuple[str, str]:
        result = scrub_with_report(self.SOURCE)
        return result.text, _reconstruct(self.SOURCE, result.removals)

    def test_a_dropped_combining_mark_is_detected(self) -> None:
        produced, authorised = self._reconstructed()
        mutated = produced.replace("é", "e")
        assert mutated != authorised, (
            "the oracle cannot see a dropped combining mark — it is measuring "
            "nothing about content preservation"
        )

    def test_a_dropped_soft_hyphen_is_detected(self) -> None:
        produced, authorised = self._reconstructed()
        mutated = produced.replace(SHY, "")
        assert mutated != authorised, (
            "the oracle cannot see a deleted invisible character"
        )

    def test_a_changed_terminator_is_detected(self) -> None:
        produced, authorised = self._reconstructed()
        mutated = produced.replace("\n", "\r\n")
        assert mutated != authorised, "the oracle cannot see a rewritten terminator"

    def test_a_widened_placeholder_is_detected(self) -> None:
        """A redaction that took MORE than its recorded span must also fail."""
        produced, authorised = self._reconstructed()
        mutated = produced.replace("[NAME]", "[NAME] extra")
        assert mutated != authorised, "the oracle cannot see an over-wide redaction"


class TestTheRecordIsSufficientToReconstructTheOutput:
    r"""REPORTED PRODUCT DEFECT — `Removal` under-describes a cued-name edit.

    The shape rules `_cued_name` handles do not replace their whole match. Given
    `Her daughter Sarah Okonkwo reports confusion.` the transformation writes
    `daughter [NAME]` over the span `daughter Sarah Okonkwo` — but the `Removal`
    it records says the span `[4, 26)` became the placeholder for kind `NAME`,
    and `Removal` has no field carrying the replacement that was actually
    written.

    So the record asserts a removal that did not happen: `daughter` is still
    standing in the output, and a reader holding only the record cannot
    reconstruct the document that was produced. `source.apply_edits` states the
    opposite in its own docstring — "a caller's removal record describes the
    document that was produced and not the one that was proposed".

    This is NOT a relaxation of O1. The assertion below is the full invariant,
    unmodified; `strict=True` means the day `mao/` is fixed this test fails and
    tells whoever fixed it to delete the marker. Reported, not patched, because
    the fix belongs in `mao/core/deident/report.py` and this worktree may not
    touch it.
    """

    CUED = (
        "Her daughter Sarah Okonkwo reports increasing confusion.",
        "His wife Margaret Whitfield attends with him.",
        "I reviewed Gordon Whitfield in the memory clinic today.",
        "We assessed Harold Nkemdirim on the ward this morning.",
    )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "PRODUCT DEFECT: Removal records the whole cued match as the "
            "redacted span but the transformation only replaced the name half, "
            "and Removal carries no replacement field to say so."
        ),
    )
    @pytest.mark.parametrize("source", CUED)
    def test_a_cued_name_removal_describes_the_edit_it_made(self, source: str) -> None:
        _assert_only_authorised_edits(source)

    @pytest.mark.parametrize("source", CUED)
    def test_the_cue_word_is_the_part_the_record_over_claims(self, source: str) -> None:
        """Pin the defect precisely, so the report above is checkable.

        This asserts what IS true today: the recorded span covers a leading cue
        word that the output still contains. It is a description of the defect,
        not a substitute for the invariant above.
        """
        result = scrub_with_report(source)
        assert result.removals, f"no removal at all on {source!r}"
        over_claimed = [
            removal
            for removal in result.removals
            if removal.value.split()[0] in result.text
        ]
        assert over_claimed, (
            "this case no longer over-claims — the defect may be fixed, in "
            "which case delete this test and the xfail above"
        )
