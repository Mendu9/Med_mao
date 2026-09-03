"""Wave 9 / B1+B2+B3 — what a labelled field's *value* actually is.

`_VALUE` was defined only by where it STOPS (`[^\\n]*?` plus a stop-lookahead),
never by what it IS. One defect, failing in both directions at once:

  B1  it can match EVERYTHING to end-of-line. A /chat query is one line, so a
      pasted `Tel:` swallowed the entire clinical question. Measured at f757375,
      7 of 8 realistic phrasings lost the whole question; the model was handed
      `Tel: [PHONE]` and answered a question nobody asked.

  B2  it can match the EMPTY STRING. A two-column letterhead extracts as
      `label \\n value`, so `\\s*$` matched immediately and the scrubber emitted
      `Patient Name: [NAME]` directly ABOVE the untouched real name — asserting
      that de-identification had happened on the line above the identifier it
      missed.

The invariant both violations break, and the one these tests hold:

    A placeholder stands for an identifier that was actually removed —
    and for nothing else.

Under-matching leaks an identifier. Over-matching destroys the clinician's
question. Both are failures; neither is traded for the other.
"""
from __future__ import annotations

import pytest

from mao.core.pii_scrubber import scrub_pii

from .pdf_fixtures import CLINICAL_LINES, IDENTIFIERS, LAYOUTS

# Real questions with a pasted header field in front, exactly as a clinician
# types them into /chat. The identifier must go; the question must stay.
QUERIES_WITH_A_PASTED_FIELD = [
    (
        "Tel: 07700 900123. Should I titrate donepezil in a patient with "
        "sinus bradycardia at 48 bpm?",
        "07700 900123",
        "bradycardia",
    ),
    (
        "Patient Name: John Smith. What is the evidence for memantine in "
        "moderate Alzheimer's disease?",
        "John Smith",
        "memantine",
    ),
    (
        "MRN: 12345678 - can I combine an SSRI with donepezil given QT prolongation?",
        "12345678",
        "QT prolongation",
    ),
    (
        "DOB: 12/03/1948, presenting with progressive aphasia. Which imaging "
        "should I order?",
        "12/03/1948",
        "imaging",
    ),
    (
        "Address: 42 Elm Street. Does anticoagulation change management in "
        "cerebral amyloid angiopathy?",
        "42 Elm Street",
        "amyloid angiopathy",
    ),
    (
        "Consultant: Dr Patel. Is amyloid PET indicated before starting lecanemab?",
        "Dr Patel",
        "lecanemab",
    ),
    (
        "Referring GP: Dr Ahmed. How should I manage agitation in advanced dementia?",
        "Dr Ahmed",
        "agitation",
    ),
    (
        "Next of Kin: Margaret Smith. Does the carer's report change the "
        "diagnostic threshold for mild cognitive impairment?",
        "Margaret Smith",
        "cognitive impairment",
    ),
]


class TestTheQuestionSurvivesScrubbing:
    """B1 — the scrubber must not delete what the clinician actually asked."""

    @pytest.mark.parametrize(
        ("query", "identifier", "clinical_term"), QUERIES_WITH_A_PASTED_FIELD
    )
    def test_the_identifier_goes_and_the_question_stays(
        self, query: str, identifier: str, clinical_term: str
    ) -> None:
        scrubbed = scrub_pii(query)
        assert identifier not in scrubbed, "the identifier survived"
        assert clinical_term in scrubbed, (
            f"the clinical question was destroyed: {scrubbed!r}"
        )

    @pytest.mark.parametrize(
        ("query", "identifier", "clinical_term"), QUERIES_WITH_A_PASTED_FIELD
    )
    def test_the_question_mark_survives(
        self, query: str, identifier: str, clinical_term: str
    ) -> None:
        """The single cheapest signal that a question still exists at all."""
        assert scrub_pii(query).rstrip().endswith("?")

    def test_a_labelled_field_does_not_eat_a_following_sentence(self) -> None:
        scrubbed = scrub_pii(
            "Patient Name: John Smith. He has a background of sinus bradycardia "
            "at 48 bpm and is not currently paced."
        )
        assert "John Smith" not in scrubbed
        assert "sinus bradycardia at 48 bpm" in scrubbed
        assert "not currently paced" in scrubbed


class TestAPlaceholderMeansSomethingWasRemoved:
    """B2 — never assert de-identification that did not happen."""

    def test_a_bare_label_with_no_value_gets_no_placeholder(self) -> None:
        """`Patient Name:` alone identifies nobody. Claiming otherwise is a lie
        the downstream prompt repeats: `clinical.extraction` tells the model
        'the report has already been de-identified'."""
        assert "[NAME]" not in scrub_pii("Patient Name:")

    def test_a_placeholder_is_never_emitted_above_the_untouched_value(self) -> None:
        text = "Patient Name:\nJonathan Aldred-Whitmore\n"
        scrubbed = scrub_pii(text)
        assert "Jonathan Aldred-Whitmore" not in scrubbed, (
            "the value survived while a [NAME] placeholder claimed it had not"
        )


class TestEveryRealPdfLayout:
    """B2 — fixtures are rendered and re-extracted, not written by hand.

    `single_column` is the layout the previous fixture built, and the only one
    that ever passed. The other four leaked at the frozen SHA.
    """

    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    @pytest.mark.parametrize("identifier", IDENTIFIERS)
    def test_no_identifier_survives_extraction_and_scrubbing(
        self, layout: str, identifier: str
    ) -> None:
        extracted = LAYOUTS[layout]()
        if identifier not in extracted:
            pytest.skip(f"{identifier!r} is not present in the {layout} layout")
        assert identifier not in scrub_pii(extracted)

    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    def test_the_clinical_content_survives(self, layout: str) -> None:
        extracted = LAYOUTS[layout]()
        scrubbed = scrub_pii(extracted)
        for marker in ("MMSE 22/30", "bradycardia"):
            if marker in extracted:
                assert marker in scrubbed, f"{marker!r} was destroyed in {layout}"

    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    def test_scrubbing_is_idempotent(self, layout: str) -> None:
        once = scrub_pii(LAYOUTS[layout]())
        assert scrub_pii(once) == once


class TestFullwidthPunctuationIsNormalised:
    """B3 — `clinical_agent` scrubbed raw extraction; only the API path
    normalised. Normalising inside the scrubber makes it impossible for a
    caller to forget."""

    def test_a_fullwidth_colon_is_treated_as_a_colon(self) -> None:
        scrubbed = scrub_pii("Patient Name：Jonathan Aldred-Whitmore")
        assert "Jonathan Aldred-Whitmore" not in scrubbed
        assert "[NAME]" in scrubbed

    def test_the_scrubber_normalises_without_its_caller(self) -> None:
        """`clinical_agent.py:356` calls scrub_pii on raw pypdf output."""
        assert "RGT/44219/B" not in scrub_pii("MRN：RGT/44219/B")


class TestLineScopeIsStructural:
    """Replaces the old assertion that some pattern carries `re.MULTILINE`.

    That test guarded the property "a value ends at the end of its line" by
    asserting the *mechanism* that implemented it. The mechanism is gone —
    `_VALUE` no longer terminates on `$`, because a value is now built from
    tokens joined by `[ \\t]+`, which cannot cross a newline at all. The
    property is stronger than before, so it is asserted directly here as
    behaviour rather than as a flag on a compiled pattern.
    """

    def test_a_value_never_crosses_a_newline_into_narrative(self) -> None:
        scrubbed = scrub_pii(
            "Patient Name: John Smith\nHe was seen in clinic on Tuesday morning."
        )
        assert "John Smith" not in scrubbed
        assert "seen in clinic on Tuesday morning" in scrubbed

    def test_a_value_never_swallows_the_next_fields_label(self) -> None:
        scrubbed = scrub_pii("Patient Name: John Smith MRN: 12345678")
        assert "12345678" not in scrubbed
        assert "[NAME]" in scrubbed and "[MRN]" in scrubbed

    @pytest.mark.parametrize(
        "tail",
        ["", "\nend of note", "\nDOB: 01/01/1950", "\nThe patient was seen today."],
    )
    def test_a_record_number_is_scrubbed_whatever_follows_it(self, tail: str) -> None:
        out = scrub_pii("MRN: RGT/44219/B" + tail)
        assert "RGT/44219/B" not in out
        assert "[MRN]" in out


class TestNamesInProseWithNoLabel:
    """B2's third row. A bare name in arbitrary prose is not regex-detectable;
    a name introduced by a relational or encounter cue is, and that covers the
    shapes a referral letter actually uses."""

    @pytest.mark.parametrize(
        ("text", "name"),
        [
            ("I reviewed Jonathan Aldred-Whitmore in clinic today.", "Jonathan Aldred-Whitmore"),
            ("His wife Margaret Aldred-Whitmore attended with him.", "Margaret Aldred-Whitmore"),
            ("I saw Arthur Kowalczyk in the memory clinic.", "Arthur Kowalczyk"),
            ("Her daughter Sarah Okonjo reports increasing confusion.", "Sarah Okonjo"),
        ],
    )
    def test_a_cued_name_is_scrubbed(self, text: str, name: str) -> None:
        assert name not in scrub_pii(text)

    @pytest.mark.parametrize(
        "text",
        [
            "I reviewed Alzheimer's Disease guidance from NICE.",
            "We assessed Braak staging on the biopsy.",
            "MRI showed medial temporal lobe atrophy, Scheltens scale 3.",
            "The patient scored 28/30 on MMSE.",
            "Amyloid PET was positive; CSF p-tau 181 elevated.",
        ],
    )
    def test_clinical_vocabulary_is_not_mistaken_for_a_name(self, text: str) -> None:
        assert "[NAME]" not in scrub_pii(text)


class TestTabularLayoutsWithNoColonAnywhere:
    """Wave 11 / A2 — a header row and a data row, each on ONE line.

    Neither line is `label \\n value` and neither is label-only, so before Wave
    11 nothing in the scrubber saw this shape at all and both the name and the
    record number reached the provider untouched.

    Tab-separated rows are asserted here rather than in the generated layouts
    because reportlab draws `\\t` as a notdef glyph, not whitespace — a tab in a
    generated PDF would test the renderer. Tabs reach `scrub_pii` from a
    clinician pasting a table into `/chat`, through the same call.
    """

    @pytest.mark.parametrize(
        ("name", "text"),
        [
            (
                "spaced_columns",
                "Patient Name    MRN    DOB\n"
                "Harold Nkemdirim    RGT/44219/B    12/03/1948\n",
            ),
            (
                "tab_columns",
                "Patient Name\tMRN\tDOB\nHarold Nkemdirim\tRGT/44219/B\t12/03/1948\n",
            ),
            (
                "data_row_above_header",
                "Harold Nkemdirim    RGT/44219/B\nPatient Name    MRN\n",
            ),
            (
                "labels_carry_colons",
                "Patient Name:    MRN:\nHarold Nkemdirim    RGT/44219/B\n",
            ),
        ],
    )
    def test_no_identifier_survives_a_table(self, name: str, text: str) -> None:
        scrubbed = scrub_pii(text)
        for identifier in ("Harold Nkemdirim", "RGT/44219/B", "12/03/1948"):
            if identifier in text:
                assert identifier not in scrubbed, f"{identifier!r} leaked from {name}"

    def test_a_header_column_with_no_data_gets_no_placeholder(self) -> None:
        """Three header columns, two data cells. The third names nobody."""
        scrubbed = scrub_pii(
            "Patient Name    MRN    Telephone\nHarold Nkemdirim    RGT/44219/B\n"
        )
        assert "[PHONE]" not in scrubbed
        assert scrubbed.split("\n")[1] == "[NAME]    [MRN]"

    def test_clinical_lines_below_a_table_are_untouched(self) -> None:
        text = (
            "Patient Name    MRN\n"
            "Harold Nkemdirim    RGT/44219/B\n"
            "Bradycardia 48 bpm untreated\n"
            "Warfarin INR 2.4\n"
        )
        scrubbed = scrub_pii(text)
        assert "Bradycardia 48 bpm untreated" in scrubbed
        assert "Warfarin INR 2.4" in scrubbed
        assert len(scrubbed.split("\n")) == len(text.split("\n"))


class TestClinicalContentIsNeverDestroyed:
    """The over-matching direction, held as its own property."""

    @pytest.mark.parametrize("line", CLINICAL_LINES)
    def test_report_narrative_is_untouched(self, line: str) -> None:
        assert scrub_pii(line) == line

    @pytest.mark.parametrize(
        "text",
        [
            "Diagnosis: Alzheimer's disease, mild stage.",
            "Impression: Progressive amnestic syndrome.",
            "Plan: Start donepezil 5 mg once daily and review in 6 weeks.",
            "Findings: Medial temporal lobe atrophy, Scheltens scale 3.",
            "History: Two-year decline in short-term recall.",
        ],
    )
    def test_a_clinical_section_heading_is_not_a_pii_label(self, text: str) -> None:
        assert scrub_pii(text) == text
