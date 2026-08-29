"""Wave 6 blocker 1 — a labelled field's value ends at the end of its LINE.

`_VALUE` terminates on `\\s*$`, but no pattern was compiled with `re.MULTILINE`,
so `$` meant end-of-*string*. A labelled field was scrubbed only when another
*recognised* label followed it, or when it was the last line of the document.
The module docstring states the rule as "it ends at the end of the line". That
rule was not implemented.

Measured at the frozen SHA — same field, only the trailing text differs:

    'MRN: RGT/44219/B'                -> 'MRN: [MRN]'
    'MRN: RGT/44219/B\\nend of note'   -> 'MRN: RGT/44219/B'    LEAKS
    'MRN: RGT/44219/B\\nDOB: 01/1950'  -> 'MRN: [MRN]'

The Wave 5 fixture placed every labelled field contiguously, so a known label
always followed and the end-of-string terminator was never exercised. It passed
while the scrubber leaked. Real PDF text extraction interleaves header fields
with prose, which is why the fixtures here are shaped that way — the shape *is*
the test.
"""
from __future__ import annotations

import re

import pytest

from mao.core import pii_scrubber
from mao.core.pii_scrubber import scrub_pii

# A discharge summary as PDF extraction actually yields it: header fields
# separated by narrative, not stacked in a contiguous block.
INTERLEAVED_LETTER = """DISCHARGE SUMMARY

Patient Name: Arthur Kowalczyk
The patient was admitted with acute confusion and a two-day history of falls.

MRN: LDS/9931/C
He was discharged on donepezil 10 mg once daily and reviewed at four weeks.

Next of Kin: Barbara Kowalczyk
Cognitive testing showed an MMSE of 21/30, down from 24/30 last year.

DOB: 12/03/1948
Follow-up in the memory clinic has been arranged.
"""

IDENTIFIERS = [
    "Arthur Kowalczyk",
    "LDS/9931/C",
    "Barbara Kowalczyk",
    "12/03/1948",
]


class TestALabelledValueStopsAtEndOfLine:
    def test_a_pattern_is_compiled_multiline(self) -> None:
        assert any(p.flags & re.MULTILINE for p, _ in pii_scrubber._COMPILED), (
            "no pattern uses re.MULTILINE, so `$` in _VALUE means end-of-string "
            "and a labelled field only scrubs when it is the last line"
        )

    @pytest.mark.parametrize(
        "tail",
        ["", "\nend of note", "\nDOB: 01/01/1950", "\nThe patient was seen today."],
    )
    def test_the_record_number_is_scrubbed_whatever_follows_it(self, tail: str) -> None:
        out = scrub_pii("MRN: RGT/44219/B" + tail)
        assert "RGT/44219/B" not in out
        assert "[MRN]" in out

    def test_prose_after_a_field_does_not_defeat_the_rule(self) -> None:
        out = scrub_pii(
            "Patient Name: John Smith\nHe was seen in clinic on Tuesday morning."
        )
        assert "John Smith" not in out
        assert "seen in clinic" in out, "the narrative must survive"


class TestTheInterleavedDischargeSummary:
    """The exact document shape the reviewer used to prove PHI reached Groq."""

    @pytest.mark.parametrize("identifier", IDENTIFIERS)
    def test_no_direct_identifier_survives(self, identifier: str) -> None:
        assert identifier not in scrub_pii(INTERLEAVED_LETTER)

    def test_the_clinical_narrative_survives(self) -> None:
        out = scrub_pii(INTERLEAVED_LETTER)
        assert "donepezil 10 mg" in out
        assert "acute confusion" in out
        assert "MMSE of 21/30" in out, "a test score is not an identifier"

    def test_the_field_labels_survive_so_structure_is_preserved(self) -> None:
        out = scrub_pii(INTERLEAVED_LETTER)
        assert "Patient Name:" in out
        assert "MRN:" in out

    def test_scrubbing_is_idempotent(self) -> None:
        once = scrub_pii(INTERLEAVED_LETTER)
        assert scrub_pii(once) == once


class TestNationalIdentifiersTheScrubberUsedToMiss:
    """Architecture review, MEDIUM: 13/15 caught; NI number and passport leaked."""

    def test_uk_national_insurance_number(self) -> None:
        out = scrub_pii("The patient's NI number is QQ123456C on file.")
        assert "QQ123456C" not in out

    def test_uk_ni_number_with_spaces(self) -> None:
        out = scrub_pii("NI: QQ 12 34 56 C")
        assert "QQ 12 34 56 C" not in out

    def test_passport_number(self) -> None:
        out = scrub_pii("Passport: 123456789 issued 2019.")
        assert "123456789" not in out

    @pytest.mark.parametrize(
        "clinical",
        [
            "The patient scored 28/30 on MMSE.",
            "Donepezil 10 mg once daily.",
            "CSF amyloid beta 42 was 412 pg/mL.",
            "MRI showed hippocampal atrophy, Scheltens scale 3.",
            "Follow up in 6 months.",
        ],
    )
    def test_clinical_vocabulary_is_not_redacted(self, clinical: str) -> None:
        assert scrub_pii(clinical) == clinical
