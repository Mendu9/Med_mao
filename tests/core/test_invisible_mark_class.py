r"""ADV16-1: NFKC composed a combining mark away before the strip could see it.

`normalise` was `_INVISIBLE_RE.sub("", unicodedata.normalize("NFKC", text))`.
`_carries_no_visible_content` correctly returned True for every `Mn`, so U+0301
WAS in the strip set — but NFKC had already composed it into its base letter, so
there was no `Mn` character left to remove. The strip therefore only ever fired
for a (mark, base) pair Unicode has no precomposed form for, which is exactly why
a cedilla on `A` held and an acute on `A` did not.

Measured consequence at `b63311d`: 132 of 570 (mark, position) probes leaked, and
`Postcode: SW1A 1AA` and `MRN: A1234567` survived byte-identical to a third-party
model.

The second class of case here is not from either review. The blanket `Mn` strip
rested on "non-spacing marks have zero advance width by definition", which is
true and is not the same property as "carries no visible content": a Devanagari
virama is a non-spacing mark that changes what the word says. Fixing the order
without fixing the class would have started destroying those.
"""
from __future__ import annotations

from mao.core.deident.text import normalise


class TestACombiningMarkCannotHideAnIdentifier:
    """The ADV16-1 reproduction."""

    MARKS = ("̀", "́", "̂", "̃", "̈", "̊")

    def test_an_accented_postcode_still_reads_as_the_postcode(self) -> None:
        value = "SW1A 1AA"
        leaks: list[str] = []
        for mark in self.MARKS:
            for position, character in enumerate(value):
                if character == " ":
                    continue
                decorated = value[: position + 1] + mark + value[position + 1 :]
                if value not in normalise(f"Postcode: {decorated}"):
                    leaks.append(f"{mark!r}@{position}")
        assert leaks == [], f"the postcode was hidden by {len(leaks)} mark(s): {leaks}"

    def test_an_accented_record_number_still_reads_as_the_record_number(self) -> None:
        leaks = [
            mark
            for mark in self.MARKS
            if "A1234567" not in normalise(f"MRN: A{mark}1234567")
        ]
        assert leaks == [], f"the record number was hidden by {leaks}"

    def test_a_mark_on_a_national_insurance_number_does_not_hide_it(self) -> None:
        leaks = [
            mark
            for mark in self.MARKS
            if "QQ123456C" not in normalise(f"NI Number: QQ{mark}123456C")
        ]
        assert leaks == [], f"the NI number was hidden by {leaks}"


class TestASemanticMarkSurvives:
    """Not a review finding. A defect the corrected mark class closes anyway.

    These marks have a non-zero canonical combining class and compose with
    nothing, so they are drawn and they carry meaning. A blanket `Mn` strip
    applied to the decomposed form would delete them, which is the silent
    content destruction this project keeps failing on, in a new script.
    """

    def test_devanagari_virama_survives(self) -> None:
        assert "्" in normalise("प्र")

    def test_devanagari_nukta_survives(self) -> None:
        assert "़" in normalise("क़")

    def test_a_hebrew_point_survives(self) -> None:
        assert "ַ" in normalise("אַ")

    def test_an_arabic_fatha_survives(self) -> None:
        assert "َ" in normalise("بَ")


class TestTheInvisibleFillerClassStillHolds:
    """The predecessor's own bypass characters. 00_RULES: these run first."""

    def test_a_grapheme_joiner_inside_a_record_number_is_removed(self) -> None:
        assert normalise("MRN: RGT/44͏219/B") == "MRN: RGT/44219/B"

    def test_a_soft_hyphen_from_pdf_hyphenation_is_removed(self) -> None:
        assert normalise("MRN: RGT/44219­/B") == "MRN: RGT/44219/B"

    def test_a_variation_selector_is_removed(self) -> None:
        assert normalise("MRN: A️1234567") == "MRN: A1234567"

    def test_a_zero_width_joiner_is_removed(self) -> None:
        assert normalise("MRN: A‍1234567") == "MRN: A1234567"

    def test_a_control_character_is_removed(self) -> None:
        assert normalise("Telephone: 0113 496 0231") == "Telephone: 0113 496 0231"

    def test_a_hangul_filler_is_removed(self) -> None:
        assert normalise("MRN: Aㅤ" "1234567") == "MRN: A1234567"

    def test_every_line_terminator_survives(self) -> None:
        assert normalise("a\r\nb\rc\ndef") == "a\r\nb\rc\ndef"

    def test_a_fullwidth_colon_still_folds_to_a_colon(self) -> None:
        assert normalise("Patient Name： Harold") == "Patient Name: Harold"
