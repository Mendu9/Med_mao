"""H1/H1b — the PII scrubber against a realistic clinic letter.

The P0-4 plumbing is correct: `clinical_agent` scrubs before extraction,
summarisation, the retrieval seed and the synthesis prompt. The scrubber itself
was the weak link — 14 of 14 identifiers in a realistic memory-clinic header
survived it.

H1b is worse than a gap: `_LABELLED_NAME` greedily consumed the *next field's
label*, so "Patient Name: John Smith MRN: 12345678" became
"Patient Name: [NAME]: 12345678" and left the MRN in the clear. Adding name
coverage made MRN coverage worse.

This runs on the path to a third-party LLM, so the bar is "no identifier
reaches the provider", not "most don't".
"""
from __future__ import annotations

import pytest

from mao.core.pii_scrubber import scrub_pii

CLINIC_LETTER = """\
Beechwood Surgery
Re: Mr John A. Smith
Patient Name: John A. Smith
DOB: 1948-03-12 (12 March 1948)
MRN: RGT/44219/B
NHS Number: 943 476 5919
Address: Flat 4, 22 Kingsway, London
Postcode: SW1A 1AA
Telephone: 07700 900123
Email: john.smith@example.com
Next of Kin: Margaret Smith (07700 900456)
Referring GP: Dr Sarah Okonjo
Consultant: Prof A. Raman
Account: ACC-2024-889231
"""

# Every identifier the adversarial review found leaking, verbatim.
LEAKED_IN_REVIEW = [
    "John A. Smith",
    "1948-03-12",
    "RGT/44219/B",
    "Kingsway",
    "07700 900123",
    "Margaret Smith",
    "Sarah Okonjo",
    "07700 900456",
    "A. Raman",
    "9434765919",
    "943 476 5919",
    "ACC-2024-889231",
    "12 March 1948",
    "john.smith@example.com",
]


class TestTheClinicLetterIsFullyScrubbed:
    @pytest.mark.parametrize("identifier", LEAKED_IN_REVIEW)
    def test_identifier_does_not_survive(self, identifier: str) -> None:
        assert identifier not in scrub_pii(CLINIC_LETTER)

    def test_no_bare_surname_survives(self) -> None:
        scrubbed = scrub_pii(CLINIC_LETTER)
        assert "Smith" not in scrubbed
        assert "Okonjo" not in scrubbed

    def test_the_clinical_content_around_it_is_preserved(self) -> None:
        text = "Patient Name: John Smith\nMMSE score was 24/30 and CSF amyloid was low."
        scrubbed = scrub_pii(text)
        assert "MMSE score was 24/30" in scrubbed
        assert "CSF amyloid was low" in scrubbed


class TestH1bLabelsAreNotEaten:
    """The name pattern must never consume the following field's label."""

    @pytest.mark.parametrize(
        ("text", "must_not_contain"),
        [
            ("Patient Name: John Smith MRN: 12345678", "12345678"),
            ("Name: Jane Doe Hospital Number: 4421900", "4421900"),
            ("Patient Name: John Smith\nMRN: 12345678", "12345678"),
            ("Name: Ann Lee NHS Number: 943 476 5919", "943 476 5919"),
        ],
    )
    def test_the_next_field_is_still_scrubbed(self, text: str, must_not_contain: str) -> None:
        assert must_not_contain not in scrub_pii(text)

    def test_the_mrn_label_survives_as_a_label(self) -> None:
        """Collapsing the field into [NAME] would scrub it but lose the schema."""
        scrubbed = scrub_pii("Patient Name: John Smith MRN: 12345678")
        assert "[MRN]" in scrubbed
        assert "[NAME]" in scrubbed

    @pytest.mark.parametrize(
        "text",
        [
            "Patient Name: John Smith MRN: 12345678",
            "Name: John Smith Hospital Number: 4421900",
            "Name: John Smith Weight: 70kg",
            "Next of Kin: Margaret Smith Tel: 07700 900456",
        ],
    )
    def test_the_whole_name_is_taken_not_just_the_first_word(self, text: str) -> None:
        """A terminator keyed on "any word then colon" stops after "John",
        because " Smith MRN:" looks like a label — and leaks the surname."""
        assert "Smith" not in scrub_pii(text)

    def test_a_multi_word_label_still_terminates_the_value(self) -> None:
        scrubbed = scrub_pii("Name: Jane Doe Hospital Number: 4421900")
        assert "4421900" not in scrubbed
        assert "Doe" not in scrubbed


class TestIdentifiersKeepTheirOwnLabel:
    """An NHS number relabelled [MRN] is scrubbed but mis-described."""

    def test_nhs_number_is_labelled_nhs(self) -> None:
        assert "[NHS]" in scrub_pii("NHS Number: 943 476 5919")

    def test_account_is_labelled_account(self) -> None:
        assert "[ACCOUNT]" in scrub_pii("Account: ACC-2024-889231")

    def test_mrn_is_labelled_mrn(self) -> None:
        assert "[MRN]" in scrub_pii("MRN: RGT/44219/B")

    def test_the_original_label_is_preserved(self) -> None:
        """"[NAME]" alone loses the schema; the model should still see the field."""
        assert scrub_pii("Patient Name: John Smith").startswith("Patient Name:")
        assert scrub_pii("Hospital Number: 4421900").startswith("Hospital Number:")

    @pytest.mark.parametrize(
        "text",
        ["MRN 004512399", "Hospital No. 4451209", "MRN #88213456"],
    )
    def test_record_numbers_without_a_colon(self, text: str) -> None:
        assert not any(ch.isdigit() for ch in scrub_pii(text))

    @pytest.mark.parametrize(
        "text",
        ["NHS Number: 943 476 5919", "Hospital Number: 4451209", "MRN Number: 12345678"],
    )
    def test_a_colonless_rule_never_eats_its_own_label(self, text: str) -> None:
        """The identifier must contain a digit, or "Number" is read as one."""
        scrubbed = scrub_pii(text)
        assert scrub_pii(scrubbed) == scrubbed
        assert scrubbed.count("[") == 1

    def test_a_label_word_alone_is_not_treated_as_an_identifier(self) -> None:
        assert scrub_pii("The NHS Number scheme was introduced nationally.") == (
            "The NHS Number scheme was introduced nationally."
        )


class TestNamedHealthcareOrganisations:
    @pytest.mark.parametrize(
        "text",
        [
            "Beechwood Surgery",
            "St Thomas' Hospital",
            "Riverside Medical Centre",
            "Oakhill Clinic",
        ],
    )
    def test_organisation_names_are_scrubbed(self, text: str) -> None:
        assert "[ORGANISATION]" in scrub_pii(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Alzheimer's disease was confirmed.",
            "Scheltens scale 3 on coronal MRI.",
            "Amyloid PET positive.",
            "The patient was admitted to hospital overnight.",
        ],
    )
    def test_clinical_vocabulary_is_not_caught_by_the_organisation_rule(
        self, text: str
    ) -> None:
        assert "[ORGANISATION]" not in scrub_pii(text)


class TestIdentifierFamilies:
    @pytest.mark.parametrize(
        "text",
        [
            "DOB: 1948-03-12",
            "DOB: 12/03/1948",
            "DOB: 12-03-1948",
            "Born 12 March 1948",
            "Born March 12, 1948",
        ],
    )
    def test_dates_of_birth(self, text: str) -> None:
        assert "1948" not in scrub_pii(text)

    @pytest.mark.parametrize(
        "text",
        [
            "MRN: 12345678",
            "MRN: RGT/44219/B",
            "Hospital Number: 4421900",
            "Medical Record Number: AB-123456",
            "Patient ID: X99/2011",
        ],
    )
    def test_record_numbers(self, text: str) -> None:
        scrubbed = scrub_pii(text)
        assert "[MRN]" in scrubbed
        assert not any(ch.isdigit() for ch in scrubbed)

    @pytest.mark.parametrize(
        "text",
        ["Telephone: 07700 900123", "Mobile: 07700900123", "Tel: +44 7700 900123"],
    )
    def test_uk_mobiles(self, text: str) -> None:
        assert "900123" not in scrub_pii(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Re: Mr John A. Smith",
            "Referring GP: Dr Sarah Okonjo",
            "Consultant: Prof A. Raman",
            "Seen by Dr Okonjo today",
            "name: john smith",
        ],
    )
    def test_names_labelled_titled_and_lowercase(self, text: str) -> None:
        scrubbed = scrub_pii(text)
        for token in ("Smith", "Okonjo", "Raman", "smith"):
            assert token not in scrubbed

    @pytest.mark.parametrize(
        "text",
        [
            "Address: Flat 4, 22 Kingsway, London",
            "Address: 12 Beechwood Street",
            "22 Kingsway Road",
        ],
    )
    def test_addresses_with_and_without_a_street_suffix(self, text: str) -> None:
        assert "Kingsway" not in scrub_pii(text) or "[ADDRESS]" in scrub_pii(text)
        assert "[ADDRESS]" in scrub_pii(text)

    def test_nhs_number_spaced_and_unspaced(self) -> None:
        assert "943 476 5919" not in scrub_pii("NHS Number: 943 476 5919")
        assert "9434765919" not in scrub_pii("NHS Number: 9434765919")

    def test_account_numbers(self) -> None:
        assert "ACC-2024-889231" not in scrub_pii("Account: ACC-2024-889231")


class TestClinicalVocabularyIsNotDestroyed:
    """Over-scrubbing a clinical note is its own failure mode."""

    @pytest.mark.parametrize(
        "text",
        [
            "Donepezil 10mg once daily was well tolerated.",
            "MMSE 24/30, MoCA 19/30.",
            "MRI showed medial temporal lobe atrophy, Scheltens scale 3.",
            "Amyloid PET was positive; CSF p-tau 181 elevated.",
            "Diagnosis: Alzheimer's disease, mild stage.",
            "The patient reported difficulty with short-term recall.",
        ],
    )
    def test_clinical_text_is_untouched(self, text: str) -> None:
        assert scrub_pii(text) == text

    def test_idempotent(self) -> None:
        once = scrub_pii(CLINIC_LETTER)
        assert scrub_pii(once) == once

    def test_empty_and_trivial_input(self) -> None:
        assert scrub_pii("") == ""
        assert scrub_pii("hello") == "hello"
