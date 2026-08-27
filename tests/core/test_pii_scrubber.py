import pytest
from mao.core.pii_scrubber import scrub_pii

def test_scrub_email():
    result = scrub_pii("Contact me at john.doe@example.com for info.")
    assert "john.doe@example.com" not in result
    assert "[EMAIL]" in result

def test_scrub_ssn():
    result = scrub_pii("SSN is 123-45-6789 ok.")
    assert "123-45-6789" not in result
    assert "[SSN]" in result

def test_scrub_nhs():
    result = scrub_pii("NHS 943 476 5919 registered.")
    assert "943 476 5919" not in result
    assert "[NHS]" in result

def test_scrub_phone():
    result = scrub_pii("Call me on +1-800-555-0123.")
    assert "+1-800-555-0123" not in result

def test_scrub_dob():
    result = scrub_pii("DOB: 01/12/1980.")
    assert "01/12/1980" not in result

def test_no_false_positive():
    text = "The patient scored 28/30 on MMSE."
    result = scrub_pii(text)
    assert "28/30" in result

def test_scrub_postcode():
    result = scrub_pii("Patient lives at SW1A 2AA.")
    assert "SW1A 2AA" not in result

def test_scrub_medicare():
    result = scrub_pii("Medicare ID: 12A345678B.")
    assert "12A345678B" not in result

def test_scrub_dob_dash():
    result = scrub_pii("DOB: 05-03-1975.")
    assert "05-03-1975" not in result


# ---------------------------------------------------------------------------
# P0-4 — direct identifiers that reach a third-party LLM via uploaded reports.
# Extracted PDF text is scrubbed before any provider call, so the scrubber has
# to cover the identifiers that actually head a medical report: the patient's
# name, their hospital/MRN number, and their address.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, name",
    [
        ("Patient Name: John Smith\nDOB: 01/02/1960", "John Smith"),
        ("Name: Jane Q. Doe", "Jane Q. Doe"),
        ("Patient: Robert Brown was admitted", "Robert Brown"),
        ("PATIENT NAME: Maria Garcia Lopez", "Maria Garcia Lopez"),
        ("patient name: Aoife O'Brien", "Aoife O'Brien"),
    ],
)
def test_scrub_labelled_patient_name(text: str, name: str):
    result = scrub_pii(text)
    assert name not in result
    assert "[NAME]" in result


def test_scrub_name_keeps_its_label():
    """The label survives so the model still knows a name field was present."""
    result = scrub_pii("Patient Name: John Smith")
    assert result.startswith("Patient Name:")


@pytest.mark.parametrize(
    "text, identifier",
    [
        ("MRN: 12345678", "12345678"),
        ("MRN 004512399", "004512399"),
        ("Medical Record Number: 88213456", "88213456"),
        ("Hospital Number: RX9087123", "RX9087123"),
        ("Hospital No. 4451209", "4451209"),
        ("Patient ID: 7788991", "7788991"),
    ],
)
def test_scrub_mrn_and_hospital_number(text: str, identifier: str):
    result = scrub_pii(text)
    assert identifier not in result
    assert "[MRN]" in result


@pytest.mark.parametrize(
    "text, address",
    [
        ("Address: 10 Downing Street", "10 Downing Street"),
        ("Lives at 221B Baker Street, London", "221B Baker Street"),
        ("1600 Pennsylvania Avenue", "1600 Pennsylvania Avenue"),
        ("Home: 45 Oak Rd", "45 Oak Rd"),
        ("742 Evergreen Terrace", "742 Evergreen Terrace"),
    ],
)
def test_scrub_street_address(text: str, address: str):
    result = scrub_pii(text)
    assert address not in result
    assert "[ADDRESS]" in result


@pytest.mark.parametrize(
    "clinical_text",
    [
        "The patient scored 28/30 on MMSE.",
        "Donepezil 10 mg daily was well tolerated.",
        "Hippocampal volume was reduced by 12 percent.",
        "Amyloid PET was positive; tau PET showed Braak stage IV.",
        "3 Tesla MRI demonstrated white matter hyperintensities.",
    ],
)
def test_clinical_text_is_left_intact(clinical_text: str):
    """Over-scrubbing destroys the clinical signal the model needs."""
    assert scrub_pii(clinical_text) == clinical_text
