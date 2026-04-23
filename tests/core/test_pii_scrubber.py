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
