"""Deterministic PII scrubbing.

Regex-only by design: this runs on the path to a third-party LLM, so it must be
fast, offline, and auditable. A model-based de-identifier would add a heavy
dependency and a second inference call to the very hop we are trying to keep
clean.

P0-4 extended coverage beyond account-style identifiers (SSN/NHS/DOB/email/
postcode/phone) to the identifiers that actually head an uploaded medical
report: the patient's name, their hospital/MRN number, and their address.

Patterns are ordered most-specific first. Labelled patterns (`Patient Name:`,
`MRN:`) are anchored on their label so they cannot fire on prose.
"""
import re

# Label -> value patterns keep the label so the model still knows the field was
# present; free-standing patterns replace the whole match.
_LABELLED_NAME = (
    r"\b((?i:patient\s+name|patient|name))[ \t]*:[ \t]*"
    r"[A-Z][A-Za-z'’\-]*(?:[ \t]+(?:[A-Z]\.|[A-Z][A-Za-z'’\-]*)){0,3}"
)

_MRN = (
    r"\b(?i:mrn|medical\s+record\s+(?:number|no\.?|#)"
    r"|hospital\s+(?:number|no\.?|#)"
    r"|patient\s+(?:id|identifier))"
    r"[ \t]*[:#]?[ \t]*[A-Z]{0,3}-?\d{4,12}\b"
)

# <house number><optional letter> <one or more capitalised words> <street suffix>
_STREET_ADDRESS = (
    r"\b\d{1,5}[A-Za-z]?[ \t]+(?:[A-Z][A-Za-z'’\-]*[ \t]+){1,4}"
    r"(?i:street|road|avenue|lane|drive|boulevard|court|close|way|place|terrace"
    r"|crescent|gardens?|square|highway|parkway|st|rd|ave|blvd|ln|hwy|pkwy)\b\.?"
)

_PATTERNS = [
    (_LABELLED_NAME, r"\1: [NAME]"),
    (_MRN, "[MRN]"),
    (_STREET_ADDRESS, "[ADDRESS]"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    (r"\b(\d{3}\s){2}\d{4}\b", "[NHS]"),
    (r"\b\d{9,10}\b(?=\s*(nhs|NHS))", "[NHS_ID]"),
    (r"\b\d{2}[A-Z]{1}\d{6}[A-Z]{1}\b", "[MEDICARE]"),
    (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", "[DOB]"),
    (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL]"),
    (r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b", "[POSTCODE]"),
    (r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}", "[PHONE]"),
    (r"\+44\s?\d{4}\s?\d{6}", "[PHONE]"),
]


def scrub_pii(text: str) -> str:
    for pattern, replacement in _PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text
