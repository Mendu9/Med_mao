import re

_PATTERNS = [
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
