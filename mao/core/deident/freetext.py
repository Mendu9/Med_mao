r"""Identifiers with no field label at all, matched by shape.

These rules are unchanged in substance from the ones the Wave 8 and Wave 10
adversarial reviews probed and could not break; they are moved here so that the
labelled path and the shape path stop sharing a regex list, which is how a fix
to one kept reopening the other.

Ordering is most-specific-first. A national insurance number must be tried
before the bare ten-digit rule, or its digits are consumed by it.

A bare name in arbitrary prose is not regex-detectable and is recorded as a
residual risk in the phase report. A name introduced by a relationship or a
clinical encounter is, and those are the shapes a referral letter actually uses.
"""
from __future__ import annotations

import re

from . import values
from .lexicon import NOT_A_NAME_RE

_TITLES = r"(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof(?:essor)?|Sir|Dame|Lord|Lady|Rev)"

_RELATION = (
    r"(?:wife|husband|spouse|partner|son|daughter|mother|father|brother|sister"
    r"|parents?|carer|caregiver|guardian|next\s+of\s+kin|neighbour|neighbor|friend)"
)
_ENCOUNTER = (
    r"(?:reviewed|saw|assessed|examined|met|admitted|discharged|referred"
    r"|consulted|treated|followed\s+up)"
)
# Eponyms and clinical vocabulary are the collision set for a name rule in a
# neurology letter: without this, "I reviewed Alzheimer's Disease guidance"
# redacts the disease.
_NAME_WORD = rf"(?!(?i:{NOT_A_NAME_RE})(?![A-Za-z]))[A-Z][a-z]+(?:-[A-Z][a-z]+)*"
# Two name-words minimum: one capitalised word after "saw" is far more often a
# place, an instrument or a drug than a patient.
_PERSON_NAME = rf"{_NAME_WORD}(?:[ \t]+[A-Z]\.)?(?:[ \t]+{_NAME_WORD})+"

_STREET = (
    r"(?i:street|road|avenue|lane|drive|boulevard|court|close|way|place|terrace"
    r"|crescent|gardens?|square|highway|parkway|st|rd|ave|blvd|ln|hwy|pkwy)"
)

_RULES: list[tuple[str, str]] = [
    # --- structured national identifiers ---
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    (r"\b\d{2}[A-Z]\d{6}[A-Z]\b", "[MEDICARE]"),
    # UK National Insurance number, closed up or spaced. Matched by SHAPE, not
    # validated against the real prefix rules: `QQ123456C` is HMRC's own example
    # and would fail a valid-prefix check, yet it is exactly what turns up in
    # referral letters. Redacting a near-miss costs nothing; missing a real one
    # is a disclosure.
    (rf"\b{values.NI_NUMBER}", "[NI_NUMBER]"),
    (r"\b(?:\d{3}[ \t]){2}\d{4}\b", "[NHS]"),
    (r"\b\d{10}\b", "[NHS_ID]"),
    # Nine bare digits is not a clinical quantity — doses, scores and lab values
    # are orders of magnitude shorter — so the false-positive cost is low.
    (r"\b\d{9}\b", "[ID_NUMBER]"),
    (r"\b[A-Z]{2,4}-\d{2,4}-\d{4,8}\b", "[ACCOUNT]"),

    # --- dates ---
    (r"\b\d{4}-\d{2}-\d{2}\b", "[DOB]"),
    (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DOB]"),
    (rf"\b\d{{1,2}}(?:st|nd|rd|th)?[ \t]+{values._MONTHS}\.?,?[ \t]+\d{{4}}\b", "[DOB]"),
    (rf"\b{values._MONTHS}\.?[ \t]+\d{{1,2}}(?:st|nd|rd|th)?,?[ \t]+\d{{4}}\b", "[DOB]"),

    # --- contact details ---
    (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL]"),
    (r"\+44[ \t]?\(?0?\)?[ \t]?\d{3,4}[ \t]?\d{6}\b", "[PHONE]"),
    (r"\b0\d{3,4}[ \t]?\d{6}\b", "[PHONE]"),
    (r"\b0\d{3}[ \t]\d{3}[ \t]\d{4}\b", "[PHONE]"),
    (r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b", "[PHONE]"),
    (r"\b[A-Z]{1,2}\d{1,2}[A-Z]?[ \t]?\d[A-Z]{2}\b", "[POSTCODE]"),

    # --- addresses ---
    (
        rf"\b\d{{1,5}}[A-Za-z]?[ \t]+(?:[A-Z][A-Za-z'\-]*[ \t]+){{1,4}}{_STREET}\b\.?",
        "[ADDRESS]",
    ),
    (r"\b(?i:flat|apt|apartment|unit|suite)[ \t]*\d+[A-Za-z]?\b", "[ADDRESS]"),

    # --- names in prose ---
    (
        rf"\b{_TITLES}\.?[ \t]+[A-Z][A-Za-z'\-]*"
        r"(?:[ \t]+(?:[A-Z]\.|[A-Z][A-Za-z'\-]*)){0,3}",
        "[NAME]",
    ),
    (rf"\b((?i:{_RELATION}))[ \t]+{_PERSON_NAME}\b", r"\1 [NAME]"),
    (rf"\b((?i:{_ENCOUNTER}))[ \t]+{_PERSON_NAME}\b", r"\1 [NAME]"),

    # --- named healthcare organisations (letterheads) ---
    #
    # Narrowly anchored on the organisation suffix. A general "two capitalised
    # words" rule would shred clinical vocabulary — "Alzheimer's Disease",
    # "Scheltens Scale", "Amyloid PET" — for no privacy gain.
    (
        r"\b(?:[A-Z][A-Za-z'\-]*[ \t]+){1,3}"
        r"(?:Surgery|Clinic|Practice|Hospital|Infirmary|Medical\s+(?:Centre|Center))\b",
        "[ORGANISATION]",
    ),
]

_COMPILED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern), replacement) for pattern, replacement in _RULES
]


def redact_by_shape(text: str) -> str:
    """Replace identifiers that no field label introduces."""
    for pattern, replacement in _COMPILED:
        text = pattern.sub(replacement, text)
    return text
