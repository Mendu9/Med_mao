"""Deterministic PII scrubbing.

Regex-only by design: this runs on the path to a third-party LLM, so it must be
fast, offline, and auditable. A model-based de-identifier would add a heavy
dependency and a second inference call to the very hop we are trying to keep
clean.

The structure that makes this tractable is that clinical documents are *labelled*.
Rather than trying to recognise what a name or a record number looks like in
free text — which is where the previous version failed, missing every identifier
in a realistic clinic header — the labelled rules key off the field label and
scrub its value whatever shape it takes. `MRN: RGT/44219/B` and `MRN: 12345678`
are both handled by knowing what `MRN:` means, not by guessing at the value.

Two rules govern the value of a labelled field:

  - it ends at the end of the line, or at the next `Label:` on the same line;
  - it never swallows that next label.

"End of the line" is why every pattern is compiled with `re.MULTILINE`. Without
that flag `$` means end of *string*, and the first rule silently became "ends at
the next recognised label, or at the end of the document" — so in a letter whose
header fields are separated by prose, which is exactly what PDF text extraction
produces, neither the patient name nor the record number was scrubbed at all.
The flag is load-bearing, not stylistic; `tests/core/test_pii_scrubber_line_scope.py`
asserts it directly.

The second rule is not a nicety. The previous name pattern greedily consumed up
to three following capitalised words, so on a single line — exactly what PDF
text extraction produces when layout collapses — "Patient Name: John Smith MRN:
12345678" matched through `MRN` and left the record number in the clear.
Adding name coverage had made record-number coverage worse.

Unlabelled identifiers (free-standing dates, emails, phone numbers, postcodes,
titled names in prose) are matched by shape, most specific first.
"""
from __future__ import annotations

import re

_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)

# Field labels whose value identifies a person.
_PERSON_LABELS = (
    r"(?:patient\s+name|patient|name|next\s+of\s+kin|nok|referring\s+gp|referred\s+by"
    r"|gp|consultant|clinician|physician|doctor|surgeon|guardian|carer|caregiver"
    r"|emergency\s+contact|mother|father|spouse|partner|informant|attending)"
)

# Field labels whose value is a clinical record identifier.
_MRN_LABELS = (
    r"(?:mrn(?:\s+(?:number|no\.?|#))?"
    r"|medical\s+record(?:\s+(?:number|no\.?|#))?"
    r"|hospital\s+(?:number|no\.?|#)"
    r"|patient\s+(?:id|identifier|number|no\.?)"
    r"|case\s+(?:number|no\.?|#)"
    r"|chart\s+(?:number|no\.?|#)"
    r"|episode\s+(?:number|no\.?|#))"
)

_NHS_LABELS = r"(?:nhs\s+(?:number|no\.?|#)|nhs)"

_ACCOUNT_LABELS = r"(?:account(?:\s+(?:number|no\.?|#))?|invoice(?:\s+(?:number|no\.?|#))?)"

_ADDRESS_LABELS = r"(?:address|home\s+address|residence|addr)"

_DOB_LABELS = r"(?:dob|d\.o\.b\.?|date\s+of\s+birth|born|birth\s+date)"

_PHONE_LABELS = r"(?:telephone|tel|phone|mobile|mob|cell|contact\s+(?:number|no\.?))"

# Every label the scrubber knows. A labelled field's value runs until the next
# *known* label, or the end of the line.
#
# Terminating on "any word followed by a colon" is wrong in both directions. On
# "Name: John Smith MRN: 12345678" it stops after "John" — because " Smith MRN:"
# looks like a label — and leaks the surname. Terminating only at end-of-line is
# wrong the other way: it lets one field swallow the next field's label, which is
# how the previous version left record numbers in the clear. Matching the known
# label set is what makes "the value is everything up to the next real field"
# expressible.
_ANY_LABEL = (
    rf"(?:{_PERSON_LABELS}|{_MRN_LABELS}|{_NHS_LABELS}|{_ACCOUNT_LABELS}"
    rf"|{_ADDRESS_LABELS}|{_DOB_LABELS}|{_PHONE_LABELS}"
    r"|postcode|post\s+code|zip(?:\s+code)?|email|e-mail|practice|surgery|clinic"
    r"|ward|bed|nhs\s+trust|hospital|department|dept)"
)

# Stop before a known label, or before any single word acting as one, or at EOL.
_VALUE = (
    r"[^\n]*?(?=\s+(?i:" + _ANY_LABEL + r")[ \t]*:"
    r"|\s+[A-Za-z][A-Za-z'\-]*[ \t]*:"
    r"|\s*$)"
)

# What a record number looks like when no colon separates it from its label.
#
# It must contain a digit. Without that, "NHS Number: [NHS]" re-matches on a
# second pass as label "NHS" + identifier "Number", and the rule eats the rest
# of its own label — non-idempotent, and on a first pass it would swallow a real
# field label the same way.
_IDENTIFIER = r"(?=[A-Za-z0-9/\-]*\d)[A-Z0-9][A-Za-z0-9/\-]{3,20}\b"

# Titles that introduce a name in prose, where no field label exists.
_TITLES = r"(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof(?:essor)?|Sir|Dame|Lord|Lady|Rev)"

_PATTERNS: list[tuple[str, str]] = [
    # --- Labelled fields (most reliable: the label tells us what the value is) ---
    #
    # The label itself is captured and re-emitted, so the model still sees which
    # field was present — "Patient Name: [NAME]" carries structure that a bare
    # "[NAME]" does not.
    (rf"\b((?i:{_DOB_LABELS}))[ \t]*:[ \t]*{_VALUE}", r"\1: [DOB]"),
    (rf"\b((?i:{_MRN_LABELS}))[ \t]*:[ \t]*{_VALUE}", r"\1: [MRN]"),
    (rf"\b((?i:{_NHS_LABELS}))[ \t]*:[ \t]*{_VALUE}", r"\1: [NHS]"),
    (rf"\b((?i:{_ACCOUNT_LABELS}))[ \t]*:[ \t]*{_VALUE}", r"\1: [ACCOUNT]"),
    (rf"\b((?i:{_ADDRESS_LABELS}))[ \t]*:[ \t]*{_VALUE}", r"\1: [ADDRESS]"),
    (rf"\b((?i:{_PHONE_LABELS}))[ \t]*:[ \t]*{_VALUE}", r"\1: [PHONE]"),
    (rf"\b((?i:{_PERSON_LABELS}))[ \t]*:[ \t]*{_VALUE}", r"\1: [NAME]"),
    # `Re: Mr John A. Smith` — a referral line, not a person label.
    (rf"\b((?i:re))[ \t]*:[ \t]*{_TITLES}\.?[ \t]+{_VALUE}", r"\1: [NAME]"),

    # Colon-less record numbers: "MRN 004512399", "Hospital No. 4451209".
    # The separator is optional ONLY for record labels, and only when what
    # follows actually looks like an identifier. Making it optional for person
    # labels would match "The patient reported difficulty..." and eat the
    # clinical narrative.
    (
        rf"\b((?i:{_MRN_LABELS}))[ \t]*[#]?[ \t]*{_IDENTIFIER}",
        r"\1: [MRN]",
    ),
    (
        rf"\b((?i:{_NHS_LABELS}))[ \t]*[#]?[ \t]*{_IDENTIFIER}",
        r"\1: [NHS]",
    ),

    # --- Structured identifiers by shape ---
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),                       # US SSN
    (r"\b\d{2}[A-Z]\d{6}[A-Z]\b", "[MEDICARE]"),
    # UK National Insurance number: two letters, six digits, an A-D suffix.
    # Written both closed up (QQ123456C) and spaced (QQ 12 34 56 C) on forms.
    #
    # Deliberately matched by SHAPE, not validated against the real prefix rules
    # (which exclude D/F/I/Q/U/V leading). De-identification is not validation:
    # `QQ123456C` is HMRC's own example number and would fail a valid-prefix
    # check, yet it is exactly what turns up in referral letters and test data.
    # Redacting a near-miss costs nothing; missing a real one is a disclosure.
    (r"\b[A-Z]{2}[ \t]?(?:\d[ \t]?){6}[A-D]\b", "[NI_NUMBER]"),
    (r"\b(?:\d{3}[ \t]){2}\d{4}\b", "[NHS]"),                  # 943 476 5919
    (r"\b\d{10}\b", "[NHS_ID]"),                               # 9434765919
    # Passport / long civil identifier. Nine bare digits is not a clinical
    # quantity — doses, scores and lab values are orders of magnitude shorter —
    # so the false-positive cost is low and the disclosure cost is not.
    (r"\b\d{9}\b", "[ID_NUMBER]"),
    (r"\b[A-Z]{2,4}-\d{2,4}-\d{4,8}\b", "[ACCOUNT]"),          # ACC-2024-889231

    # --- Dates ---
    (r"\b\d{4}-\d{2}-\d{2}\b", "[DOB]"),                       # ISO 1948-03-12
    (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DOB]"),
    (rf"\b\d{{1,2}}(?:st|nd|rd|th)?[ \t]+{_MONTHS}\.?,?[ \t]+\d{{4}}\b", "[DOB]"),
    (rf"\b{_MONTHS}\.?[ \t]+\d{{1,2}}(?:st|nd|rd|th)?,?[ \t]+\d{{4}}\b", "[DOB]"),

    # --- Contact details ---
    (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL]"),
    (r"\+44[ \t]?\(?0?\)?[ \t]?\d{3,4}[ \t]?\d{6}\b", "[PHONE]"),   # +44 7700 900123
    (r"\b0\d{3,4}[ \t]?\d{6}\b", "[PHONE]"),                        # 07700 900123
    (r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b", "[PHONE]"),
    (r"\b[A-Z]{1,2}\d{1,2}[A-Z]?[ \t]?\d[A-Z]{2}\b", "[POSTCODE]"),

    # --- Addresses ---
    (
        r"\b\d{1,5}[A-Za-z]?[ \t]+(?:[A-Z][A-Za-z'\-]*[ \t]+){1,4}"
        r"(?i:street|road|avenue|lane|drive|boulevard|court|close|way|place|terrace"
        r"|crescent|gardens?|square|highway|parkway|st|rd|ave|blvd|ln|hwy|pkwy)\b\.?",
        "[ADDRESS]",
    ),
    (r"\b(?i:flat|apt|apartment|unit|suite)[ \t]*\d+[A-Za-z]?\b", "[ADDRESS]"),

    # --- Titled names in prose (no field label to key off) ---
    (
        rf"\b{_TITLES}\.?[ \t]+[A-Z][A-Za-z'\-]*"
        r"(?:[ \t]+(?:[A-Z]\.|[A-Z][A-Za-z'\-]*)){0,3}",
        "[NAME]",
    ),

    # --- Named healthcare organisations (letterheads) ---
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

# MULTILINE is required, not cosmetic: `_VALUE` terminates on `\s*$`, and
# without it `$` matches only at end-of-string. See the module docstring.
_COMPILED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.MULTILINE), replacement) for pattern, replacement in _PATTERNS
]


def scrub_pii(text: str) -> str:
    """Replace personal identifiers with typed placeholders.

    Placeholders keep the field's *shape* so the model can still tell that a
    patient name or record number was present without learning whose.
    """
    if not text:
        return text
    for pattern, replacement in _COMPILED:
        text = pattern.sub(replacement, text)
    return text
