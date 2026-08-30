r"""Deterministic PII scrubbing.

Regex-only by design: this runs on the path to a third-party LLM, so it must be
fast, offline, and auditable. A model-based de-identifier would add a heavy
dependency and a second inference call to the very hop we are trying to keep
clean.

The structure that makes this tractable is that clinical documents are *labelled*.
Rather than trying to recognise what a name or a record number looks like in
free text — which is where an earlier version failed, missing every identifier
in a realistic clinic header — the labelled rules key off the field label and
scrub its value whatever shape it takes. `MRN: RGT/44219/B` and `MRN: 12345678`
are both handled by knowing what `MRN:` means, not by guessing at the value.

## The invariant

    A placeholder stands for an identifier that was actually removed —
    and for nothing else.

Both directions are failures. Under-matching leaks an identifier. Over-matching
destroys the clinician's question, and does it silently, because `state["user_query"]`
*is* the scrubbed string: no control downstream can see what the original said.

## Why a value is defined positively

The previous `_VALUE` was `[^\n]*?` plus a stop-lookahead — defined only by
where it stopped, never by what it was. One defect, failing both ways at once:

  - it could match EVERYTHING to end-of-line. A `/chat` query is one line, so a
    pasted `Tel:` swallowed the whole question. Measured: 7 of 8 realistic
    phrasings lost the question entirely, and the model answered a question
    nobody asked, confidently, with a clinical disclaimer attached.

  - it could match the EMPTY STRING. `\s*$` matches at once when a label ends
    its line, which is exactly what a two-column letterhead extracts as. The
    scrubber emitted `Patient Name: [NAME]` directly ABOVE the untouched real
    name — asserting de-identification on the line above the identifier it
    missed, while the registered `clinical.extraction` prompt told the model the
    report "has already been de-identified".

So a value is now built up from tokens rather than carved out by terminators. A
value token is a name-, number-, or code-shaped word: it is not a function word,
and it is not the label of the next field. A value is one to eight such tokens
joined by spaces or commas. It therefore cannot be empty, cannot cross a
newline (the separator is `[ \t]+`), and cannot run into prose.

## Why the document is reflowed first

pypdf reads a two-column letterhead column by column, so every value lands on
the line *after* its label — and in a wide layout, every label can arrive before
any value. Rather than teach every rule about that, `_reflow_orphan_labels`
rejoins `label \n value` into `label: value` and hands the rules the shape they
are defined for. One normalisation, not a second rule set.

NFKC normalisation happens here too, not only in `input_guardrails`:
`clinical_agent` scrubs raw pypdf output directly, and a fullwidth colon
(U+FF1A, routine in scanned documents) is preserved by pypdf and matches no
ASCII-colon rule at all. Normalising inside the scrubber makes it impossible
for a caller to forget.

Unlabelled identifiers (free-standing dates, emails, phone numbers, postcodes,
titled names in prose) are matched by shape, most specific first.
"""
from __future__ import annotations

import re
import unicodedata

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

# Every label the scrubber knows. A value never absorbs one of these, because
# doing so is how an earlier version turned "Patient Name: John Smith MRN:
# 12345678" into "Patient Name: [NAME]: 12345678" and left the record number in
# the clear. Adding name coverage had made record-number coverage worse.
_ANY_LABEL = (
    rf"(?:{_PERSON_LABELS}|{_MRN_LABELS}|{_NHS_LABELS}|{_ACCOUNT_LABELS}"
    rf"|{_ADDRESS_LABELS}|{_DOB_LABELS}|{_PHONE_LABELS}"
    r"|postcode|post\s+code|zip(?:\s+code)?|email|e-mail|practice|surgery|clinic"
    r"|ward|bed|nhs\s+trust|hospital|department|dept)"
)

# Function words. These open almost every clinical sentence and appear inside no
# identifier, so they are what separates "Robert Brown was admitted" (value:
# "Robert Brown") from "an 84-year-old man with..." (no value at all).
#
# "May" is deliberately absent despite being a function word: it is also a
# month, and excluding it would truncate "DOB: 12 May 1948" after the day and
# leave the year in the clear.
_PROSE_WORDS = (
    r"(?:a|an|the|and|or|but|if|is|are|was|were|be|been|being|has|have|had|do|does"
    r"|did|can|could|should|would|will|shall|might|must|in|on|at|to|for|with"
    r"|without|from|by|of|as|that|this|these|those|there|their|his|her|its|he|she"
    r"|they|them|him|who|whom|which|what|when|where|why|how|not|no|nor|so|than"
    r"|then|about|after|before|during|since|until|while|per|via|presenting"
    r"|presented|referred|admitted|discharged|reviewed|seen|known|reports"
    r"|reported|complains|complaining|diagnosed|treated|started|stopped|continues"
    r"|denies|attended|attends|please|regarding|re)"
)

# A token may not be the next field's label. Both forms matter: the known
# multi-word labels ("NHS Number:") and any bare word acting as one ("Weight:").
_NOT_A_LABEL = (
    rf"(?!(?i:{_ANY_LABEL})[ \t]*:)"
    r"(?![A-Za-z][A-Za-z0-9'\-]*[ \t]*:)"
)
_NOT_PROSE = rf"(?!(?i:{_PROSE_WORDS})\b)"

# A single initial ("A." in "John A. Smith") must be tried first: the general
# word form would match the bare "A" and stop at the period, splitting the name
# and leaving the surname in the clear.
#
# An internal period is part of the token only when a letter or digit follows it
# ("12.03.1948"). A period followed by a space ends the value — which is what
# keeps "Smith. What is the evidence..." from being swallowed.
_WORD = r"\+?[A-Za-z0-9](?:[A-Za-z0-9'/\-]|\.(?=[A-Za-z0-9]))*"
_CAP_WORD = r"[A-Z0-9](?:[A-Za-z0-9'/\-]|\.(?=[A-Za-z0-9]))*"

# The initial is matched before the prose check, not after it: "A." in
# "Prof A. Raman" is a single capital, and a case-insensitive test against the
# function words reads it as the article "a" and rejects it — truncating the
# value to "Prof" and leaving the surname in the clear.
_VALUE_TOKEN = rf"(?:[A-Z]\.|{_NOT_A_LABEL}{_NOT_PROSE}{_WORD})"
# After a comma the value may continue only into a capitalised or numeric token
# — "Flat 4, 22 Kingsway, London" is one address, but "12/03/1948, presenting
# with progressive aphasia" is a date followed by the clinician's sentence.
_CAP_TOKEN = rf"{_NOT_A_LABEL}(?:[A-Z]\.|{_CAP_WORD})"

_VALUE = (
    rf"{_VALUE_TOKEN}"
    rf"(?:[ \t]*,[ \t]+{_CAP_TOKEN}|[ \t]+{_VALUE_TOKEN}){{0,7}}"
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

# --- Names in prose with no field label at all -----------------------------
#
# A bare name in arbitrary prose is not regex-detectable and is recorded as a
# residual. A name introduced by a relationship or a clinical encounter is, and
# those are the shapes a referral letter actually uses.
_RELATION = (
    r"(?:wife|husband|spouse|partner|son|daughter|mother|father|brother|sister"
    r"|parents?|carer|caregiver|guardian|next\s+of\s+kin|neighbour|neighbor|friend)"
)
_ENCOUNTER = (
    r"(?:reviewed|saw|assessed|examined|met|admitted|discharged|referred"
    r"|consulted|treated|followed\s+up)"
)
# Capitalised words that are not people. Eponyms are exactly the collision set
# for a name rule in a neurology letter — without this, "I reviewed Alzheimer's
# Disease guidance" redacts the disease.
_NOT_A_PERSON = (
    r"(?:Alzheimer|Parkinson|Lewy|Braak|Scheltens|Montreal|Mini|Creutzfeldt"
    r"|Huntington|Wernicke|Korsakoff|Broca|Binswanger|Pick|Down|Charcot"
    r"|Hachinski|Rankin|Glasgow|Barthel|Addenbrooke|Boston|Trail|Stroop"
    r"|Hospital|Clinic|Surgery|Practice|Medical|NHS|MRI|CT|PET|EEG|CSF)"
)
_NAME_WORD = rf"(?!{_NOT_A_PERSON}\b)[A-Z][a-z]+(?:-[A-Z][a-z]+)*"
# Two name-words minimum: one capitalised word after "saw" is far more often a
# place, an instrument or a drug than a patient.
_PERSON_NAME = rf"{_NAME_WORD}(?:[ \t]+[A-Z]\.)?(?:[ \t]+{_NAME_WORD})+"

_PATTERNS: list[tuple[str, str]] = [
    # --- Labelled fields (most reliable: the label tells us what the value is) ---
    #
    # The label itself is captured and re-emitted, so the model still sees which
    # field was present — "Patient Name: [NAME]" carries structure that a bare
    # "[NAME]" does not. `_VALUE` requires at least one token, so a label with
    # nothing after it produces no placeholder at all.
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

    # --- Untitled names in prose, introduced by a relationship or an encounter ---
    (rf"\b((?i:{_RELATION}))[ \t]+{_PERSON_NAME}\b", r"\1 [NAME]"),
    (rf"\b((?i:{_ENCOUNTER}))[ \t]+{_PERSON_NAME}\b", r"\1 [NAME]"),

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

_COMPILED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern), replacement) for pattern, replacement in _PATTERNS
]

# A line holding a label and nothing else — the left column of a letterhead.
_LABEL_ONLY_LINE = re.compile(rf"^[ \t]*(?i:{_ANY_LABEL})[ \t]*:[ \t]*$")
# A line that is entirely a plausible field value — the right column. Requiring
# the WHOLE line to parse as a value is what keeps prose out: "The patient was
# admitted with acute confusion" stops at "was" and so never matches to `$`.
_VALUE_ONLY_LINE = re.compile(rf"^[ \t]*{_VALUE}[ \t]*$")


def _reflow_orphan_labels(text: str) -> str:
    """Rejoin `label \\n value` pairs produced by two-column PDF extraction.

    pypdf reads a letterhead column by column, so a label can be separated from
    its value by a newline, and in a wide layout every label can arrive before
    any value. Both are handled by pairing a run of label-only lines with the
    run of value-only lines immediately following it, in order.

    A label that cannot be paired is left exactly as it is. It then matches no
    labelled rule, and so produces no placeholder — which is the point: a
    placeholder must never claim a removal that did not happen.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        run_end = i
        while run_end < len(lines) and _LABEL_ONLY_LINE.match(lines[run_end]):
            run_end += 1
        labels = run_end - i
        if not labels:
            out.append(lines[i])
            i += 1
            continue

        paired = 0
        while (
            paired < labels
            and run_end + paired < len(lines)
            and _VALUE_ONLY_LINE.match(lines[run_end + paired])
        ):
            paired += 1

        for offset in range(labels):
            label = lines[i + offset].rstrip()
            if offset < paired:
                out.append(f"{label} {lines[run_end + offset].strip()}")
            else:
                out.append(lines[i + offset])
        i = run_end + paired
    return "\n".join(out)


def scrub_pii(text: str) -> str:
    """Replace personal identifiers with typed placeholders.

    Placeholders keep the field's *shape* so the model can still tell that a
    patient name or record number was present without learning whose.

    Normalisation and reflow happen here rather than in the callers:
    `clinical_agent` scrubs raw pypdf output directly, and a guarantee a caller
    has to remember to establish is not a guarantee.
    """
    if not text:
        return text
    text = unicodedata.normalize("NFKC", text)
    text = _reflow_orphan_labels(text)
    for pattern, replacement in _COMPILED:
        text = pattern.sub(replacement, text)
    return text
