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
from collections.abc import Callable

from . import values
from .fields import is_ambiguous_person_label
from .lexicon import NOT_A_NAME_RE, has_clinical_head
from .text import map_lines

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

def _nhs_number(match: re.Match[str]) -> str:
    """Redact ten digits only when they check out as an NHS number.

    Modulus 11: weight the first nine digits by 10..2, sum, take 11 minus the
    remainder mod 11; 11 means 0 and 10 means the number is invalid. A column of
    lab values passes the SHAPE and fails this, which is the whole point.
    """
    digits = [int(character) for character in match.group() if character.isdigit()]
    if len(digits) != 10:
        return match.group()
    total = sum(digit * weight for digit, weight in zip(digits[:9], range(10, 1, -1), strict=True))
    check = (11 - total % 11) % 11
    if check == 10 or check != digits[9]:
        return match.group()
    return "[NHS]"


def _cued_name(match: re.Match[str]) -> str:
    """Redact a name introduced by a relationship or an encounter — but only
    when the cue is unambiguous, or the name itself looks like one.

    `Carer Strain Index` is a validated instrument. `carer` is a relation cue
    and `Strain Index` is two capitalised words, so this rule rewrote it as
    `Carer [NAME]` and deleted the instrument's name from 276 generated
    documents. `Carer`, `Partner`, `Mother` and `Guardian` are ordinary clinical
    words; `wife`, `daughter`, `reviewed` and `saw` are not, and those keep
    working on shape alone so the prose coverage is not narrowed.
    """
    cue, gap = match.group(1), match.group(2)
    name = match.group()[len(cue) + len(gap) :]
    tokens = values.name_tokens(name, 0)
    if not tokens:
        return match.group()

    # The clinical head may sit just OUTSIDE the match, because `_NAME_WORD`
    # excludes clinical vocabulary and so stops before it: in `Next of Kin
    # Rockwood Frailty Scale` the rule matched `Rockwood Frailty` and `Scale`
    # was already excluded. Looking one token past the match is what tells a
    # relative's name from the instrument named after its author.
    #
    # The following word must be CAPITALISED to count. An instrument's name is
    # a title-case noun phrase — `Rockwood Frailty Scale` — whereas `Her
    # daughter Sarah Okonjo reports increasing confusion` continues into a
    # lower-case VERB that happens to share a stem with the head noun `report`.
    # Without the case test this rule stopped redacting real relatives' names.
    tail = match.string[match.end() :].lstrip(" \t")
    following = re.match(r"[^\W\d_]+", tail)
    if (
        following is not None
        and following.group()[:1].isupper()
        and has_clinical_head([following.group().lower()])
    ):
        return match.group()
    if values.is_clinical_phrase(name, tokens):
        return match.group()
    if is_ambiguous_person_label(cue) and not values.person_evidence(name, tokens):
        return match.group()
    return f"{cue}{gap}[NAME]"


_RULES: list[tuple[str, str | Callable[[re.Match[str]], str]]] = [
    # --- structured national identifiers ---
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    (r"\b\d{2}[A-Z]\d{6}[A-Z]\b", "[MEDICARE]"),
    # UK National Insurance number, closed up or spaced. Matched by SHAPE, not
    # validated against the real prefix rules: `QQ123456C` is HMRC's own example
    # and would fail a valid-prefix check, yet it is exactly what turns up in
    # referral letters. Redacting a near-miss costs nothing; missing a real one
    # is a disclosure.
    (rf"\b{values.NI_NUMBER}", "[NI_NUMBER]"),
    # An NHS number is ten digits with a modulus-11 check digit. Validating it
    # is what separates a real one from a column of lab values: `138 102 2024`
    # became `[NHS]`, which deletes three clinical numbers and asserts a removal
    # that never happened. De-identification is not validation in general — a
    # near-miss NI number is still redacted, because missing a real one is a
    # disclosure — but here the shape is SO common in clinical data that the
    # false-positive cost is the larger one.
    (r"\b(?:\d{3}[ \t]){2}\d{4}\b", _nhs_number),
    # The closed-up form keeps its unconditional rule. Ten CONTIGUOUS digits is
    # not a clinical quantity — the spaced 3-3-4 form is the one that collides
    # with a column of lab values, and it is the only one gated on the checksum.
    (r"\b\d{10}\b", "[NHS_ID]"),
    # Nine bare digits is not a clinical quantity — doses, scores and lab values
    # are orders of magnitude shorter — so the false-positive cost is low.
    (r"\b\d{9}\b", "[ID_NUMBER]"),
    (r"\b[A-Z]{2,4}-\d{2,4}-\d{4,8}\b", "[ACCOUNT]"),

    # --- further HIPAA Safe Harbor categories ---
    #
    # Eight of the eighteen categories had no rule at all, and 13 of 15 probes
    # leaked on the plainest `Label: value` layout — the one shape every earlier
    # wave DID handle. Coverage of layouts had been mistaken for coverage of
    # identifiers.
    #
    # Each of these is a well-specified format matched by its own shape, not a
    # guess: a fixed list of FORMATS is a different thing from a fixed list of
    # layouts, because a format is defined by the body that issues it.
    (r"\b\d{3}[ \t]\d{2}[ \t]\d{4}\b", "[SSN]"),                    # 078 05 1120
    (r"\b\d[A-Z]{2}\d[-. ]?[A-Z]{2}\d[-. ]?[A-Z]{2}\d{2}\b", "[MEDICARE]"),  # MBI
    (r"\b[A-Z]{2,3}\d{6,9}\b(?![A-Za-z0-9])", "[PASSPORT]"),        # GBR123456789
    (r"\b[A-Z]{5}\d{6}[A-Z0-9]{5}\b", "[DRIVING_LICENCE]"),         # DVLA
    (r"\b[A-Z]{2}\d{2}[ \t]?(?:[A-Z0-9]{4}[ \t]?){2,7}[A-Z0-9]{1,4}\b", "[IBAN]"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP_ADDRESS]"),
    (r"\b(?:SN|S/N|Serial)[ \t]*[-:#]?[ \t]*[A-Z0-9][A-Z0-9\-]{4,}\b", "[DEVICE_ID]"),
    (r"\bhttps?://\S+", "[URL]"),

    # --- dates ---
    (r"\b\d{4}-\d{2}-\d{2}\b", "[DOB]"),
    (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DOB]"),
    (rf"\b\d{{1,2}}(?:st|nd|rd|th)?[ \t]+{values._MONTHS}\.?,?[ \t]+\d{{4}}\b", "[DOB]"),
    (rf"\b{values._MONTHS}\.?[ \t]+\d{{1,2}}(?:st|nd|rd|th)?,?[ \t]+\d{{4}}\b", "[DOB]"),

    # --- contact details ---
    #
    # `[ \t]` throughout, never `\s`. `\s` matches a newline, and these rules
    # used to run over the whole document at once, so a column of lab values
    # (`138\n102\n2024`) collapsed into one `[PHONE]` and two lines of clinical
    # data were DELETED. `redact_by_shape` is now applied line by line as well,
    # so this is belt and braces rather than the only guard.
    # The lookbehind is load-bearing for PERFORMANCE, not for correctness.
    # Without it the engine restarts the greedy local-part scan at every
    # character of every token and only then discovers there is no `@`, which is
    # quadratic: 8.2 seconds on one 60 KB line, and `_extract_pdf_text` puts no
    # size cap on an uploaded report. Anchoring the start to a token boundary
    # makes each position O(1) to reject.
    (
        r"(?<![a-zA-Z0-9_.+-])[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+",
        "[EMAIL]",
    ),
    # Any UK area-code length: 020 (London), 028 (Belfast), 029 (Cardiff) all
    # leaked because the rule demanded 3-4 digits after the leading zero.
    (r"\+44[ \t]?\(?0?\)?[ \t]?(?:\d[ \t]?){9,10}\d\b", "[PHONE]"),
    (r"\b0(?:[ \t]?\d){9,10}\b", "[PHONE]"),
    # Space-separated North American form. Removing it to stop `138 102 2024`
    # becoming `[PHONE]` traded a false placeholder for a RAW DISCLOSURE: real
    # US numbers in prose then leaked, 186/200. Under the agreed policy an
    # ambiguous shape is redacted, not left; the lab-value collision is bounded
    # by requiring a plausible NANP area and exchange code (neither starts 0 or 1).
    (r"\b[2-9]\d{2}[ \t][2-9]\d{2}[ \t]\d{4}\b", "[PHONE]"),
    # A North American number needs a real telephone separator — parentheses, a
    # dash, a dot, or a +1 country code. SPACE-separated 3-3-4 is not accepted,
    # because that is also what a column of lab values looks like: `138 102 2024`
    # became `[PHONE]`, deleting three clinical numbers and asserting a removal
    # that never happened. UK numbers are unaffected; they start 0 or +44 and
    # have their own rules above.
    (r"\+1[-. \t]?\(?\d{3}\)?[-. \t]?\d{3}[-. \t]?\d{4}\b", "[PHONE]"),
    (r"\(\d{3}\)[-. \t]?\d{3}[-. \t]?\d{4}\b", "[PHONE]"),
    (r"\b\d{3}[-.]\d{3}[-.]\d{4}\b", "[PHONE]"),
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
    (rf"\b((?i:{_RELATION}))([ \t]+){_PERSON_NAME}\b", _cued_name),
    (rf"\b((?i:{_ENCOUNTER}))([ \t]+){_PERSON_NAME}\b", _cued_name),

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

_COMPILED: list[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]]] = [
    (re.compile(pattern), replacement) for pattern, replacement in _RULES
]


def _redact_line(line: str) -> str:
    for pattern, replacement in _COMPILED:
        line = pattern.sub(replacement, line)
    return line


def redact_by_shape(text: str) -> str:
    """Replace identifiers that no field label introduces.

    Applied LINE BY LINE. Run over the whole document these rules could and did
    join lines — `\\s` matches `\\n`, so a column of numbers became one
    `[PHONE]` and the lines between were deleted. `layout` guaranteed it never
    joined a line; this pass did not, and the module docstring claimed the
    guarantee for the whole scrubber. Going through `map_lines` makes the claim
    true by construction instead of by inspection of each rule.
    """
    return map_lines(text, _redact_line)
