r"""What a value of a given type looks like — positively, and bounded.

The design this replaces had ONE value grammar for every label, defined as "up
to eight tokens that are not function words". That is a definition of a value by
what it is not, and it failed in both directions at once:

  - `GP: Dr Fairbanks Amyloid PET indication threshold question` -> `GP: [NAME]`
    absorbed the clinician's whole question, because none of those words is a
    function word;
  - a tabular layout with no colon matched nothing at all, because the grammar
    was reachable only through `label[ \t]*:`.

Here each type answers for itself. A phone value is digits in a phone's shape; a
person value is name-shaped words that are not clinical vocabulary; a record
number is one token containing a digit. Widening how a label is *found* (see
`layout`) therefore cannot widen what a value may *swallow*, which is the
coupling that made A2's fix reopen A1 last wave.

Every grammar is bounded by construction:

  - tokens are joined by `[ \t]+`, so no value can cross a newline;
  - no grammar admits the empty string, so a label with nothing beside it can
    never produce a placeholder;
  - a value stops at the next known label, at a sentence terminator, at the
    first token carrying a digit where the type forbids one, and at the first
    word in the clinical lexicon.
"""
from __future__ import annotations

import re

from .fields import ANY_LABEL, IMPROVISED_LABEL, FieldType
from .lexicon import NOT_A_NAME_RE, PARTICLES_RE, SUFFIXES_RE, TITLES_RE

# --- token-level building blocks -------------------------------------------

#: Between two tokens of one value: whitespace, optionally after a comma.
#: `[ \t]` and not `\s`, so line scope is structural rather than a flag.
_JOIN = r"[ \t]*,?[ \t]+"

# A value token may not be the next field's label. Both forms matter: a known
# label (`NHS Number:`) and any bare word acting as one (`Weight:`).
_NOT_LABEL = rf"(?!(?i:{ANY_LABEL})(?![A-Za-z0-9]))(?!{IMPROVISED_LABEL})"
# ...and may not be clinical vocabulary or capitalised common English. The
# lookahead ends at the first non-letter, so `Alzheimer's` is rejected by
# `alzheimer` while `Ali` is not rejected by the particle `al`.
_NOT_CLINICAL = rf"(?!(?i:{NOT_A_NAME_RE})(?![A-Za-z]))"

#: `A.` in `John A. Smith`. Matched before the clinical check, because a single
#: capital read case-insensitively is the article "a" and would truncate the
#: value, leaving the surname in the clear.
_INITIAL = r"[A-Z]\."
#: Titlecase, ALL-CAPS or internally hyphenated/apostrophised. Two characters
#: minimum, so a bare capital is only ever an initial.
_NAME_SHAPE = r"[A-Z](?:['’\-]?[A-Za-z])+"
#: The same shape typed without capitals. Admitted only on a line that carries
#: no capital letter anywhere — see `uncapitalised` below.
_NAME_SHAPE_ANY_CASE = r"[A-Za-z](?:['’\-]?[A-Za-z])+"
_TITLE = rf"(?i:{TITLES_RE})\.?"
_PARTICLE = rf"(?i:{PARTICLES_RE})(?![A-Za-z])"
_SUFFIX = rf"(?i:{SUFFIXES_RE})\.?(?![A-Za-z])"


def _name(shape: str) -> str:
    """A person value: at most five tokens — title, forename, initial, surname,
    suffix.

    A title or a particle may sit INSIDE a name but may not END one, so
    `Consultant: Dr` names nobody and emits no placeholder.
    """
    word = rf"{_NOT_LABEL}{_NOT_CLINICAL}{shape}"
    inner = rf"(?:{_INITIAL}|{_TITLE}|{_PARTICLE}|{_SUFFIX}|{word})"
    final = rf"(?:{_INITIAL}|{_SUFFIX}|{word})"
    return rf"(?:{inner}{_JOIN}){{0,4}}{final}"


_NAME_WORD = rf"{_NOT_LABEL}{_NOT_CLINICAL}{_NAME_SHAPE}"
NAME = _name(_NAME_SHAPE)
NAME_ANY_CASE = _name(_NAME_SHAPE_ANY_CASE)

# --- dates ------------------------------------------------------------------
_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_DATE_SHAPE = (
    r"(?:\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?[ \t]+{_MONTHS}\.?,?[ \t]+\d{{2,4}}"
    rf"|{_MONTHS}\.?[ \t]+\d{{1,2}}(?:st|nd|rd|th)?,?[ \t]+\d{{2,4}})"
)
DOB = rf"{_DATE_SHAPE}(?![A-Za-z0-9])"

# --- record numbers ---------------------------------------------------------
#
# One token, and it must contain a digit. Without the digit `NHS Number: [NHS]`
# re-matches on a second pass as label `NHS` plus identifier `Number`, and the
# rule eats its own label.
#
# A record number is the LEAST specific of the identifier shapes — every date
# and every national-insurance number also fits `[A-Za-z0-9/\-]+ containing a
# digit`. Those are excluded here rather than resolved downstream, so that
# `Hospital Number:` can never claim `12/03/1948` and report a date of birth as
# a record number.
_NI_SHAPE = r"[A-Z]{2}[ \t]?(?:\d[ \t]?){6}[A-D]"
_NOT_MORE_SPECIFIC = rf"(?!(?:{_DATE_SHAPE}|{_NI_SHAPE})(?![A-Za-z0-9]))"
_CODE = (
    rf"{_NOT_LABEL}{_NOT_MORE_SPECIFIC}(?=[A-Za-z0-9/\-]*\d)"
    r"[A-Za-z0-9][A-Za-z0-9/\-]{2,23}(?![A-Za-z0-9/\-])"
)
MRN = _CODE
ACCOUNT = _CODE
# An NHS number has a shape of its own — ten digits, usually spaced 3-3-4. It is
# deliberately NOT the generic code shape: when it was, every record number also
# parsed as an NHS number and the more specific type could never win.
NHS = r"(?:\d{3}[ \t\-]\d{3}[ \t\-]\d{4}|\d{10}(?!\d))(?![A-Za-z0-9])"
NI_NUMBER = rf"{_NI_SHAPE}(?![A-Za-z0-9])"

# --- contact ----------------------------------------------------------------
PHONE = (
    r"(?:\+\d{1,3}[ \t]?\(?0?\)?[ \t]?\d{2,5}[ \t]?\d{3}[ \t]?\d{3,4}"
    r"|\b0\d{3,4}[ \t]?\d{3}[ \t]?\d{3,4}"
    r"|\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4})"
    r"(?![A-Za-z0-9])"
)
EMAIL = r"[A-Za-z0-9_.+\-]+@[A-Za-z0-9\-]+\.[A-Za-z0-9.\-]*[A-Za-z0-9]"
POSTCODE = r"[A-Z]{1,2}\d{1,2}[A-Z]?[ \t]?\d[A-Z]{2}(?![A-Za-z0-9])"

# --- addresses --------------------------------------------------------------
#
# Number-first by design. A rule that accepted a leading word would match any
# two capitalised words in a letter; a house number is the one token an address
# reliably starts with.
_HOUSE = r"(?:(?i:flat|apt|apartment|unit|suite)[ \t]*\d{1,4}[A-Za-z]?[, \t]+)?"
_ADDRESS_WORD = rf"(?:{_NAME_WORD}|\d{{1,5}}[A-Za-z]?)"
ADDRESS = rf"{_HOUSE}\d{{1,5}}[A-Za-z]?(?:{_JOIN}{_ADDRESS_WORD}){{1,5}}"

_GRAMMARS: dict[FieldType, str] = {
    FieldType.NAME: NAME,
    FieldType.MRN: MRN,
    FieldType.NHS: NHS,
    FieldType.ACCOUNT: ACCOUNT,
    FieldType.ADDRESS: ADDRESS,
    FieldType.DOB: DOB,
    FieldType.PHONE: PHONE,
    FieldType.EMAIL: EMAIL,
    FieldType.POSTCODE: POSTCODE,
    FieldType.NI_NUMBER: NI_NUMBER,
}

#: Anchored at a position — used for `label: value` on one line.
_AT: dict[FieldType, re.Pattern[str]] = {
    field_type: re.compile(grammar) for field_type, grammar in _GRAMMARS.items()
}
#: The same, for a line typed entirely without capitals.
_AT_ANY_CASE: dict[FieldType, re.Pattern[str]] = {
    **_AT,
    FieldType.NAME: re.compile(NAME_ANY_CASE),
}
#: The whole of a line — used to pair an orphan label with a column cell. The
#: WHOLE line must parse as a value of the label's own type, which is what stops
#: `MRN:` from claiming `Bradycardia 48 bpm untreated`.
_WHOLE: dict[FieldType, re.Pattern[str]] = {
    field_type: re.compile(rf"[ \t]*(?:{grammar})[ \t]*\Z")
    for field_type, grammar in _GRAMMARS.items()
}


def uncapitalised(line: str) -> bool:
    """True for a line typed with no capital letters at all.

    `name: john smith` is a real query and its value must still be removed, but
    admitting lowercase name words everywhere would let `Patient: elderly
    gentleman` become `Patient: [NAME]` and destroy the description. Case is the
    discriminator: a lowercase name only occurs in text that is not
    capitalising anything, and such a line has no capitalised prose to lose.
    """
    return not any(character.isupper() for character in line)


def value_at(
    field_type: FieldType, line: str, position: int, *, any_case: bool = False
) -> tuple[int, int] | None:
    """The span of a value of `field_type` starting exactly at `position`."""
    pattern = (_AT_ANY_CASE if any_case else _AT).get(field_type)
    if pattern is None:
        return None
    match = pattern.match(line, position)
    if match is None or match.end() == match.start():
        return None
    return match.start(), match.end()


def whole_value(field_type: FieldType, line: str) -> tuple[int, int] | None:
    """The span of `line` when the entire line is one value of `field_type`."""
    pattern = _WHOLE.get(field_type)
    if pattern is None:
        return None
    match = pattern.fullmatch(line)
    if match is None:
        return None
    stripped = line.strip(" \t")
    if not stripped:
        return None
    start = line.index(stripped)
    return start, start + len(stripped)


# Most specific first. Where two grammars can both accept one string, the more
# specific type owns it: `943 476 5919` is an NHS number that also has a phone's
# shape, and reporting it as `[PHONE]` is a mislabelled placeholder — a (b4)
# failure — even though the digits are removed either way.
_SPECIFICITY: tuple[FieldType, ...] = (
    FieldType.EMAIL,
    FieldType.NI_NUMBER,
    FieldType.DOB,
    FieldType.POSTCODE,
    FieldType.NHS,
    FieldType.PHONE,
    FieldType.ADDRESS,
    FieldType.MRN,
    FieldType.ACCOUNT,
    FieldType.NAME,
)
# MRN and ACCOUNT are the same shape; neither is more specific than the other.
_INTERCHANGEABLE: frozenset[FieldType] = frozenset({FieldType.MRN, FieldType.ACCOUNT})


def _equivalent(left: FieldType, right: FieldType) -> bool:
    return left is right or {left, right} <= _INTERCHANGEABLE


def matches_exclusively(field_type: FieldType, line: str) -> tuple[int, int] | None:
    """`whole_value`, but only when no MORE SPECIFIC type also claims the line.

    Used for association across lines, where the label is not adjacent to its
    value and a shape collision is therefore a genuine mis-association rather
    than a redundant one. An orphan `Telephone` must not adopt the NHS number
    two rows down just because ten digits can be dialled.
    """
    span = whole_value(field_type, line)
    if span is None:
        return None
    for candidate in _SPECIFICITY:
        if _equivalent(candidate, field_type):
            break
        if whole_value(candidate, line) is not None:
            return None
    return span
