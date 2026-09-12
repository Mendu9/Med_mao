"""What field labels exist, and what type of value each one carries.

A label does two jobs, and conflating them is how Wave 8's `[NAME]: 12345678`
happened. A label:

  1. names the TYPE of the value beside it, which is what makes `MRN: RGT/44219/B`
     tractable without guessing at the value's shape;
  2. is a BOUNDARY — no value may absorb one, because the next field's label is
     where the previous field's value ends.

So every label is registered once, here, and both jobs read the same table.
Labels that carry no identifier at all (`Ward`, `Department`) are registered as
`FieldType.STRUCTURAL`: they are boundaries but are never redacted.
"""
from __future__ import annotations

import re
from enum import Enum


class FieldType(Enum):
    """The type of identifier a label's value holds. The value is the
    placeholder emitted, so adding a type adds a placeholder."""

    NAME = "NAME"
    MRN = "MRN"
    NHS = "NHS"
    ACCOUNT = "ACCOUNT"
    ADDRESS = "ADDRESS"
    DOB = "DOB"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    POSTCODE = "POSTCODE"
    NI_NUMBER = "NI_NUMBER"
    #: A real field boundary that identifies nobody. Never redacted.
    STRUCTURAL = "STRUCTURAL"


#: label spelling -> type. Spellings are regex fragments so a label can carry its
#: own optional words (`MRN`, `MRN No.`, `MRN #`) without a second entry.
_LABELS: tuple[tuple[str, FieldType], ...] = (
    # --- people ---
    (r"patient\s+name", FieldType.NAME),
    (r"next\s+of\s+kin", FieldType.NAME),
    (r"nok", FieldType.NAME),
    (r"referring\s+gp", FieldType.NAME),
    (r"referred\s+by", FieldType.NAME),
    (r"emergency\s+contact", FieldType.NAME),
    (r"gp", FieldType.NAME),
    (r"consultant", FieldType.NAME),
    (r"clinician", FieldType.NAME),
    (r"physician", FieldType.NAME),
    (r"doctor", FieldType.NAME),
    (r"surgeon", FieldType.NAME),
    (r"guardian", FieldType.NAME),
    (r"carer", FieldType.NAME),
    (r"caregiver", FieldType.NAME),
    (r"informant", FieldType.NAME),
    (r"attending", FieldType.NAME),
    (r"mother", FieldType.NAME),
    (r"father", FieldType.NAME),
    (r"spouse", FieldType.NAME),
    (r"partner", FieldType.NAME),
    # The NHS Data Dictionary spellings. Their absence was a CRITICAL: a
    # `Surname:` / `Forename:` header is the canonical UK patient banner, and
    # nothing matched it, so the full name leaked from a layout that was
    # otherwise handled. A fixed list of labels remains a fixed list; what
    # bounds the damage is that an unrecognised label produces no placeholder
    # rather than a wrong one.
    (r"family\s+name", FieldType.NAME),
    (r"given\s+name", FieldType.NAME),
    (r"first\s+name", FieldType.NAME),
    (r"last\s+name", FieldType.NAME),
    (r"middle\s+name", FieldType.NAME),
    (r"maiden\s+name", FieldType.NAME),
    (r"preferred\s+name", FieldType.NAME),
    (r"full\s+name", FieldType.NAME),
    (r"known\s+as", FieldType.NAME),
    (r"surname", FieldType.NAME),
    (r"forename", FieldType.NAME),
    (r"name", FieldType.NAME),
    (r"patient", FieldType.NAME),
    # --- record numbers ---
    (r"mrn(?:\s+(?:number|no\.?|#))?", FieldType.MRN),
    (r"medical\s+record(?:\s+(?:number|no\.?|#))?", FieldType.MRN),
    (r"hospital\s+(?:number|no\.?|#)", FieldType.MRN),
    (r"patient\s+(?:id|identifier|number|no\.?)", FieldType.MRN),
    (r"case\s+(?:number|no\.?|#)", FieldType.MRN),
    (r"chart\s+(?:number|no\.?|#)", FieldType.MRN),
    (r"episode\s+(?:number|no\.?|#)", FieldType.MRN),
    (r"nhs\s+(?:number|no\.?|#)", FieldType.NHS),
    (r"nhs", FieldType.NHS),
    (r"ni\s+(?:number|no\.?|#)", FieldType.NI_NUMBER),
    (r"national\s+insurance(?:\s+(?:number|no\.?|#))?", FieldType.NI_NUMBER),
    (r"account(?:\s+(?:number|no\.?|#))?", FieldType.ACCOUNT),
    (r"invoice(?:\s+(?:number|no\.?|#))?", FieldType.ACCOUNT),
    # --- contact ---
    (r"home\s+address", FieldType.ADDRESS),
    (r"address", FieldType.ADDRESS),
    (r"residence", FieldType.ADDRESS),
    (r"addr", FieldType.ADDRESS),
    (r"post\s*code", FieldType.POSTCODE),
    (r"zip(?:\s+code)?", FieldType.POSTCODE),
    (r"e-?mail", FieldType.EMAIL),
    (r"contact\s+(?:number|no\.?)", FieldType.PHONE),
    (r"telephone", FieldType.PHONE),
    (r"mobile", FieldType.PHONE),
    (r"phone", FieldType.PHONE),
    (r"tel", FieldType.PHONE),
    (r"mob", FieldType.PHONE),
    (r"cell", FieldType.PHONE),
    # --- dates ---
    (r"date\s+of\s+birth", FieldType.DOB),
    (r"birth\s+date", FieldType.DOB),
    (r"d\.o\.b\.?", FieldType.DOB),
    (r"dob", FieldType.DOB),
    (r"born", FieldType.DOB),
    # --- boundaries that identify nobody ---
    (r"nhs\s+trust", FieldType.STRUCTURAL),
    (r"practice", FieldType.STRUCTURAL),
    (r"surgery", FieldType.STRUCTURAL),
    (r"clinic", FieldType.STRUCTURAL),
    (r"ward", FieldType.STRUCTURAL),
    (r"bed", FieldType.STRUCTURAL),
    (r"hospital", FieldType.STRUCTURAL),
    (r"department", FieldType.STRUCTURAL),
    (r"dept", FieldType.STRUCTURAL),
)

# Longest spelling first, so `Patient Name` wins over `Patient` and
# `Hospital Number` (a record number) wins over `Hospital` (a boundary).
_BY_LENGTH = sorted(_LABELS, key=lambda pair: len(pair[0]), reverse=True)

#: Matches any known label. Used both to find fields and to stop a value.
ANY_LABEL: str = "(?:" + "|".join(spelling for spelling, _ in _BY_LENGTH) + ")"

#: A bare word acting as a label — `Weight:`, `Height:`. Not in the table, but a
#: value must still stop at one, or `Name: John Smith Weight: 78kg` puts the
#: weight inside the name.
IMPROVISED_LABEL: str = r"[A-Za-z][A-Za-z0-9'\-]*[ \t]*:"

# Person labels that are ALSO ordinary clinical nouns.
#
# `Carer Strain Index` is a validated instrument; `Carer` is a person label; so
# a colon-less match read the instrument's name as the carer's and produced
# `Carer [NAME]`, deleting the score's name from the report. The same holds for
# `Patient` ("Patient reported..."), `Guardian`, `Partner`, `Mother`.
#
# The distinction is not cosmetic: `Surname`, `Next of Kin` and `Patient Name`
# occur in running clinical prose essentially never, so a colon-less match on
# one of those is safe. These need either explicit punctuation after the label
# or positive evidence that what follows is a person.
AMBIGUOUS_PERSON_LABELS: frozenset[str] = frozenset(
    {
        "patient", "carer", "caregiver", "guardian", "partner", "spouse",
        "mother", "father", "doctor", "surgeon", "physician", "clinician",
        "informant", "attending", "gp",
    }
)


def is_ambiguous_person_label(label: str) -> bool:
    """Whether this label spelling is also an ordinary clinical word."""
    return " ".join(label.split()).lower() in AMBIGUOUS_PERSON_LABELS


#: A label is a WORD, so it may not begin inside a token.
#:
#: The alphanumeric boundaries were not enough, because `@` and `.` are
#: token-INTERNAL in the values this module exists to protect. In
#: `Email: harold@nhs.uk` the `nhs` of the domain is preceded by `@` and
#: followed by `.`, so it satisfied both boundaries and was read as an NHS
#: field label — which then clipped the email's own value span at it:
#:
#:     'Email: harold@nhs.uk'  ->  'Email: [EMAIL]nhs.uk'
#:
#: A placeholder standing for half an address, with the domain left beside it,
#: and a removal record saying `EMAIL = "harold@"` — so the run-scoped egress
#: backstop was watching a string that is not what remained. Excluding `@` and
#: `.` from what a label may follow is a statement about TOKENS, not about
#: spellings: no vocabulary is added and none is widened.
_LABEL_RE = re.compile(
    rf"(?<![A-Za-z0-9@.]){ANY_LABEL}(?![A-Za-z0-9])", re.IGNORECASE
)
_TYPE_OF = tuple(
    (re.compile(rf"^{spelling}$", re.IGNORECASE), field_type)
    for spelling, field_type in _BY_LENGTH
)


def type_of(label: str) -> FieldType | None:
    """The type carried by a label spelling, or None if it is not a label."""
    text = " ".join(label.split())
    for pattern, field_type in _TYPE_OF:
        if pattern.match(text):
            return field_type
    return None


#: The placeholder spellings this package emits, upper-cased.
_PLACEHOLDER_NAMES = frozenset(
    field.value for field in FieldType if field is not FieldType.STRUCTURAL
)


def _is_inside_a_placeholder(line: str, start: int, end: int) -> bool:
    """Whether this label match is really the inside of `[NAME]`-style output."""
    return (
        start > 0
        and line[start - 1] == "["
        and end < len(line)
        and line[end] == "]"
        and line[start:end].upper() in _PLACEHOLDER_NAMES
    )


def find_labels(line: str) -> list[tuple[int, int, FieldType]]:
    """Every known label in `line`, as (start, end, type), non-overlapping.

    Overlaps are resolved by taking the longest spelling at each position, which
    is what keeps `Hospital Number` from being read as the boundary `Hospital`
    followed by a stray word.
    """
    found: list[tuple[int, int, FieldType]] = []
    end_of_previous = 0
    for match in _LABEL_RE.finditer(line):
        if match.start() < end_of_previous:
            continue
        if _is_inside_a_placeholder(line, match.start(), match.end()):
            # `[NAME]` is this module's OWN output, not a field label. Reading
            # it as one made every line carrying a placeholder label-bearing,
            # so it stopped being an unclaimed cell — and a caller who simply
            # appended ` [NAME]` to each line of a pasted banner suppressed
            # redaction of the real name AND the record number beside it.
            # All ten emittable placeholders worked as decoys that way.
            continue
        field_type = type_of(match.group())
        if field_type is None:  # pragma: no cover - alternation is exhaustive
            continue
        found.append((match.start(), match.end(), field_type))
        end_of_previous = match.end()
    return found
