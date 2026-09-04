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
import unicodedata

from .fields import ANY_LABEL, IMPROVISED_LABEL, FieldType, type_of
from .lexicon import (
    GIVEN_NAMES,
    NOT_A_NAME,
    has_clinical_head,
    NOT_A_NAME_RE,
    PARTICLES,
    SUFFIXES,
    TITLES,
)
from .text import INVISIBLE

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
#: A name word, matched by Unicode letter class rather than `[A-Za-z]`.
#:
#: The ASCII class was a silent, total failure for anyone outside Anglophone
#: naming: `Patient Name: José Muñoz` produced `Patient Name: [NAME]é Muñoz` —
#: a placeholder ASSERTING de-identification with the name still legible beside
#: it — and Jürgen Müller, Ольга Петрова and 張偉玲 were not touched at all.
#: `[^\W\d_]` is every Unicode letter and no digit or underscore.
_LETTER = r"[^\W\d_]"
# Written as `letters (punct letters)*` rather than `letter (punct letters | letter)*`:
# the second form lets two alternatives match the same character, which is the
# shape that backtracks catastrophically. This one is unambiguous.
#
# The invisible characters are inside the token, not between tokens: a soft
# hyphen or zero-width space survives a real reportlab -> pypdf round trip, and
# splitting `Nkem<U+00AD>dirim` into two tokens produced `[NAME]<U+00AD>dirim` —
# a placeholder asserting de-identification beside the still-legible surname.
_NAME_TOKEN_RE = re.compile(
    rf"{_LETTER}+(?:[’'\-{INVISIBLE}]{_LETTER}+)*"
)
#: Whitespace, optionally after a comma: `MACDONALD, Fiona` is one name.
_NAME_GAP_RE = re.compile(r"[ \t]*,?[ \t]+")

#: Kept for the address grammar, which still needs a capitalised-word fragment.
_NAME_WORD = rf"{_NOT_LABEL}{_NOT_CLINICAL}[A-Z](?:['’\-]?[A-Za-z])+"

#: At most eight tokens. The cap is a backstop for a word the lexicon does not
#: know; it is NOT what protects the clinician's question — the lexicon and the
#: case rule are. A cap that is too tight is worse than none, because a name
#: LONGER than the cap failed the whole-line match entirely and then leaked
#: untouched: `Maria Del Carmen Gonzalez Rodriguez Perez` is six.
_MAX_NAME_TOKENS = 8

_FOLD = str.maketrans({"’": "'"})


def _fold(word: str) -> str:
    """Lower-case and strip accents, for gazetteer and lexicon lookup.

    `Jürgen` must find `jurgen` and `José` must find `jose`, or the positive
    signal misses exactly the names the ASCII matcher already missed.
    """
    stripped = "".join(ch for ch in word if ch not in INVISIBLE)
    plain = unicodedata.normalize("NFKD", stripped.translate(_FOLD).lower())
    return "".join(ch for ch in plain if not unicodedata.combining(ch))


def _parts(word: str) -> list[str]:
    """A token and its hyphen/apostrophe components: `Trail-Making` is clinical
    because `trail` is, and `Aldred-Whitmore` is not because neither part is."""
    folded = _fold(word)
    return [folded, *re.split(r"[-']", folded)]


def _token_kind(token: str, *, any_case: bool) -> str | None:
    """What role this token can play in a person's name, or None if it cannot."""
    folded = _fold(token)
    if folded in TITLES:
        return "title"
    if folded in PARTICLES:
        return "particle"
    if folded in SUFFIXES:
        return "suffix"
    if any(part in NOT_A_NAME for part in _parts(token) if part):
        return None
    if type_of(token) is not None:  # the next field's label ends this value
        return None
    if len(token) < 2:
        return None
    # A lower-case token is prose, except on a line typed without any capitals.
    if token.islower() and not any_case:
        return None
    return "word"


def _has_non_latin(token: str) -> bool:
    """Cyrillic, Greek, Han, Arabic, Devanagari and beyond.

    Clinical English does not contain these, so a token that does is a name in
    a script the gazetteer cannot be expected to enumerate.
    """
    return any(ord(character) > 0x036F for character in token)


def name_tokens(
    line: str, start: int, *, any_case: bool = False
) -> list[tuple[str, int, int]]:
    """The (kind, start, end) of each token of a person value from `start`."""
    tokens: list[tuple[str, int, int]] = []
    position = start
    while len(tokens) < _MAX_NAME_TOKENS:
        match = _NAME_TOKEN_RE.match(line, position)
        if match is None:
            break
        token = match.group()
        end = match.end()
        # `A.` and `Dr.` keep their period; `Smith.` does not — that period ends
        # a sentence, and swallowing it would delete punctuation the clinician
        # wrote and let the value run into the next one.
        followed_by_dot = end < len(line) and line[end] == "."
        if len(token) == 1 and followed_by_dot:
            # An initial is settled by its period, BEFORE any word test. A bare
            # capital fails the two-character minimum and reads case-insensitively
            # as the article "a", so testing it as a word truncated the value:
            # `Consultant: Prof A. Raman` stopped at `Prof` and left the surname
            # in the clear.
            kind = "initial"
            end += 1
        else:
            word_kind = _token_kind(token, any_case=any_case)
            if word_kind is None:
                break
            kind = word_kind
            if followed_by_dot and kind in ("title", "suffix"):
                end += 1
        tokens.append((kind, match.start(), end))
        position = end
        if followed_by_dot and end == match.end():
            break  # a sentence ended here
        gap = _NAME_GAP_RE.match(line, position)
        if gap is None:
            break
        position = gap.end()
    # A title or a particle may sit INSIDE a name but may not END one, so
    # `Consultant: Dr` names nobody and emits no placeholder.
    while tokens and tokens[-1][0] in ("title", "particle"):
        tokens.pop()
    return tokens


def is_clinical_phrase(line: str, tokens: list[tuple[str, int, int]]) -> bool:
    """Whether this span is positively a CLINICAL phrase rather than a name.

    This replaces `person_evidence` as the decision for cross-line association,
    and the inversion is the whole point. Requiring positive proof of personhood
    — a given name in a gazetteer — leaked 49.3% of names over 186,624 generated
    documents, because `Gordon Whitfield` and `Esme Fairhurst` are perfectly
    ordinary names that no gazetteer of any size reliably contains. That was a
    REGRESSION introduced by the fix for the destroy direction: the two halves
    of the invariant traded places for the fifth time.

    Asking the opposite question does not have that failure mode. A clinical
    noun phrase is recognisable from its own vocabulary and morphology:

      - any token in the clinical lexicon (`Sinus`, `Frontal`, `Geriatric`);
      - a clinical head noun (`... Syndrome`, `... Test`, `... Index`), which
        generalises to phrases nobody enumerated;
      - a morphological ending (`-itis`, `-osis`, `-aemia`, `-pathy`).

    A name that matches none of those is redacted, which is the safe direction:
    an unrecognised clinical phrase becomes a placeholder that still tells the
    model a field was present, while an unrecognised name would otherwise reach
    a third-party provider in the clear.
    """
    words = [line[start:end] for _, start, end in tokens]
    if any(part in NOT_A_NAME for word in words for part in _parts(word) if part):
        return True
    return has_clinical_head([_fold(word) for word in words])


def person_evidence(line: str, tokens: list[tuple[str, int, int]]) -> bool:
    """Positive evidence that this span is a PERSON and not a clinical phrase.

    Shape cannot separate `Harold Nkemdirim` from `Peptic Ulcer Bleeding`, and a
    stop-list of clinical words loses: the adversarial review destroyed 46 of 60
    real clinical phrases that way, including three cardiac contraindications to
    the drug this system advises on. So the burden is inverted — a bare line is
    a person only on evidence, and the absence of evidence means "leave it
    alone", which is the safe direction for clinical content.
    """
    for kind, start, end in tokens:
        if kind in ("title", "particle", "initial", "suffix"):
            return True
        word = line[start:end]
        if _fold(word) in GIVEN_NAMES:
            return True
        if _has_non_latin(word):
            return True
    return False

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
# UK numbers are 10 or 11 digits after the leading 0, but the AREA CODE is 2, 3,
# 4 or 5 digits: London is 020, Belfast 028, Cardiff 029. Requiring 3-4 digits
# after the 0 missed every one of them, so `Telephone: 020 7946 0958` leaked even
# with its label attached. Written as "leading 0, then 9 or 10 more digits in any
# grouping" instead of guessing the split.
PHONE = (
    r"(?:\+\d{1,3}[ \t]?\(?0?\)?[ \t]?(?:\d[ \t]?){9,11}"
    r"|\b0(?:[ \t]?\d){9,10}"
    r"|\(?\d{3}\)?[-. \t]\d{3}[-. \t]\d{4})"
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
#: An address ENDS at its street type. Without this the generic run kept going
#: and swallowed whatever followed on the line: `Home Address: 148 Beckett Road
#: Waterlow Score` became `[ADDRESS] Score`, deleting the name of the pressure-
#: ulcer risk assessment. The street-type form is tried FIRST so it wins; the
#: open-ended run remains for an address that has no street type at all.
_STREET_TYPE = (
    r"(?i:street|road|avenue|lane|drive|boulevard|court|close|way|place|terrace"
    r"|crescent|gardens?|square|highway|parkway|row|walk|rise|view|park|hill"
    r"|green|grove|mews|parade|vale|wharf|st|rd|ave|blvd|ln|hwy|pkwy)"
)
ADDRESS = (
    rf"(?:{_HOUSE}\d{{1,5}}[A-Za-z]?(?:{_JOIN}{_ADDRESS_WORD}){{0,4}}"
    rf"{_JOIN}{_STREET_TYPE}(?![A-Za-z])"
    rf"|{_HOUSE}\d{{1,5}}[A-Za-z]?(?:{_JOIN}{_ADDRESS_WORD}){{1,5}})"
)

#: NAME is deliberately ABSENT: it is matched by `name_tokens`, not by a regex.
#: A regex cannot express "a Unicode letter run that is not lower-case unless
#: the whole line is", cannot be asked whether a span carries positive person
#: evidence, and failed open when a name exceeded its token cap.
_GRAMMARS: dict[FieldType, str] = {
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


# A NAME VALUE IS NEVER TRUNCATED TO A PREFIX.
#
# Bounding an embedded name at two words looked like a way to stop it eating the
# clinician's sentence. It was worse than the disease: a three-part name lost its
# last part and the SURNAME stayed legible beside the `[NAME]` that claimed to
# have removed it — 300/300 measured, and proven on the wire at the Groq egress
# (`PHI REACHING THE PROVIDER: ['Ogunlana']`). A placeholder covering PART of an
# identifier is the worst outcome available: the name leaks AND the output
# asserts it did not.
#
# So the run ends where a CLINICAL signal says it ends — a lexicon word, a
# clinical head noun, a digit, a lower-case word, a sentence terminator, the
# next label — or it is taken whole. Where no signal is found the value is
# genuinely ambiguous, and under the agreed policy that resolves by REDACTING on
# the chat path (the clinician sees the result) and by QUARANTINING on the upload
# path (see `ambiguity`), never by guessing a boundary.


def value_at(
    field_type: FieldType, line: str, position: int, *, any_case: bool = False
) -> tuple[int, int] | None:
    """The span of a value of `field_type` starting exactly at `position`."""
    if field_type is FieldType.NAME:
        tokens = name_tokens(line, position, any_case=any_case)
        if not tokens:
            return None
        # Greedy to the end of the line is right for a header — a six-token name
        # is one field. Greedy in the MIDDLE of a line is wrong: `Name: Mary
        # Okonkwo Sick Sinus Syndrome` swallowed `Sick` and left `Sinus
        # Syndrome`, and 77 of 87 clinical phrases were mutilated this way.
        # Reaching the end of the line is what tells the two apart, and it needs
        # no vocabulary at all.
        return tokens[0][1], tokens[-1][2]
    pattern = _AT.get(field_type)
    if pattern is None:
        return None
    match = pattern.match(line, position)
    if match is None or match.end() == match.start():
        return None
    return match.start(), match.end()


def longest_value_at(
    field_type: FieldType, line: str, position: int, *, any_case: bool = False
) -> tuple[int, int] | None:
    """`value_at`, extended to the longest span any MORE SPECIFIC type accepts.

    `MRN: 943 476 5919` produced `MRN: [MRN] 476 5919` — the generic record-code
    grammar matched only the first group, and the partial redaction then
    destroyed the very shape the free-text NHS rule needed to catch the rest. A
    placeholder that covers part of an identifier is the worst of both
    directions: the number still leaks and the output claims it did not.
    """
    best = value_at(field_type, line, position, any_case=any_case)
    for candidate in _SPECIFICITY:
        if _equivalent(candidate, field_type):
            break
        span = value_at(candidate, line, position, any_case=any_case)
        if span is not None and (best is None or span[1] > best[1]):
            best = span
    return best


#: An editorial annotation appended to a column cell: `[SIC]`, `[NB]`, `[?]`.
#: Left in the content span it made the cell fail every whole-line value test,
#: so a name carrying one was never associated with its label and leaked.
_ANNOTATION = re.compile(r"[ \t]*\[[A-Za-z? ]{1,12}\][ \t]*\Z")


def _content_span(line: str) -> tuple[int, int] | None:
    """The line minus surrounding whitespace, invisibles and any annotation."""
    stripped = line.strip(" \t\r" + INVISIBLE)
    if not stripped:
        return None
    start = line.index(stripped)
    end = start + len(stripped)
    annotation = _ANNOTATION.search(line, start, end)
    if annotation is not None and annotation.start() > start:
        end = annotation.start()
    return start, end


def whole_value(field_type: FieldType, line: str) -> tuple[int, int] | None:
    """The span of `line` when the entire line is one value of `field_type`."""
    content = _content_span(line)
    if content is None:
        return None
    start, end = content
    if field_type is FieldType.NAME:
        tokens = name_tokens(line, start)
        if not tokens or tokens[-1][2] != end or tokens[0][1] != start:
            return None
        return content
    pattern = _WHOLE.get(field_type)
    if pattern is None or pattern.fullmatch(line[start:end]) is None:
        return None
    return content


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
