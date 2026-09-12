r"""Where in a document each labelled identifier is — and redaction in place.

## Why this replaces `_reflow_orphan_labels`

pypdf reads a two-column letterhead column by column, so a label and its value
land on different lines. Wave 9 handled that by REWRITING the text: a run of
label-only lines was joined to the run of lines after it, turning
`label \n value` into `label: value` before the rules ran.

Joining is a deletion. When the pairing was wrong the clinical line was gone,
and a measured 90 combinations out of 90 lost one:

    'Patient Name:' / 'MRN:' / 'Harold Nkemdirim' / 'Bradycardia 48 bpm'
      -> 'Patient Name: [NAME]' / 'MRN: [MRN]'

`Bradycardia 48 bpm` was deleted, and `[MRN]` asserted a record number had been
removed that was never there. Both invariants broken by one line of pairing.

So nothing is joined here. A label is ASSOCIATED with a value, and the value is
redacted where it already sits. The transform's only effect on any line is to
replace a matched span with a placeholder, so:

  - no line can be deleted, because no line is ever consumed;
  - no line can be altered unless a value was matched ON THAT LINE;
  - a mis-association can at worst redact a line that did not need it, never
    remove one.

The line count is therefore invariant, and that is asserted directly by
`tests/core/test_pii_scrubber_layouts.py::test_the_line_structure_is_preserved`.

## Why association is type-checked

An association only fires when the candidate parses ENTIRELY as a value of the
label's own type. `MRN:` needs one token containing a digit; `Bradycardia 48 bpm
untreated` is four tokens and is not one. That is what makes it safe to widen
how a label is FOUND — colon-less, pipe-separated, value-before-label, tabular —
without widening what a label may swallow. The two directions of the invariant
are enforced by different mechanisms, so closing one cannot reopen the other.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .fields import FieldType, find_labels, is_ambiguous_person_label, type_of
from .report import Removal
from .text import INVISIBLE, join_lines, split_lines
from .values import (
    is_clinical_phrase,
    longest_value_at,
    matches_exclusively,
    name_tokens,
    person_evidence,
    uncapitalised,
    whole_value,
)

#: Characters that may sit between a label and its value — `MRN: 4451209`,
#: `MRN #88213456`, `Patient Name | Harold Nkemdirim`, or a bare column gap.
#:
#: `\r` is in the set because CRLF input otherwise left a carriage return on the
#: end of every line, which made `_is_label_only` false for every label line and
#: disabled this entire module — the raw name then reached the application log
#: and the database. The invisible characters are here for the same reason: a
#: leading byte-order mark is not whitespace to `str.strip()`.
_SEPARATORS = " \t\r:：|=#–—-" + INVISIBLE


#: A line that already carries a placeholder of OURS has had its value removed.
#:
#: Matching `\[[A-Z_]+\]` matched anything in square brackets, so an editorial
#: `[SIC]`, `[NB]` or `[X]` beside a name made the line look already-scrubbed
#: and disabled cross-line association for it — the name AND the record number
#: below it then leaked. Only the placeholders this module emits count.
_PLACEHOLDER = re.compile(
    "|".join(rf"\[{field.value}\]" for field in FieldType if field is not FieldType.STRUCTURAL)
)


class Certainty(Enum):
    """Whether a claim was established on evidence or guessed.

    ONE decision procedure produces both, and the two paths read the same output
    under different policies. That is the fix for the finding that the refusal
    and the scrubber "disagree about the same string": `find_ambiguities` used to
    re-derive the question with its own narrower rule, so the quarantine fired
    when the lexicon already solved the problem and did not fire when it did not
    — 49 of 60 clinical phrases destroyed on the path whose stated policy is to
    refuse rather than guess.

        SETTLED  a structural signal established the value and its extent: a
                 non-person type matched by shape, a run bounded by the next
                 field label or a sentence terminator, a short run, or a span
                 carrying its own person evidence.

        GUESSED  the value could be a name or could be clinical content, or its
                 extent could not be established. Nothing here decides it.

    Upload path: a GUESSED claim is REFUSED and the caller is asked for
    structured fields, because the document is processed unseen and a wrong
    guess is silent in both directions.

    Chat path: a GUESSED claim is REDACTED, because the clinician wrote the text
    and reads the answer, so an over-redaction is visible and recoverable while
    a leak is not.
    """

    SETTLED = "settled"
    GUESSED = "guessed"


@dataclass(frozen=True)
class _Claim:
    """A span on one line that will become a placeholder."""

    line: int
    start: int
    end: int
    field_type: FieldType
    certainty: Certainty = Certainty.SETTLED
    #: The line carrying the label that made this claim. Equal to `line` for an
    #: inline value; different for a two-column extraction, where a refusal has
    #: to be able to name BOTH — the field it could not resolve and the line
    #: whose content it is protecting.
    label_line: int = -1
    label: str = ""
    words: int = 0
    reason: str = ""
    #: Where on `label_line` the label sits. The Safe Handoff accounting needs
    #: this: `Patient Name` left standing beside `[NAME]` is the label that
    #: ATTRIBUTED the removal, not clinical content the projection dropped, and
    #: the only non-guessing way to know that is to ask the claim that made it.
    #: Reading "everything before the first colon" instead is the predicate that
    #: made `Complete heart block, pacemaker implanted: <date>` invisible.
    label_start: int = -1
    label_end: int = -1
    #: Whether the clinician must be TOLD about this removal, independently
    #: of whether the upload path would REFUSE it. True for any person value
    #: of `_UNBOUNDED_NAME_WORDS` or more words. See `_extent_certainty`.
    announce: bool = False


#: Separator characters that only a FIELD uses. A colon, pipe, equals or hash
#: after a word is an explicit declaration that a value follows; a single space
#: is not, and reading one as a declaration is what turned `Carer Strain Index`
#: into `Carer [NAME]`.
_EXPLICIT_SEPARATORS = ":：|=#–—-"


def _skip_separator(line: str, position: int) -> tuple[int, bool]:
    """Advance past the separator, and say whether it was an EXPLICIT one."""
    index = position
    explicit = False
    while index < len(line) and line[index] in _SEPARATORS:
        explicit = explicit or line[index] in _EXPLICIT_SEPARATORS
        index += 1
    # Two or more spaces is a column gap, which is as deliberate as a colon.
    if not explicit and index - position >= 2:
        explicit = True
    return index, explicit


#: Characters that decorate a value without being part of one.
#:
#: Markdown emphasis, a quotation mark of any nationality, a bullet, a bracket,
#: a redaction bar. `_skip_separator` advanced over `_SEPARATORS` only and then
#: required the grammar to match EXACTLY there, so any other character at that
#: position made `longest_value_at` return `None`, no claim was made, and —
#: because the line is not label-only — no orphan or type fallback ever looked
#: at it either. 41 of 95 (prefix, field) cells leaked, and the two field types
#: that leaked for EVERY prefix were NAME and MRN: precisely the two with no
#: free-text shape rule to catch them a second time.
#:
#: No adversary is required. pypdf emits typographic quotes for a PDF that uses
#: them, and markdown emphasis and bullets are ordinary pasted clinical text.
_DECORATION = "*_`~/&@%+$!?^" + "\\" + "•«»“”‘’\"'()[]{}<>█|"

#: How many decoration characters may sit between the separator and the value.
#: Bounded so a line of punctuation cannot be scanned indefinitely.
_MAX_DECORATION = 6

#: A short bracketed word, however many brackets and whatever the case:
#: `[NAME]`, `[name]`, `[[NAME]]`, `[SIC]`, `[NB]`, `[?]`.
_BRACKETED = re.compile(r"\[+[A-Za-z_?]{1,16}\]+")


def _skip_noise(line: str, position: int) -> int:
    """Advance past separators, decorations and any placeholder OF OURS.

    A placeholder is skipped over, NOT treated as an answer. That distinction is
    the whole of the fix for the decoy:

        trust the marker    `Patient Name: [NAME] Harold Nkemdirim` is declared
                            already-scrubbed, and the real name leaks — the
                            caller controls the input, so this is forgeable in
                            any position a rule can name;

        skip the marker     the scanner steps over `[NAME]` and finds
                            `Harold Nkemdirim` behind it, which is redacted.

    Idempotence survives without trusting anything: on genuinely scrubbed text
    there is nothing behind the placeholder, so no claim is made. The property
    is established by what is THERE rather than by believing a marker, which is
    what `_already_redacted` did and what made three insertion shapes leak.
    """
    index = position
    for _ in range(_MAX_DECORATION + 2):
        moved = False
        index, _ = _skip_separator(line, index)
        # A bracketed short word, before the bare-decoration skip. Matching the
        # brackets as a UNIT is what handles the nested and doubled spellings:
        # skipping `[` and `[` separately from `[[NAME]]` leaves the scanner on
        # `NAME`, which then parses as a name and produces a placeholder around
        # the word "NAME". `[+ ... ]+` consumes the whole decoy at once.
        #
        # Case-insensitive, and wider than the placeholders this module emits:
        # `[name]`, `[SIC]` and `[NB]` are all non-values, and stepping over a
        # non-value can never suppress a redaction — it can only reveal what is
        # behind it.
        bracketed = _BRACKETED.match(line, index)
        if bracketed is not None:
            index = bracketed.end()
            moved = True
        else:
            decorated = 0
            while (
                index < len(line)
                and line[index] in _DECORATION
                and decorated < _MAX_DECORATION
            ):
                index += 1
                decorated += 1
                moved = True
        if not moved:
            break
    return index


def _value_after(
    line: str, label_end: int, field_type: FieldType
) -> tuple[tuple[int, int] | None, bool]:
    start, explicit = _skip_separator(line, label_end)
    any_case = uncapitalised(line)
    span = longest_value_at(field_type, line, start, any_case=any_case)
    if span is None:
        # Only as a FALLBACK, so a value that legitimately begins with one of
        # these characters is unaffected: `(020) 7946 0958` and `+44 7700 900456`
        # match at the separator and never reach here.
        relaxed = _skip_noise(line, label_end)
        if relaxed > start:
            # Matched against the REMAINDER, not at an offset into the line.
            # `_` is a decoration and also a word character, so `\b` at the
            # start of the telephone grammar looked back at the underscore we
            # had just stepped over and failed — `Telephone: _0113 496 0231_`
            # leaked with the label right there. Matching the remainder makes
            # the skipped decoration invisible to a word-boundary assertion,
            # which is what "skipped" should mean.
            found = longest_value_at(
                field_type, line[relaxed:], 0, any_case=any_case
            )
            if found is not None:
                span = (found[0] + relaxed, found[1] + relaxed)
    return span, explicit


def _value_before(line: str, label_start: int, field_type: FieldType) -> tuple[int, int] | None:
    """A value that sits BEFORE its label — `Harold Nkemdirim | Patient Name`.

    The whole of the line before the label must parse as one value, which is a
    much stronger requirement than "a value ends here" and is what keeps this
    from firing inside narrative.

    Decoration is stepped over on this side too, symmetrically with
    `_skip_noise` on the after-side. Only separators were stripped here, so a
    BRACKETED trailing label was not reachable at all:

        'Harold Nkemdirim (Patient Name)'  ->  unchanged
        'A1234567 (MRN)'                   ->  unchanged
        'Harold Nkemdirim - Patient Name'  ->  '[NAME] - Patient Name'

    The label was found in every one of those; the prefix simply ended in `(`
    and stopped parsing. NHS numbers, postcodes and telephones in the same shape
    were caught anyway by their free-text shape rules — the two types that
    leaked are NAME and MRN, which have none, which is the same pair that leaked
    when `_skip_noise` was missing on the other side.

    This widens where a label may be FOUND and not what one may absorb: the
    prefix must still parse ENTIRELY as one value of the label's own type, so a
    clinical phrase before a bracketed word is still not claimable.
    """
    prefix = line[:label_start].rstrip(_SEPARATORS + _DECORATION)
    if not prefix.strip():
        return None
    return whole_value(field_type, prefix)


# `_already_redacted` is GONE.
#
# It answered "has this label's value already been replaced?" by looking for a
# placeholder beside the label — `after = line[end:].lstrip(_SEPARATORS)` then
# `_PLACEHOLDER.match(after)`, plus a `before.endswith("]")` test for the
# line-leading form. The caller controls the input and there is no
# authentication on `mao/api/`, so both were forgeable: inserting `[NAME]`
# after the label made the module declare the field scrubbed and the real name
# went out in the clear. All ten emittable placeholders worked, in every
# insertion shape, for the two field types that have no free-text shape rule to
# catch them a second time.
#
# The property it was reaching for — "a second scrub is a no-op" — is not
# obtainable from a marker in caller-controlled text, and neither is it
# obtainable from a POSITION in caller-controlled text, which is what replaced
# it and which leaked in three further shapes. It is obtained instead by
# scrubbing exactly once and carrying the typed result
# (`mao/trust/inputs/boundary.py`), and, at this level, by `_skip_noise`
# STEPPING OVER a placeholder rather than believing it.


def _is_label_only(line: str, labels: list[tuple[int, int, FieldType]]) -> bool:
    """A line holding exactly one label and no value — a letterhead's left column.

    A remainder consisting only of placeholders counts as "no value". Without
    that, `Patient Name: [NAME]` with the real name on the NEXT line was neither
    an inline claim nor an orphan label, so nothing cross-line ever looked at
    it and the name below leaked. Stripping our own placeholders is not trusting
    them: a placeholder is not a value whichever way it got there, and the line
    still yields no claim of its own either way.
    """
    if len(labels) != 1:
        return False
    start, end, _ = labels[0]
    # `_BRACKETED`, not `_PLACEHOLDER`: the latter matches only the exact
    # upper-case spellings this module emits, so `Patient Name: [name]` was
    # still "a line with a value", never became an orphan, and the real name on
    # the next line was never looked for. A bracketed short word is not a value
    # in any spelling.
    remainder = _BRACKETED.sub("", line[end:])
    return not line[:start].strip() and not remainder.strip(_SEPARATORS)


def _clip_to_labels(
    span: tuple[int, int] | None,
    label_spans: list[tuple[int, int, FieldType]],
    own: tuple[int, int],
) -> tuple[int, int] | None:
    """A claimed value may not overlap ANOTHER FIELD'S LABEL on the same line.

    `_token_kind` already refuses a token that is a whole label by itself, but a
    label is often several words — `Medical Record Number` — and none of those
    words is a label alone. With the clinical lexicon no longer terminating a
    name run (see `values._token_kind`), a header row of a table extraction

        'Consultant:    Medical Record Number:    Date of Birth:'

    let `Consultant:` claim `Medical Record` and emit `Consultant: [NAME]:`,
    which destroys a column heading and asserts a removal that never happened.

    `find_labels` has already located every label on the line, so the bound is
    available and needs no vocabulary: a value stops where the next field
    begins. Structural, and it holds for a label of any length.
    """
    if span is None:
        return None
    start, end = span
    for label_start, label_end, _ in label_spans:
        if (label_start, label_end) == own:
            continue
        if start < label_start < end:
            end = label_start
        elif label_start <= start < label_end:
            return None
    return (start, end) if end > start else None


def _same_line_claims(
    lines: list[str], labels: list[list[tuple[int, int, FieldType]]]
) -> list[_Claim]:
    """Values sharing a line with their label, in either order.

    Two passes. Every type except PERSON is settled first, because those are
    unambiguous by shape, and the count of what they settled is the evidence
    that this document is a FORM. Only then are the person labels resolved,
    which is what lets a bare surname be recognised in a patient banner without
    letting a clinical phrase be mistaken for a name in ordinary prose.
    """
    claims: list[_Claim] = []
    deferred: list[tuple[int, int, int, FieldType]] = []
    for index, line in enumerate(lines):
        for start, end, field_type in labels[index]:
            if field_type is FieldType.STRUCTURAL:
                continue
            if field_type is FieldType.NAME:
                deferred.append((index, start, end, field_type))
                continue
            span, _ = _value_after(line, end, field_type)
            if span is None:
                span = _value_before(line, start, field_type)
            span = _clip_to_labels(span, labels[index], (start, end))
            if span is not None:
                claims.append(
                    _Claim(
                        index,
                        span[0],
                        span[1],
                        field_type,
                        label_line=index,
                        label=" ".join(line[start:end].split()),
                        label_start=start,
                        label_end=end,
                    )
                )

    form_evidence = _form_evidence(lines, claims)
    for index, start, end, field_type in deferred:
        line = lines[index]
        # BEFORE first, for a person label only. `_value_before` demands that the
        # ENTIRE prefix parse as one value; `_value_after` takes a greedy run
        # from wherever the label ends. When both can match, the first is much
        # the stronger evidence — and preferring the second got it exactly
        # backwards: `Jonathan Aldred-Whitmore: Name Rockwood Frailty Scale`
        # redacted `Rockwood Frailty` as the name and left the real one, in the
        # clear, at the start of the line.
        # A value BEFORE its label is credibility-checked too.
        #
        # It used to be taken as settled on the strength of the structure —
        # `_value_before` demands that the ENTIRE prefix parse as one value,
        # which is strong evidence — and that was safe only because the clinical
        # lexicon was refusing capitalised common English inside `name_tokens`.
        # With the lexicon out of the tokeniser (it truncated names, which is
        # what produced every partial redaction), `The` parses as a name token,
        # `Patient` is an ambiguous person label, and
        #
        #     'The patient scored 28/30 on MMSE.'  ->  '[NAME] patient scored...'
        #
        # The vocabulary still has the answer; it just has to be asked as a
        # CLASSIFICATION question about the span rather than allowed to cut a
        # name in half.
        span = _value_before(line, start, field_type)
        before = span is not None
        explicit = True
        if span is None:
            span, explicit = _value_after(line, end, field_type)
        span = _clip_to_labels(span, labels[index], (start, end))
        if span is None:
            continue
        verdict = (
            _CELL_PERSON
            if explicit and not before
            else _person_label_may_claim(line, line[start:end], span, form_evidence)
        )
        if verdict == _CELL_CLINICAL:
            continue
        certainty, words, reason = _extent_certainty(line, span)
        if verdict == _CELL_UNDECIDED:
            certainty = Certainty.GUESSED
            reason = reason or (
                "the value starts like a name and continues into text that "
                "reads as clinical content, and nothing establishes where one "
                "ends and the other begins"
            )
        claims.append(
            _Claim(
                line=index,
                start=span[0],
                end=span[1],
                field_type=field_type,
                certainty=certainty,
                label_line=index,
                label=" ".join(line[start:end].split()),
                words=words,
                reason=reason,
                label_start=start,
                label_end=end,
                announce=words >= _UNBOUNDED_NAME_WORDS,
            )
        )
    return claims


#: A run this long, with more than a bounded amount of content behind it, is the
#: shape whose end cannot be established. Two words is an ordinary forename and
#: surname and needs no adjudication.
_UNBOUNDED_NAME_WORDS = 3


def _extent_certainty(line: str, span: tuple[int, int]) -> tuple[Certainty, int, str]:
    """Whether this person value's EXTENT was established, or guessed.

    The predecessor asked this question with an extra escape hatch — a run
    reaching the end of the line was declared settled, on the reasoning that "a
    header value runs to the end of its line". That reasoning is what
    `values.value_at` uses to justify taking the whole run, so using it here too
    means the two halves agree by construction and neither of them has actually
    established anything:

        'Patient Name: Harold Nkemdirim Rockwood Frailty'
          tail is empty -> declared unambiguous -> the whole run is taken
          -> the instrument's name is DELETED, unseen, with no refusal

    Reaching the end of the line does not tell a six-token name from a name
    followed by a clinical phrase. Nothing does. So it is reported as guessed,
    and 49 of 60 measured phrases stop being destroyed silently.

    What DOES settle an extent is a structural boundary the caller wrote: the
    next known field label, or a sentence terminator. `Patient Name: John
    Michael Smith MRN: RGT/44219/B` is an ordinary banner and is settled — which
    matters, because a refusal that fires on the shape every real patient banner
    has is a broken product rather than a policy.

    ## Two questions, and only one of them has an answer

    This function was asked one question and used to answer a different one.

        where does the RUN end?          `name_tokens` answers it, and a
                                         following label or a mid-line sentence
                                         terminator corroborates it.
        is the run ALL NAME?             nothing answers it, for a run of three
                                         or more name-shaped words.

    The escape hatches answer the FIRST. Treating them as answers to the second
    is what produced the silent destructions, and the two measured shapes differ
    by one character:

        'Patient Name: Sarah Okonkwo Complete Heart Block.'      announced
        'Patient Name: Sarah Okonkwo Complete Heart Block. X'    silent

    A newline after the full stop announced; a space silenced. The discriminator
    had moved from the lexicon to punctuation, which is not an improvement.

    So the two questions are now reported separately. `Certainty` answers the
    first and still drives the upload REFUSAL, which keeps an ordinary banner
    processing rather than refusing. `announce` answers the second — it is true
    for every run of `_UNBOUNDED_NAME_WORDS` or more, unconditionally — and it
    drives the clinician's NOTICE, whose wording has always been "the removal
    may have taken more than the identifier with it". That sentence is true of
    every such span, and there is no punctuation that makes it false.
    """
    from .fields import find_labels as _find_labels

    tokens = name_tokens(line, span[0])
    words = len([token for token in tokens if token[0] == "word"])
    if words < _UNBOUNDED_NAME_WORDS:
        return Certainty.SETTLED, words, ""
    reason = (
        f"{words} name-shaped words were removed as one value — the end of the "
        "name cannot be established from the text, so the removal may have "
        "taken clinical content with it"
    )
    tail = line[span[1] :]
    leading = tail.lstrip()
    if leading[:1] in _SENTENCE_END and leading[1:].strip():
        return Certainty.SETTLED, words, reason
    if _find_labels(tail):
        return Certainty.SETTLED, words, reason
    return Certainty.GUESSED, words, reason


_SENTENCE_END = ".?!;"


#: Types whose shape is unambiguous — a line that is wholly one of these is an
#: identifier and could not be clinical narrative.
_UNAMBIGUOUS: tuple[FieldType, ...] = (
    FieldType.MRN,
    FieldType.NHS,
    FieldType.DOB,
    FieldType.PHONE,
    FieldType.EMAIL,
    FieldType.POSTCODE,
    FieldType.NI_NUMBER,
    FieldType.ACCOUNT,
    FieldType.ADDRESS,
)


def _form_evidence(lines: list[str], claims: list[_Claim]) -> int:
    """How much this document looks like a patient banner rather than prose.

    Counted from the identifiers that are PRESENT, not from the ones that
    happened to pair. A record number and a date of birth sitting in a column
    are what make a document a form; whether the label beside them was matched
    is a fact about this module, not about the document. Counting pairings
    instead undercounted a banner whose date was caught by the free-text date
    rule rather than by its label, and a bare surname then leaked.

    Only unambiguous types count. A name cannot be its own evidence — that is
    the question being asked.
    """
    lines_with_evidence = {
        claim.line for claim in claims if claim.field_type is not FieldType.NAME
    }
    for index, line in enumerate(lines):
        if index in lines_with_evidence:
            continue
        for field_type in _UNAMBIGUOUS:
            if matches_exclusively(field_type, line) is not None:
                lines_with_evidence.add(index)
                break
    return len(lines_with_evidence)


def _minimal_name(tokens: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """The shortest prefix of `tokens` that could be a name on its own.

    Counted in WORD tokens, not in tokens. `van der Grinten` begins with two
    particles, so a flat "first two tokens" gave `van der` — and `name_tokens`
    pops trailing particles, so that span parsed as nothing at all, the
    credibility test failed, and a real Dutch surname was not redacted.
    Titles and initials have the same shape of problem: `Dr A. Raman`.
    """
    words = 0
    for index, (kind, _, _) in enumerate(tokens):
        if kind == "word":
            words += 1
            if words == 2:
                return tokens[: index + 1]
    return tokens


def _person_label_may_claim(
    line: str, label: str, span: tuple[int, int], form_evidence: int
) -> str:
    """Whether a person label with NO explicit separator may take this value.

    `Carer Strain Index` is a validated instrument and `Carer` is a person
    label, so with the separator ignored the instrument's name was read as the
    carer's and deleted. An explicit separator settles it; without one, an
    ambiguous label needs the value to look like a person, and even an
    unambiguous one needs the ordinary cross-line credibility test.
    """
    tokens = name_tokens(line, span[0])
    if not tokens:
        return _CELL_CLINICAL
    if is_ambiguous_person_label(label):
        return (
            _CELL_PERSON if person_evidence(line, tokens) else _CELL_CLINICAL
        )
    # Two questions, on two different spans, and conflating them cost a leak.
    #
    # "Does a person value START here?" is about the MINIMAL name — the first
    # two tokens. `Carer Strain Index` fails it, which is the case this guard
    # exists for.
    #
    # "Is the whole run clinical content?" is about the greedy span, and its
    # answer is no longer allowed to DROP the claim. A run-on extraction —
    # `Surname Okonkwo-Achebe Rockwood Frailty Scale...` all on one line — ran
    # to the next label, the instrument vocabulary in the tail made the span
    # look clinical, the claim was dropped, and the surname was not redacted at
    # all. Judging the span is right; discarding the claim on it was the leak.
    #
    # So a credible start with a clinical-looking tail is UNDECIDED: redacted on
    # the chat path, refused on the upload path, and never silently dropped.
    minimal = _minimal_name(tokens)
    if not _name_is_credible(
        line[minimal[0][1] : minimal[-1][2]], FieldType.NAME, form_evidence
    ):
        return _CELL_CLINICAL
    if _name_is_credible(line[span[0] : span[1]], FieldType.NAME, form_evidence):
        return _CELL_PERSON
    return _CELL_UNDECIDED


def _runs_of_orphan_labels(
    lines: list[str], labels: list[list[tuple[int, int, FieldType]]], paired: set[int]
) -> list[list[int]]:
    """Maximal runs of consecutive label-only lines that found no value inline."""
    runs: list[list[int]] = []
    current: list[int] = []
    for index, line in enumerate(lines):
        orphan = (
            index not in paired
            and labels[index]
            and _is_label_only(line, labels[index])
            and labels[index][0][2] is not FieldType.STRUCTURAL
        )
        if orphan:
            current.append(index)
            continue
        if current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


#: A column gap inside one line: a tab, or two or more spaces.
_CELL_GAP = re.compile(r"[ \t]{2,}|\t")


def _cells(line: str) -> list[tuple[str, int, int]]:
    """Split a line into column cells, keeping each cell's offsets."""
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for gap in _CELL_GAP.finditer(line):
        bounds.append((cursor, gap.start()))
        cursor = gap.end()
    bounds.append((cursor, len(line)))

    cells: list[tuple[str, int, int]] = []
    for start, end in bounds:
        raw = line[start:end]
        text = raw.strip()
        if not text:
            continue
        offset = start + raw.index(text)
        cells.append((text, offset, offset + len(text)))
    return cells


def _cell_label(text: str) -> FieldType | None:
    """The type of a cell that is entirely one label, `Born:` colon included."""
    return type_of(text.rstrip(":：").strip())


def _label_row(line: str) -> list[FieldType] | None:
    """The column types of a line that is nothing but labels in columns."""
    cells = _cells(line)
    if len(cells) < 2:
        return None
    types: list[FieldType] = []
    for text, _, _ in cells:
        field_type = _cell_label(text)
        if field_type is None:
            return None
        types.append(field_type)
    return types


def _match_columns(
    types: list[FieldType], cells: list[tuple[str, int, int]]
) -> list[_Claim]:
    """Assign data cells to header columns left to right, by type.

    Greedy and in order, so a header column with no data beneath it — a real
    shape, and the one that would otherwise shift every later column onto the
    wrong type — is skipped rather than filled with its neighbour's value.
    """
    matched: list[_Claim] = []
    column = 0
    for text, start, _ in cells:
        for position in range(column, len(types)):
            span = matches_exclusively(types[position], text)
            if span is None:
                continue
            matched.append(
                _Claim(-1, start + span[0], start + span[1], types[position])
            )
            column = position + 1
            break
    return matched


def _table_claims(
    lines: list[str],
    labels: list[list[tuple[int, int, FieldType]]],
    consumed: set[int],
) -> list[_Claim]:
    """A header row and a data row, each on ONE line, cells in wide-gap columns.

    Neither line is `label \\n value` and neither is label-only, so nothing in
    the line-pair machinery sees this at all: a discharge summary in this shape
    matched no rule whatever and both the name and the record number reached the
    provider untouched. It arrives from PDFs whose text operators emit a whole
    row at once, and from a clinician pasting a table into `/chat`.
    """
    claims: list[_Claim] = []
    for index, line in enumerate(lines):
        types = _label_row(line)
        if types is None:
            continue
        for neighbour in (index + 1, index - 1):
            if not 0 <= neighbour < len(lines) or neighbour in consumed:
                continue
            cells = _cells(lines[neighbour])
            if len(cells) < 2 or any(_cell_label(text) for text, _, _ in cells):
                continue
            matched = _match_columns(types, cells)
            if not matched:
                continue
            claims += [
                _Claim(
                    neighbour,
                    claim.start,
                    claim.end,
                    claim.field_type,
                    label_line=index,
                    label_start=0,
                    label_end=len(line),
                )
                for claim in matched
            ]
            consumed.add(neighbour)
            break
    return claims


def _pair_in_direction(
    lines: list[str],
    labels: list[list[tuple[int, int, FieldType]]],
    runs: list[list[int]],
    consumed: set[int],
    forwards: bool,
    form_evidence: int,
) -> list[tuple[int, _Claim]]:
    """Pair every orphan label with the column cell `forwards` or backwards of it.

    Returns (label line, claim) pairs, so a caller can tell which labels are
    still unplaced without re-deriving the association.

    Alignment is by position within the run, which is the order a two-column
    extraction preserves, and every pairing must match the label's own type
    EXCLUSIVELY — and, for a PERSON label, must also be a credible name rather
    than a clinical noun phrase of the same shape. A label that cannot be paired
    is left exactly as it is: it then produces no placeholder, which is the
    point.
    """
    paired: list[tuple[int, _Claim]] = []
    taken = set(consumed)
    satisfied: set[int] = set()
    # Non-person columns first: they are unambiguous by shape, and how many of
    # them pair is the evidence that this document is a patient banner rather
    # than a page of narrative that happens to contain a label word.
    for names_now in (False, True):
        evidence = form_evidence
        for run in runs:
            length = len(run)
            for offset in range(length):
                label_line = run[offset]
                start, end, field_type = labels[label_line][0]
                if (field_type is FieldType.NAME) is not names_now:
                    continue
                candidate = (
                    run[-1] + 1 + offset if forwards else run[0] - length + offset
                )
                if not _is_free_cell(lines, labels, candidate, taken):
                    continue
                span = matches_exclusively(field_type, lines[candidate])
                if span is None:
                    continue
                verdict = _person_cell_verdict(
                    lines[candidate],
                    field_type,
                    evidence,
                    ambiguous=is_ambiguous_person_label(lines[label_line][start:end]),
                )
                if verdict == _CELL_CLINICAL:
                    continue
                paired.append(
                    (
                        label_line,
                        _cross_line_claim(
                            lines, labels, label_line, candidate, span, field_type,
                            verdict,
                        ),
                    )
                )
                taken.add(candidate)
    # A satisfied label is PLACED with no claim, so `_cross_line_claims` does
    # not hand it on to the type-matched fallback. The sentinel line is -1 and
    # is filtered out before any span is applied.
    paired.extend(
        (label_line, _Claim(-1, 0, 0, FieldType.STRUCTURAL))
        for label_line in satisfied
    )
    return paired


def _cross_line_claim(
    lines: list[str],
    labels: list[list[tuple[int, int, FieldType]]],
    label_line: int,
    candidate: int,
    span: tuple[int, int],
    field_type: FieldType,
    verdict: str,
) -> _Claim:
    """A claim from an orphan label onto a cell on another line.

    Records BOTH lines. A refusal has to be answerable for what it excludes, and
    for a two-column extraction those are different lines: the field that could
    not be resolved is on one, and the content being protected is on the other.
    A report naming only the label line looks like a blanket over a failure
    somewhere else.
    """
    start, end, _ = labels[label_line][0]
    label = " ".join(lines[label_line][start:end].split())
    return _Claim(
        line=candidate,
        start=span[0],
        end=span[1],
        field_type=field_type,
        certainty=(
            Certainty.GUESSED if verdict == _CELL_UNDECIDED else Certainty.SETTLED
        ),
        label_line=label_line,
        label=label,
        label_start=start,
        label_end=end,
        words=len(lines[candidate].split()),
        reason=(
            "a person label on another line was paired with this one by "
            "position, and the line carries no evidence that it is a name "
            "rather than clinical content"
        ),
    )


def _is_free_cell(
    lines: list[str],
    labels: list[list[tuple[int, int, FieldType]]],
    index: int,
    taken: set[int],
) -> bool:
    """A line that could be an unclaimed column cell.

    Non-empty and carrying no label.

    It deliberately does NOT reject a line merely for containing a placeholder.
    That test was here for idempotence, and it was forgeable: the caller
    controls the input and there is no authentication on `mao/api/`, so
    appending ` [NAME]` to each line of a pasted banner made every line look
    already-scrubbed and suppressed redaction of the name AND the record number
    — 10 of 10 of this module's own placeholders worked as decoys.

    A marker in the input cannot establish "this span was produced by THIS run".
    Idempotence is established positionally instead, by `_value_is_placeholder`:
    a label whose OWN candidate cell holds nothing but a placeholder has already
    been satisfied. Appending a decoy leaves the real value on the line, so the
    cell is not placeholder-only and is still redacted.
    """
    return (
        0 <= index < len(lines)
        and index not in taken
        and not labels[index]
        and bool(lines[index].strip())
    )


def _value_is_placeholder(line: str) -> bool:
    """Whether this line holds a placeholder and nothing else of substance.

    NOT used to decide that a label is already satisfied. It was, and that was a
    forgery: the caller controls the input and there is no authentication on
    `mao/api/`, so INSERTING a placeholder-only line shifted the column
    alignment by one and the real name and record number went out in the clear.

    The two properties are in genuine tension and neither is free:

      - trust the marker  -> scrubbing is idempotent, and an inserted line
                             suppresses redaction (a LEAK, caller-triggerable);
      - distrust it       -> no forgery is possible, and a second scrub
                             over-redacts a clinical line it cannot type
                             (a DESTRUCTION, on a path `/chat` really takes).

    Distrust wins. A leak is unrecoverable and reaches a third party; an
    over-redaction is visible in the answer and recoverable. The residual is
    recorded rather than hidden, and closing it properly means telling a
    clinical phrase from a name — the vocabulary problem this module has not
    solved and which the phase report names as the open design question.
    """
    return bool(_PLACEHOLDER.search(line)) and not _PLACEHOLDER.sub("", line).strip(
        " \t\r.,;:" + INVISIBLE
    )


def _name_is_credible(
    line: str,
    field_type: FieldType,
    form_evidence: int,
    *,
    ambiguous: bool = False,
    near: bool = True,
) -> bool:
    """Whether a bare line may be paired with a PERSON label across lines.

    This is the hardest judgement in the module and the one both reviewers broke.
    `Harold Nkemdirim` and `Peptic Ulcer Bleeding` are the same shape, so an
    orphan `Patient Name:` above a clinical line claimed the clinical line and
    destroyed it — 46 of 60 real phrases, including `Sick Sinus Syndrome`,
    `Complete Heart Block` and `Prolonged QTc 520 ms`, which are the cardiac
    contraindications to the drug this system is asked about.

    Shape is therefore not enough, and a stop-list of clinical words is not
    either — clinical English does not enumerate. Two positive tests instead:

      - the span carries person evidence of its own: a title, a name particle,
        a given name, or a non-Latin script;
      - or it is at most two tokens AND the document has already proved itself
        a FORM by pairing at least two other fields correctly. A surname alone
        (`Okonkwo-Achebe`) has no internal evidence, but a banner that also
        yielded an MRN and a date of birth is a patient banner.

    A clinical noun phrase is three tokens or more far more often than a bare
    surname is, so the token bound is doing real work rather than papering over
    the gap. What neither test catches is recorded as a residual.

    `near` is the distance guard, and it applies ONLY to the weaker second test.
    A span carrying its own person evidence may be matched anywhere in the
    document — narrative interleaved into a wide two-column header puts six
    lines between `Patient Name:` and the name — but a span relying on form
    evidence alone must be close to its label, or `Lasting Power of Attorney`
    thirteen lines below an orphan `Consultant:` starts looking like a person.
    """
    return _person_cell_verdict(line, field_type, form_evidence,
                                ambiguous=ambiguous, near=near) is not _CELL_CLINICAL


#: The three answers a bare column cell can get. `_CELL_UNDECIDED` is the one
#: that used to be silently folded into "person": `_name_is_credible` ended in a
#: bare `return True`, so an orphan `Patient Name:` above `Rockwood Frailty`
#: claimed the instrument and deleted it — with no refusal, on the path whose
#: stated policy is to refuse rather than guess.
_CELL_PERSON = "person"
_CELL_CLINICAL = "clinical"
_CELL_UNDECIDED = "undecided"


def _person_cell_verdict(
    line: str,
    field_type: FieldType,
    form_evidence: int,
    *,
    ambiguous: bool = False,
    near: bool = True,
) -> str:
    """Person, clinical, or genuinely undecided — see `_name_is_credible`."""
    if field_type is not FieldType.NAME:
        return _CELL_PERSON
    tokens = name_tokens(line, 0) or name_tokens(line, len(line) - len(line.lstrip()))
    if not tokens:
        return _CELL_CLINICAL
    # STRONG person evidence is consulted BEFORE the clinical test, and the
    # order is the fix for a leak rather than a preference.
    #
    # `is_clinical_phrase` fires when ANY token is in the lexicon, and a
    # dementia service's lexicon is full of surnames — so `Mary Parkinson` in a
    # letterhead column was classified as clinical content and left completely
    # unredacted, 160 of 192 measured. Asking "is there a given name, a title,
    # an initial, a suffix, a non-Latin script?" first settles that case on
    # evidence about a person, and leaves every span WITHOUT such evidence to
    # the clinical test exactly as before.
    #
    # Strong evidence only. A particle would let `Lasting Power of Attorney` and
    # `Activities of Daily Living` outvote their own clinical vocabulary on the
    # strength of the word "of".
    if person_evidence(line, tokens, strong_only=True):
        return _CELL_PERSON
    if is_clinical_phrase(line, tokens):
        return _CELL_CLINICAL
    if person_evidence(line, tokens):
        return _CELL_PERSON
    if ambiguous:
        # `Carer`, `Patient`, `Mother` are ordinary clinical words, so the label
        # itself carries no weight. The value must look like a person, or the
        # document must be a form and the value must be nearby and short.
        if near and len(tokens) <= 2 and form_evidence >= 2:
            return _CELL_UNDECIDED
        return _CELL_CLINICAL
    # Neither positively a person nor positively clinical.
    #
    # This branch used to `return True`, and that bare `True` is where 49 of 60
    # measured clinical phrases were destroyed: an orphan `Patient Name:` above
    # `Rockwood Frailty`, `Substantia Nigra` or `Medtronic Azure Pacemaker`
    # claimed the line and deleted it, unseen, with no refusal — on the path
    # whose stated policy is that an unresolvable header is refused.
    #
    # `Harold Nkemdirim` and `Rockwood Frailty` are the same shape and no rule
    # separates them. Saying so is the only honest answer; what to DO about it
    # is a policy question, and the two paths answer it differently.
    return _CELL_UNDECIDED


#: How far from a PERSON label a value may be found when position has broken
#: down.
#:
#: The search was unbounded — `for candidate in range(len(lines))` — so an
#: orphan `Consultant:` on line 0 reached down to line 13 and redacted `Lasting
#: Power of Attorney`. The bound applies to PERSON labels only, because that is
#: the only type whose shape clinical prose also has.
#:
#: An unambiguous type needs no window. A line that is wholly and exclusively a
#: record number, a date of birth or a postcode holds no clinical content to
#: lose, so redacting it is safe however far from its label it sits — and it
#: does sit far, because narrative interleaved into a wide two-column header
#: puts three clinical lines between the label block and the value block. A
#: window tight enough to protect names left record numbers leaking in 35
#: generated documents.
#:
#: The bound is derived from the document, not chosen: in an all-labels-then-
#: all-values extraction a value sits exactly as far from its label as the
#: header block is tall, so the reach is that block's size plus a small margin
#: for interleaved narrative. A lone orphan label therefore reaches a few lines;
#: a six-field banner reaches across its own block; and neither reaches thirteen
#: lines down into the discussion, which is where `Lasting Power of Attorney`
#: was found and destroyed.
_NAME_FALLBACK_MARGIN = 4


def _pair_by_type(
    lines: list[str],
    labels: list[list[tuple[int, int, FieldType]]],
    unpaired: list[int],
    taken: set[int],
    form_evidence: int,
) -> list[_Claim]:
    """Last resort: match a leftover label to a nearby leftover cell by TYPE.

    Narrative interleaved into a wide two-column header destroys the positional
    alignment entirely — a label can be four lines above its own value with two
    medication lines and another label in between. Position cannot recover that,
    but type can, within a bounded window, and doing so is safe in both
    directions:

      - the cell is redacted only if the WHOLE line parses as one value of that
        type and of no more specific type, so nothing but an identifier is
        touched and the placeholder names what was really there;
      - a PERSON label additionally needs `_name_is_credible`, because a
        person's shape is the one shape clinical prose also has;
      - the label itself is still not rewritten, so an unmatched label continues
        to produce no placeholder.
    """
    claims: list[_Claim] = []
    reach = _NAME_FALLBACK_MARGIN + sum(1 for found in labels if found)
    for label_line in unpaired:
        field_type = labels[label_line][0][2]
        for candidate in range(len(lines)):
            if not _is_free_cell(lines, labels, candidate, taken):
                continue
            span = matches_exclusively(field_type, lines[candidate])
            if span is None:
                continue
            start, end, _ = labels[label_line][0]
            verdict = _person_cell_verdict(
                lines[candidate],
                field_type,
                form_evidence,
                ambiguous=is_ambiguous_person_label(lines[label_line][start:end]),
                near=abs(candidate - label_line) <= reach,
            )
            if verdict == _CELL_CLINICAL:
                continue
            claims.append(
                _cross_line_claim(
                    lines, labels, label_line, candidate, span, field_type, verdict
                )
            )
            taken.add(candidate)
            break
    return claims


def _cross_line_claims(
    lines: list[str],
    labels: list[list[tuple[int, int, FieldType]]],
    consumed: set[int],
    form_evidence: int,
) -> list[_Claim]:
    """Associate orphan labels with their column, choosing ONE order per document.

    A document has one column order, not one per row. Deciding per run reads an
    alternating `value / label / value / label` extraction as if each label
    owned the row below it, which walks every association one row out of step —
    the first value is never redacted and the last label adopts a value that is
    not its own.

    So both orders are scored across the whole document and the better one wins.
    Ties go forwards, which is the commoner extraction. Whatever the winning
    order cannot place is then matched by type, within a bounded window.

    `form_evidence` is how many fields already paired unambiguously — inline or
    tabular. It is what lets a bare surname be recognised in a patient banner
    without letting a clinical noun phrase be mistaken for one in a document
    that is not a form at all.
    """
    runs = _runs_of_orphan_labels(lines, labels, consumed)
    if not runs:
        return []
    forwards = _pair_in_direction(
        lines, labels, runs, consumed, forwards=True, form_evidence=form_evidence
    )
    backwards = _pair_in_direction(
        lines, labels, runs, consumed, forwards=False, form_evidence=form_evidence
    )
    chosen = forwards if len(forwards) >= len(backwards) else backwards

    claims = [claim for _, claim in chosen if claim.line >= 0]
    placed_labels = {label_line for label_line, _ in chosen}
    taken = set(consumed) | {claim.line for claim in claims}
    unpaired = [
        label_line
        for run in runs
        for label_line in run
        if label_line not in placed_labels
    ]
    return claims + _pair_by_type(lines, labels, unpaired, taken, form_evidence)


def _apply(
    lines: list[str], claims: list[_Claim], removed: list[Removal] | None = None
) -> list[str]:
    """Replace each claimed span with its placeholder. Lines are never joined.

    `removed` collects what was actually replaced. The spans are already here
    and already correct; returning only the rewritten text threw them away and
    forced every downstream control to re-derive them with a grammar of its own.
    See `mao/core/deident/report.py`.
    """
    by_line: dict[int, list[_Claim]] = {}
    for claim in claims:
        by_line.setdefault(claim.line, []).append(claim)

    out = list(lines)
    for index, line_claims in by_line.items():
        line = lines[index]
        rebuilt: list[str] = []
        cursor = 0
        for claim in sorted(line_claims, key=lambda c: (c.start, -c.end)):
            if claim.start < cursor:  # overlapping claim; the first one wins
                continue
            rebuilt.append(line[cursor : claim.start])
            rebuilt.append(f"[{claim.field_type.value}]")
            if removed is not None:
                removed.append(
                    Removal(
                        kind=claim.field_type.value,
                        value=line[claim.start : claim.end],
                        line=index,
                    )
                )
            cursor = claim.end
        rebuilt.append(line[cursor:])
        out[index] = "".join(rebuilt)
    return out


def build_claims(text: str) -> tuple[list[str], list[str], list[_Claim]]:
    """Every span this module would redact, with how certain each one is.

    Exposed so `ambiguity` reads the SAME decision rather than re-deriving it.
    The predecessor had two implementations of "is this resolvable" and they
    disagreed about the same string — the refusal fired when the lexicon already
    solved the problem and did not fire when it did not, which is the exact
    inverse of what a quarantine is for.
    """
    lines, terminators = split_lines(text)
    labels = [find_labels(line) for line in lines]

    claims = _same_line_claims(lines, labels)
    consumed = {claim.line for claim in claims}
    claims += _table_claims(lines, labels, consumed)
    claims += _cross_line_claims(
        lines, labels, consumed, _form_evidence(lines, claims)
    )
    return lines, terminators, claims


@dataclass(frozen=True)
class LabelledSpan:
    """One claim, in absolute offsets of the text it was found in.

    `build_claims` reports `(line, start, end)`, which is the right shape for
    rewriting a line in place and the wrong one for mapping back to an immutable
    source. This is the same decision, re-expressed once, here — rather than
    every caller re-deriving line offsets and getting it subtly different.
    """

    start: int
    end: int
    kind: str
    label_start: int = -1
    label_end: int = -1
    label: str = ""
    guessed: bool = False
    announce: bool = False
    words: int = 0
    reason: str = ""
    line: int = 0
    label_line: int = -1


def _line_offsets(contents: list[str], terminators: list[str]) -> list[int]:
    offsets: list[int] = []
    position = 0
    for content, terminator in zip(contents, terminators, strict=True):
        offsets.append(position)
        position += len(content) + len(terminator)
    return offsets


def labelled_spans(text: str) -> list[LabelledSpan]:
    """Every span a field label identifies, as offsets into `text` itself."""
    lines, terminators, claims = build_claims(text)
    offsets = _line_offsets(lines, terminators)
    found: list[LabelledSpan] = []
    for claim in claims:
        if claim.line < 0 or claim.start >= claim.end:
            continue
        base = offsets[claim.line]
        label_start = label_end = -1
        if claim.label_line >= 0 and claim.label_start >= 0:
            label_base = offsets[claim.label_line]
            label_start = label_base + claim.label_start
            label_end = label_base + claim.label_end
        found.append(
            LabelledSpan(
                start=base + claim.start,
                end=base + claim.end,
                kind=claim.field_type.value,
                label_start=label_start,
                label_end=label_end,
                label=claim.label,
                guessed=claim.certainty is Certainty.GUESSED,
                announce=claim.announce,
                words=claim.words,
                reason=claim.reason,
                line=claim.line,
                label_line=claim.label_line,
            )
        )
    return found


def redact_labelled_fields_with_report(text: str) -> tuple[str, list[Removal]]:
    """Redact every value that a field label identifies, and say which."""
    lines, terminators, claims = build_claims(text)
    removed: list[Removal] = []
    return join_lines(_apply(lines, claims, removed), terminators), removed


def redact_labelled_fields(text: str) -> str:
    """Redact every value that a field label identifies, in place."""
    return redact_labelled_fields_with_report(text)[0]
