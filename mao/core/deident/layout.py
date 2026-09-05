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

from .fields import FieldType, find_labels, is_ambiguous_person_label, type_of
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


@dataclass(frozen=True)
class _Claim:
    """A span on one line that will become a placeholder."""

    line: int
    start: int
    end: int
    field_type: FieldType


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


def _value_after(
    line: str, label_end: int, field_type: FieldType
) -> tuple[tuple[int, int] | None, bool]:
    start, explicit = _skip_separator(line, label_end)
    span = longest_value_at(
        field_type, line, start, any_case=uncapitalised(line)
    )
    return span, explicit


def _value_before(line: str, label_start: int, field_type: FieldType) -> tuple[int, int] | None:
    """A value that sits BEFORE its label — `Harold Nkemdirim | Patient Name`.

    The whole of the line before the label must parse as one value, which is a
    much stronger requirement than "a value ends here" and is what keeps this
    from firing inside narrative.
    """
    prefix = line[:label_start].rstrip(_SEPARATORS)
    if not prefix.strip():
        return None
    return whole_value(field_type, prefix)


def _already_redacted(line: str, start: int, end: int) -> bool:
    """Whether this label's value has already been replaced by a placeholder.

    `/chat` scrubs twice — `apply_input_guardrails` and then `query_decomposer`
    — so a second pass is a real code path, not a curiosity. Without this check
    the first pass turned `Jonathan Aldred-Whitmore: Name Rockwood Frailty
    Scale` into `[NAME]: Name Rockwood Frailty Scale`, and then the second pass
    could no longer parse the prefix as a value, fell through to the text AFTER
    the label, and redacted the clinical instrument instead: `[NAME]: Name
    [NAME] Scale`. Scrubbing an already-scrubbed string must be a no-op.
    """
    before = line[:start].rstrip(_SEPARATORS)
    if before.endswith("]") and _PLACEHOLDER.search(before[-16:]):
        return True
    after = line[end:].lstrip(_SEPARATORS)
    return bool(_PLACEHOLDER.match(after))


def _is_label_only(line: str, labels: list[tuple[int, int, FieldType]]) -> bool:
    """A line holding exactly one label and no value — a letterhead's left column."""
    if len(labels) != 1:
        return False
    start, end, _ = labels[0]
    return not line[:start].strip() and not line[end:].strip(_SEPARATORS)


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
            if _already_redacted(line, start, end):
                continue
            if field_type is FieldType.NAME:
                deferred.append((index, start, end, field_type))
                continue
            span, _ = _value_after(line, end, field_type)
            if span is None:
                span = _value_before(line, start, field_type)
            if span is not None:
                claims.append(_Claim(index, span[0], span[1], field_type))

    form_evidence = _form_evidence(lines, claims)
    for index, start, end, field_type in deferred:
        line = lines[index]
        if _already_redacted(line, start, end):
            continue
        # BEFORE first, for a person label only. `_value_before` demands that the
        # ENTIRE prefix parse as one value; `_value_after` takes a greedy run
        # from wherever the label ends. When both can match, the first is much
        # the stronger evidence — and preferring the second got it exactly
        # backwards: `Jonathan Aldred-Whitmore: Name Rockwood Frailty Scale`
        # redacted `Rockwood Frailty` as the name and left the real one, in the
        # clear, at the start of the line.
        span = _value_before(line, start, field_type)
        explicit = True
        if span is None:
            span, explicit = _value_after(line, end, field_type)
        if span is None:
            continue
        if not explicit and not _person_label_may_claim(
            line, line[start:end], span, form_evidence
        ):
            continue
        claims.append(_Claim(index, span[0], span[1], field_type))
    return claims


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


def _person_label_may_claim(
    line: str, label: str, span: tuple[int, int], form_evidence: int
) -> bool:
    """Whether a person label with NO explicit separator may take this value.

    `Carer Strain Index` is a validated instrument and `Carer` is a person
    label, so with the separator ignored the instrument's name was read as the
    carer's and deleted. An explicit separator settles it; without one, an
    ambiguous label needs the value to look like a person, and even an
    unambiguous one needs the ordinary cross-line credibility test.
    """
    tokens = name_tokens(line, span[0])
    if not tokens:
        return False
    if is_ambiguous_person_label(label):
        return person_evidence(line, tokens)
    return _name_is_credible(line[span[0] : span[1]], FieldType.NAME, form_evidence)


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
                _Claim(neighbour, claim.start, claim.end, claim.field_type)
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
                # Already satisfied by an earlier pass: this label's OWN cell
                # holds a placeholder and nothing else. Recording it as placed
                # is what stops it walking on to the next line and redacting a
                # clinical one — the residual idempotence defect, which
                # destroyed a clinical line in 880 of 82,764 documents on the
                # second scrub `/chat` actually performs.
                #
                # Positional, so a decoy cannot exploit it: appending `[NAME]`
                # to a line leaves the real value there, the cell is not
                # placeholder-ONLY, and it is still redacted.
                if 0 <= candidate < len(lines) and _value_is_placeholder(
                    lines[candidate]
                ):
                    satisfied.add(label_line)
                    continue
                if not _is_free_cell(lines, labels, candidate, taken):
                    continue
                span = matches_exclusively(field_type, lines[candidate])
                if span is None:
                    continue
                if not _name_is_credible(
                    lines[candidate],
                    field_type,
                    evidence,
                    ambiguous=is_ambiguous_person_label(lines[label_line][start:end]),
                ):
                    continue
                paired.append(
                    (label_line, _Claim(candidate, span[0], span[1], field_type))
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
    """Whether this line holds a placeholder and nothing else of substance."""
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
    if field_type is not FieldType.NAME:
        return True
    tokens = name_tokens(line, 0) or name_tokens(line, len(line) - len(line.lstrip()))
    if not tokens:
        return False
    if is_clinical_phrase(line, tokens):
        return False
    if person_evidence(line, tokens):
        return True
    if ambiguous:
        # `Carer`, `Patient`, `Mother` are ordinary clinical words, so the label
        # itself carries no weight. The value must look like a person, or the
        # document must be a form and the value must be nearby and short.
        return near and len(tokens) <= 2 and form_evidence >= 2
    return True


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
            if not _name_is_credible(
                lines[candidate],
                field_type,
                form_evidence,
                ambiguous=is_ambiguous_person_label(lines[label_line][start:end]),
                near=abs(candidate - label_line) <= reach,
            ):
                continue
            claims.append(_Claim(candidate, span[0], span[1], field_type))
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


def _apply(lines: list[str], claims: list[_Claim]) -> list[str]:
    """Replace each claimed span with its placeholder. Lines are never joined."""
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
            cursor = claim.end
        rebuilt.append(line[cursor:])
        out[index] = "".join(rebuilt)
    return out


def redact_labelled_fields(text: str) -> str:
    """Redact every value that a field label identifies, in place."""
    lines, terminators = split_lines(text)
    labels = [find_labels(line) for line in lines]

    claims = _same_line_claims(lines, labels)
    consumed = {claim.line for claim in claims}
    claims += _table_claims(lines, labels, consumed)
    claims += _cross_line_claims(
        lines, labels, consumed, _form_evidence(lines, claims)
    )
    return join_lines(_apply(lines, claims), terminators)
