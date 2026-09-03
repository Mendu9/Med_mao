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

from .fields import FieldType, find_labels, type_of
from .values import matches_exclusively, uncapitalised, value_at, whole_value

#: Characters that may sit between a label and its value — `MRN: 4451209`,
#: `MRN #88213456`, `Patient Name | Harold Nkemdirim`, or a bare column gap.
_SEPARATORS = " \t:：|=#–—-"


@dataclass(frozen=True)
class _Claim:
    """A span on one line that will become a placeholder."""

    line: int
    start: int
    end: int
    field_type: FieldType


def _skip_separator(line: str, position: int) -> int:
    """Advance past the separator between a label and its value."""
    index = position
    while index < len(line) and line[index] in _SEPARATORS:
        index += 1
    return index


def _value_after(line: str, label_end: int, field_type: FieldType) -> tuple[int, int] | None:
    return value_at(
        field_type,
        line,
        _skip_separator(line, label_end),
        any_case=uncapitalised(line),
    )


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


def _is_label_only(line: str, labels: list[tuple[int, int, FieldType]]) -> bool:
    """A line holding exactly one label and no value — a letterhead's left column."""
    if len(labels) != 1:
        return False
    start, end, _ = labels[0]
    return not line[:start].strip() and not line[end:].strip(_SEPARATORS)


def _same_line_claims(
    lines: list[str], labels: list[list[tuple[int, int, FieldType]]]
) -> list[_Claim]:
    """Values sharing a line with their label, in either order."""
    claims: list[_Claim] = []
    for index, line in enumerate(lines):
        for start, end, field_type in labels[index]:
            if field_type is FieldType.STRUCTURAL:
                continue
            span = _value_after(line, end, field_type)
            if span is None and not line[end:].strip(_SEPARATORS):
                span = _value_before(line, start, field_type)
            if span is not None:
                claims.append(_Claim(index, span[0], span[1], field_type))
    return claims


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
) -> list[tuple[int, _Claim]]:
    """Pair every orphan label with the column cell `forwards` or backwards of it.

    Returns (label line, claim) pairs, so a caller can tell which labels are
    still unplaced without re-deriving the association.

    Alignment is by position within the run, which is the order a two-column
    extraction preserves, and every pairing must match the label's own type
    EXCLUSIVELY. A label that cannot be paired is left exactly as it is: it then
    produces no placeholder, which is the point.
    """
    paired: list[tuple[int, _Claim]] = []
    taken = set(consumed)
    for run in runs:
        length = len(run)
        for offset in range(length):
            candidate = (
                run[-1] + 1 + offset if forwards else run[0] - length + offset
            )
            if not _is_free_cell(lines, labels, candidate, taken):
                continue
            field_type = labels[run[offset]][0][2]
            span = matches_exclusively(field_type, lines[candidate])
            if span is None:
                continue
            paired.append((run[offset], _Claim(candidate, span[0], span[1], field_type)))
            taken.add(candidate)
    return paired


def _is_free_cell(
    lines: list[str],
    labels: list[list[tuple[int, int, FieldType]]],
    index: int,
    taken: set[int],
) -> bool:
    """A line that could be an unclaimed column cell: non-empty, no label on it."""
    return (
        0 <= index < len(lines)
        and index not in taken
        and not labels[index]
        and bool(lines[index].strip())
    )


def _pair_by_type(
    lines: list[str],
    labels: list[list[tuple[int, int, FieldType]]],
    unpaired: list[int],
    taken: set[int],
) -> list[_Claim]:
    """Last resort: match a leftover label to a leftover cell by TYPE alone.

    Narrative interleaved into a wide two-column header destroys the positional
    alignment entirely — a label can be four lines above its own value with two
    medication lines and another label in between. Position cannot recover that,
    but type can, and doing so is safe in both directions:

      - the cell is redacted only if the WHOLE line parses as one value of that
        type and of no more specific type, so nothing but an identifier is
        touched and the placeholder names what was really there;
      - the label itself is still not rewritten, so an unmatched label continues
        to produce no placeholder.
    """
    claims: list[_Claim] = []
    for label_line in unpaired:
        field_type = labels[label_line][0][2]
        for candidate in range(len(lines)):
            if not _is_free_cell(lines, labels, candidate, taken):
                continue
            span = matches_exclusively(field_type, lines[candidate])
            if span is None:
                continue
            claims.append(_Claim(candidate, span[0], span[1], field_type))
            taken.add(candidate)
            break
    return claims


def _cross_line_claims(
    lines: list[str],
    labels: list[list[tuple[int, int, FieldType]]],
    consumed: set[int],
) -> list[_Claim]:
    """Associate orphan labels with their column, choosing ONE order per document.

    A document has one column order, not one per row. Deciding per run reads an
    alternating `value / label / value / label` extraction as if each label
    owned the row below it, which walks every association one row out of step —
    the first value is never redacted and the last label adopts a value that is
    not its own.

    So both orders are scored across the whole document and the better one wins.
    Ties go forwards, which is the commoner extraction. Whatever the winning
    order cannot place is then matched by type alone.
    """
    runs = _runs_of_orphan_labels(lines, labels, consumed)
    if not runs:
        return []
    forwards = _pair_in_direction(lines, labels, runs, consumed, forwards=True)
    backwards = _pair_in_direction(lines, labels, runs, consumed, forwards=False)
    chosen = forwards if len(forwards) >= len(backwards) else backwards

    claims = [claim for _, claim in chosen]
    placed_labels = {label_line for label_line, _ in chosen}
    taken = set(consumed) | {claim.line for claim in claims}
    unpaired = [
        label_line
        for run in runs
        for label_line in run
        if label_line not in placed_labels
    ]
    return claims + _pair_by_type(lines, labels, unpaired, taken)


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
    lines = text.split("\n")
    labels = [find_labels(line) for line in lines]

    claims = _same_line_claims(lines, labels)
    consumed = {claim.line for claim in claims}
    claims += _table_claims(lines, labels, consumed)
    claims += _cross_line_claims(lines, labels, consumed)
    return "\n".join(_apply(lines, claims))
