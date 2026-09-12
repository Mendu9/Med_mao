r"""Deterministic PII scrubbing.

Regex-only by design: this runs on the path to a third-party LLM, so it must be
fast, offline, and auditable. A model-based de-identifier would add a heavy
dependency and a second inference call to the very hop we are trying to keep
clean.

## The invariant

    A placeholder stands for an identifier that was actually removed —
    and for nothing else.

Both directions are failures. Under-matching leaks an identifier. Over-matching
destroys the clinician's question, and does it silently, because
`state["user_query"]` *is* the scrubbed string: no control downstream can see
what the original said.

## Why this is a pipeline and not a regex list

Four waves of this project were defeated by a single list of patterns doing two
jobs at once. Every fix to the leak direction widened what a value could
swallow, and every fix to the destroy direction narrowed what could be found —
so the two halves of the invariant kept trading places, and the fix for one
shipped as a regression in the other.

They are separated here, into `mao.core.deident`:

    normalise  NFKC, so a fullwidth colon is a colon before anything reads it
    layout     find labelled values and redact them WHERE THEY ARE
    freetext   identifiers no label introduces, matched by shape

`layout` owns the destroy direction structurally. It never joins, splits or
deletes a line — its only effect on any line is to replace a matched span with a
placeholder — so a mis-association can redact something it should not have, but
it cannot delete clinical content and it cannot emit a placeholder where no
value matched. The Wave 10 CRITICAL, in which every orphan label deleted the
clinical line after it in 90 of 90 generated combinations, is unreachable by
construction rather than merely untested.

`values` owns the leak direction. Each field type states positively what its
value looks like, so widening how a label is FOUND — colon-less, pipe-separated,
tabular, value-before-label — cannot widen what a label may absorb.

## What holds this

`tests/core/test_pii_scrubber_layouts.py` asserts both directions on every
document in a generated cartesian product of separator, label/value order,
column arrangement, orphan labels, clinical placement and field set — rendered
with reportlab and read back with pypdf, the pair production uses. The layouts
are not chosen by the implementer, which is the failure this project repeated
four times.

NFKC normalisation happens here, not only in `input_guardrails`: `clinical_agent`
scrubs raw pypdf output directly, and a fullwidth colon (U+FF1A, routine in
scanned documents) is preserved by pypdf and matches no ASCII-colon rule at all.
Normalising inside the scrubber makes it impossible for a caller to forget.
"""
from __future__ import annotations

from mao.core.deident.freetext import shape_spans
from mao.core.deident.layout import labelled_spans
from mao.core.deident.report import RedactionEvent, Removal, ScrubResult
from mao.core.deident.source import (
    AppliedEdit,
    Edit,
    MatchView,
    apply_edits,
    build_match_view,
    place,
)
from mao.core.deident.text import split_lines

__all__ = ["ScrubResult", "scrub_pii", "scrub_with_report"]


def scrub_with_report(text: str) -> ScrubResult:
    """`scrub_pii`, plus the record of every span it replaced.

    ## The source is immutable

    The detectors read a normalised MATCH VIEW and never the document. Every
    span they find is mapped back through `MatchView` and written over the
    ORIGINAL characters, so the output differs from the input at redaction spans
    and nowhere else — measured byte for byte by
    `tests/core/test_source_is_preserved.py`.

    That ordering is what ends the six-wave trade-off. The match view may fold
    an accent as aggressively as an ASCII identifier grammar needs, because
    folding it costs nothing now: `café` is matched as `cafe` and shipped as
    `café`, and `Postcode: SW1<acute>A 1AA` is matched as `SW1A 1AA` and the
    ORIGINAL five characters are replaced by `[POSTCODE]`. At `ff34722` the
    folded form was the output, so 608 of 615 precomposed Latin letters, every
    Greek and Cyrillic diacritic, and every Devanagari, Thai, Telugu and Sinhala
    vowel sign were destroyed in text a clinician reads.

    ## The record

    Each `Removal` carries its SOURCE span, the label that attributed it, and
    whether its extent was established or guessed. The egress boundary compares
    values; the Safe Handoff accounting and the clinician's redaction notice
    read the spans. Both therefore describe the transformation that actually
    happened rather than a second opinion about the same string.
    """
    if not text:
        return ScrubResult(text=text)

    view = build_match_view(text)
    edits = _labelled_edits(view) + _shape_edits(view)
    scrubbed, merged = apply_edits(text, edits)
    return ScrubResult(
        text=scrubbed,
        removals=_removals(text, merged),
        events=_events(scrubbed, place(text, merged)),
    )


def _labelled_edits(view: MatchView) -> list[Edit]:
    """Every value a field label identifies, mapped onto the source."""
    edits: list[Edit] = []
    for span in labelled_spans(view.text):
        label = None
        if span.label_start >= 0 and span.label_end > span.label_start:
            label = view.source_span(span.label_start, span.label_end)
        edits.append(
            Edit(
                span=view.source_span(span.start, span.end),
                replacement=f"[{span.kind}]",
                kind=span.kind,
                label=label,
                certainty="guessed" if span.guessed else "settled",
                announce=span.announce,
                words=span.words,
                reason=span.reason,
            )
        )
    return edits


def _shape_edits(view: MatchView) -> list[Edit]:
    """Identifiers no label introduces, mapped onto the source.

    Run against the LABELLED-REDACTED match text rather than the raw match text,
    because the rules are ordered against each other and a value already claimed
    by its label must not be re-examined by a shape rule that would type it
    differently. The offsets come back through the same staged map, so both
    passes land in one coordinate system.
    """
    labelled = labelled_spans(view.text)
    staged, origin = _stage(view.text, labelled)
    edits: list[Edit] = []
    for span in shape_spans(staged):
        start, end = _unstage(origin, span.start, span.end)
        if start >= end:
            continue
        label = _introducing_label(view.text, start)
        edits.append(
            Edit(
                span=view.source_span(start, end),
                replacement=span.replacement,
                kind=span.kind,
                label=view.source_span(*label) if label else None,
            )
        )
    return edits


def _introducing_label(text: str, start: int) -> tuple[int, int] | None:
    """The field label that introduced a SHAPE-matched removal, if there is one.

    `Contact: <email>` is redacted by the shape rules rather than by the layout
    pass, because no label claimed it — but if the scrubber's own label grammar
    does recognise a field name earlier on the line, that name is the thing that
    introduced the identifier and is not clinical content the projection failed
    to carry.

    Attribution comes from `find_labels`, which is the SAME grammar `layout`
    uses, applied at the position of a removal that actually happened. It is a
    vocabulary of IDENTIFIER FIELD NAMES, which is the only kind of vocabulary
    admissible here: it defines what an identifier field is, and it says nothing
    about clinical content.

    There is deliberately no positional fallback. "Text running up to a removed
    identifier with only separators in between" would make `Complete heart
    block, permanent pacemaker implanted: <date>` structural, which is AR17-1
    exactly. A label the grammar does not know leaves its line accountable, and
    that is the pessimistic direction: it can make a projection look less
    complete than it is, never more.
    """
    from mao.core.deident.fields import find_labels

    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    line = text[line_start : line_end if line_end >= 0 else len(text)]
    best: tuple[int, int] | None = None
    for found_start, found_end, _ in find_labels(line):
        if line_start + found_end <= start and (best is None or found_end > best[1]):
            best = (found_start, found_end)
    if best is None:
        return None
    return line_start + best[0], line_start + best[1]


def _stage(text: str, spans: list) -> tuple[str, list[tuple[int, int]]]:
    """Apply the labelled spans, keeping each character's match-view origin."""
    ordered = sorted(spans, key=lambda span: (span.start, -span.end))
    out: list[str] = []
    origin: list[tuple[int, int]] = []
    cursor = 0
    for span in ordered:
        if span.start < cursor:
            continue
        out.append(text[cursor : span.start])
        origin.extend((index, index + 1) for index in range(cursor, span.start))
        placeholder = f"[{span.kind}]"
        out.append(placeholder)
        origin.extend([(span.start, span.end)] * len(placeholder))
        cursor = span.end
    out.append(text[cursor:])
    origin.extend((index, index + 1) for index in range(cursor, len(text)))
    return "".join(out), origin


def _unstage(origin: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    covered = origin[start:end]
    if not covered:
        return 0, 0
    return min(item[0] for item in covered), max(item[1] for item in covered)


def _removals(source: str, applied: list[Edit]) -> tuple[Removal, ...]:
    """The applied edits as the record the protected plane carries."""
    starts = _line_starts(source)
    removals: list[Removal] = []
    for edit in applied:
        label_start = edit.label.start if edit.label else -1
        label_end = edit.label.end if edit.label else -1
        removals.append(
            Removal(
                kind=edit.kind,
                value=source[edit.span.start : edit.span.end],
                line=_line_of(starts, edit.span.start),
                start=edit.span.start,
                end=edit.span.end,
                label_start=label_start,
                label_end=label_end,
                label=(
                    " ".join(source[label_start:label_end].split())
                    if edit.label
                    else ""
                ),
                guessed=edit.certainty == "guessed",
                announce=edit.announce,
                words=edit.words,
                reason=edit.reason,
            )
        )
    return tuple(removals)


def _events(scrubbed: str, applied: list[AppliedEdit]) -> tuple[RedactionEvent, ...]:
    """The applied edits, content-free, in the scrubbed text's coordinates."""
    starts = _line_starts(scrubbed)
    return tuple(
        RedactionEvent(
            kind=item.edit.kind,
            start=item.out_start,
            end=item.out_end,
            label_start=item.label_out.start if item.label_out else -1,
            label_end=item.label_out.end if item.label_out else -1,
            label=(
                " ".join(
                    scrubbed[item.label_out.start : item.label_out.end].split()
                )
                if item.label_out
                else ""
            ),
            line=_line_of(starts, item.out_start),
            guessed=item.edit.certainty == "guessed",
            announce=item.edit.announce,
            words=item.edit.words,
            reason=item.edit.reason,
        )
        for item in applied
    )


def _line_starts(text: str) -> list[int]:
    contents, terminators = split_lines(text)
    starts: list[int] = []
    position = 0
    for content, terminator in zip(contents, terminators, strict=True):
        starts.append(position)
        position += len(content) + len(terminator)
    return starts


def _line_of(starts: list[int], offset: int) -> int:
    from bisect import bisect_right

    return max(0, bisect_right(starts, offset) - 1)


def scrub_pii(text: str) -> str:
    """Replace personal identifiers with typed placeholders.

    Placeholders keep the field's *shape* so the model can still tell that a
    patient name or record number was present without learning whose. The line
    structure of the input is preserved exactly — every terminator, CRLF
    included — so nothing the clinician wrote can disappear on this path.

    A leading byte-order mark is detached and restored rather than removed. It
    is not whitespace to `str.strip()`, so left in place it made the first line
    unrecognisable as a label line and disabled the labelled path for the whole
    document; but it is not an identifier either, and the invariant is that the
    output differs from the input ONLY where an identifier was removed.
    """
    return scrub_with_report(text).text
