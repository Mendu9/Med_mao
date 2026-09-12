r"""Where every part of the source went. An account, not an estimate.

## The defect this replaces

`coverage()` has now been unable to see the loss it exists to detect at three
consecutive gates, by a different mechanism each time:

    b63311d  the denominator was a LEXICON LOOKUP, so a clinical line the
             vocabulary did not recognise was in neither the numerator nor the
             denominator. Measured: `_is_clinical_line` true for 0 of 49
             clinical phrases.                                        ADV16-6

    ff34722  the denominator was a line count, and a new predicate excluded
             lines BEFORE they reached it. `_carries_nothing_to_lose` read only
             the value half of the first colon, so every
             `<clinical statement>: <redacted identifier>` line vanished from
             both sides. Measured: Complete Heart Block and an implanted
             pacemaker dropped from the projection handed to a model asked
             whether a bradycardic drug was safe, at coverage 1.0000, with
             `uncertainties` empty and no notice to the clinician.
                                                          AR17-1 · ADV17-2

    ff34722  and a second route into the same exclusion: `\[[A-Z_]+\]` matched
             any bracketed uppercase token, so a clinician's `Pacing mode:
             [DDDR]` was accounted as an identifier the boundary had removed.

`00_RULES.md` names the response: *"If repeated fixes to one mechanism
repeatedly introduce new failures in the same invariant class, stop patching
symptoms and escalate the abstraction/design before another remediation round."*
A fourth predicate is the thing that clause forbids.

## What replaces it

Not a better predicate. An ACCOUNT.

Every character of the protected text is attributed to exactly one state, and
the states are exhaustive by construction rather than by a rule someone wrote:

    IDENTIFIER_REMOVED   the boundary actually redacted this span. Known from
                         the transformation's own event record, not from the
                         shape of what it left behind.
    STRUCTURAL           a field label that attributed one of those removals, or
                         a run with no word characters at all. Both are
                         positional facts.
    TYPED_FACT           this segment became a field of the projection, and the
                         projection can be asked to show it.
    UNRESOLVED           none of the above. Meaning-bearing text that the
                         projection does not carry.

Completeness is then not a judgement. A projection is complete when nothing is
`UNRESOLVED`, and the number reported is the share of accountable segments that
became typed facts. No vocabulary decides it, no colon decides it, and no
regex over placeholder-shaped text decides it — the only inputs are the spans
the transformation recorded and the fields the projection actually carries.

## What it deliberately does not do

It does not decide whether an unresolved segment MATTERED. It cannot: that needs
the clinical understanding `02_IMPLEMENTATION_PLAN` allocates to Phase 3. What
it guarantees instead is the property `00_RULES.md` actually requires — that an
incomplete projection is never represented as a complete one. The count is
reported, the status is reported, and the external synthesis is told.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from mao.core.deident.report import RedactionEvent
from mao.core.deident.text import split_lines


class SegmentState(str, Enum):
    """The four states a source segment can be in. Exhaustive by construction."""

    IDENTIFIER_REMOVED = "identifier_removed"
    STRUCTURAL = "structural"
    TYPED_FACT = "typed_fact"
    UNRESOLVED = "unresolved"


#: Characters that cannot carry meaning on their own. A run consisting only of
#: these is structure — a separator, a rule, a bullet, the punctuation left
#: behind when a label's value was removed.
#:
#: This is a STRUCTURAL set and not a vocabulary: it names no word, and adding a
#: word to it is not possible. `str.isalnum()` decides the rest, which covers
#: every script rather than the Latin one.
_PUNCTUATION = " \t\r\n|=#*_`~/\\<>[](){}«»“”‘’\"'.,;:!?-–—•■□▪●○+&@%$^"


def _carries_meaning(text: str) -> bool:
    """Whether this run has any character a reader could take meaning from."""
    return any(character.isalnum() for character in text)


@dataclass(frozen=True)
class Segment:
    """One accounted run of the protected text.

    `text` is protected content. This object stays inside the protected plane:
    `CompletenessReport` is what crosses a boundary, and it carries counts and
    line numbers only.
    """

    line: int
    start: int
    end: int
    state: SegmentState
    text: str = ""
    #: Which projection field carried it, when `TYPED_FACT`. Field names only.
    carried_as: str = ""
    kind: str = ""


@dataclass(frozen=True)
class CompletenessReport:
    """The account, with every piece of source text taken out.

    This is the ONLY thing the external synthesis, the response metadata, the
    Redis cache and the trace ever see about what the projection did not carry.
    At `ff34722` they saw the text itself: `uncertainties` was populated from
    the residue — the lines selected precisely FOR being unparseable, which is
    the population with the highest residual-identifier density — and
    `SafeSynthesisContext.render()` emitted it verbatim to the model. Measured:
    a full personal name and a contact extension in the rendered payload, inside
    the class whose own docstring prohibits raw or scrubbed report text.

    Measuring the loss and transmitting the residue are different requirements
    and this is where they separate. The count and the line references satisfy
    the first; nothing here can satisfy the second, because nothing here is text
    from the document.
    """

    carried: int = 0
    unresolved: int = 0
    identifiers_removed: int = 0
    structural: int = 0
    #: 1-based line numbers, so a clinician who has the document can find them.
    #: A line NUMBER is not content: it says where to look, not what is there.
    unresolved_lines: tuple[int, ...] = ()

    @property
    def accountable(self) -> int:
        """Segments that could have become a fact: carried plus unresolved."""
        return self.carried + self.unresolved

    @property
    def complete(self) -> bool:
        """Whether every accountable segment became a typed fact."""
        return self.unresolved == 0

    def coverage(self) -> float:
        """The share of accountable segments the projection carries.

        1.0 when there was nothing accountable — a document of letterhead and
        nothing else has genuinely lost nothing, and that is the case the A-1
        remediation note warned would otherwise collapse and start refusing
        ordinary documents. It does not collapse here, because an identifier the
        boundary removed and the label that introduced it are accounted in
        neither the numerator nor the denominator: they are accounted in their
        OWN states, which is the difference between an account and an exclusion.
        """
        if self.accountable == 0:
            return 1.0
        return self.carried / self.accountable

    def describe(self) -> str:
        """A statement safe for a model, a log, a cache and a trace.

        Names counts and line numbers. Never text.
        """
        if self.complete:
            return (
                f"Complete: all {self.carried} clinically accountable source "
                "segment(s) are represented in this projection."
            )
        where = ", ".join(str(line) for line in self.unresolved_lines)
        return (
            f"INCOMPLETE: {self.unresolved} of {self.accountable} clinically "
            f"accountable source segment(s) could not be represented in this "
            f"projection and are NOT included below"
            + (f" (source line(s) {where})" if where else "")
            + ". Treat the case as incompletely described and say so; do not "
            "assume the omitted content was immaterial."
        )


@dataclass(frozen=True)
class SourceAccounting:
    """Every segment of one protected text, with its state.

    Protected-plane only. `report()` is the projection of it that may travel.
    """

    segments: tuple[Segment, ...] = ()

    def in_state(self, state: SegmentState) -> tuple[Segment, ...]:
        return tuple(segment for segment in self.segments if segment.state is state)

    def unresolved_text(self) -> tuple[str, ...]:
        """The residue, for LOCAL measurement only.

        Deliberately a method on the protected object and not a field on
        anything typed: reaching this text requires holding the accounting,
        which never leaves the process. `SafeSynthesisContext` has no field it
        could be assigned to.
        """
        return tuple(
            segment.text for segment in self.in_state(SegmentState.UNRESOLVED)
        )

    def report(self) -> CompletenessReport:
        unresolved = self.in_state(SegmentState.UNRESOLVED)
        return CompletenessReport(
            carried=len(self.in_state(SegmentState.TYPED_FACT)),
            unresolved=len(unresolved),
            identifiers_removed=len(self.in_state(SegmentState.IDENTIFIER_REMOVED)),
            structural=len(self.in_state(SegmentState.STRUCTURAL)),
            unresolved_lines=tuple(
                sorted({segment.line + 1 for segment in unresolved})
            ),
        )


#: Markdown heading syntax. A `#` at the start of a line is structure the author
#: typed deliberately, and recognising it reads a CHARACTER, not a word.
_MARKDOWN_HEADING = re.compile(r"^\s*#+\s")


def account(
    protected_text: str,
    events: tuple[RedactionEvent, ...],
    carried: dict[int, tuple[str, str]],
) -> SourceAccounting:
    """Attribute every part of `protected_text` to exactly one state.

    `events` are the redactions that actually happened, in this text's own
    coordinates. `carried` maps a line index to `(field_name, text)` for every
    line the extractor turned into a typed fact.

    The walk is positional from beginning to end. Nothing is skipped, nothing is
    excluded before it is counted, and there is no branch in which a segment is
    dropped from both the numerator and the denominator — which is the shape of
    every failure this module replaces.
    """
    lines, terminators = split_lines(protected_text)
    offsets: list[int] = []
    position = 0
    for content, terminator in zip(lines, terminators, strict=True):
        offsets.append(position)
        position += len(content) + len(terminator)

    by_line: dict[int, list[tuple[int, int, str, bool]]] = {}
    for event in events:
        base = offsets[event.line] if event.line < len(offsets) else 0
        by_line.setdefault(event.line, []).append(
            (event.start - base, event.end - base, event.kind, False)
        )
        if event.has_label() and 0 <= event.label_start:
            label_line = _line_containing(offsets, event.label_start)
            label_base = offsets[label_line]
            by_line.setdefault(label_line, []).append(
                (
                    event.label_start - label_base,
                    event.label_end - label_base,
                    event.kind,
                    True,
                )
            )

    segments: list[Segment] = []
    for index, line in enumerate(lines):
        segments.extend(
            _account_line(index, offsets[index], line, by_line.get(index, []), carried)
        )
    return SourceAccounting(segments=tuple(segments))


def _line_containing(offsets: list[int], position: int) -> int:
    from bisect import bisect_right

    return max(0, bisect_right(offsets, position) - 1)


def _account_line(
    index: int,
    base: int,
    line: str,
    marks: list[tuple[int, int, str, bool]],
    carried: dict[int, tuple[str, str]],
) -> list[Segment]:
    """Account one line: the marked spans, then everything between them."""
    segments: list[Segment] = []
    ordered = sorted(
        (
            (max(0, start), min(len(line), end), kind, is_label)
            for start, end, kind, is_label in marks
            if start < len(line) and end > 0 and start < end
        )
    )

    cursor = 0
    residue: list[tuple[int, int]] = []
    for start, end, kind, is_label in ordered:
        if start < cursor:
            continue
        if start > cursor:
            residue.append((cursor, start))
        segments.append(
            Segment(
                line=index,
                start=base + start,
                end=base + end,
                state=(
                    SegmentState.STRUCTURAL
                    if is_label
                    else SegmentState.IDENTIFIER_REMOVED
                ),
                text=line[start:end],
                kind=kind,
            )
        )
        cursor = end
    if cursor < len(line):
        residue.append((cursor, len(line)))

    remainder = "".join(line[start:end] for start, end in residue)
    if not _carries_meaning(remainder):
        # Whitespace, separators and the punctuation a removed value left
        # behind. Nothing here can be lost because nothing here says anything.
        if remainder.strip(_PUNCTUATION) or residue:
            segments.append(
                Segment(
                    line=index,
                    start=base + (residue[0][0] if residue else 0),
                    end=base + (residue[-1][1] if residue else 0),
                    state=SegmentState.STRUCTURAL,
                    text=remainder,
                )
            )
        return segments

    if _MARKDOWN_HEADING.match(line):
        # `## Findings` is a structure marker the author typed. The test reads
        # the `#` character and not the words after it, so no heading vocabulary
        # exists to be widened — which is what `_HEADING`'s Titlecase-plus-colon
        # rule was, and it was one of the routes clinical lines disappeared
        # through.
        segments.append(
            Segment(
                line=index,
                start=base + residue[0][0],
                end=base + residue[-1][1],
                state=SegmentState.STRUCTURAL,
                text=remainder,
            )
        )
        return segments

    field, _ = carried.get(index, ("", ""))
    segments.append(
        Segment(
            line=index,
            start=base + residue[0][0],
            end=base + residue[-1][1],
            state=SegmentState.TYPED_FACT if field else SegmentState.UNRESOLVED,
            text=remainder,
            carried_as=field,
        )
    )
    return segments
