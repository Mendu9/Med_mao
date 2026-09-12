r"""The source text is immutable. Matching happens on a separate representation.

## The defect this closes

`normalise()` was applied to the document and its output SHIPPED. It ran NFD,
deleted every mark it classified as invisible, folded every mark it classified
as a decoration, and then ran NFKC — and the result was what the scrubber
redacted, what the compiler read, and what a third-party model received.

That is two jobs in one string, and each one broke the other:

  - classifying too FEW marks as decoration left an accented postcode
    unreadable to an ASCII identifier grammar, so the identifier leaked
    (ADV16-1);
  - classifying too MANY destroyed the document. Measured at `ff34722`:
    608 of 615 precomposed Latin letters altered in shipped output, every
    Greek and Cyrillic diacritic dropped, `año` shipped as `ano`, `café au
    lait` as `cafe au lait`, and — through a different clause — Devanagari,
    Thai, Telugu and Sinhala vowel signs deleted, so `बुखार` ("fever") shipped
    as `बखार` and `มี` ("has") lost its vowel.

Four successive enumerations of "invisible" were each defeated by a character
the enumerator had not looked at, and the fifth would be too. The enumeration is
not the problem. Applying it to the text that ships is.

## The contract

    The protected source string is IMMUTABLE except at spans this module is
    explicitly told to replace.

    Canonicalisation, decomposition and folding exist ONLY inside a separate
    matching representation, which never leaves this package, and every span
    found in it maps back to a span of the original.

So a mark may be folded as aggressively as the identifier grammars need —
because folding it now costs nothing. Nothing downstream sees the folded form.
`AccentedPostcode` is found in the match view and `[POSTCODE]` is written over
the ORIGINAL characters, accent included, leaving every other byte of the
document exactly as the clinician wrote it.

## Why the map is chunk-aligned and not character-aligned

Normalisation is not a per-character function: NFD expands, NFKC composes and
folds compatibility forms, and canonical reordering permutes. What IS true is
that every one of those operations is confined between two *starters* — a
character whose compatibility decomposition begins with canonical combining
class zero. That is the standard incremental-normalisation boundary.

So the source is cut at starters, each chunk is normalised, and every match
character records the chunk it came from. A span in the match view therefore
maps to the union of its chunks' source ranges — which is exactly right, since
a chunk is one user-perceived character and redacting half of one is not a
thing that can be meant.

`_STAGES_AGREE_WITH_NORMALISE` in `tests/core/` holds the staged pipeline here
byte-identical to `text.normalise` over a generated corpus, so this module
cannot silently become a different matcher from the one the reproductions were
measured against.
"""
from __future__ import annotations

import unicodedata
from bisect import bisect_right
from dataclasses import dataclass

from mao.core.deident.text import _fold_decorations, _INVISIBLE_RE


@dataclass(frozen=True, order=True)
class SourceSpan:
    """A half-open range of the IMMUTABLE source string."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start > self.end:  # pragma: no cover - construction error
            raise ValueError(f"inverted span {self.start}:{self.end}")

    def overlaps(self, other: SourceSpan) -> bool:
        return self.start < other.end and other.start < self.end

    def union(self, other: SourceSpan) -> SourceSpan:
        return SourceSpan(min(self.start, other.start), max(self.end, other.end))


def _composition_seconds() -> frozenset[str]:
    """Every character that can be the SECOND element of a canonical composition.

    Combining class zero does not mean "cannot compose". Measured while
    building this module: U+0BD7 TAMIL AU LENGTH MARK, U+0BBE TAMIL VOWEL SIGN
    AA and U+1B35 BALINESE VOWEL SIGN TEDUNG all have class zero and all
    recompose with the letter before them, so cutting in front of one produced
    a match view that differed from `text.normalise` on 93 of 6000 generated
    strings. Hangul V and T jamo compose algorithmically and appear in no
    decomposition table at all, so they are named.

    Derived from the decomposition table rather than listed, for the same
    reason every other class in this package is computed: a character added to
    Unicode tomorrow is handled without anyone noticing it.
    """
    seconds: set[str] = set()
    for code in range(0x30000):
        decomposition = unicodedata.decomposition(chr(code))
        if not decomposition or decomposition.startswith("<"):
            continue
        parts = decomposition.split()
        if len(parts) == 2:
            seconds.add(chr(int(parts[1], 16)))
    seconds.update(chr(code) for code in range(0x1160, 0x11A8))  # Hangul V
    seconds.update(chr(code) for code in range(0x11A8, 0x1200))  # Hangul T
    return frozenset(seconds)


_COMPOSITION_SECONDS = _composition_seconds()


def _is_starter(character: str) -> bool:
    """Whether normalisation may be flushed immediately before this character.

    Two conditions, and both were established by measurement rather than by
    reading the standard optimistically:

      the compatibility decomposition begins with combining class zero
          `unicodedata.combining(character) == 0` alone is not sufficient:
          U+FF9E HALFWIDTH KATAKANA VOICED SOUND MARK has class zero and
          decomposes to U+3099, which has class 8 and composes with the kana
          BEFORE it.

      and that leading character cannot be a composition second
          see `_composition_seconds`.
    """
    decomposed = unicodedata.normalize("NFKD", character)
    if not decomposed:
        return False
    head = decomposed[0]
    return unicodedata.combining(head) == 0 and head not in _COMPOSITION_SECONDS


def _chunk(text: str) -> list[int]:
    """Offsets at which `text` may be cut without changing its normalisation."""
    starts = [0]
    for index in range(1, len(text)):
        if _is_starter(text[index]):
            starts.append(index)
    return starts


@dataclass(frozen=True)
class MatchView:
    """The matching representation of a source string, and the way back.

    `text` is what every detector reads. `source` is what ships. They are
    different strings on purpose, and this object is the only thing that knows
    how one relates to the other.
    """

    source: str
    text: str
    #: For each character of `text`, the source offsets of the chunk it came
    #: from. Parallel arrays rather than tuples: this is indexed per character
    #: of every document the boundary sees.
    _starts: tuple[int, ...]
    _ends: tuple[int, ...]

    def source_span(self, start: int, end: int) -> SourceSpan:
        """The smallest source range covering match characters `[start, end)`.

        An EMPTY match range still has a position, and it maps to the empty
        source range at the corresponding offset — which is what lets an
        insertion be expressed without claiming to have replaced anything.
        """
        if start >= end:
            at = self._ends[start - 1] if start else 0
            if start < len(self._starts):
                at = self._starts[start]
            return SourceSpan(at, at)
        return SourceSpan(self._starts[start], self._ends[end - 1])

    def match_offset_of(self, source_offset: int) -> int:
        """The first match offset whose chunk starts at or after `source_offset`."""
        return bisect_right(self._starts, source_offset - 1)


def build_match_view(source: str) -> MatchView:
    """Normalise `source` for matching, keeping every character's origin.

    The stages are `text.normalise`'s, in `text.normalise`'s order, and the
    equivalence is asserted by test rather than assumed:

        NFD  ->  drop invisibles  ->  fold decorations  ->  NFKC

    NFD and NFKC are applied chunk-wise, which is exact because the cuts are at
    starters. The two deletion stages are applied to the WHOLE string, because
    `_fold_decorations` reads the character before the one it is deciding about
    and a chunk-local view of that would give a different answer at a chunk
    boundary.
    """
    if not source:
        return MatchView(source="", text="", _starts=(), _ends=())

    # Stage 1 — NFD, chunk-wise, recording each output character's origin.
    decomposed: list[str] = []
    origin_start: list[int] = []
    origin_end: list[int] = []
    cuts = _chunk(source)
    for index, cut in enumerate(cuts):
        stop = cuts[index + 1] if index + 1 < len(cuts) else len(source)
        piece = unicodedata.normalize("NFD", source[cut:stop])
        decomposed.append(piece)
        origin_start.extend([cut] * len(piece))
        origin_end.extend([stop] * len(piece))
    staged = "".join(decomposed)

    # Stages 2 and 3 — deletions over the whole string. Both are expressed as
    # "keep this character", so the origin arrays are filtered, never rebuilt.
    kept = [
        index
        for index, character in enumerate(staged)
        if not _INVISIBLE_RE.match(character)
    ]
    stripped = "".join(staged[index] for index in kept)
    folded = _fold_decorations(stripped)
    kept = _surviving(stripped, folded, kept)

    # Stage 4 — NFKC, chunk-wise over what survived.
    starts = [origin_start[index] for index in kept]
    ends = [origin_end[index] for index in kept]
    out: list[str] = []
    out_starts: list[int] = []
    out_ends: list[int] = []
    cuts = _chunk(folded)
    for index, cut in enumerate(cuts):
        stop = cuts[index + 1] if index + 1 < len(cuts) else len(folded)
        piece = unicodedata.normalize("NFKC", folded[cut:stop])
        out.append(piece)
        out_starts.extend([min(starts[cut:stop], default=0)] * len(piece))
        out_ends.extend([max(ends[cut:stop], default=0)] * len(piece))

    return MatchView(
        source=source,
        text="".join(out),
        _starts=tuple(out_starts),
        _ends=tuple(out_ends),
    )


def _surviving(before: str, after: str, origins: list[int]) -> list[int]:
    """Which of `before`'s characters `after` kept, for a DELETE-ONLY stage.

    `_fold_decorations` only ever drops characters, so the two strings align by
    a single forward walk. Asserting that rather than diffing keeps this exact:
    if the stage ever starts substituting, the walk runs off the end and the
    caller gets an error instead of a silently wrong map.
    """
    kept: list[int] = []
    cursor = 0
    for index, character in enumerate(before):
        if cursor < len(after) and after[cursor] == character:
            kept.append(origins[index])
            cursor += 1
    if cursor != len(after):  # pragma: no cover - guards a future refactor
        raise AssertionError("fold stage is no longer delete-only")
    return kept


@dataclass(frozen=True)
class Edit:
    """One authorised replacement of a source span.

    `kind` is the placeholder's field type. `label` is the source span of the
    field label that ATTRIBUTED this removal, when one did — the accounting
    needs to know that `Patient Name` on the same line was introducing this
    identifier and is not clinical content left behind.
    """

    span: SourceSpan
    replacement: str
    kind: str
    label: SourceSpan | None = None
    certainty: str = "settled"
    #: How many name-shaped words the removed span covered. 0 for every type
    #: whose shape is unambiguous.
    words: int = 0
    reason: str = ""


@dataclass(frozen=True)
class AppliedEdit:
    """An edit that was actually written, with where it landed in the output."""

    edit: Edit
    out_start: int
    out_end: int
    #: The attributing label's span in OUTPUT coordinates. A label is never
    #: redacted, so it survives into the scrubbed text — shifted by whatever the
    #: edits before it did to the length.
    label_out: SourceSpan | None = None


def apply_edits(source: str, edits: list[Edit]) -> tuple[str, list[Edit]]:
    """Rewrite `source` at the given spans and nowhere else.

    Overlapping edits are merged rather than dropped, and the merged edit keeps
    the LONGEST replacement's kind, because a span claimed twice was claimed by
    two grammars that agree something is there. Dropping the second would leave
    part of it standing.

    Returns the rewritten text and the edits actually applied, in source order,
    so a caller's removal record describes the document that was produced and
    not the one that was proposed.
    """
    ordered = sorted(edits, key=lambda edit: (edit.span.start, -edit.span.end))
    merged: list[Edit] = []
    for edit in ordered:
        if edit.span.start == edit.span.end:
            continue
        if merged and edit.span.overlaps(merged[-1].span):
            previous = merged[-1]
            merged[-1] = Edit(
                span=previous.span.union(edit.span),
                replacement=previous.replacement,
                kind=previous.kind,
                label=previous.label or edit.label,
                certainty=(
                    "guessed"
                    if "guessed" in (previous.certainty, edit.certainty)
                    else "settled"
                ),
                words=max(previous.words, edit.words),
                reason=previous.reason or edit.reason,
            )
            continue
        merged.append(edit)

    out: list[str] = []
    cursor = 0
    for edit in merged:
        out.append(source[cursor : edit.span.start])
        out.append(edit.replacement)
        cursor = edit.span.end
    out.append(source[cursor:])
    return "".join(out), merged


def place(source: str, merged: list[Edit]) -> list[AppliedEdit]:
    """Where each applied edit, and its label, landed in the rewritten text.

    The Safe Handoff accounting reads the PROTECTED text, so it needs the spans
    in that text's coordinates rather than the source's. Deriving them here —
    from the same merged edit list that produced the output — is what stops the
    accounting from re-deriving "which of these characters is a placeholder" with
    a regex. `\\[[A-Z_]+\\]` was that regex, and a clinician's `Pacing mode:
    [DDDR]` satisfied it, so a pacing mode was accounted as a removed identifier
    and the line vanished from the measurement entirely.
    """
    applied: list[AppliedEdit] = []
    drift = 0
    for edit in merged:
        out_start = edit.span.start + drift
        out_end = out_start + len(edit.replacement)
        label_out = None
        if edit.label is not None:
            label_drift = _drift_at(merged, edit.label.start)
            label_out = SourceSpan(
                edit.label.start + label_drift, edit.label.end + label_drift
            )
        applied.append(AppliedEdit(edit, out_start, out_end, label_out))
        drift += len(edit.replacement) - (edit.span.end - edit.span.start)
    return applied


def _drift_at(merged: list[Edit], offset: int) -> int:
    drift = 0
    for edit in merged:
        if edit.span.end > offset:
            break
        drift += len(edit.replacement) - (edit.span.end - edit.span.start)
    return drift
