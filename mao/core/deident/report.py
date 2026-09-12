r"""What a scrub actually removed, as data rather than as a diff.

## Why the scrubber has to report

Every control in this project so far has re-derived "is there an identifier in
this text?" at whichever point it was standing. Each re-derivation is a grammar,
each grammar is a list, and the six waves before this one are a record of inputs
the list's author had not seen.

A removal record replaces the question. The scrubber already knows exactly which
spans it replaced — `layout` holds them as `_Claim`s and `freetext` produces them
as match objects — and throwing that away at the return statement is what forced
everything downstream to guess.

With the record kept, the egress boundary can ask a question that has an exact
answer: *is any of the text this request's boundary removed present in what is
about to be sent?* That is a lookup, not a judgement. It cannot be widened by a
caller, it cannot be forged by a marker in the document, and it holds for a
channel nobody remembered to scrub.

## Why it is not a second privacy wall on its own

A removal record can only contain what the scrubber found. It is therefore
exactly as complete as the detector, and it is not evidence that a document is
clean. Its value is different: it turns "the scrubber ran here" into "these
specific strings must not appear downstream", which is a property the sink can
check and the detector cannot.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Removal:
    """One span the scrubber replaced with a placeholder.

    `value` is the raw text that was removed, and it is sensitive — this object
    stays inside the protected plane. It is never logged, never serialised into
    a trace, and never put in an exception message; `mao/trust/egress` uses it
    for comparison only and reports the KIND when a comparison fires.

    The offsets are into the ORIGINAL, immutable source string, and they are the
    reason this record now exists in two places rather than one. Everything that
    needs to know what the transformation did — the Safe Handoff accounting, the
    clinician's redaction notice, the completeness report — reads THESE, so it
    cannot disagree with the transformation the way a second detector can.
    `ff34722` had exactly that disagreement: `find_ambiguities` announced one
    chat destruction and stayed silent on an identically shaped one.
    """

    kind: str
    value: str
    line: int
    #: Half-open offsets into the source string. `-1` only for a removal that
    #: predates the span pipeline — no production path produces one.
    start: int = -1
    end: int = -1
    #: The field label that attributed this removal, where one did. Offsets into
    #: the same source string.
    label_start: int = -1
    label_end: int = -1
    label: str = ""
    #: Whether the extent of this span was established or guessed. Carried from
    #: `layout.Certainty`, so the notice and the refusal read one answer.
    guessed: bool = False
    #: Whether the clinician must be TOLD about this removal. A SUPERSET of
    #: `guessed`: the refusal answers "where does the run end", which a
    #: following label or a mid-line stop can settle; the notice answers "may
    #: this span have taken more than the identifier", which nothing settles
    #: for a run of three or more name-shaped words.
    announce: bool = False
    #: How many name-shaped words the span covered, for a person value.
    words: int = 0
    #: Why the extent could not be established. Names no value — this string is
    #: returned over HTTP and written to a log.
    reason: str = ""

    def span(self) -> tuple[int, int]:
        return self.start, self.end

    def has_label(self) -> bool:
        return self.label_start >= 0 and self.label_end > self.label_start


@dataclass(frozen=True)
class RedactionEvent:
    """Where one redaction landed in the PROTECTED text. Carries no value.

    `Removal` is sensitive and stays in the protected plane. This is the same
    event with the value taken out and the offsets re-expressed in the scrubbed
    text's own coordinates, so it can travel with the `SafeDerivedText` it
    describes without putting an identifier inside a class whose whole purpose
    is not carrying one.

    Everything that needs to know what the transformation did reads these:

      - the Safe Handoff accounting, to decide which characters of a line are
        an identifier the boundary removed, which are the label that introduced
        it, and which are clinical content still to be accounted for;
      - the clinician's redaction notice, to say what was taken;
      - the completeness report, whose counts must describe the same events.

    One record, three readers. At `ff34722` the notice had a reader of its own —
    a second pass with its own predicate — and it announced one chat destruction
    while staying silent on an identically shaped one.
    """

    kind: str
    #: Half-open offsets into the SCRUBBED text.
    start: int
    end: int
    #: The attributing field label's span in the same coordinates, or -1.
    label_start: int = -1
    label_end: int = -1
    label: str = ""
    line: int = 0
    guessed: bool = False
    announce: bool = False
    words: int = 0
    reason: str = ""

    def has_label(self) -> bool:
        return self.label_start >= 0 and self.label_end > self.label_start


@dataclass(frozen=True)
class ScrubResult:
    """The de-identified text and the record of what produced it."""

    text: str
    removals: tuple[Removal, ...] = ()
    #: The same events, content-free and in the scrubbed text's coordinates.
    events: tuple[RedactionEvent, ...] = ()

    def kinds(self) -> tuple[str, ...]:
        """The identifier types removed. Safe to log — no values."""
        return tuple(sorted({removal.kind for removal in self.removals}))

    def guessed(self) -> tuple[Removal, ...]:
        """Removals whose extent the transformation could not establish.

        This is the single source of the clinician's redaction notice AND of the
        upload path's refusal. Both used to be derived from a second pass over
        the text, which is how one of them could fire while the other did not on
        the same string.
        """
        return tuple(removal for removal in self.removals if removal.guessed)
