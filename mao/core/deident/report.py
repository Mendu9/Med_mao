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
    """

    kind: str
    value: str
    line: int


@dataclass(frozen=True)
class ScrubResult:
    """The de-identified text and the record of what produced it."""

    text: str
    removals: tuple[Removal, ...] = ()

    def kinds(self) -> tuple[str, ...]:
        """The identifier types removed. Safe to log — no values."""
        return tuple(sorted({removal.kind for removal in self.removals}))
