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

from mao.core.deident.freetext import redact_by_shape_with_report
from mao.core.deident.layout import redact_labelled_fields_with_report
from mao.core.deident.report import Removal, ScrubResult
from mao.core.deident.text import normalise, strip_leading_bom

__all__ = ["ScrubResult", "scrub_pii", "scrub_with_report"]


def scrub_with_report(text: str) -> ScrubResult:
    """`scrub_pii`, plus the record of every span it replaced.

    The record is what lets the egress boundary assert, at the real sink, that
    no identifier THIS request's boundary removed is in an outgoing payload.
    That check is a lookup against what was actually found, not another grammar
    applied to the characters — which is the difference between it and the six
    detectors that preceded it. See `mao/core/deident/report.py`.
    """
    if not text:
        return ScrubResult(text=text)
    bom, body = strip_leading_bom(text)
    body = normalise(body)
    body, labelled = redact_labelled_fields_with_report(body)
    body, shaped = redact_by_shape_with_report(body)
    removals: list[Removal] = [*labelled, *shaped]
    return ScrubResult(text=bom + body, removals=tuple(removals))


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
