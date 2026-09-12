r"""Where de-identification had to GUESS, and what to do about it.

## Why this module exists

Three remediation rounds failed on one irreducible question: given a run of
capitalised words beside a person label, is it a name or clinical content?
`Gordon Whitfield` and `Rockwood Frailty Scale` are the same shape. Every
mechanism tried decided it from a list somebody wrote — a clinical stop-list
(46/60 clinical phrases destroyed), a given-name gazetteer (49.3% of names
leaked), clinical head nouns (72.9% destroyed) — and each failed in one
direction because no such list is complete. Local NER was measured too: 24/30
name recall, 2/30 false positives. Better, still leaky.

There is no vocabulary-free rule and no complete list, so the scrubber must stop
pretending it knows. What it CAN do reliably is say when it did not know.

## The policy this implements

Ambiguity is resolved differently depending on who can see the outcome:

  UPLOAD PATH — the report is processed unseen, so a wrong guess is silent. An
  ambiguous header is REFUSED, and the caller is asked for structured patient
  fields instead of having them inferred from extracted text. The route answers
  422 and names the fields.

  CHAT PATH — the clinician wrote the text and reads the answer, so an
  over-redaction is visible and recoverable while a leak is not. Ambiguity
  resolves by REDACTING, and the whole run is taken rather than a prefix,
  because a placeholder covering PART of an identifier leaks the rest while
  asserting it did not.

Neither path guesses a boundary. That is the point, and it is what makes both
invariants satisfiable on the path where they must both hold.

## What changed, and why it had to

This module used to RE-DERIVE the question with a rule of its own, narrower than
the scrubber's. The two halves then disagreed about the same string:

    'Patient Name: Harold Nkemdirim Rockwood Frailty'
      find_ambiguities  tail is empty -> "unambiguous"
      values.value_at   "a header value runs to the end of its line" -> take it
      result            the instrument's name is DELETED, unseen, no refusal

and, measured across a hold-out corpus, the refusal fired on 9 of 60 documents —
exactly the 9 whose last word the clinical lexicon happened to contain. *The
refusal fired when the lexicon already solved the problem and did not fire when
it did not.* An independent review then showed the refused set was a
98%-precision, 100%-recall oracle for the documents that would FAIL the suite: a
quarantine tuned, in effect, to remove the failures.

So there is now ONE decision procedure. `layout.build_claims` marks every claim
`SETTLED` or `GUESSED`, this module reports the GUESSED ones, and the scrubber
redacts all of them. The refusal and the redaction cannot disagree about a
string because they are reading the same answer.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .report import RedactionEvent


class AmbiguousDocument(Exception):
    """An uploaded document whose header cannot be de-identified confidently."""

    def __init__(self, report: AmbiguityReport) -> None:
        super().__init__(report.describe())
        self.report = report


@dataclass(frozen=True)
class Ambiguity:
    """One decision the scrubber could not make on evidence.

    Deliberately carries NO text from the document. This object is raised as an
    exception, and the exception's message is logged and returned to the caller
    — so quoting the span here put the patient's name into the application log
    and the HTTP body, on the one code path whose entire purpose is that the
    name could not be safely handled. The position and the shape are enough to
    act on; the content is exactly what must not travel.
    """

    line: int
    words: int
    label: str
    reason: str
    #: The line carrying the label, when it is not `line` itself. A two-column
    #: extraction puts the field and the content it protects on different rows,
    #: and a refusal that names only one of them looks like a blanket over a
    #: failure somewhere else.
    label_line: int = -1


@dataclass
class AmbiguityReport:
    """Every decision that had to be guessed, in document order."""

    items: list[Ambiguity] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.items)

    def covered_lines(self) -> set[int]:
        """Every line this refusal is answerable for.

        A refusal must NAME what it is protecting, or it cannot be told from a
        blanket. Both the field that could not be resolved and the content line
        it was about are included, because in a two-column extraction they are
        different rows and excluding either would make an honest refusal look
        like an oracle — or let an oracle look honest.
        """
        lines = {item.line for item in self.items}
        lines |= {item.label_line for item in self.items if item.label_line >= 0}
        return lines

    def describe(self) -> str:
        """A description safe to log and to return to the caller.

        Names the LINE and the FIELD, never the value. A caller who sent the
        document can find the line; a log reader who did not, cannot learn
        anything from it.
        """
        if not self.items:
            return "no ambiguous fields"
        return "; ".join(
            f"line {item.line + 1}, field {item.label!r}: {item.reason}"
            for item in self.items
        )


def ambiguities_from(events: Sequence[RedactionEvent]) -> AmbiguityReport:
    """The report, built from redactions that ACTUALLY HAPPENED.

    This is the whole of the "unify redaction evidence" decision. The notice a
    clinician reads, the 422 an upload path answers with, and the completeness
    accounting now all consume one event list — the one the transformation
    emitted while it was rewriting the text — rather than each re-deriving the
    question from the characters.

    At `ff34722` they did not. `find_ambiguities` re-normalised the text and
    re-ran the claim builder, and the two halves disagreed in measured, opposite
    directions on identically shaped input:

        'Patient Name: Sarah May Okonkwo Complete Heart Block.'
            -> 'Patient Name: [NAME].'          redaction_notice = []
        'Patient Name: Sarah May Okonkwo Rockwood Frailty Scale 6.'
            -> 'Patient Name: [NAME] 6.'        redaction_notice = [...]

    Whether a destruction was announced turned out to be a function of which
    clinical phrase was destroyed, which is a lexicon property — the ADV16-6
    shape, reappearing in the announcement instead of in the measurement.
    """
    report = AmbiguityReport()
    for event in sorted(
        (event for event in events if event.guessed or event.announce),
        key=lambda event: event.start,
    ):
        report.items.append(
            Ambiguity(
                line=event.line,
                words=event.words,
                label=event.label,
                reason=event.reason,
                label_line=event.line,
            )
        )
    return report


def unresolved_from(events: Sequence[RedactionEvent]) -> AmbiguityReport:
    """Only the removals whose EXTENT could not be established.

    The upload path refuses on this and the chat path announces on
    `ambiguities_from`, which is a strict superset. The two are separated
    because they answer different questions:

        refuse    "can this document be de-identified without guessing?"
                  A following field label or a mid-line sentence terminator
                  genuinely bounds the run, so an ordinary
                  `Patient Name: John Michael Smith MRN: ...` banner is not
                  refused - and refusing on the shape every real patient
                  banner has would be a broken product rather than a policy.

        announce  "may this removal have taken more than the identifier?"
                  True of every run of three or more name-shaped words,
                  bounded or not, because the boundary between the name and
                  whatever follows it inside the run is what was never
                  established.

    Collapsing them is what made the notice depend on punctuation: a newline
    after the full stop announced and a space silenced, on the same
    destruction.
    """
    return ambiguities_from(
        [event for event in events if event.guessed]
    )


def find_ambiguities(text: str) -> AmbiguityReport:
    """Every point where a person value's identity or extent had to be guessed.

    Runs the SCRUBBER and reads what it did, rather than re-deriving the
    question with a pass of its own. A caller that has already scrubbed should
    use `ambiguities_from` with the events it already holds; this convenience
    exists for callers that have only the text, and it is defined as the same
    computation so the two cannot drift.
    """
    if not text:
        return AmbiguityReport()
    from mao.core.pii_scrubber import scrub_with_report

    return ambiguities_from(scrub_with_report(text).events)
