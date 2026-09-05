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
  fields instead of having them inferred from extracted text. `clinical_agent`
  raises `AmbiguousDocument` and the route answers 422.

  CHAT PATH — the clinician wrote the text and reads the answer, so an
  over-redaction is visible and recoverable while a leak is not. Ambiguity
  resolves by REDACTING, and the whole run is taken rather than a prefix,
  because a placeholder covering PART of an identifier leaks the rest while
  asserting it did not.

Neither path guesses a boundary. That is the point, and it is what makes both
invariants satisfiable on the path where they must both hold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .fields import FieldType
from .lexicon import has_clinical_head
from .text import split_lines
from .values import _fold, name_tokens


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


@dataclass
class AmbiguityReport:
    """Every decision that had to be guessed, in document order."""

    items: list[Ambiguity] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.items)

    def describe(self) -> str:
        """A description safe to log and to return to the caller.

        Names the LINE and the FIELD, never the value. A caller who sent the
        document can find the line; a log reader who did not, cannot learn
        anything from it.
        """
        if not self.items:
            return "no ambiguous fields"
        return "; ".join(
            f"line {item.line + 1}, field {item.label!r}: {item.words} "
            f"name-shaped words — {item.reason}"
            for item in self.items
        )


#: A name run this long with no clinical signal inside it and more content after
#: it on the same line is the shape that cannot be bounded. Two words is an
#: ordinary forename and surname and needs no adjudication; three or more,
#: followed by yet more words, is `Harold Nkemdirim Rockwood Frailty` — where the
#: name stops is exactly what nothing can tell us.
_UNBOUNDED_NAME_WORDS = 3

_SENTENCE_END = ".?!;"


def find_ambiguities(text: str) -> AmbiguityReport:
    """Every point where a person value's extent could not be established.

    Deliberately narrow. It reports the ONE case that defeated three rounds —
    an unpunctuated run of name-shaped words continuing into more content — and
    not every judgement the scrubber makes, because a refusal that fires on
    ordinary documents is a refusal nobody can ship.
    """
    from .fields import find_labels

    report = AmbiguityReport()
    lines, _ = split_lines(text)
    for index, line in enumerate(lines):
        for _start, end, field_type in find_labels(line):
            if field_type is not FieldType.NAME:
                continue
            position = end
            while position < len(line) and line[position] in " \t:：|=#-":
                position += 1
            tokens = name_tokens(line, position)
            words = [token for token in tokens if token[0] == "word"]
            if len(words) < _UNBOUNDED_NAME_WORDS:
                continue
            tail = line[tokens[-1][2] :]
            # A sentence terminator ends the value unambiguously; so does the
            # end of the line. Only an unpunctuated continuation is unresolvable.
            if not tail.strip() or tail.lstrip()[:1] in _SENTENCE_END:
                continue
            # ...and so does the NEXT FIELD. `Patient Name: John Michael Smith
            # MRN: RGT/44219/B` is a perfectly ordinary banner: the name ends
            # where the next label begins, which `name_tokens` already knows.
            # Refusing it made the quarantine fire on the shape real patient
            # banners always have — 30/30 in review — and a refusal that common
            # is a broken product rather than a policy.
            if find_labels(tail):
                continue
            if has_clinical_head([_fold(line[s:e]) for _, s, e in tokens]):
                continue
            report.items.append(
                Ambiguity(
                    line=index,
                    words=len(words),
                    label=" ".join(line[_start:end].split()),
                    reason=(
                        "followed by more text with no punctuation and no "
                        "following field — the end of the name cannot be established"
                    ),
                )
            )
    return report
