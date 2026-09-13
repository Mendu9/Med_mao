r"""Local, deterministic extraction of clinical facts from protected text.

## Why this is local and why it is deterministic

The approved M-1 policy names what no external model may receive, and
"raw/scrubbed free-text clinical reports" is on the list. That rules out the
route this system used: send the scrubbed note to `CLINICAL_SYNTHESIS` and ask
it for JSON. The extraction step was itself an egress of the whole document, so
migrating only the final synthesis call would have left the note leaving the
process one call earlier.

An extractor that runs in-process has no such problem, and it does not need to
be clever to be useful. `SafeSynthesisContext` is a MINIMUM-NECESSARY
projection: a handful of medications with their doses, the vitals and labs that
change a recommendation, the findings a clinician wrote down, and the question.
Those have shapes. Where a shape is not found, this module says so and the
compiler refuses rather than inventing one.

## What it deliberately does not do

It does not attempt to understand the note. It does not infer a diagnosis, it
does not normalise to an ontology, and it does not resolve negation — a line
saying "no bradycardia" contributes the line, and the model reasoning over it
sees the negation because the line is carried whole.

Phase 3 replaces this with the real structured pathway. `02_IMPLEMENTATION_PLAN`
allocates "full ProtectedCaseContext" and "structured clinical extraction"
there; §E of Phase 1 says the compiler "may be conservative", and this is what
conservative means: extract what has a shape, refuse the rest, and never let a
gap be filled by guessing.

## The honest limitation, stated

A fact this module does not recognise is absent from the projection, and a
projection missing a clinically material fact is the failure mode
`00_RULES.md` calls out — "a transformation may not silently delete clinically
material content and then present the result as equivalent to the original".
That is why `coverage()` exists and why the compiler refuses a projection that
covers too little of the source: the alternative is a confident answer about a
case the model was told half of.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from mao.core.deident.lexicon import NOT_A_NAME, has_clinical_head
from mao.core.deident.report import RedactionEvent
from mao.core.deident.text import split_lines
from mao.core.deident.values import _fold, _parts
from mao.trust.handoff.accounting import (
    CompletenessReport,
    SourceAccounting,
    account,
)

#: A dose: a number, a unit, and optionally a frequency.
_DOSE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|microgram|g|ml|units?|iu)\b"
    r"(?:\s*(od|bd|tds|qds|prn|once daily|twice daily|nocte|mane|daily|weekly))?",
    re.IGNORECASE,
)

#: A vital sign written the way a clinician writes it.
_VITALS = (
    ("heart_rate", re.compile(r"\b(?:hr|heart rate|pulse)\D{0,4}(\d{2,3})\b", re.I)),
    ("blood_pressure", re.compile(r"\b(?:bp|blood pressure)\D{0,4}(\d{2,3}/\d{2,3})\b", re.I)),
    ("temperature", re.compile(r"\b(?:temp|temperature)\D{0,4}(\d{2}(?:\.\d)?)\b", re.I)),
    ("oxygen_saturation", re.compile(r"\b(?:sats?|spo2|o2 sat\w*)\D{0,4}(\d{2,3})\s*%?", re.I)),
    ("respiratory_rate", re.compile(r"\b(?:rr|resp rate|respiratory rate)\D{0,4}(\d{1,2})\b", re.I)),
    ("weight", re.compile(r"\bweight\D{0,4}(\d{2,3}(?:\.\d)?)\s*kg\b", re.I)),
)

#: A laboratory or cognitive score. The named instruments are the ones a memory
#: clinic letter reports numerically, where the NUMBER is the clinical fact.
_LABS = (
    ("mmse", re.compile(r"\bmmse\D{0,4}(\d{1,2})\s*(?:/\s*30)?\b", re.I)),
    ("moca", re.compile(r"\bmoca\D{0,4}(\d{1,2})\s*(?:/\s*30)?\b", re.I)),
    ("acer", re.compile(r"\bace-?(?:iii|3|r)?\D{0,4}(\d{1,3})\s*(?:/\s*100)?\b", re.I)),
    ("inr", re.compile(r"\binr\D{0,4}(\d\.\d)\b", re.I)),
    ("egfr", re.compile(r"\begfr\D{0,4}(\d{1,3})\b", re.I)),
    ("hba1c", re.compile(r"\bhba1c\D{0,4}(\d{1,3})\b", re.I)),
    ("b12", re.compile(r"\b(?:b12|vitamin b12)\D{0,4}(\d{2,4})\b", re.I)),
    ("tsh", re.compile(r"\btsh\D{0,4}(\d+(?:\.\d+)?)\b", re.I)),
    ("creatinine", re.compile(r"\bcreatinine\D{0,4}(\d{1,3})\b", re.I)),
    ("sodium", re.compile(r"\b(?:na|sodium)\D{0,4}(\d{3})\b", re.I)),
)

#: An age band rather than an age. `SafeSynthesisContext` carries `age_group`
#: precisely because an exact age is an identifier under HIPAA Safe Harbor over
#: 89 and is rarely what changes a recommendation below it.
_AGE = re.compile(r"\b(?:age[ds]?|aged)\D{0,4}(\d{1,3})\b", re.I)

#: Section headings are no longer excluded HERE.
#:
#: The old `_is_heading` matched a Titlecase line ending in a colon and dropped
#: it before anything counted it, which is one more "exclude before measuring"
#: branch of exactly the kind `accounting` exists to end — and the exclusion was
#: never safe: `Complete Heart Block:`, `Sick Sinus Syndrome:` and `Mild
#: Cognitive Impairment:` are all Titlecase noun phrases ending in a colon, and
#: the first is a contraindication to the drug this system is asked about.
#:
#: Nor are they excluded in `accounting`, which briefly recognised structure by
#: reading the `#` of a markdown heading until both `46a198a` reviewers measured
#: `# Permanent pacemaker in situ` out of a payload that still called itself
#: complete (AR18-1 / ADV18-1). Structure is now attributable ONLY from the
#: transformation's own record — the field label an event says introduced the
#: identifier it removed — never from how a line is spelled.
#:
#: So a heading-shaped line is carried as a finding if the extractor recognises
#: it and is UNRESOLVED if it does not, and the module's own asymmetry argument
#: says that is the right way round: carrying `Current medications:` into the
#: findings list adds a word to a prompt, and dropping `Complete Heart Block`
#: removes a contraindication. The cost is that `## Findings` is now reported
#: unresolved — pessimism in the safe direction, which is the R-2 trade the
#: control plane has already accepted.

#: What the caller was asking. Kept as the clinical question when nothing more
#: specific is available.
_QUESTION = re.compile(r"[^.?!]*\?")


def age_group_for(age: int) -> str:
    """Bands, not years. Wide enough that the band is not an identifier."""
    if age >= 90:
        return "90_or_over"
    if age >= 75:
        return "older_adult_75_89"
    if age >= 65:
        return "older_adult_65_74"
    if age >= 18:
        return "adult"
    return "under_18"


@dataclass(frozen=True)
class ExtractedFacts:
    """What a deterministic pass could establish, and where the rest went."""

    age_group: str = ""
    medications: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()
    vitals: tuple[tuple[str, str], ...] = ()
    labs: tuple[tuple[str, str], ...] = ()
    clinical_question: str = ""
    #: Every segment of the source, in exactly one state. The account replaces
    #: the content-line denominator AND the exclusion predicate that used to sit
    #: in front of it: see `mao/trust/handoff/accounting.py` for why a fourth
    #: predicate was not the available move.
    accounting: SourceAccounting = field(default_factory=SourceAccounting)

    def completeness(self) -> CompletenessReport:
        """The account with every piece of source text removed.

        This is what may travel. `accounting` may not.
        """
        return self.accounting.report()

    def accounted_against(
        self, represented: Callable[[str, str], bool]
    ) -> ExtractedFacts:
        """These facts with the account re-stated against the shipped projection.

        The extractor's own view of what it carried is a CLAIM about an
        intermediate object. What ships is the compiled projection, in which a
        caller-supplied structured field may have displaced an extracted one, so
        the claim has to be put to the projection before it can be reported as
        completeness. See `accounting.against` and ADV19-1.

        The typed fields are untouched: this changes what is ACCOUNTED, never
        what was extracted.
        """
        return replace(self, accounting=self.accounting.against(represented))

    def coverage(self) -> float:
        """The share of accountable segments the projection carries."""
        return self.completeness().coverage()

    def is_complete(self) -> bool:
        return self.completeness().complete

    def unresolved_text(self) -> tuple[str, ...]:
        """The residue, for LOCAL measurement only. Never an egress payload."""
        return self.accounting.unresolved_text()


def _is_clinical_line(line: str) -> bool:
    """Whether this line says something clinical, by the shared lexicon.

    Uses the SAME vocabulary the de-identifier uses, on purpose: a line that
    vocabulary calls clinical is a line the redaction machinery was willing to
    protect, so a projection that drops it is dropping something the system
    already decided was clinical content.
    """
    words = re.findall(r"[^\W\d_]+", line)
    if not words:
        return False
    if any(part in NOT_A_NAME for word in words for part in _parts(word) if part):
        return True
    return has_clinical_head([_fold(word) for word in words])


def extract(
    text: str,
    question: str = "",
    events: tuple[RedactionEvent, ...] = (),
) -> ExtractedFacts:
    """Everything a deterministic pass can establish, plus where the rest went.

    `events` are the redactions the boundary actually performed on this text. A
    caller that has them gets an account in which an identifier the boundary
    removed and the label that introduced it are recognised as such — which is
    what keeps an ordinary letterhead from collapsing to a refusal without
    excluding anything from the measurement.

    A caller without them still gets a truthful account, and a more pessimistic
    one: with no record of a removal, `Patient Name: [NAME]` is meaning-bearing
    text the projection does not carry, so it counts as UNRESOLVED and coverage
    falls. That direction is the safe one, and it is the reason the parameter
    is not required: a missing record can never make a projection look MORE
    complete than it is.
    """
    lines, _ = split_lines(text)

    medications: list[str] = []
    findings: list[str] = []
    vitals: dict[str, str] = {}
    labs: dict[str, str] = {}
    carried: dict[int, list[tuple[int, int, str, str]]] = {}
    age_group = ""

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        #: Where `stripped` sits inside `line`, so every span recorded below is
        #: in the LINE's coordinates and not the stripped copy's.
        shift = line.index(stripped) if stripped else 0
        #: `(start, end, field, represents)`. `represents` is the VALUE the
        #: field would carry for this span, which is not always the span's text:
        #: an age ships as a band and a vital ships as a number. The account
        #: later puts that value to the projection that actually ships, so
        #: recording it here is what makes the TYPED_FACT claim checkable.
        spans: list[tuple[int, int, str, str]] = []

        # A rule that carries a NUMBER records the number's span. A rule that
        # carries the LINE records the whole line. The distinction is the whole
        # of the sub-line accounting fix: `Contraindicated per cardiology, HR
        # 36` gives `heart_rate 36` and nothing else, so `Contraindicated per
        # cardiology` is residue and must be accounted as such rather than
        # covered by its neighbour's success.
        age = _AGE.search(stripped)
        if age is not None:
            try:
                # The BAND this span represents, not the band that is kept. A
                # second, differing age on a later line represents a band the
                # projection does not carry, and saying so is the same property
                # as saying so when a caller's `age` displaces it.
                band = age_group_for(int(age.group(1)))
                age_group = age_group or band
                spans.append(
                    (shift + age.start(), shift + age.end(), "age_group", band)
                )
            except ValueError:  # pragma: no cover - the pattern guarantees digits
                pass

        for name, pattern in _VITALS:
            found = pattern.search(stripped)
            if found is not None and name not in vitals:
                vitals[name] = found.group(1)
                spans.append(
                    (
                        shift + found.start(),
                        shift + found.end(),
                        "vitals",
                        f"{name}={found.group(1)}",
                    )
                )

        for name, pattern in _LABS:
            found = pattern.search(stripped)
            if found is not None and name not in labs:
                labs[name] = found.group(1)
                spans.append(
                    (
                        shift + found.start(),
                        shift + found.end(),
                        "labs",
                        f"{name}={found.group(1)}",
                    )
                )

        if _DOSE.search(stripped):
            medications.append(stripped)
            spans = [(shift, shift + len(stripped), "medications", stripped)]
        elif _is_clinical_line(stripped):
            findings.append(stripped)
            spans = [(shift, shift + len(stripped), "findings", stripped)]

        if spans:
            carried[index] = spans

    clinical_question = question.strip()
    if not clinical_question:
        found = _QUESTION.search(text)
        clinical_question = found.group().strip() if found else ""

    return ExtractedFacts(
        age_group=age_group,
        medications=tuple(dict.fromkeys(medications)),
        findings=tuple(dict.fromkeys(findings)),
        vitals=tuple(sorted(vitals.items())),
        labs=tuple(sorted(labs.items())),
        clinical_question=clinical_question,
        accounting=account(text, events, carried),
    )
