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
from dataclasses import dataclass

from mao.core.deident.lexicon import NOT_A_NAME, has_clinical_head
from mao.core.deident.text import split_lines
from mao.core.deident.values import _fold, _parts

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

#: A line that is a section heading rather than a fact.
#:
#: It must END IN A COLON (or be a markdown heading). Matching a bare Titlecase
#: line instead was the same mistake this project has made in five other places:
#: `Complete Heart Block`, `Sick Sinus Syndrome` and `Mild Cognitive Impairment`
#: are all Titlecase noun phrases, all section-heading-shaped, and all cardiac
#: or cognitive findings — the first is a contraindication to the drug this
#: system is asked about. Requiring the colon costs nothing (a real heading has
#: one, or is `## Findings`) and `_is_clinical_line` is consulted as a second
#: guard, so a colon-terminated clinical line is still carried.
_HEADING = re.compile(r"^\s*(?:#+\s*[A-Z][A-Za-z /]{2,30}|[A-Z][A-Za-z /]{2,30}:)\s*$")


#: Markdown heading syntax. Explicit structure a clinician typed deliberately,
#: so it needs no tie-break: `## Findings` is a heading whatever words follow.
_MARKDOWN_HEADING = re.compile(r"^\s*#+\s*\S")


def _is_heading(line: str) -> bool:
    """Structure, not content. Clinical vocabulary wins every ambiguous tie.

    A line that is heading-SHAPED but says something clinical is kept, because
    the cost of the two errors is not symmetric: carrying `Current medications:`
    into the findings list adds a word to a prompt, and dropping
    `Complete Heart Block` removes a contraindication to the drug under review.
    """
    if _MARKDOWN_HEADING.match(line):
        return True
    return _HEADING.match(line) is not None and not _is_clinical_line(line)

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
    """What a deterministic pass could establish, and how much it covered."""

    age_group: str = ""
    medications: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()
    vitals: tuple[tuple[str, str], ...] = ()
    labs: tuple[tuple[str, str], ...] = ()
    clinical_question: str = ""
    #: Every non-empty, non-heading line the document had. The DENOMINATOR, and
    #: deliberately a count of document content rather than of lexicon hits.
    content_lines: int = 0
    #: Content lines that no rule above turned into a typed fact.
    #: Reported, not discarded: it is the measure of what the projection does
    #: NOT carry, and the compiler refuses when it is most of the document.
    uncovered: tuple[str, ...] = ()

    def coverage(self) -> float:
        """The share of the document's content lines that became a typed fact.

        The denominator is DOCUMENT CONTENT, and that is the whole of the fix.

        It used to be `carried + len(uncovered)`, where `uncovered` was
        populated only from lines `_is_clinical_line` marked clinical - the same
        lexicon the de-identifier uses. A clinical line that lexicon does not
        recognise was therefore in neither the numerator nor the denominator:
        not "uncovered" but invisible, so this returned 1.00 exactly when the
        loss was total (ADV16-6). The guard was bound to the detector it exists
        to check, which is the one thing a guard may not be. Separately, the
        branch that populated `uncovered` at all was unreachable, so the value
        was unconditionally empty and every caller read a constant 1.0 (A-1).

        A line count is a measurement. It cannot be defeated by a word nobody
        put in a list, which is the property six remediation rounds could not
        obtain from a vocabulary - and widening that vocabulary again is what
        `00_RULES.md` forbids rather than what it asks for.

        1.0 when the document had no content lines: there was genuinely nothing
        to lose. NOT 1.0 when there was content and none of it was carried,
        which is the case that mattered.
        """
        if self.content_lines == 0:
            return 1.0
        return (self.content_lines - len(self.uncovered)) / self.content_lines


#: A redaction placeholder this system's own boundary emitted.
_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")


def _carries_nothing_to_lose(line: str) -> bool:
    """Whether this line had a value at all once its identifier was removed.

    `Patient Name: [NAME]` and `MRN: [MRN]` are lines whose entire value was an
    identifier. The boundary removed it, correctly, and what remains is a label
    with nothing behind it. Such a line cannot contribute clinical content to a
    projection, so counting it in `coverage()`'s denominator would measure the
    de-identifier's success as the extractor's failure - and on an ordinary
    letterhead, where most lines are exactly this shape, it collapses coverage
    and refuses documents that are perfectly answerable. The architecture
    review predicted this precise trap in its A-1 remediation note.

    This is a STRUCTURAL test and not a vocabulary one, which is what makes it
    admissible here at all. It asks whether a placeholder THIS SYSTEM emitted
    accounts for the whole of the line's value. It does not ask what the line
    is about, consults no lexicon, and cannot be widened by adding words.
    """
    _, separator, value = line.partition(":")
    candidate = value if separator else line
    return not _PLACEHOLDER.sub("", candidate).strip(" \t.,;|-_")


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


def extract(text: str, question: str = "") -> ExtractedFacts:
    """Everything a deterministic pass can establish from protected text."""
    lines, _ = split_lines(text)

    medications: list[str] = []
    findings: list[str] = []
    vitals: dict[str, str] = {}
    labs: dict[str, str] = {}
    uncovered: list[str] = []
    content_lines = 0
    age_group = ""

    for line in lines:
        stripped = line.strip()
        if not stripped or _is_heading(stripped):
            continue
        if _carries_nothing_to_lose(stripped):
            # A label whose only value was an identifier the boundary removed.
            # Neither carried nor lost: there was nothing there to carry.
            continue
        content_lines += 1

        matched = False

        age = _AGE.search(stripped)
        if age is not None:
            try:
                age_group = age_group or age_group_for(int(age.group(1)))
                matched = True
            except ValueError:  # pragma: no cover - the pattern guarantees digits
                pass

        for name, pattern in _VITALS:
            found = pattern.search(stripped)
            if found is not None and name not in vitals:
                vitals[name] = found.group(1)
                matched = True

        for name, pattern in _LABS:
            found = pattern.search(stripped)
            if found is not None and name not in labs:
                labs[name] = found.group(1)
                matched = True

        if _DOSE.search(stripped):
            medications.append(stripped)
            matched = True
        elif _is_clinical_line(stripped):
            findings.append(stripped)
            matched = True

        if not matched:
            # Not "and _is_clinical_line(stripped)". That conjunct was
            # unsatisfiable - the `elif` above has already set `matched` for
            # every line the lexicon calls clinical - so `uncovered` was
            # unconditionally empty and the whole thin-projection refusal was
            # dead code (A-1). Asking the lexicon again here would also
            # reintroduce ADV16-6, because the question is whether this line
            # became a fact, and that is answered by `matched`, not by a
            # vocabulary that has never seen the words in question.
            uncovered.append(stripped)

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
        content_lines=content_lines,
        uncovered=tuple(uncovered),
    )
