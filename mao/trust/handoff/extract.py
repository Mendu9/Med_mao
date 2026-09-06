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
    #: Lines carrying clinical vocabulary that no rule above turned into a fact.
    #: Reported, not discarded: it is the measure of what the projection does
    #: NOT carry, and the compiler refuses when it is most of the document.
    uncovered: tuple[str, ...] = ()

    def coverage(self) -> float:
        """The share of clinically-marked lines that became a typed fact.

        1.0 when there was nothing clinical to lose. A low number means the
        projection would be a summary of a document it mostly did not read,
        which is what `00_RULES.md` forbids presenting as equivalent.
        """
        carried = len(self.medications) + len(self.findings) + len(self.vitals) + len(self.labs)
        total = carried + len(self.uncovered)
        return 1.0 if total == 0 else carried / total


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
    age_group = ""

    for line in lines:
        stripped = line.strip()
        if not stripped or _is_heading(stripped):
            continue

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

        if not matched and _is_clinical_line(stripped):
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
        uncovered=tuple(uncovered),
    )
