r"""Purpose-specific safe projections from protected clinical facts.

    ProtectedCaseContext
      -> SafeEvidenceQuery       # minimum facts needed to FIND evidence
      -> SafeSynthesisContext    # minimum facts needed to APPLY evidence

Neither is a redacted copy of the note, and that is the whole point of having
two of them. Six waves tried to make one rewritten document safe for every
destination; the projections differ because the destinations need different
things, and a rewritten note is the wrong shape for both.

## Refusal is a first-class outcome

`05_CASE_TO_EVIDENCE_SPEC.md`:

    If a safe evidence or synthesis projection cannot be established without
    either identifier leakage or clinically material information loss,
    clarify/request structured input or refuse that external handoff.

So `HandoffRefused` carries the structured fields that would resolve it. A
refusal a caller cannot act on is an outage with a polite message; one that
names the three fields it needs is a request.

Two conditions refuse:

  *empty* — the projection carries no clinical fact at all. Sending it would ask
  a model to reason about a case it has been told nothing about, and the answer
  would be generic advice wearing the appearance of case-specific advice.

  *thin* — the projection carries some facts but leaves most of the document's
  clinical lines behind. This is the condition `00_RULES.md` names: "a
  transformation may not silently delete clinically material content and then
  present the result as equivalent to the original". The compiler cannot know
  whether the missing lines mattered, and confidently answering anyway is
  exactly the failure it must not have.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from mao.trust.classes import (
    ProtectedCaseContext,
    SafeEvidenceQuery,
    SafeSynthesisContext,
)
from mao.trust.handoff.extract import ExtractedFacts, age_group_for, extract

logger = logging.getLogger(__name__)

#: Below this share of clinically-marked lines carried into the projection, the
#: projection is a summary of a document it mostly did not read.
#:
#: Not tuned to make a test pass: it is the point at which the majority of what
#: the shared clinical vocabulary marked as content is absent, and answering
#: from a minority of a clinical document is the behaviour the rules forbid.
MINIMUM_COVERAGE = 0.5


class HandoffRefused(Exception):
    """No safe projection could be established. Names what would fix it."""

    def __init__(self, reason: str, wanted: tuple[str, ...]) -> None:
        super().__init__(
            f"{reason} Send these as structured patient fields and the request "
            f"can be answered: {', '.join(wanted)}."
        )
        self.reason = reason
        self.wanted = wanted


#: The structured fields that resolve a refusal. Named, not "provide more
#: detail": a caller can act on a list of field names.
_WANTED = (
    "clinical_question",
    "age",
    "conditions",
    "medications (with dose)",
    "findings",
    "vitals",
    "labs",
)


def case_from_facts(
    facts: ExtractedFacts,
    *,
    structured: dict[str, str] | None = None,
    question: str = "",
    jurisdiction: str = "",
    provenance: tuple[str, ...] = (),
) -> ProtectedCaseContext:
    """Assemble the protected case from typed facts and structured fields.

    Structured fields WIN over extracted ones. The caller stating a patient's
    age is better evidence than a regex finding a two-digit number near the word
    "aged", and the structured pathway is what the ambiguity policy points a
    refused caller at — so it has to be worth using.
    """
    structured = structured or {}
    age_group = facts.age_group
    if structured.get("age"):
        try:
            age_group = age_group_for(int(str(structured["age"]).strip()))
        except ValueError:
            logger.debug("structured age %r is not a number", structured.get("age"))

    def _listed(key: str) -> tuple[str, ...]:
        raw = structured.get(key, "")
        return tuple(part.strip() for part in raw.split(";") if part.strip())

    return ProtectedCaseContext(
        case_id_internal=uuid.uuid4().hex,
        clinical_question=(
            question.strip()
            or structured.get("clinical_question", "").strip()
            or facts.clinical_question
        ),
        demographics={"age_group": age_group} if age_group else {},
        conditions=_listed("conditions"),
        medications=_listed("medications") or facts.medications,
        allergies=_listed("allergies"),
        findings=_listed("findings") or facts.findings,
        vitals=dict(facts.vitals),
        labs=dict(facts.labs),
        jurisdiction=jurisdiction or structured.get("jurisdiction", ""),
        uncertainties=facts.uncovered,
        source_provenance=provenance,
    )


def _require_substance(context: ProtectedCaseContext, facts: ExtractedFacts) -> None:
    if not context.has_clinical_facts():
        raise HandoffRefused(
            "No clinical fact could be established from this request without "
            "sending the document itself, which the egress policy does not "
            "permit.",
            _WANTED,
        )
    coverage = facts.coverage()
    if coverage < MINIMUM_COVERAGE:
        logger.warning(
            "safe projection covers %.0f%% of the clinical lines — refusing",
            coverage * 100,
        )
        raise HandoffRefused(
            f"Only {coverage:.0%} of this document's clinical content could be "
            "carried into a payload that excludes direct identifiers, so an "
            "answer would be based on a minority of the case.",
            _WANTED,
        )


def compile_evidence_query(
    context: ProtectedCaseContext,
    facts: ExtractedFacts,
    *,
    freshness: str = "",
) -> SafeEvidenceQuery:
    """The minimum needed to FIND evidence.

    Concepts, not sentences. A finding is carried as the line the clinician
    wrote because normalising to an ontology is Phase 3 work and a wrong
    normalisation is worse than a verbatim clinical phrase — but the phrase is
    only here because it survived the protected boundary, and the boundary
    removed the identifiers from it.
    """
    _require_substance(context, facts)
    return SafeEvidenceQuery(
        query_id=uuid.uuid4().hex,
        search_intent=context.clinical_question or "clinical evidence",
        population={k: v for k, v in context.demographics.items() if v},
        condition=context.conditions,
        intervention=context.medications,
        outcomes=(),
        findings=context.findings,
        jurisdiction=context.jurisdiction,
        freshness=freshness,
        provenance=tuple(
            name
            for name, present in (
                ("demographics", bool(context.demographics)),
                ("conditions", bool(context.conditions)),
                ("medications", bool(context.medications)),
                ("findings", bool(context.findings)),
            )
            if present
        ),
    )


def compile_synthesis_context(
    context: ProtectedCaseContext, facts: ExtractedFacts
) -> SafeSynthesisContext:
    """The minimum needed to APPLY evidence to this case.

    Built field by field from `context`. There is deliberately no parameter
    through which a caller could pass free text, and no branch in which the note
    is substituted for the projection — which is what makes "external clinical
    synthesis receives only SafeSynthesisContext" a property of the type rather
    than a rule somebody has to remember.
    """
    _require_substance(context, facts)
    return SafeSynthesisContext(
        context_id=uuid.uuid4().hex,
        clinical_question=context.clinical_question or "clinical decision support",
        age_group=context.demographics.get("age_group", ""),
        conditions=context.conditions,
        medications_and_doses=context.medications,
        allergies=context.allergies,
        vitals=dict(context.vitals),
        labs=dict(context.labs),
        clinically_relevant_findings=context.findings,
        imaging_findings=context.imaging_findings,
        risk_factors=context.risk_factors,
        jurisdiction=context.jurisdiction,
        uncertainties=context.uncertainties,
        provenance=context.source_provenance,
    )


@dataclass(frozen=True)
class SafeHandoff:
    """Both projections plus the facts they were built from."""

    facts: ExtractedFacts
    case: ProtectedCaseContext
    evidence_query: SafeEvidenceQuery
    synthesis_context: SafeSynthesisContext


def compile_handoff(
    protected_text: str,
    *,
    question: str = "",
    structured: dict[str, str] | None = None,
    jurisdiction: str = "",
    provenance: tuple[str, ...] = (),
) -> SafeHandoff:
    """Protected text and structured fields in, two safe projections out.

    Raises `HandoffRefused` rather than producing a projection that would be
    empty or would leave most of the document behind.
    """
    facts = extract(protected_text, question=question)
    case = case_from_facts(
        facts,
        structured=structured,
        question=question,
        jurisdiction=jurisdiction,
        provenance=provenance,
    )
    return SafeHandoff(
        facts=facts,
        case=case,
        evidence_query=compile_evidence_query(case, facts),
        synthesis_context=compile_synthesis_context(case, facts),
    )
