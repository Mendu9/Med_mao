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
from mao.core.deident.report import RedactionEvent
from mao.trust.handoff.accounting import CompletenessReport
from mao.trust.handoff.effective import EffectiveProjection
from mao.trust.handoff.extract import ExtractedFacts, extract

logger = logging.getLogger(__name__)

#: Below this share of ACCOUNTABLE segments carried into the projection, the
#: projection is a summary of a document it mostly did not read.
#:
#: A JUDGEMENT, and recorded as one. It is deliberately NOT re-tuned: tuning it
#: to make one example refuse would make the threshold a function of whichever
#: examples someone happened to try, which is the failure mode this phase exists
#: to end. Phase 2's Evaluation Lab owes this number a measurement.
#:
#: It is NOT what closes the false-completeness class, and the distinction
#: matters because the threshold argument has now been defeated twice. Both
#: `ff34722` reviews observed that where the loss was invisible, coverage was
#: 1.00 — so no threshold at any value would have refused. What closes the class
#: is `_require_complete_projection` below, which does not consult a number at
#: all: a projection with any unresolved segment is never REPRESENTED as
#: complete, on any path, to any reader.
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

    What CHANGED at ADV19-1 is not that they win. It is that winning is no
    longer invisible. The resolution happens once, in
    `EffectiveProjection.resolve`, which records for every field what it carries
    and which extracted values it therefore does not — and the account is
    re-stated against that same model before any completeness claim is made. A
    displaced finding becomes `UNRESOLVED` and is named, instead of remaining
    `TYPED_FACT` in a payload that calls itself complete.
    """
    structured = structured or {}
    projection = EffectiveProjection.resolve(facts, structured)
    age_group = projection.values_for("age_group")

    return ProtectedCaseContext(
        case_id_internal=uuid.uuid4().hex,
        # `clinical_question` and `jurisdiction` are resolved here rather than
        # through the effective model, and the reason is the model's scope
        # rather than an oversight: the model exists to make DISPLACEMENT
        # accountable, and neither of these can displace anything. The extractor
        # records no carried span against either — `_QUESTION` sets
        # `clinical_question` without claiming a segment — so no source segment
        # is counted carried on the strength of them, and there is nothing for
        # the account to be wrong about. A field that ever acquires a carried
        # span must move into the model, and `represents` fails closed for any
        # field the model does not name, so that move is forced rather than
        # remembered.
        clinical_question=(
            question.strip()
            or structured.get("clinical_question", "").strip()
            or facts.clinical_question
        ),
        demographics={"age_group": age_group[0]} if age_group else {},
        conditions=projection.values_for("conditions"),
        medications=projection.values_for("medications"),
        allergies=projection.values_for("allergies"),
        findings=projection.values_for("findings"),
        vitals=dict(facts.vitals),
        labs=dict(facts.labs),
        jurisdiction=jurisdiction or structured.get("jurisdiction", ""),
        # Caller-STATED uncertainties only. This used to be `facts.uncovered` —
        # the residue, selected precisely for being unparseable, which is the
        # population with the highest residual-identifier density — and
        # `compile_synthesis_context` forwarded it into the external payload
        # verbatim. Measured at `ff34722`: a full personal name and a contact
        # extension rendered to the model inside the class whose own docstring
        # prohibits raw or scrubbed report text.
        #
        # Measuring the loss and transmitting the residue are different
        # requirements. The measurement lives on `facts.accounting`, which never
        # leaves the process; what crosses the boundary is
        # `CompletenessReport`, which is counts and line numbers.
        uncertainties=projection.values_for("uncertainties"),
        source_provenance=provenance,
    )


def accounted_projection(
    context: ProtectedCaseContext, facts: ExtractedFacts
) -> ExtractedFacts:
    """These facts with the account re-stated against the projection that ships.

    THE seam ADV19-1 turns on, and the reason it is a named function called on
    every path rather than a line inside `compile_handoff`.

    The extractor's account is a claim about an intermediate object: it says
    which source spans IT turned into facts. The payload is built from `context`
    — in which a caller-supplied structured field may have displaced an
    extracted one — so the claim has to be put to `context` before any
    completeness statement derives from it. `EffectiveProjection.of_case` reads
    the compiled context, `accounting.against` re-states each `TYPED_FACT`
    segment the projection cannot be asked to show, and completeness, coverage
    and the named source lines then all describe the object the external model
    actually receives.

    Idempotent, so calling it on the refusal path and again on the compile path
    cannot compound: a segment already `UNRESOLVED` stays `UNRESOLVED`, and one
    the projection still carries is unaffected by being asked twice. That is
    what lets every entry point apply it without any of them having to know
    whether another already did.
    """
    return facts.accounted_against(EffectiveProjection.of_case(context, facts).represents)


def _require_substance(context: ProtectedCaseContext, facts: ExtractedFacts) -> None:
    """Refuse rather than answer from a projection that carries too little.

    Two conditions, and a third obligation that is not a condition at all.

      empty     nothing was established. Answering would be generic advice
                wearing the appearance of case-specific advice.

      thin      a minority of the accountable segments were carried. A
                JUDGEMENT, with `MINIMUM_COVERAGE` behind it.

    The obligation is `_require_complete_projection`, and it is separate on
    purpose. A threshold can only refuse what it can see, and both `ff34722`
    reviews defeated the threshold argument the same way: in every case where
    clinical content vanished, coverage was 1.0000, so no value of the threshold
    would have refused. Truthful completeness is therefore enforced
    unconditionally and everywhere, and the threshold is left to do the smaller
    job it can actually do.
    """
    if not context.has_clinical_facts():
        raise HandoffRefused(
            "No clinical fact could be established from this request without "
            "sending the document itself, which the egress policy does not "
            "permit.",
            _WANTED,
        )
    completeness = facts.completeness()
    coverage = completeness.coverage()
    if coverage < MINIMUM_COVERAGE:
        logger.warning(
            "safe projection carries %d of %d accountable source segment(s) "
            "(%.0f%%) — refusing",
            completeness.carried,
            completeness.accountable,
            coverage * 100,
        )
        raise HandoffRefused(
            f"Only {coverage:.0%} of this document's clinically accountable "
            "content could be carried into a payload that excludes direct "
            "identifiers, so an answer would be based on a minority of the "
            "case.",
            _WANTED,
        )


def _require_complete_projection(
    context: ProtectedCaseContext, facts: ExtractedFacts
) -> CompletenessReport:
    """Establish the completeness statement that MUST travel with the payload.

    `00_RULES.md` and M-1 require that a projection which cannot represent the
    case is clarified or refused — and, above that, that an incomplete
    projection is never presented as equivalent to the original. Phase 1 cannot
    decide whether an unresolved segment MATTERED; that needs the clinical
    understanding Phase 3 allocates. What it can do, and what this guarantees,
    is never to claim otherwise.

    So the report is built here, once, from the account, and
    `SafeSynthesisContext` has no constructor that omits it. A caller cannot
    forget to state it and there is no branch in which a projection is emitted
    without one.
    """
    report = facts.completeness()
    if not report.complete:
        logger.info(
            "safe projection is INCOMPLETE: %d of %d accountable segment(s) "
            "unresolved on source line(s) %s",
            report.unresolved,
            report.accountable,
            report.unresolved_lines,
        )
    return report


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
    _require_substance(context, accounted_projection(context, facts))
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

    The account is re-stated against `context` HERE, and not left to the caller,
    for the same reason: a projection whose completeness was measured against
    something other than itself is the ADV19-1 defect, and the only way to
    guarantee it cannot be built is for the re-statement to have no branch that
    omits it.
    """
    facts = accounted_projection(context, facts)
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
        completeness=_require_complete_projection(context, facts),
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
    events: tuple[RedactionEvent, ...] = (),
) -> SafeHandoff:
    """Protected text and structured fields in, two safe projections out.

    Raises `HandoffRefused` rather than producing a projection that would be
    empty or would leave most of the document behind.

    `events` are the redactions the boundary performed on `protected_text`. See
    `extract` for why they are optional and why omitting them can only make the
    account more pessimistic, never less.
    """
    facts = extract(protected_text, question=question, events=events)
    case = case_from_facts(
        facts,
        structured=structured,
        question=question,
        jurisdiction=jurisdiction,
        provenance=provenance,
    )
    # The account the handoff CARRIES is the one re-stated against the compiled
    # case, so `handoff.facts.completeness()` describes the payload rather than
    # the extractor's intermediate view. Without this the response metadata, the
    # Redis cache and the trace would keep reporting the pre-override account
    # even though the projection itself no longer does.
    facts = accounted_projection(case, facts)
    return SafeHandoff(
        facts=facts,
        case=case,
        evidence_query=compile_evidence_query(case, facts),
        synthesis_context=compile_synthesis_context(case, facts),
    )
