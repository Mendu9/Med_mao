r"""The seven trust classes, as types rather than as a naming convention.

## Why these are types

Five remediation rounds treated "the scrubbed note" and "text that is safe to
send to a third party" as the same thing, because both were `str`. Every control
that tried to tell them apart had to re-derive the distinction by inspecting the
characters, which is the losing game this project has played six times.

A trust class is a fact about where a value CAME FROM, not about what it looks
like. `SafeSynthesisContext` is safe because it was built field by field out of
typed protected facts — not because a regex failed to find an identifier in it.
That is the whole difference, and giving it a type is what makes it checkable at
the boundary instead of guessable at the sink.

## The rule the egress gateway enforces

    RawSensitiveInput      never leaves the process
    ProtectedCaseContext   never leaves the process
    SafeEvidenceQuery      may go to approved evidence sources
    SafeSynthesisContext   may go to approved synthesis models, with evidence
    PublicEvidence         came from outside; may go back outside
    SafeDerivedText        text minted by the protected input boundary
    VerifiedOutput         an answer that passed verification

`SafeDerivedText` is not in `01_ARCHITECTURE.md`'s list and is not a widening of
it. It is the honest name for what Phase 1 still has: routes whose payload is a
de-identified string rather than a typed projection. Naming it separately means
the egress policy can say exactly which purposes may still accept one, and a
later phase can retire it by migrating those purposes — which is visible, rather
than being hidden inside a `str` that looks like every other `str`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrustClass(str, Enum):
    """Where a value came from, and therefore where it may go."""

    RAW_SENSITIVE_INPUT = "raw_sensitive_input"
    PROTECTED_CASE_CONTEXT = "protected_case_context"
    SAFE_EVIDENCE_QUERY = "safe_evidence_query"
    SAFE_SYNTHESIS_CONTEXT = "safe_synthesis_context"
    PUBLIC_EVIDENCE = "public_evidence"
    SAFE_DERIVED_TEXT = "safe_derived_text"
    VERIFIED_OUTPUT = "verified_output"


#: Classes that no external destination may receive under the approved M-1
#: policy. Changing this requires a control-document decision, not a code edit;
#: `tests/trust/test_egress_policy.py` asserts the membership so that a silent
#: widening fails the build.
NEVER_EXTERNAL: frozenset[TrustClass] = frozenset(
    {TrustClass.RAW_SENSITIVE_INPUT, TrustClass.PROTECTED_CASE_CONTEXT}
)


class InputChannel(str, Enum):
    """Every surface a caller can put sensitive text through.

    Enumerated as a type so that adding a modality is a change to this list
    rather than a new unguarded path. The audio finding was exactly that: a
    branch added later that never joined the boundary the other channels use.
    """

    QUERY = "query"
    CHAT_HISTORY = "chat_history"
    REPORT = "report"
    TRANSCRIPT = "transcript"
    STRUCTURED_FIELDS = "structured_fields"
    ATTACHMENT_METADATA = "attachment_metadata"


@dataclass(frozen=True)
class RawSensitiveInput:
    """What the caller actually sent, before anything has been decided about it.

    Holds every channel together rather than one object per endpoint, because
    the defect this closes is a channel that never reached the boundary at all.
    A route constructs exactly one of these, and the boundary consumes it.
    """

    trust_class = TrustClass.RAW_SENSITIVE_INPUT

    query: str = ""
    chat_history: tuple[tuple[str, str], ...] = ()
    report_text: str = ""
    transcript_text: str = ""
    structured_fields: dict[str, str] = field(default_factory=dict)
    attachment_metadata: dict[str, Any] = field(default_factory=dict)

    def channels(self) -> dict[InputChannel, str]:
        """The free text on each channel, for the boundary to transform.

        Chat history is flattened into one string per turn so that no channel
        gets a bespoke code path. The turns are reassembled by the boundary from
        the transformed pieces, in order, so nothing is joined or lost.
        """
        present: dict[InputChannel, str] = {}
        if self.query:
            present[InputChannel.QUERY] = self.query
        if self.report_text:
            present[InputChannel.REPORT] = self.report_text
        if self.transcript_text:
            present[InputChannel.TRANSCRIPT] = self.transcript_text
        return present


@dataclass(frozen=True)
class PrivateIdentifier:
    """One identifier the boundary found, kept apart from the clinical facts.

    `01_ARCHITECTURE.md`: "Private identifiers remain separately classified from
    clinical facts." Keeping the values here — rather than only their
    placeholders — is what lets the egress gateway assert, at the sink, that no
    identifier from THIS request is in an outgoing payload. That check is
    run-scoped, so unlike a marker in the text it cannot be forged by a caller.
    """

    kind: str
    value: str
    channel: InputChannel


@dataclass(frozen=True)
class ProtectedCaseContext:
    """The internal representation. Not an externally approved payload.

    Phase 1 populates the identifier and provenance halves and leaves the typed
    clinical fields to Phase 3, which is what `05_CASE_TO_EVIDENCE_SPEC.md`
    allocates. The class exists now because the egress policy has to be able to
    name it, and because `SafeSynthesisContext` must be built FROM something
    typed rather than from a scrubbed string.
    """

    trust_class = TrustClass.PROTECTED_CASE_CONTEXT

    case_id_internal: str
    clinical_question: str = ""
    demographics: dict[str, str] = field(default_factory=dict)
    conditions: tuple[str, ...] = ()
    medications: tuple[str, ...] = ()
    allergies: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()
    vitals: dict[str, str] = field(default_factory=dict)
    labs: dict[str, str] = field(default_factory=dict)
    imaging_findings: tuple[str, ...] = ()
    timeline: tuple[str, ...] = ()
    risk_factors: tuple[str, ...] = ()
    jurisdiction: str = ""
    uncertainties: tuple[str, ...] = ()
    private_identifiers: tuple[PrivateIdentifier, ...] = ()
    source_provenance: tuple[str, ...] = ()

    def has_clinical_facts(self) -> bool:
        """Whether anything survives a projection that excludes identifiers.

        A projection with no clinical content is not a safe payload, it is an
        empty one — and sending it would ask a model to reason about a case it
        has been told nothing about. The compiler refuses instead.
        """
        return bool(
            self.clinical_question
            or self.conditions
            or self.medications
            or self.findings
            or self.labs
            or self.vitals
            or self.imaging_findings
        )


@dataclass(frozen=True)
class SafeEvidenceQuery:
    """The minimum needed to FIND evidence. Never a redacted copy of the note."""

    trust_class = TrustClass.SAFE_EVIDENCE_QUERY

    query_id: str
    search_intent: str
    population: dict[str, str] = field(default_factory=dict)
    condition: tuple[str, ...] = ()
    intervention: tuple[str, ...] = ()
    comparator: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()
    jurisdiction: str = ""
    freshness: str = ""
    #: Which protected facts each field came from. Field names only — a
    #: provenance record that quoted the source text would put the protected
    #: content back into an object whose whole purpose is not carrying it.
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class SafeSynthesisContext:
    """The minimum needed to APPLY evidence to the case.

    Different from `SafeEvidenceQuery` on purpose, and the difference is the
    reason M-1 needed a decision: finding evidence about bradycardia and
    donepezil needs the drug and the finding, while applying it needs the dose,
    the heart rate and the patient's age band. One is not a superset of the
    other and neither is a rewritten note.

    Explicitly prohibited, and asserted by `test_egress_policy.py`: direct
    identifiers, free-text patient headers, raw or scrubbed report text, raw
    chat history, raw transcript text, identifying metadata.

    `uncertainties` is CALLER-STATED only. At `ff34722` it was populated from
    the extractor's residue — every line no grammar could parse, which is the
    population selected for being unparseable and therefore the one with the
    highest residual-identifier density — and `render()` emitted it verbatim.
    Measured: a full personal name and a contact extension in the payload. The
    compiler now builds it from structured fields the caller supplied, and the
    account of what was NOT carried travels as `completeness`, which is counts
    and line numbers and no text at all.
    """

    trust_class = TrustClass.SAFE_SYNTHESIS_CONTEXT

    context_id: str
    clinical_question: str
    age_group: str = ""
    sex_if_relevant: str = ""
    conditions: tuple[str, ...] = ()
    medications_and_doses: tuple[str, ...] = ()
    allergies: tuple[str, ...] = ()
    vitals: dict[str, str] = field(default_factory=dict)
    labs: dict[str, str] = field(default_factory=dict)
    clinically_relevant_findings: tuple[str, ...] = ()
    imaging_findings: tuple[str, ...] = ()
    timeline_summary: str = ""
    risk_factors: tuple[str, ...] = ()
    jurisdiction: str = ""
    uncertainties: tuple[str, ...] = ()
    #: The account of what this projection does and does not carry. Counts,
    #: statuses and source LINE NUMBERS — never source text.
    #:
    #: `Any` rather than the concrete type because `mao.trust.handoff` imports
    #: this module; the compiler is the only producer and it supplies a
    #: `CompletenessReport`. A projection built without one renders as
    #: "completeness not established", which is a true statement about a
    #: payload nobody accounted for and is the fail-closed direction.
    completeness: Any = None
    provenance: tuple[str, ...] = ()

    def render(self) -> str:
        """The payload as text for a model, built field by field.

        Rendering happens HERE and not at a call site, so that the text a
        synthesis model receives is a function of the typed fields and of
        nothing else. A caller cannot append to it, and there is no branch in
        which a free-text note is substituted for it.

        The completeness statement is emitted UNCONDITIONALLY and first. A model
        asked whether a drug is safe must not be handed a projection that reads
        as the whole case when it is not, and the only way to guarantee that is
        for the statement to have no branch that omits it.
        """
        parts: list[str] = [
            f"Clinical question: {self.clinical_question}",
            f"Case completeness: {self._completeness_statement()}",
        ]
        for label, values in (
            ("Conditions", self.conditions),
            ("Medications and doses", self.medications_and_doses),
            ("Allergies", self.allergies),
            ("Findings", self.clinically_relevant_findings),
            ("Imaging findings", self.imaging_findings),
            ("Risk factors", self.risk_factors),
            ("Uncertainties", self.uncertainties),
        ):
            if values:
                parts.append(f"{label}: " + "; ".join(values))
        for label, mapping in (("Vitals", self.vitals), ("Labs", self.labs)):
            if mapping:
                rendered = "; ".join(f"{k} {v}" for k, v in sorted(mapping.items()))
                parts.append(f"{label}: {rendered}")
        for label, value in (
            ("Age group", self.age_group),
            ("Sex", self.sex_if_relevant),
            ("Timeline", self.timeline_summary),
            ("Jurisdiction", self.jurisdiction),
        ):
            if value:
                parts.append(f"{label}: {value}")
        return "\n".join(parts)

    def _completeness_statement(self) -> str:
        describe = getattr(self.completeness, "describe", None)
        if describe is None:
            return (
                "NOT ESTABLISHED: no account was made of how much of the source "
                "this projection represents. Treat the case as incompletely "
                "described."
            )
        return describe()

    def is_complete(self) -> bool:
        """Whether the account says every accountable segment is represented."""
        return bool(getattr(self.completeness, "complete", False))


@dataclass(frozen=True)
class PublicEvidence:
    """Evidence retrieved from an approved public source.

    It came from outside the trust boundary, so returning it outside is not a
    disclosure. It is still UNTRUSTED DATA for the safety verifier — those are
    different properties, and conflating them is what let a retrieved document
    forge the evaluator's control block.
    """

    trust_class = TrustClass.PUBLIC_EVIDENCE

    evidence_id: str
    text: str
    source_id: str = ""
    title: str = ""
    url: str = ""

    @classmethod
    def from_evidence(cls, item: Any) -> PublicEvidence:
        """Adapt the canonical `mao.schemas.evidence.Evidence` or a raw chunk."""
        from mao.schemas.evidence import as_text

        return cls(
            evidence_id=str(getattr(item, "evidence_id", "") or ""),
            text=as_text(item),
            source_id=str(getattr(item, "source_id", "") or ""),
            title=str(getattr(item, "title", "") or ""),
            url=str(getattr(item, "url", "") or ""),
        )


@dataclass(frozen=True)
class SafeDerivedText:
    """De-identified text minted by the protected input boundary.

    The honest name for what a not-yet-migrated route sends. It is a TYPE and
    not a convention so that the egress policy can name the purposes still
    allowed to accept one, and so that a plain `str` arriving at the gateway is
    refused rather than assumed to have been through the boundary.

    `origin` records which channel minted it. That is what makes a later
    migration measurable: the purposes still accepting `SafeDerivedText` are
    exactly the ones Phase 1 did not finish typing.
    """

    trust_class = TrustClass.SAFE_DERIVED_TEXT

    text: str
    origin: InputChannel
    #: What the boundary actually removed from this text, as content-free
    #: `RedactionEvent`s in `text`'s own coordinates.
    #:
    #: Carried on the value rather than looked up later because every reader of
    #: "what did the transformation do" must read the SAME record. At `ff34722`
    #: the Safe Handoff accounting re-derived it with a placeholder regex and
    #: the clinician's notice re-derived it with a second detector pass, and
    #: both disagreed with the transformation in measured, opposite ways.
    #:
    #: Content-free by construction: `RedactionEvent` has offsets, a kind and a
    #: label, and no value. The sensitive `Removal` record stays behind in the
    #: protected plane, which is why this can ride on the class that reaches the
    #: egress gateway.
    events: tuple[Any, ...] = ()

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.text


@dataclass(frozen=True)
class ExternalSafePayload:
    """A destination-specific envelope. The only thing the gateway will send.

    Carries the policy decision with the data, so a trace can answer "what was
    sent, to where, for what purpose, under which policy version" without
    re-deriving any of it — and so that no call site can construct the envelope
    while leaving the decision implicit.
    """

    destination: str
    purpose: str
    #: `None` when the flow carries no data from the request at all — a
    #: liveness probe. Distinct from carrying something harmless.
    payload_trust_class: TrustClass | None
    safe_payload: Any
    policy_version: str
    trace_id: str = ""
    evidence: tuple[PublicEvidence, ...] = ()
