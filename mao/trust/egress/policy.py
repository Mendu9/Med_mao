r"""Which trust class may reach which destination, for which purpose.

## Why a table and not a predicate

`00_RULES.md` requires that the egress layer "enforce trust class, destination,
and purpose and reject unnamed/unapproved flows by default". A predicate that
answers from the data cannot do that: it can only ever say what it was written
to notice, and the six waves before this one are a record of what a predicate
does not notice.

A table can. Every allowed flow is a row somebody wrote down, and everything
that is not a row is refused — including a destination or a purpose added later
by code that never heard of this module. That is what "fail closed" means here.

## The M-1 decision, encoded

    No external model or service is approved to receive RawSensitiveInput or
    ProtectedCaseContext. External clinical synthesis receives only
    SafeSynthesisContext plus PublicEvidence.

`NEVER_EXTERNAL` in `mao.trust.classes` states the first half and no row here
may contradict it — `_validate` refuses to build a table that does, so the two
statements cannot drift. The second half is the `CLINICAL_SYNTHESIS` row: it
admits `SAFE_SYNTHESIS_CONTEXT` and nothing else. Notably it does NOT admit
`SAFE_DERIVED_TEXT`, which is what makes "a scrubbed free-text report is not a
valid synthesis payload" an executable rule rather than a sentence in a
document.

## SAFE_DERIVED_TEXT, and what it is doing here

Phase 1 does not finish typing every route. The purposes below that still accept
`SAFE_DERIVED_TEXT` are exactly the ones whose payload is a de-identified string
— routing, general synthesis, safety verification, evidence search. Recording
that in the table makes the residual countable and gives a later phase a precise
list, instead of leaving it as an untyped `str` indistinguishable from any
other. `CLINICAL_SYNTHESIS` is deliberately absent from that list, because it is
the one M-1 decided.
"""
from __future__ import annotations

from enum import Enum

from mao.trust.classes import NEVER_EXTERNAL, TrustClass

#: Bumped whenever a row changes. A trace citing a version must be able to
#: reconstruct the decision that was made, which means the version has to move
#: when the decision does.
EGRESS_POLICY_VERSION = "2026.09-2"


class Destination(str, Enum):
    """A kind of external service, not a hostname.

    Kinds rather than URLs because the policy question is "may this class of
    data go to a third-party model at all", and answering it per hostname would
    make the table a configuration file that drifts.
    """

    MODEL_PROVIDER = "model_provider"
    WEB_SEARCH = "web_search"
    SCHOLARLY_API = "scholarly_api"
    VECTOR_STORE = "vector_store"
    EXTERNAL_MEMORY = "external_memory"
    MCP_TOOL = "mcp_tool"
    ANALYTICS = "analytics"


class EgressPurpose(str, Enum):
    """Why a call is being made. Required, and never defaulted.

    A default would be the unnamed flow the rules forbid: whichever purpose was
    chosen as the default would silently become the one every new call site
    inherits, and the table would stop meaning anything.
    """

    CLINICAL_SYNTHESIS = "clinical_synthesis"
    GENERAL_SYNTHESIS = "general_synthesis"
    EVIDENCE_SEARCH = "evidence_search"
    SAFETY_VERIFICATION = "safety_verification"
    ROUTING = "routing"
    EXTRACTION = "extraction"
    IMAGE_ANALYSIS = "image_analysis"
    EMBEDDING = "embedding"
    MEMORY_WRITE = "memory_write"
    MEMORY_READ = "memory_read"
    HEALTH_PROBE = "health_probe"


#: (destination, purpose) -> the trust classes that flow may carry.
#:
#: Absent key => refused. Empty value => refused. Both spellings mean the same
#: thing and both are tested, because "I forgot to add a row" and "I added a row
#: that allows nothing" must not behave differently.
_ALLOWED: dict[tuple[Destination, EgressPurpose], frozenset[TrustClass]] = {
    # The M-1 row. SafeSynthesisContext only: not a scrubbed note, not history,
    # not a transcript, not the protected case.
    (Destination.MODEL_PROVIDER, EgressPurpose.CLINICAL_SYNTHESIS): frozenset(
        {TrustClass.SAFE_SYNTHESIS_CONTEXT}
    ),
    # Not-yet-typed routes. Each is a de-identified string and is named as one.
    (Destination.MODEL_PROVIDER, EgressPurpose.GENERAL_SYNTHESIS): frozenset(
        {TrustClass.SAFE_DERIVED_TEXT, TrustClass.PUBLIC_EVIDENCE}
    ),
    (Destination.MODEL_PROVIDER, EgressPurpose.ROUTING): frozenset(
        {TrustClass.SAFE_DERIVED_TEXT}
    ),
    (Destination.MODEL_PROVIDER, EgressPurpose.EXTRACTION): frozenset(
        {TrustClass.SAFE_DERIVED_TEXT}
    ),
    # The verifier judges an answer against retrieved evidence. Both are already
    # outside the protected plane by the time they reach it.
    (Destination.MODEL_PROVIDER, EgressPurpose.SAFETY_VERIFICATION): frozenset(
        {TrustClass.SAFE_DERIVED_TEXT, TrustClass.PUBLIC_EVIDENCE,
         TrustClass.VERIFIED_OUTPUT}
    ),
    # A liveness check sends a fixed string and no request data at all.
    (Destination.MODEL_PROVIDER, EgressPurpose.HEALTH_PROBE): frozenset(),
    # Vision. DISABLED FOR PHASE 1 - control decision M-2.
    #
    # There WAS a row here admitting SAFE_DERIVED_TEXT, and it is why both
    # b63311d reviews raised the same finding independently (A-5, ADV16-7). A
    # base64 patient scan is not "de-identified text minted by the protected
    # input boundary": it has been through no boundary at all, it has no
    # InputChannel origin, and `gateway._outgoing_text` reads only the text
    # parts of a multipart turn - so the run-scoped identifier assertion, the
    # second wall, is a structural no-op for the image half. `_validate` refuses
    # any row admitting a NEVER_EXTERNAL class and was satisfied here only
    # because the call site declared a class the payload does not have. A
    # fail-closed table is defeated by one false declaration, invisibly.
    #
    # The row is REMOVED rather than re-typed. Removing it makes the refusal
    # structural: `authorise` rejects an unnamed flow before it considers a
    # trust class at all, so no future call site can re-enable imagery by
    # declaring something. Adding a truthful raw-imagery class and an approved
    # row would be RE-ENABLING the flow, and that is a control-plane decision
    # this wave does not have. PROJECT_STATE.md records the four conditions.
    #
    # Nothing here forecloses VISION as a capability. It forecloses carrying it
    # under a class it does not have.
    # Evidence retrieval. `SafeEvidenceQuery` is the typed form; the derived
    # string is the Phase 1 residual for routes not yet migrated.
    (Destination.WEB_SEARCH, EgressPurpose.EVIDENCE_SEARCH): frozenset(
        {TrustClass.SAFE_EVIDENCE_QUERY, TrustClass.SAFE_DERIVED_TEXT}
    ),
    (Destination.SCHOLARLY_API, EgressPurpose.EVIDENCE_SEARCH): frozenset(
        {TrustClass.SAFE_EVIDENCE_QUERY, TrustClass.SAFE_DERIVED_TEXT}
    ),
    (Destination.VECTOR_STORE, EgressPurpose.EVIDENCE_SEARCH): frozenset(
        {TrustClass.SAFE_EVIDENCE_QUERY, TrustClass.SAFE_DERIVED_TEXT}
    ),
    (Destination.VECTOR_STORE, EgressPurpose.EMBEDDING): frozenset(
        {TrustClass.SAFE_EVIDENCE_QUERY, TrustClass.SAFE_DERIVED_TEXT,
         TrustClass.PUBLIC_EVIDENCE}
    ),
    # External memory. `01_ARCHITECTURE.md`: protected patient context must not
    # flow into external/general memory by default, so this row is narrow.
    (Destination.EXTERNAL_MEMORY, EgressPurpose.MEMORY_WRITE): frozenset(
        {TrustClass.SAFE_DERIVED_TEXT}
    ),
    # A memory READ sends this request's query to the same third party the
    # write sends its content to, and there was no row for it at b63311d - so
    # `search_memories` could not have been authorised even by a call site that
    # wanted to be. A-4 counted the write; the read is the same sink.
    (Destination.EXTERNAL_MEMORY, EgressPurpose.MEMORY_READ): frozenset(
        {TrustClass.SAFE_DERIVED_TEXT, TrustClass.SAFE_EVIDENCE_QUERY}
    ),
}


def _validate() -> None:
    """No row may contradict `NEVER_EXTERNAL`.

    Run at import. The policy document and the table are two statements of one
    decision, and this is what stops them drifting apart silently — which is how
    `PROJECT_STATE.md` came to record a blocking finding as deferred.
    """
    for key, classes in _ALLOWED.items():
        forbidden = classes & NEVER_EXTERNAL
        if forbidden:
            raise ValueError(
                f"egress policy row {key} admits {sorted(c.value for c in forbidden)}, "
                "which no external destination is approved to receive. Changing "
                "that is a control-plane decision, not a code edit."
            )


_validate()


def allowed_classes(
    destination: Destination, purpose: EgressPurpose
) -> frozenset[TrustClass]:
    """What this flow may carry. Empty for an unnamed or empty-valued flow."""
    return _ALLOWED.get((destination, purpose), frozenset())


def is_named(destination: Destination, purpose: EgressPurpose) -> bool:
    """Whether anybody wrote this flow down at all.

    Distinguished from "wrote it down as carrying nothing" so a refusal can say
    which it was. `HEALTH_PROBE` is named and carries nothing; a typo is neither.
    """
    return (destination, purpose) in _ALLOWED


def named_flows() -> list[tuple[Destination, EgressPurpose, frozenset[TrustClass]]]:
    """Every approved flow, for the trace, the report and the policy test."""
    return [
        (destination, purpose, classes)
        for (destination, purpose), classes in sorted(
            _ALLOWED.items(), key=lambda item: (item[0][0].value, item[0][1].value)
        )
    ]
