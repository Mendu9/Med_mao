r"""One protected entry for every sensitive input channel.

## The defect this closes

`apply_input_guardrails` was applied to `request.query` and to nothing else.
`chat_history` reached three agents verbatim, an audio transcript reached the
synthesis prompt with the scrubber never called at all, and each was a *missing
call site on a second channel* rather than a defect in the detector. Every
de-identification finding in every previous wave was irrelevant to them.

So a channel does not get to be a call site. `RawSensitiveInput` names every
channel; `protect()` is the only thing that consumes one; and a modality added
later is a member of `InputChannel` or it does not reach a model.

## Exactly once

`/chat` scrubbed twice — once in the guardrails and again in the query
decomposer — and the second pass over-redacted a clinical line the first had
left alone. The fix attempted for that was idempotence: make a second scrub a
no-op by recognising the first one's output. Both mechanisms tried were
forgeable, because both had to read a marker or a position in text the caller
controls, and the caller can write either. Removing the forgeable mechanism then
reopened the destruction: at the post-gate head
`Patient Name:\nMRN:\nHarold Nkemdirim\nRockwood Frailty\n` loses the clinical
line on the second pass.

Neither property is achievable while a second pass exists. So there is no second
pass. Each channel is transformed once, here, and the result is carried as a
`SafeDerivedText` — a type, so a later stage that wanted to "make sure" cannot
re-run the transformation without saying so in its signature.

## What it hands to the egress boundary

Every span removed is recorded in a `RequestProtection` bound to the request.
The egress gateway then answers "is any of what we removed in what we are about
to send?" by lookup rather than by another grammar. That is the control that
holds for a channel nobody remembered, and it is unforgeable in the way a marker
in a document is not: the caller does not populate the record, the boundary
does, from what it actually found.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from mao.core.deident.ambiguity import AmbiguityReport, AmbiguousDocument, find_ambiguities
from mao.core.pii_scrubber import scrub_with_report
from mao.trust.classes import (
    InputChannel,
    RawSensitiveInput,
    SafeDerivedText,
)
from mao.trust.egress.gateway import RequestProtection
from mao.trust.inputs import limits

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProtectedInput:
    """Every channel, de-identified once, plus the record of what was removed."""

    protection: RequestProtection
    query: SafeDerivedText
    chat_history: tuple[tuple[str, SafeDerivedText], ...] = ()
    report: SafeDerivedText | None = None
    transcript: SafeDerivedText | None = None
    structured_fields: dict[str, str] = field(default_factory=dict)
    ambiguities: AmbiguityReport = field(default_factory=AmbiguityReport)

    def history_as_messages(self) -> list[dict[str, str]]:
        """The de-identified history in the shape the agents already consume."""
        return [{"role": role, "content": text.text} for role, text in self.chat_history]


def _protect_channel(
    text: str, channel: InputChannel, protection: RequestProtection
) -> SafeDerivedText:
    """De-identify one channel exactly once and record what came out of it."""
    result = scrub_with_report(text)
    for removal in result.removals:
        protection.record_identifier(removal.kind, removal.value, channel)
    return SafeDerivedText(text=result.text, origin=channel)


def protect_channel(
    text: str, channel: InputChannel, *, refuse_ambiguity: bool = True
) -> SafeDerivedText:
    """Take one channel through the boundary, inside an established request.

    For content that only becomes available deep in the graph — a PDF's text is
    extracted by the agent that received the attachment, not by the route. The
    identifiers it removes are recorded against the CURRENT request's
    protection, so the egress assertion covers them exactly as it covers the
    query and the history.

    Outside a request there is no protection to record against and no patient to
    protect: a CLI run or an ingestion job still gets the de-identified text,
    and the egress assertion has nothing to assert, which `authorise` reports
    rather than silently treating as a pass.
    """
    from mao.trust.egress.gateway import current_protection

    if refuse_ambiguity:
        report = find_ambiguities(text)
        if report:
            raise AmbiguousDocument(report)

    protection = current_protection()
    if protection is None:
        result = scrub_with_report(text)
        logger.debug(
            "no request protection bound — %s de-identified without recording",
            channel.value,
        )
        return SafeDerivedText(text=result.text, origin=channel)
    return _protect_channel(text, channel, protection)


def protect(
    raw: RawSensitiveInput,
    *,
    trace_id: str = "",
    refuse_ambiguity: bool = True,
) -> ProtectedInput:
    """Take every channel through the boundary once. Raises rather than guesses.

    `refuse_ambiguity` distinguishes the two paths the agreed policy defines,
    and it is the ONLY thing that distinguishes them:

      upload — the document is processed unseen, so a wrong guess is silent. An
               unresolvable patient header is refused and the caller is asked
               for structured fields.
      chat   — the clinician wrote the text and reads the answer, so an
               over-redaction is visible and recoverable while a leak is not.

    Both paths run the same transformation over the same channels. Only the
    response to an unresolved boundary differs, which is what stops the two
    drifting apart the way `/chat` and `/chat/stream` did.
    """
    protection = RequestProtection(trace_id=trace_id or uuid.uuid4().hex)

    limits.check("query", len(raw.query), limits.MAX_QUERY_CHARS)
    query = _protect_channel(raw.query, InputChannel.QUERY, protection)

    history: list[tuple[str, SafeDerivedText]] = []
    # Most recent turns kept. The agents already read only the last three or
    # four, so this discards nothing any consumer would have seen — see
    # `limits`, which explains why history is the one channel that is bounded by
    # dropping rather than by refusing.
    for role, content in list(raw.chat_history)[-limits.MAX_HISTORY_TURNS :]:
        limits.check("a chat_history turn", len(content), limits.MAX_HISTORY_TURN_CHARS)
        history.append(
            (role, _protect_channel(content, InputChannel.CHAT_HISTORY, protection))
        )

    report: SafeDerivedText | None = None
    ambiguities = AmbiguityReport()
    if raw.report_text:
        limits.check(
            "the extracted report text",
            len(raw.report_text),
            limits.MAX_EXTRACTED_TEXT_CHARS,
        )
        ambiguities = find_ambiguities(raw.report_text)
        if ambiguities and refuse_ambiguity:
            raise AmbiguousDocument(ambiguities)
        report = _protect_channel(raw.report_text, InputChannel.REPORT, protection)

    transcript: SafeDerivedText | None = None
    if raw.transcript_text:
        limits.check(
            "the transcript",
            len(raw.transcript_text),
            limits.MAX_TRANSCRIPT_CHARS,
        )
        # A spoken patient header is exactly as unresolvable as a written one,
        # and the transcript is processed unseen for the same reason a report
        # is. The audio path previously called neither the ambiguity check nor
        # the scrubber, so an ambiguous spoken header was neither refused nor
        # redacted; it is on the same contract now.
        transcript_ambiguities = find_ambiguities(raw.transcript_text)
        if transcript_ambiguities and refuse_ambiguity:
            raise AmbiguousDocument(transcript_ambiguities)
        transcript = _protect_channel(
            raw.transcript_text, InputChannel.TRANSCRIPT, protection
        )

    structured: dict[str, str] = {}
    for key, value in raw.structured_fields.items():
        limits.check(
            f"structured field {key!r}", len(value), limits.MAX_STRUCTURED_FIELD_CHARS
        )
        # A structured field is not free text and is not scrubbed. It is
        # CLASSIFIED: the caller has said what it is, so an identifier field is
        # recorded as one and never reaches an external payload, while a
        # clinical field is carried intact. This is the pathway the ambiguity
        # policy points a refused caller at, and guessing inside it would defeat
        # the point of asking.
        structured[key] = value
        if _is_identifying(key):
            protection.record_identifier(
                key.upper(), value, InputChannel.STRUCTURED_FIELDS
            )

    if protection.identifiers:
        logger.info(
            "protected boundary removed %d identifier(s) across %s",
            len(protection.identifiers),
            sorted({i.channel.value for i in protection.identifiers}),
        )

    return ProtectedInput(
        protection=protection,
        query=query,
        chat_history=tuple(history),
        report=report,
        transcript=transcript,
        structured_fields=structured,
        ambiguities=ambiguities,
    )


#: Structured field names that carry a direct identifier.
#:
#: A field the caller NAMES as an identifier needs no detection at all, which is
#: the whole advantage of the structured pathway over the free-text one: there
#: is no boundary to establish, so there is nothing to guess and nothing to
#: refuse.
_IDENTIFYING_FIELDS: frozenset[str] = frozenset(
    {
        "patient_name", "name", "surname", "forename", "family_name",
        "given_name", "next_of_kin", "carer_name", "consultant",
        "mrn", "hospital_number", "medical_record_number", "case_no",
        "nhs_number", "nhs_no", "ni_number", "national_insurance",
        "date_of_birth", "dob", "born",
        "address", "home_address", "postcode", "post_code",
        "telephone", "phone", "mobile", "contact_number", "email",
    }
)


def _is_identifying(key: str) -> bool:
    return key.strip().lower().replace(" ", "_").replace("-", "_") in _IDENTIFYING_FIELDS
