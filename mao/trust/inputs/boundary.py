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
    #: Ambiguities on the REPORT channel. Unchanged meaning.
    ambiguities: AmbiguityReport = field(default_factory=AmbiguityReport)
    #: Ambiguities on the QUERY and CHAT_HISTORY channels.
    #:
    #: These two channels consulted `find_ambiguities` for NOTHING at all, so
    #: the `refuse_ambiguity=False` the chat route passes was not even the
    #: operative control for them - there was no check to disable, and the
    #: scrubber's guess was taken unconditionally. 227 of 1476 hold-out cases
    #: lose clinical content here, across 38 distinct phrases, six of which are
    #: implanted cardiac devices and leads (A-2).
    #:
    #: The chat posture is unchanged and stays "do not refuse". Refusing would
    #: apply the three-token-name refusal rate - measured at 100% for an
    #: ordinary inline banner (A-7) - to the interactive route, which is the
    #: worse trade and is not what either review asks for. What changes is that
    #: the report is CARRIED instead of discarded, so the third option
    #: `00_RULES.md` actually prescribes for this case is available: say what
    #: was removed.
    chat_ambiguities: AmbiguityReport = field(default_factory=AmbiguityReport)

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


#: How a structured identifier is spelled once it has been removed by name.
#:
#: Typed rather than generic, so the placeholder still tells a reader — and a
#: model — which KIND of field was present, exactly as the detected ones do.
_STRUCTURED_PLACEHOLDER = {
    "patient_name": "[NAME]", "name": "[NAME]", "surname": "[NAME]",
    "forename": "[NAME]", "family_name": "[NAME]", "given_name": "[NAME]",
    "next_of_kin": "[NAME]", "carer_name": "[NAME]", "consultant": "[NAME]",
    "mrn": "[MRN]", "hospital_number": "[MRN]",
    "medical_record_number": "[MRN]", "case_no": "[MRN]",
    "nhs_number": "[NHS]", "nhs_no": "[NHS]",
    "ni_number": "[NI_NUMBER]", "national_insurance": "[NI_NUMBER]",
    "date_of_birth": "[DOB]", "dob": "[DOB]", "born": "[DOB]",
    "address": "[ADDRESS]", "home_address": "[ADDRESS]",
    "postcode": "[POSTCODE]", "post_code": "[POSTCODE]",
    "telephone": "[PHONE]", "phone": "[PHONE]", "mobile": "[PHONE]",
    "contact_number": "[PHONE]", "email": "[EMAIL]",
}


def remove_known_identifiers(
    text: str, structured: dict[str, str], protection: RequestProtection | None
) -> str:
    r"""Remove identifiers the caller NAMED, by exact match. No guessing at all.

    ## Why this is the first thing the boundary does

    The hard case in this whole subsystem is a patient header whose extent
    cannot be established from the text:

        'Patient Name: Harold Nkemdirim Rockwood Frailty'

    Taking the run deletes an instrument's name; stopping short leaves half a
    surname beside a placeholder claiming it was removed. `00_RULES.md` says
    such content "must not be guessed into a supposedly safe form", and lists
    the remedies in order — **structured fields first**, then refusal.

    A caller who states `patient_name: "Harold Nkemdirim"` has removed the
    question. There is no boundary to establish: the identifier is known, it is
    removed by exact match, and what remains on the line is clinical content
    that survives untouched. Both directions of the invariant are satisfied at
    once, by not having to decide anything.

    This is what makes the 422 an actionable request rather than a wall. The
    refusal names the fields it wants; supplying them makes the same document
    process correctly.

    Longest value first, so a full name is removed before a forename that is a
    prefix of it and cannot leave a dangling remainder.

    ## Why a sentinel and not the placeholder

    Substituting `[NAME]` here and then scrubbing does not work, and the reason
    is instructive: `layout._skip_noise` deliberately STEPS OVER a placeholder
    rather than believing it, because a caller can write one. So
    `Patient Name: [NAME] Rockwood Frailty` had the label reach past the
    placeholder, find a name-shaped run behind it, and redact the instrument —
    reintroducing the exact destruction this pathway exists to prevent.

    Distrusting a marker in caller-controlled text is correct. But a span THIS
    RUN has just decided is not caller-controlled, and the difference has to be
    carried out of band rather than in the characters. So the removal leaves a
    private-use sentinel that no value grammar can match and no name token can
    contain, and the typed placeholders are substituted back after the scrub.
    """
    if not structured:
        return text
    named = sorted(
        (
            (str(value).strip(), _STRUCTURED_PLACEHOLDER[key])
            for key, value in (
                (k.strip().lower().replace(" ", "_").replace("-", "_"), v)
                for k, v in structured.items()
            )
            if key in _STRUCTURED_PLACEHOLDER and str(value).strip()
        ),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for index, (value, placeholder) in enumerate(named):
        if value in text:
            text = text.replace(value, _sentinel(index))
            if protection is not None:
                protection.record_identifier(
                    placeholder.strip("[]"), value, InputChannel.STRUCTURED_FIELDS
                )
    return text


#: Private-use codepoints. `\w` does not match category `Co`, so `values._LETTER`
#: cannot take one into a name token, and no identifier grammar admits one — the
#: span is inert to every rule in the scrubber. `text.normalise` leaves them
#: alone because they are neither zero-width nor a control character.
_SENTINEL_OPEN = ""
_SENTINEL_CLOSE = ""


def _sentinel(index: int) -> str:
    return f"{_SENTINEL_OPEN}{index}{_SENTINEL_CLOSE}"


def restore_known_identifiers(text: str, structured: dict[str, str]) -> str:
    """Swap each sentinel for the typed placeholder its field deserves."""
    if not structured:
        return text
    named = sorted(
        (
            (str(value).strip(), _STRUCTURED_PLACEHOLDER[key])
            for key, value in (
                (k.strip().lower().replace(" ", "_").replace("-", "_"), v)
                for k, v in structured.items()
            )
            if key in _STRUCTURED_PLACEHOLDER and str(value).strip()
        ),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for index, (_, placeholder) in enumerate(named):
        text = text.replace(_sentinel(index), placeholder)
    return text


def protect_channel(
    text: str,
    channel: InputChannel,
    *,
    refuse_ambiguity: bool = True,
    structured: dict[str, str] | None = None,
) -> SafeDerivedText:
    """Take one channel through the boundary, inside an established request.

    For content that only becomes available deep in the graph — a PDF's text is
    extracted by the agent that received the attachment, not by the route. The
    identifiers it removes are recorded against the CURRENT request's
    protection, so the egress assertion covers them exactly as it covers the
    query and the history.

    `structured` is applied FIRST and by exact match, and the ambiguity check
    then runs on what is left. That ordering is the point: a header the caller
    has already named is not ambiguous, so the refusal fires only where nothing
    has told us the answer. See `remove_known_identifiers`.

    Outside a request there is no protection to record against and no patient to
    protect: a CLI run or an ingestion job still gets the de-identified text,
    and the egress assertion has nothing to assert, which `authorise` reports
    rather than silently treating as a pass.
    """
    from mao.trust.egress.gateway import current_protection

    protection = current_protection()
    fields = structured or {}
    masked = remove_known_identifiers(text, fields, protection)

    if refuse_ambiguity:
        report = find_ambiguities(masked)
        if report:
            raise AmbiguousDocument(report)

    if protection is None:
        result = scrub_with_report(masked)
        logger.debug(
            "no request protection bound — %s de-identified without recording",
            channel.value,
        )
        return SafeDerivedText(
            text=restore_known_identifiers(result.text, fields), origin=channel
        )
    protected = _protect_channel(masked, channel, protection)
    return SafeDerivedText(
        text=restore_known_identifiers(protected.text, fields), origin=channel
    )


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
      chat   - the clinician wrote the text and reads the answer, so an
               over-redaction is recoverable while a leak is not - PROVIDED the
               route says what it removed. It did not. `ChatResponse` carries
               no protected text and no list of removed spans, and
               `api/protected_input.py` states as a design premise that "the
               server never returns the de-identified query to the client", so
               "visible" was not supported by the response contract at all: the
               clinician saw an answer, not the question the model was asked
               (A-2). The judgement that a leak is worse than an over-redaction
               is sound, but the boundary was not choosing between those two -
               the third option, and the one `00_RULES.md` prescribes here, is
               to say what was removed. `chat_ambiguities` is how.

    Both paths run the same transformation over the same channels. Only the
    response to an unresolved boundary differs, which is what stops the two
    drifting apart the way `/chat` and `/chat/stream` did.
    """
    protection = RequestProtection(trace_id=trace_id or uuid.uuid4().hex)
    chat_ambiguities = AmbiguityReport()

    limits.check("query", len(raw.query), limits.MAX_QUERY_CHARS)
    # A-2. The detector runs on this channel now. It did not before, which is
    # why `refuse_ambiguity` was not the operative control here: there was no
    # check to disable. The report is not raised on the chat posture - it is
    # carried, so the route can tell the caller what was removed.
    chat_ambiguities.items.extend(find_ambiguities(raw.query).items)
    query = _protect_channel(raw.query, InputChannel.QUERY, protection)

    history: list[tuple[str, SafeDerivedText]] = []
    # Most recent turns kept. The agents already read only the last three or
    # four, so this discards nothing any consumer would have seen — see
    # `limits`, which explains why history is the one channel that is bounded by
    # dropping rather than by refusing.
    for role, content in list(raw.chat_history)[-limits.MAX_HISTORY_TURNS :]:
        limits.check("a chat_history turn", len(content), limits.MAX_HISTORY_TURN_CHARS)
        chat_ambiguities.items.extend(find_ambiguities(content).items)
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
        chat_ambiguities=chat_ambiguities,
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
