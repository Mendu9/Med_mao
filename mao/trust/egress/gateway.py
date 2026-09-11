r"""The one place an external, data-bearing call is authorised.

## Three controls, and why each is needed

**1. The flow must be named.** `authorise` refuses any (destination, purpose)
pair that is not a row in `policy._ALLOWED`, and `purpose` has no default. A
call site that does not say why it is calling out cannot call out. This is what
`00_RULES.md` means by "reject unnamed/unapproved flows by default".

**2. The payload's trust class must be admitted by that row.** This is where the
approved M-1 decision becomes executable: the `CLINICAL_SYNTHESIS` row admits
`SAFE_SYNTHESIS_CONTEXT` and nothing else, so a scrubbed free-text report is
refused by type rather than by inspection.

**3. No identifier from THIS request may be in the outgoing text.**

The third control is the one that closes the class of defect the previous six
waves kept reopening, and it is worth being precise about why it works where a
re-scrub does not.

Every earlier control asked a question ABOUT THE TEXT: does this string look
like it contains an identifier? That question is answered by a grammar, a
grammar is a list, and every list in this project has been defeated by an input
the list's author had not seen — a Hangul filler, a markdown asterisk, an eponym
surname, a middle name that is also an English word.

This control asks a different question: is any of the text the protected
boundary REMOVED FROM THIS REQUEST present in what is about to be sent? That is
not a judgement about the characters, it is a lookup in a run-scoped record. It
cannot be widened by a caller, because the caller does not populate it — the
boundary does, from what it actually found. It cannot be forged by a marker in
the input, which is the failure mode of every idempotence mechanism tried so
far. And it catches a channel nobody remembered to scrub, because it does not
care which channel the text came from.

It is a SECOND wall and not the first one. If it fires, something upstream is
broken and the request is refused rather than repaired: a de-identification
defect that reaches the sink is not something to paper over at the sink.

## What is exempt, and why

Retrieved evidence is exempt. It came from outside the trust boundary before
this request began, so it cannot be a disclosure of this patient — and a memory
clinic's evidence corpus contains the word `Parkinson` on almost every page. A
guard that refused every answer whose evidence mentions Parkinson's disease
because the patient is also called Parkinson would be a false refusal on the
clinical path, which is a patient-safety cost paid for no privacy gain.

Evidence is subtracted by REGISTERED SPAN, not by vocabulary: the retrieval
layer registers the text it retrieved, and the assertion masks exactly that text
out of the outgoing payload before looking. So the exemption covers what
retrieval actually returned and nothing else, and a caller cannot claim it.
"""
from __future__ import annotations

import logging
import unicodedata
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from mao.trust.classes import (
    ExternalSafePayload,
    InputChannel,
    PrivateIdentifier,
    PublicEvidence,
    TrustClass,
)
from mao.trust.egress.policy import (
    EGRESS_POLICY_VERSION,
    Destination,
    EgressPurpose,
    allowed_classes,
    is_named,
)

logger = logging.getLogger(__name__)


class EgressRefused(Exception):
    """An external call was not authorised. Never carries the payload.

    The message names the flow and the reason. It deliberately does not quote
    the offending text: this exception is raised on the de-identification
    failure path, it is logged, and it reaches an HTTP handler — which is
    exactly the route by which a patient's name previously reached the
    application log.
    """


#: Identifier values shorter than this are not searched for in outgoing text.
#:
#: A three-character span matches inside ordinary words often enough that the
#: guard would refuse legitimate traffic, and a three-character identifier is
#: not one an attacker learns anything from. Real record numbers, names,
#: postcodes, dates and telephone numbers are all longer.
_MIN_ASSERTABLE = 4

#: Identifier kinds whose values are never ordinary English, so a single token
#: of one is safe to search for on its own.
#:
#: NAME is absent on purpose. A surname is a word — `Parkinson`, `Pick`, `Day`,
#: `Down`, `Rankin` are all neurology vocabulary as well as names — so a NAME is
#: asserted whole, and in contiguous multi-token pieces, but never one token at
#: a time. Emitting a fragment of a name is a de-identification defect and is
#: caught by `tests/trust/test_holdout_deidentification.py`, which is the layer
#: that owns it.
_TOKENWISE_KINDS: frozenset[str] = frozenset(
    {"MRN", "NHS", "DOB", "PHONE", "EMAIL", "POSTCODE", "NI_NUMBER", "ACCOUNT"}
)


def _visible(text: str) -> str:
    """The text as a reader recovers it, normalised as hard as possible.

    DECOMPOSES FIRST, and then removes every mark. The previous form applied
    NFKC without NFD, so a combining acute on the `A` of a record number was
    composed into `A-acute` before the strip could see it and this wall
    reported no leak on a payload carrying the record number in full - the
    same mechanism, in the same direction, as the scrubber defect it exists to
    back stop (ADV16-1). Both walls shared one detection step, so neither could
    catch what the other missed.

    This function is therefore deliberately STRICTER than
    `mao.core.deident.text.normalise`, and the difference must be kept. That
    function is a transformation whose output a clinician reads, so it
    preserves marks that carry meaning. This one never ships anything - it only
    answers "is an identifier this request removed readable in what we are
    about to send" - so over-normalising has no cost and only widens what the
    wall catches. A control and its backstop must not share a detection step.

    Defined against the Unicode property "occupies no advance width" rather than
    against a category denylist. `Cf` was the previous definition and it is a
    proper subset: `U+034F` is `Mn`, the Hangul fillers are `Lo`, the Braille
    blank is `So`, and every one of them split an identifier grammar.

    Comparing visible forms on both sides is what stops
    `[NAME]<U+034F>dirim Okonkwo` counting as a redaction on the ground that
    `Nkemdirim` is not a literal substring of it.
    """
    out: list[str] = []
    for character in unicodedata.normalize("NFD", text):
        category = unicodedata.category(character)
        if character in "\t\n\r":
            out.append(character)
        elif category in ("Cf", "Mn", "Me", "Cc") or character in "ᅟᅠㅤﾠ⠀":
            continue
        else:
            out.append(character)
    return unicodedata.normalize("NFKC", "".join(out))


def _assertable_spans(identifier: PrivateIdentifier) -> list[str]:
    """Every form of this identifier worth searching for in outgoing text."""
    whole = _visible(identifier.value).strip()
    if len(whole) < _MIN_ASSERTABLE:
        return []
    spans = [whole]
    tokens = whole.split()
    if identifier.kind in _TOKENWISE_KINDS:
        spans += [t for t in tokens if len(t) >= _MIN_ASSERTABLE]
    elif len(tokens) > 2:
        # Contiguous pairs and longer: `Mary Parkinson` from
        # `Mary Jane Parkinson`. Never `Parkinson` alone — see _TOKENWISE_KINDS.
        spans += [
            " ".join(tokens[start : start + length])
            for length in range(2, len(tokens))
            for start in range(len(tokens) - length + 1)
        ]
    return sorted(set(spans), key=len, reverse=True)


@dataclass
class RequestProtection:
    """What the protected input boundary removed from THIS request.

    Run-scoped and populated only by the boundary. A caller can neither read it
    nor add to it, which is what makes the outgoing assertion unforgeable in the
    way a marker in the document is not.
    """

    trace_id: str = ""
    identifiers: list[PrivateIdentifier] = field(default_factory=list)
    #: Text retrieved from approved public sources during this request. Masked
    #: out of an outgoing payload before the identifier assertion runs.
    evidence_texts: list[str] = field(default_factory=list)
    #: Every authorised flow, for the trace and the phase report.
    authorised: list[tuple[str, str, str]] = field(default_factory=list)

    def record_identifier(self, kind: str, value: str, channel: InputChannel) -> None:
        if value and value.strip():
            self.identifiers.append(PrivateIdentifier(kind, value.strip(), channel))

    def register_evidence(self, text: str) -> None:
        """Declare text that came from outside the boundary before this request."""
        if text and text.strip():
            self.evidence_texts.append(_visible(text))

    def leaked_in(self, text: str) -> list[str]:
        """Which of this request's identifiers are readable in `text`.

        Evidence is masked first, so an identifier that appears ONLY inside
        retrieved public text is not reported — it is not a disclosure of this
        patient and refusing it would be a false refusal on the clinical path.
        """
        haystack = _visible(text)
        for evidence in self.evidence_texts:
            if evidence:
                haystack = haystack.replace(evidence, " ")
        found: list[str] = []
        for identifier in self.identifiers:
            for span in _assertable_spans(identifier):
                if span in haystack:
                    found.append(f"{identifier.kind}:{identifier.channel.value}")
                    break
        return sorted(set(found))


_protection: ContextVar[RequestProtection | None] = ContextVar(
    "mao_request_protection", default=None
)


@contextmanager
def protected_request(protection: RequestProtection) -> Iterator[RequestProtection]:
    """Bind this request's protection record for the duration of the block.

    A `ContextVar` for the same reason `mao.core.deadline` uses one: the sink is
    four or five frames below the route, through the gateway and the provider,
    and threading an argument through every signature means any call site that
    forgot it is unguarded again. `run_graph` copies the context into the
    executor worker explicitly, so this reaches the threads that make the calls.
    """
    token = _protection.set(protection)
    try:
        yield protection
    finally:
        _protection.reset(token)


def current_protection() -> RequestProtection | None:
    """This request's protection record, or None outside a request.

    None for a CLI run, an ingestion job or a unit test — none of which has a
    patient to protect. The flow and trust-class checks still apply; only the
    identifier assertion is skipped, and `authorise` says so in the trace.
    """
    return _protection.get()


def authorise(
    *,
    destination: Destination,
    purpose: EgressPurpose,
    trust_class: TrustClass | None,
    texts: Sequence[str] = (),
    payload: object = None,
    evidence: Sequence[PublicEvidence] = (),
) -> ExternalSafePayload:
    """Authorise one external call, or raise `EgressRefused`.

    Returns the envelope that was approved, so the caller has the policy
    decision in hand rather than having made it implicitly.

    `trust_class=None` means "this call carries no data from the request" and is
    valid only for a flow whose policy row is empty — a liveness probe sending a
    fixed string. Spelling it as `None` rather than inventing a trust class
    keeps "carries nothing" and "carries something harmless" distinguishable.
    """
    if not is_named(destination, purpose):
        raise EgressRefused(
            f"no approved flow for {destination.value} / {purpose.value}. "
            "An external call must name a destination and a purpose that the "
            "egress policy records; unnamed flows are refused by default."
        )

    permitted = allowed_classes(destination, purpose)
    if trust_class is None:
        if permitted:
            raise EgressRefused(
                f"{destination.value} / {purpose.value} carries request data "
                f"({sorted(c.value for c in permitted)}), so the caller must "
                "declare its trust class."
            )
    elif trust_class not in permitted:
        raise EgressRefused(
            f"{destination.value} / {purpose.value} may carry "
            f"{sorted(c.value for c in permitted) or 'nothing'}, "
            f"not {trust_class.value}."
        )

    protection = current_protection()
    if protection is not None:
        for text in texts:
            leaked = protection.leaked_in(text)
            if leaked:
                logger.error(
                    "egress refused: %s identifier(s) from this request would "
                    "have reached %s for %s",
                    ",".join(leaked), destination.value, purpose.value,
                )
                raise EgressRefused(
                    f"an identifier this request's protected boundary removed "
                    f"({', '.join(leaked)}) is present in a payload bound for "
                    f"{destination.value}. The call is refused rather than "
                    "repaired: a de-identification defect that reaches the sink "
                    "is a defect upstream."
                )
        protection.authorised.append(
            (destination.value, purpose.value,
             trust_class.value if trust_class else "none")
        )

    return ExternalSafePayload(
        destination=destination.value,
        purpose=purpose.value,
        payload_trust_class=trust_class,
        safe_payload=payload,
        policy_version=EGRESS_POLICY_VERSION,
        trace_id=protection.trace_id if protection else "",
        evidence=tuple(evidence),
    )
