r"""Model gateway — the single entry point for role-addressed LLM calls.

Agents call `complete(role=..., messages=..., purpose=...)`. They never name a
model id and never import a provider SDK. Provider-specific code stays behind
`mao.providers.llm`.

## Why `purpose` is required and has no default

This function is the seam every byte of model traffic crosses, so it is where
the egress policy can actually be enforced rather than described. A default
purpose would defeat that immediately: whichever value was chosen would become
the one every new call site silently inherits, and "no trust class reaches an
unnamed destination by default" would be false the first time somebody added an
agent. A call that cannot say why it is talking to a third party does not talk
to a third party.

## Why clinical synthesis is a different function

`complete()` takes free-text messages. The approved M-1 decision says external
clinical synthesis receives `SafeSynthesisContext` plus `PublicEvidence` and
nothing else — so for that purpose there must be no parameter through which
free text can arrive at all. `synthesise_clinical()` takes the typed projection
and renders the user turn itself; `complete()` refuses the purpose outright.

That is the difference between a rule and a type. The previous six waves each
enforced a rule of this shape by remembering to call a scrubber, and each was
defeated by a call site that did not.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from mao.providers import usage
from mao.providers.llm.base import ChatProvider
from mao.providers.registry import ModelRecord, ModelRegistry, ModelRole
from mao.trust.classes import PublicEvidence, SafeSynthesisContext, TrustClass
from mao.trust.egress.gateway import EgressRefused, authorise
from mao.trust.egress.policy import Destination, EgressPurpose

logger = logging.getLogger(__name__)

_registry: ModelRegistry | None = None
_provider: ChatProvider | None = None

# How much extra reasoning room a retry gets after a truncated, empty reply.
# Multiplies the declared overhead rather than the answer budget: it is the
# analysis channel that overran, not the answer.
_TRUNCATION_RETRY_FACTOR = 3


def registry() -> ModelRegistry:
    """Process-wide ModelRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry.default()
    return _registry


def reset_registry() -> None:
    """Drop the cached registry so env changes take effect. Test/CLI use only."""
    global _registry
    _registry = None


def resolve(role: ModelRole) -> ModelRecord:
    return registry().resolve(role)


def model_id_for(role: ModelRole) -> str:
    """The concrete, active model id bound to `role`."""
    return registry().model_id_for(role)


@dataclass(frozen=True)
class Completion:
    """A gateway response plus the provenance a trace needs."""

    text: str
    model_id: str
    provider: str
    role: ModelRole
    input_tokens: int = 0
    output_tokens: int = 0
    # The model hit its ceiling before finishing. An empty `text` then means
    # "cut off mid-thought", which is an infrastructure failure — not "the model
    # declined to answer", which is a verdict. Wave 6 blocker 3 was invisible
    # precisely because those two were the same empty string.
    truncated: bool = False

    @property
    def estimated_cost_usd(self) -> float:
        record = registry().get(self.model_id)
        if record is None:
            return 0.0
        return (
            self.input_tokens * record.cost_per_1m_input_usd
            + self.output_tokens * record.cost_per_1m_output_usd
        ) / 1_000_000


# ---------------------------------------------------------------------------
# Provider binding
# ---------------------------------------------------------------------------


def provider() -> ChatProvider:
    """The active chat provider, defaulting to Groq."""
    global _provider
    if _provider is None:
        from mao.providers.llm.groq_provider import GroqChatProvider

        _provider = GroqChatProvider()
    return _provider


def set_provider(new_provider: ChatProvider) -> None:
    """Bind a provider. Used by tests and by deployment wiring."""
    global _provider
    _provider = new_provider


def reset_provider() -> None:
    """Restore the default provider."""
    global _provider
    _provider = None


def _outgoing_text(messages: list[dict]) -> list[str]:
    """Every string a message carries, including inside multipart content.

    Multipart content is how the vision role sends `{"type": "text", ...}`
    alongside an image part. Reading only `str` contents would have skipped the
    text half of exactly the request most likely to carry a patient's details.
    """
    texts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
    return texts


def complete(
    *,
    role: ModelRole,
    messages: list[dict],
    purpose: EgressPurpose,
    # No default. A trust class a call site did not state is a trust class
    # this signature asserted on its behalf, and `multimodal_agent` sending a
    # base64 patient scan as SAFE_DERIVED_TEXT is exactly what that produced
    # (A-5, ADV16-7). `EgressPurpose` already refuses to carry a default for
    # this reason - "whichever purpose was chosen as the default would silently
    # become the one every new call site inherits" - and the same argument
    # applies with more force to the class, because the class is what
    # `policy._validate` checks and therefore what makes the table fail closed.
    trust_class: TrustClass | None,
    evidence: Sequence[PublicEvidence] = (),
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Completion:
    """Run a completion for a capability role.

    `purpose` is required and is checked against the egress policy before the
    provider is touched. `trust_class` is required and has no default: it
    declares what the messages carry, and the policy table decides whether that
    is admissible for this purpose. `SAFE_DERIVED_TEXT` is the honest name for
    text that has been through the protected input boundary; it is not
    admissible for clinical synthesis - see `synthesise_clinical` - and it was
    never an honest name for a base64 patient scan.

    `max_tokens` is the budget for the **answer**. The bound model's analysis
    channel is paid for on top of it, from the record's declared
    `reasoning_overhead_tokens`, because the provider counts both against one
    ceiling. A call site therefore states what it needs to read back, and
    rebinding a role to a model that thinks harder re-sizes every one of that
    role's call sites at once.

    Getting this wrong is silent: the model spends the ceiling reasoning and
    returns an empty `content`, which every parser downstream reads as "said
    nothing". That is Wave 6 blocker 3, and it refused 5 of 5 ordinary clinical
    questions while the test suite stayed green.

    The declared overhead is sized from measurement, not from worst-case
    paranoia, because the provider's rate limiter charges the *requested*
    ceiling and not the tokens actually produced — a 429 observed during
    remediation reported "Limit 200000, Used 199385, Requested 2333". An
    over-generous ceiling therefore costs real quota on every call, including
    the overwhelming majority that never approach it.

    So the tail is handled by retrying rather than by pre-paying for it: if the
    model is cut off mid-thought and returns nothing readable, the call is made
    once more with a materially larger allowance. The common case stays cheap,
    the rare case self-heals, and neither one returns the silent empty string
    that was Wave 6 blocker 3.

    There is deliberately no `model_id` parameter: business logic must not be
    able to bypass role resolution and hand a raw (possibly retired) id to a
    provider.
    """
    if purpose is EgressPurpose.CLINICAL_SYNTHESIS:
        raise EgressRefused(
            "clinical synthesis does not accept free-text messages. Build a "
            "SafeSynthesisContext through the Safe Handoff Compiler and call "
            "gateway.synthesise_clinical(). Under the approved M-1 policy an "
            "external synthesis model receives SafeSynthesisContext plus "
            "PublicEvidence and nothing else — a scrubbed report is not a "
            "synthesis payload."
        )
    authorise(
        destination=Destination.MODEL_PROVIDER,
        purpose=purpose,
        trust_class=trust_class,
        texts=_outgoing_text(messages),
        evidence=evidence,
    )
    record = resolve(role)
    active = provider()
    raw = active.complete(
        model_id=record.model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens + record.reasoning_overhead_tokens,
    )

    # Cut off before it said anything. Not a verdict — a budget failure.
    if raw.truncated and not raw.text.strip():
        retry_ceiling = max_tokens + record.reasoning_overhead_tokens * _TRUNCATION_RETRY_FACTOR
        logger.warning(
            "role=%s model=%s produced no content before its ceiling (answer %d "
            "+ overhead %d); retrying once at %d",
            role.name, record.model_id, max_tokens,
            record.reasoning_overhead_tokens, retry_ceiling,
        )
        raw = active.complete(
            model_id=record.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=retry_ceiling,
        )
        if raw.truncated and not raw.text.strip():
            # Report it loudly. Every parser downstream fails closed on an empty
            # reply, so this becomes a refused clinical answer, and the declared
            # overhead for this model is the thing to fix.
            logger.error(
                "role=%s model=%s returned no content even at %d tokens — raise "
                "reasoning_overhead_tokens for this model",
                role.name, record.model_id, retry_ceiling,
            )
    elif raw.truncated:
        logger.warning(
            "role=%s model=%s hit its ceiling (answer budget %d + reasoning "
            "overhead %d) — the answer may be cut short",
            role.name, record.model_id, max_tokens, record.reasoning_overhead_tokens,
        )

    completion = Completion(
        text=raw.text,
        model_id=record.model_id,
        provider=active.name,
        role=role,
        input_tokens=raw.input_tokens,
        output_tokens=raw.output_tokens,
        truncated=raw.truncated,
    )
    # Accounting happens here so no agent has to carry it. A no-op unless the
    # request bound a collector.
    usage.record(
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cost_usd=completion.estimated_cost_usd,
    )
    return completion


#: The system instruction for a typed clinical synthesis.
#:
#: The model is told what it is receiving, because it is receiving something
#: different from what it used to: a minimum-necessary projection rather than a
#: de-identified document. Telling it the case description is deliberately
#: partial is what stops it filling the gaps — the previous prompt said the
#: report "has already been de-identified", which invited exactly that.
_CLINICAL_SYNTHESIS_FRAMING = (
    "The case below is a minimum-necessary projection of a patient record. It "
    "contains no identifiers and it is not the full record: facts that could "
    "not be carried safely are absent rather than summarised. Reason only from "
    "what is stated and from the evidence supplied. Where a fact you would need "
    "is missing, say which fact and do not assume it."
)


def synthesise_clinical(
    *,
    system_prompt: str,
    context: SafeSynthesisContext,
    evidence: Sequence[PublicEvidence] = (),
    role: ModelRole = ModelRole.CLINICAL_SYNTHESIS,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> Completion:
    """External clinical synthesis, from typed facts only.

    The user turn is rendered HERE, from `context.render()` and the evidence.
    There is no parameter through which a caller can add free text, so the
    approved M-1 policy is a property of the signature rather than a rule a call
    site has to remember — which is the difference between this and every
    previous attempt at the same guarantee.
    """
    if not isinstance(context, SafeSynthesisContext):
        raise EgressRefused(
            "clinical synthesis requires a SafeSynthesisContext built by the "
            f"Safe Handoff Compiler, not {type(context).__name__}."
        )

    rendered_evidence = "\n\n".join(
        f"[{index}] {item.title or item.source_id or item.evidence_id}\n{item.text}"
        for index, item in enumerate(evidence, 1)
    )
    user_turn = (
        f"{_CLINICAL_SYNTHESIS_FRAMING}\n\n"
        f"## Case\n{context.render()}\n\n"
        f"## Evidence\n{rendered_evidence or 'No evidence was retrieved.'}"
    )

    authorise(
        destination=Destination.MODEL_PROVIDER,
        purpose=EgressPurpose.CLINICAL_SYNTHESIS,
        trust_class=TrustClass.SAFE_SYNTHESIS_CONTEXT,
        # Only the case half is asserted against this request's identifiers.
        # The evidence came from outside the boundary before the request began.
        texts=[context.render()],
        payload=context,
        evidence=tuple(evidence),
    )

    record = resolve(role)
    active = provider()
    raw = active.complete(
        model_id=record.model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_turn},
        ],
        temperature=temperature,
        max_tokens=max_tokens + record.reasoning_overhead_tokens,
    )
    completion = Completion(
        text=raw.text,
        model_id=record.model_id,
        provider=active.name,
        role=role,
        input_tokens=raw.input_tokens,
        output_tokens=raw.output_tokens,
        truncated=raw.truncated,
    )
    usage.record(
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cost_usd=completion.estimated_cost_usd,
    )
    return completion


def stream(
    *,
    role: ModelRole,
    messages: list[dict],
    purpose: EgressPurpose,
    # No default. A trust class a call site did not state is a trust class
    # this signature asserted on its behalf, and `multimodal_agent` sending a
    # base64 patient scan as SAFE_DERIVED_TEXT is exactly what that produced
    # (A-5, ADV16-7). `EgressPurpose` already refuses to carry a default for
    # this reason - "whichever purpose was chosen as the default would silently
    # become the one every new call site inherits" - and the same argument
    # applies with more force to the class, because the class is what
    # `policy._validate` checks and therefore what makes the table fail closed.
    trust_class: TrustClass | None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Iterator[str]:
    """Stream a completion for a capability role.

    Same contract as `complete`, including the egress authorisation and the
    reasoning-overhead budgeting. Streaming exists on the gateway so the API
    layer never has to reach past it into `mao.core.llm` to get incremental
    delivery — which it did, and which also meant that path resolved its model
    from a config alias rather than from the registry.

    A streamed response is subject to the SAME egress policy as a buffered one.
    Transport is not a safety input, and a purpose that may not send free text
    when the answer is buffered may not send it when the answer is streamed.
    """
    if purpose is EgressPurpose.CLINICAL_SYNTHESIS:
        raise EgressRefused(
            "clinical synthesis does not accept free-text messages on the "
            "streaming path either. Transport is not a safety input."
        )
    authorise(
        destination=Destination.MODEL_PROVIDER,
        purpose=purpose,
        trust_class=trust_class,
        texts=_outgoing_text(messages),
    )
    record = resolve(role)
    return provider().stream(
        model_id=record.model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens + record.reasoning_overhead_tokens,
    )
