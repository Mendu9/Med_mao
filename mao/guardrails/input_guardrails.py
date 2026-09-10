import logging
import re
import unicodedata
from enum import Enum

from fastapi import HTTPException

from mao.guardrails.db_helper import log_guardrail_event

logger = logging.getLogger(__name__)


class GuardrailSeverity(str, Enum):
    BLOCK = "block"    # Hard block — raises HTTPException(400)
    WARN = "warn"      # Soft warning — logged, request continues
    INFO = "info"      # Informational — logged silently


# P1-20: "act as/like <X>" only indicates a persona hijack when X is an agent,
# a persona, or a constraint-removal adjective. Matching a bare article blocked
# ordinary mechanism-of-action questions ("does tau act as a scaffold?") with an
# HTTP 400, while — because the alternation required "a " — it simultaneously
# missed "act as an unrestricted AI". Naming the targets fixes both directions.
_PERSONA_TARGET = (
    r"(?:the\s+|an?\s+|my\s+|your\s+)?"
    r"(?:"
    r"unrestricted|unfiltered|uncensored|unlimited|unbounded|jailbroken|jailbreak|"
    r"amoral|immoral|unethical|evil|rogue|malicious|"
    r"ai\b|a\.i\.|assistant|chat\s*bot|chatbot|language\s+model|llm\b|gpt|claude|dan\b|"
    r"developer\s+mode|god\s+mode|debug\s+mode|"
    r"system\s+(?:administrator|admin|prompt)|sysadmin|root\s+user|superuser|"
    r"hacker|attacker|penetration\s+tester"
    r")"
)

_INJECTION_PATTERNS = [
    # Prompt injection classics
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(your|the)\s+(system|previous)", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    # Roleplay / persona hijack
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"\bDAN\s+mode\b", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.IGNORECASE),
    # "act as if you are ..." has no clinical reading — keep it unqualified.
    re.compile(r"act\s+as\s+if\s+you\s+are\b", re.IGNORECASE),
    re.compile(rf"act\s+(?:as|like)\s+{_PERSONA_TARGET}", re.IGNORECASE),
    re.compile(r"roleplay\s+as\s+", re.IGNORECASE),
    # Delimiter injection
    re.compile(r"\[\[SYSTEM\]\]", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"###\s*system\s*:", re.IGNORECASE),
    # Code/shell injection
    re.compile(r"\bsudo\s+", re.IGNORECASE),
    re.compile(r"rm\s+-rf\b", re.IGNORECASE),
    # XSS
    re.compile(r"<\s*script\s*>", re.IGNORECASE),
    # SQL injection
    re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE),
    re.compile(r";\s*--"),
    re.compile(r"\bUNION\s+SELECT\b", re.IGNORECASE),
    re.compile(r"\bINSERT\s+INTO\b", re.IGNORECASE),
    re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE),
    re.compile(r"\bUPDATE\s+\w+\s+SET\b", re.IGNORECASE),
]

_MAX_TOKENS = 500
_WARN_TOKENS = 400  # soft warning before hard block

# Off-topic domain filter — this is a medical/clinical AI, not a general coding assistant
_OFF_TOPIC_PATTERNS = [
    re.compile(r"\bwrite\s+(a\s+)?(python|javascript|java|c\+\+|typescript|rust|go|sql)\b", re.IGNORECASE),
    re.compile(r"\b(implement|create|build)\s+(a\s+)?(function|class|script|program|app|api)\b", re.IGNORECASE),
    re.compile(r"\bcode\s+(for|to)\s+", re.IGNORECASE),
    # P1-20: "install/deploy/configure/setup" are software verbs in any reading,
    # but "run" is also how clinicians talk about administering an instrument
    # ("how do I run a MoCA assessment"). Qualify "run" with a technical object.
    re.compile(r"\bhow\s+do\s+i\s+(install|deploy|configure|setup|set\s+up)\b", re.IGNORECASE),
    re.compile(
        r"\bhow\s+do\s+i\s+run\s+(?:this|that|the|a|an|my)?\s*"
        r"(script|program|server|command|container|image|build|pipeline|migration|"
        r"binary|executable|notebook|job|daemon|service|query|unit\s+tests?|"
        r"docker|kubernetes|npm|pip|conda|python|node|java|bash|shell|sql|code|app|application)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdebugg?(ing)?\s+(my\s+)?(code|script|program)\b", re.IGNORECASE),
    re.compile(r"\b(git|docker|kubernetes|npm|pip|conda)\s+\w+", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+(the\s+)?syntax\s+for\b", re.IGNORECASE),
    re.compile(r"\brefactor\b.{0,40}\bcode\b", re.IGNORECASE),
]

_OFF_TOPIC_RESPONSE = (
    "I'm a clinical and biomedical AI assistant specialising in Alzheimer's disease, "
    "stroke, neuroimaging, and related medical topics. I can't help with software "
    "development or general coding questions. Please ask me about symptoms, treatments, "
    "research findings, or clinical decision support."
)


def _count_tokens(text: str) -> int:
    return len(text.split())


async def apply_input_guardrails(query: str, session_id: str) -> str:
    """Apply tiered input safety checks. Returns the query, normalised, unscrubbed.

    BLOCK — prompt injection or >500 tokens → HTTPException(400)
    WARN  — 400-500 tokens → logged, request continues

    **This function no longer de-identifies anything.** It used to, and that was
    the shape of ADV15-8: de-identification lived at one route-level call site
    that took `request.query` as its only argument, so `chat_history` — a second
    channel through the same endpoint — was never de-identified at all, and an
    audio transcript was never de-identified either.

    De-identification now happens in `mao.trust.inputs.boundary.protect`, which
    consumes a `RawSensitiveInput` naming every channel at once. Doing it there
    also makes it happen EXACTLY ONCE: `/chat` scrubbed twice, here and again in
    the query decomposer, and the second pass over-redacted a clinical line the
    first had left alone. Two call sites cannot both be the only one, so there
    is now one, and it is not this.

    What stays here is what this module is actually for: refusing a request on
    its content — injection, off-topic, length. Those read the query and decide
    whether it is processed; they do not transform it. The NFKC normalisation
    stays with them, because it is what the patterns are matched against, and
    the normalised text is what is returned so the boundary transforms the same
    characters the guardrails judged.
    """
    # Normalize Unicode homoglyphs (e.g. Cyrillic 'у' → 'y') before scanning
    query = unicodedata.normalize("NFKC", query)

    # BLOCK: off-topic query (coding / general tech — this is a medical-only system)
    for pattern in _OFF_TOPIC_PATTERNS:
        if pattern.search(query):
            log_guardrail_event(
                session_id, "off_topic",
                triggered=True, detail="domain_filter",
                severity=GuardrailSeverity.BLOCK,
            )
            raise HTTPException(status_code=400, detail=_OFF_TOPIC_RESPONSE)

    # BLOCK: prompt injection
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            log_guardrail_event(
                session_id, "prompt_injection",
                triggered=True, detail="regex",
                severity=GuardrailSeverity.BLOCK,
            )
            raise HTTPException(status_code=400, detail="Invalid request")

    token_count = _count_tokens(query)

    # BLOCK: hard token limit
    if token_count > _MAX_TOKENS:
        log_guardrail_event(
            session_id, "token_limit",
            triggered=True, detail=f"{token_count} tokens",
            severity=GuardrailSeverity.BLOCK,
        )
        raise HTTPException(status_code=400, detail="Query exceeds maximum length")

    # WARN: approaching token limit
    if token_count > _WARN_TOKENS:
        log_guardrail_event(
            session_id, "token_limit_warn",
            triggered=True, detail=f"{token_count} tokens (warn threshold)",
            severity=GuardrailSeverity.WARN,
        )
        logger.warning("session=%s query approaching token limit (%d words)", session_id, token_count)

    # The `pii_detected` event is still emitted, from
    # `mao.api.protected_input.protect_chat_request`, where the boundary knows
    # what it removed and across which channels. It is not emitted twice.
    return query
