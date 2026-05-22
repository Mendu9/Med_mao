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


_INJECTION_PATTERNS = [
    # Prompt injection classics
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(your|the)\s+(system|previous)", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    # Roleplay / persona hijack
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"\bDAN\s+mode\b", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+are|a\s+)", re.IGNORECASE),
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
    # Persona bypass variants
    re.compile(r"act\s+like\s+", re.IGNORECASE),
]

_MAX_TOKENS = 500
_WARN_TOKENS = 400  # soft warning before hard block

# Off-topic domain filter — this is a medical/clinical AI, not a general coding assistant
_OFF_TOPIC_PATTERNS = [
    re.compile(r"\bwrite\s+(a\s+)?(python|javascript|java|c\+\+|typescript|rust|go|sql)\b", re.IGNORECASE),
    re.compile(r"\b(implement|create|build)\s+(a\s+)?(function|class|script|program|app|api)\b", re.IGNORECASE),
    re.compile(r"\bcode\s+(for|to)\s+", re.IGNORECASE),
    re.compile(r"\bhow\s+do\s+i\s+(install|deploy|configure|setup|run)\b", re.IGNORECASE),
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


def _scrub_pii(text: str) -> str:
    from mao.core.pii_scrubber import scrub_pii
    return scrub_pii(text)


async def apply_input_guardrails(query: str, session_id: str) -> str:
    """Apply tiered input safety checks. Returns the PII-scrubbed query.

    BLOCK — prompt injection or >500 tokens → HTTPException(400)
    WARN  — 400-500 tokens → logged, request continues
    INFO  — PII detected → logged, never blocked
    Returns the scrubbed query string (original if no PII found).
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

    # INFO: PII detected — scrub and log, never block
    scrubbed = query
    try:
        scrubbed = _scrub_pii(query)
        if scrubbed != query:
            log_guardrail_event(
                session_id, "pii_detected",
                triggered=True, detail="scrubbed",
                severity=GuardrailSeverity.INFO,
            )
    except Exception as exc:
        logger.warning("PII scrubber failed (non-fatal): %s", exc)

    return scrubbed
