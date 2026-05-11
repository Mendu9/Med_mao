import logging
import re
from enum import Enum

from fastapi import HTTPException

from mao.guardrails.db_helper import log_guardrail_event

logger = logging.getLogger(__name__)


class GuardrailSeverity(str, Enum):
    BLOCK = "block"    # Hard block — raises HTTPException(400)
    WARN = "warn"      # Soft warning — logged, request continues
    INFO = "info"      # Informational — logged silently


_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"disregard\s+(your|the)\s+(system|previous)", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bDAN\s+mode\b", re.IGNORECASE),
]

_MAX_TOKENS = 500
_WARN_TOKENS = 400  # soft warning before hard block


def _count_tokens(text: str) -> int:
    return len(text.split())


def _scrub_pii(text: str) -> str:
    from mao.core.pii_scrubber import scrub_pii
    return scrub_pii(text)


async def apply_input_guardrails(query: str, session_id: str) -> None:
    """Apply tiered input safety checks.

    BLOCK — prompt injection or >500 tokens → HTTPException(400)
    WARN  — 400-500 tokens → logged, request continues
    INFO  — PII detected → logged, never blocked
    """
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

    # INFO: PII detected — log but never block
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
