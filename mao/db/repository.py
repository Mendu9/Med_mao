"""Typed persistence for chat sessions, report cards, and feedback.

All access goes through the SQLAlchemy session factory, so there is exactly one
engine, one URL, and one connection pool. The previous raw-asyncpg endpoints
used a second, differently-configured connection and depended on a package that
was in neither requirements file (P1-7).
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, cast

from mao.db import get_db_session
from mao.db.models import ChatSession, ResponseFeedback

logger = logging.getLogger(__name__)


def _as_uuid(value: str | uuid.UUID) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def save_chat_session(
    *,
    user_id: str,
    user_query: str,
    pii_scrubbed_query: str,
    response: str,
    agent_used: str,
    domain: str | None = None,
    uncertainty_flag: bool = False,
    council_verdict: dict[str, Any] | None = None,
    nli_flags: list[dict[str, Any]] | None = None,
    report_card: dict[str, Any] | None = None,
) -> uuid.UUID:
    """Persist one turn with its full clinical audit trail.

    Writes the four columns the audit found permanently NULL — pii_scrubbed_query,
    council_verdict, nli_flags, report_card (P1-9) — so a streamed clinical answer
    leaves the same record as a non-streamed one, and /export/report can succeed (P1-8).
    """
    session_id = uuid.uuid4()
    with get_db_session() as db:
        db.add(
            ChatSession(
                session_id=session_id,
                user_id=user_id,
                user_query=user_query,
                pii_scrubbed_query=pii_scrubbed_query,
                response=response,
                agent_used=agent_used,
                domain=domain,
                uncertainty_flag=bool(uncertainty_flag),
                council_verdict=council_verdict,
                nli_flags=nli_flags,
                report_card=report_card,
            )
        )
    return session_id


def get_chat_session(session_id: str | uuid.UUID) -> ChatSession | None:
    sid = _as_uuid(session_id)
    if sid is None:
        return None
    with get_db_session() as db:
        row = db.get(ChatSession, sid)
        if row is not None:
            db.expunge(row)
        return row


def get_report_card(session_id: str | uuid.UUID) -> dict[str, Any] | None:
    row = get_chat_session(session_id)
    if row is None:
        return None
    # models.py uses legacy Column() rather than Mapped[], so the attribute
    # types as Column[Any] to mypy even though it is a plain value at runtime.
    return cast("dict[str, Any] | None", row.report_card)


def save_feedback(*, session_id: str | uuid.UUID, thumbs_up: bool, comment: str = "") -> bool:
    """Record thumbs up/down against an existing session.

    Returns False rather than raising when the session id is malformed or
    unknown — the previous implementation violated both the NOT NULL user_id
    and the Uuid foreign key, so every call failed with a 500 (P1-7).
    """
    sid = _as_uuid(session_id)
    if sid is None:
        logger.info("Feedback rejected: malformed session_id")
        return False
    try:
        with get_db_session() as db:
            parent = db.get(ChatSession, sid)
            if parent is None:
                logger.info("Feedback rejected: unknown session")
                return False
            db.add(
                ResponseFeedback(
                    session_id=sid,
                    user_id=parent.user_id,   # NOT NULL, inherited from the session
                    rating=1 if thumbs_up else -1,
                    comment=comment,
                )
            )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Feedback insert failed: %s", exc)
        return False


def get_feedback_user_id(session_id: str | uuid.UUID) -> str | None:
    sid = _as_uuid(session_id)
    if sid is None:
        return None
    with get_db_session() as db:
        row = (
            db.query(ResponseFeedback)
            .filter(ResponseFeedback.session_id == sid)
            .order_by(ResponseFeedback.id.desc())
            .first()
        )
        if row is None:
            return None
        return cast("str", row.user_id)
