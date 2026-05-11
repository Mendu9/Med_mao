import logging

logger = logging.getLogger(__name__)


def log_guardrail_event(
    session_id: str,
    name: str,
    triggered: bool,
    detail: str | None = None,
) -> None:
    """Fire-and-forget write to guardrail_events table. Never raises."""
    try:
        from mao.db import get_db_session
        from mao.db.models import GuardrailEvent
        with get_db_session() as session:
            event = GuardrailEvent(
                session_id=session_id,
                guardrail_name=name,
                triggered=triggered,
                detail=detail,
            )
            session.add(event)
            session.commit()
    except Exception as exc:
        logger.warning("Failed to log guardrail event %s: %s", name, exc)
