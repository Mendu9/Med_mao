import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_guardrail_event(
    session_id: str,
    name: str,
    triggered: bool,
    detail: str | None = None,
    severity: Any = None,
) -> None:
    """Fire-and-forget write to guardrail_events table. Never raises.

    severity is optional (GuardrailSeverity enum or str).  Its value is
    prepended to detail as ``[block] ...`` so it is visible in the DB without
    a schema change.
    """
    try:
        from mao.db import get_db_session
        from mao.db.models import GuardrailEvent
        sev = severity.value if hasattr(severity, "value") else (str(severity) if severity else None)
        stored_detail: str | None
        if sev and detail:
            stored_detail = f"[{sev}] {detail}"
        elif sev:
            stored_detail = f"[{sev}]"
        else:
            stored_detail = detail
        with get_db_session() as session:
            event = GuardrailEvent(
                session_id=session_id,
                guardrail_name=name,
                triggered=triggered,
                detail=stored_detail,
            )
            session.add(event)
            # get_db_session context manager commits on exit — no explicit commit needed
    except Exception as exc:
        logger.warning("Failed to log guardrail event %s: %s", name, exc)
