"""Stored-session endpoints: user feedback and report-card export."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from mao.api.executor import get_executor
from mao.db.repository import get_report_card, save_feedback

logger = logging.getLogger(__name__)

router = APIRouter(tags=["records"])


class FeedbackRequest(BaseModel):
    session_id: str
    thumbs_up: bool
    comment: str = ""


@router.post("/feedback")
async def post_feedback(req: FeedbackRequest) -> dict[str, str]:
    """Record thumbs up/down against a stored session.

    Goes through the shared repository rather than a second raw asyncpg
    connection: the old implementation depended on a package in neither
    requirements file, omitted the NOT NULL user_id, and passed a str against a
    Uuid foreign key — so every call failed with a 500 (P1-7).
    """
    loop = asyncio.get_running_loop()
    saved = await loop.run_in_executor(
        get_executor(),
        lambda: save_feedback(
            session_id=req.session_id, thumbs_up=req.thumbs_up, comment=req.comment
        ),
    )
    if not saved:
        raise HTTPException(status_code=404, detail="Unknown or malformed session_id")
    return {"status": "ok"}


@router.get("/export/report/{session_id}")
async def export_report(session_id: str) -> Response:
    """Render a stored report card as PDF.

    This could never succeed before: nothing wrote `report_card`, so the column
    was always NULL and the endpoint was a permanent 404 (P1-8). The repository
    now persists it on every clinical turn.
    """
    from mao.report.report_card import build_report_card

    try:
        loop = asyncio.get_running_loop()
        card_data = await loop.run_in_executor(
            get_executor(), lambda: get_report_card(session_id)
        )
        if not card_data:
            raise HTTPException(status_code=404, detail="Report not found")
        card = build_report_card(**card_data)
        return Response(
            content=card.to_pdf(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=report_{session_id}.pdf"
            },
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("PDF export failed: %s", exc)
        raise HTTPException(status_code=500, detail="PDF export failed") from exc
