from __future__ import annotations
import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, Column, DateTime,
    Float, ForeignKey, Integer, JSON, SmallInteger, String, Text,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Uuid


class Base(DeclarativeBase):
    pass


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> datetime:
    # Always UTC — critical for clinical audit trail
    return datetime.now(timezone.utc)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    session_id         = Column(Uuid(as_uuid=True), primary_key=True, default=_uuid)
    user_id            = Column(String, nullable=False)
    user_query         = Column(Text, nullable=False)
    pii_scrubbed_query = Column(Text)
    response           = Column(Text)
    agent_used         = Column(String)
    domain             = Column(String)
    report_card        = Column(JSON)
    council_verdict    = Column(JSON)
    nli_flags          = Column(JSON)
    uncertainty_flag   = Column(Boolean, default=False)
    created_at         = Column(DateTime(timezone=True), default=_now, nullable=False)


class ResponseMetrics(Base):
    __tablename__ = "response_metrics"
    id                = Column(Integer, primary_key=True, autoincrement=True)
    request_id        = Column(String, nullable=False)
    session_id        = Column(Uuid(as_uuid=True), ForeignKey("chat_sessions.session_id"), nullable=False)
    user_id           = Column(String, nullable=False)
    agent_used        = Column(String)
    faithfulness      = Column(Float)
    answer_relevancy  = Column(Float)
    context_precision = Column(Float)
    context_recall    = Column(Float)
    latency_ms        = Column(Float)
    created_at        = Column(DateTime(timezone=True), default=_now, nullable=False)


class ResponseFeedback(Base):
    __tablename__ = "response_feedback"
    __table_args__ = (
        CheckConstraint("rating IN (-1, 1)", name="ck_feedback_rating"),
    )
    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Uuid(as_uuid=True), ForeignKey("chat_sessions.session_id"), nullable=False)
    user_id    = Column(String, nullable=False)
    rating     = Column(SmallInteger, nullable=False)
    comment    = Column(Text)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)


class LLMJudgeScore(Base):
    __tablename__ = "llm_judge_scores"
    __table_args__ = (
        CheckConstraint("accuracy BETWEEN 1 AND 10", name="ck_accuracy_range"),
        CheckConstraint("completeness BETWEEN 1 AND 10", name="ck_completeness_range"),
        CheckConstraint("safety BETWEEN 1 AND 10", name="ck_safety_range"),
        CheckConstraint("clarity BETWEEN 1 AND 10", name="ck_clarity_range"),
    )
    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(Uuid(as_uuid=True), ForeignKey("chat_sessions.session_id"), nullable=False)
    accuracy     = Column(SmallInteger)
    completeness = Column(SmallInteger)
    safety       = Column(SmallInteger)
    clarity      = Column(SmallInteger)
    notes        = Column(Text)
    created_at   = Column(DateTime(timezone=True), default=_now, nullable=False)


class GuardrailEvent(Base):
    __tablename__ = "guardrail_events"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    session_id     = Column(String)  # nullable — fire-and-forget
    guardrail_name = Column(String, nullable=False)
    triggered      = Column(Boolean, nullable=False)
    detail         = Column(Text)
    created_at     = Column(DateTime(timezone=True), default=_now, nullable=False)
