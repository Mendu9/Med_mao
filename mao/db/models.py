from __future__ import annotations
import uuid
from datetime import datetime, timezone
from sqlalchemy import BigInteger, Boolean, Column, Float, ForeignKey, Integer, JSON, SmallInteger, String, Text, Uuid
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def _uuid():
    return uuid.uuid4()


def _now():
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
    created_at         = Column(Text, default=lambda: _now().isoformat())


class ResponseMetrics(Base):
    __tablename__ = "response_metrics"
    id                = Column(Integer, primary_key=True, autoincrement=True)
    request_id        = Column(String, nullable=False)
    session_id        = Column(Uuid(as_uuid=True), ForeignKey("chat_sessions.session_id"))
    user_id           = Column(String, nullable=False)
    agent_used        = Column(String)
    faithfulness      = Column(Float)
    answer_relevancy  = Column(Float)
    context_precision = Column(Float)
    context_recall    = Column(Float)
    latency_ms        = Column(Float)
    created_at        = Column(Text, default=lambda: _now().isoformat())


class ResponseFeedback(Base):
    __tablename__ = "response_feedback"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Uuid(as_uuid=True), ForeignKey("chat_sessions.session_id"))
    user_id    = Column(String, nullable=False)
    rating     = Column(SmallInteger, nullable=False)
    comment    = Column(Text)
    created_at = Column(Text, default=lambda: _now().isoformat())


class LLMJudgeScore(Base):
    __tablename__ = "llm_judge_scores"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(Uuid(as_uuid=True), ForeignKey("chat_sessions.session_id"))
    accuracy     = Column(SmallInteger)
    completeness = Column(SmallInteger)
    safety       = Column(SmallInteger)
    clarity      = Column(SmallInteger)
    notes        = Column(Text)
    created_at   = Column(Text, default=lambda: _now().isoformat())


class GuardrailEvent(Base):
    __tablename__ = "guardrail_events"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    session_id     = Column(Uuid(as_uuid=True))
    guardrail_name = Column(String, nullable=False)
    triggered      = Column(Boolean, nullable=False)
    detail         = Column(Text)
    created_at     = Column(Text, default=lambda: _now().isoformat())
