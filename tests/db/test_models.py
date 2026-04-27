import pytest
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from mao.db.models import (
    Base, ChatSession, ResponseMetrics,
    ResponseFeedback, LLMJudgeScore, GuardrailEvent,
)

@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)

@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s

def test_chat_session_insert(session):
    row = ChatSession(session_id=uuid.uuid4(), user_id="user-1", user_query="What is amyloid?")
    session.add(row)
    session.commit()
    result = session.get(ChatSession, row.session_id)
    assert result.user_id == "user-1"
    assert result.uncertainty_flag is False

def test_response_metrics_insert(session):
    sid = uuid.uuid4()
    session.add(ChatSession(session_id=sid, user_id="u", user_query="q"))
    session.commit()
    row = ResponseMetrics(request_id="req-1", session_id=sid, user_id="u", faithfulness=0.85)
    session.add(row)
    session.commit()
    assert session.get(ResponseMetrics, row.id).faithfulness == pytest.approx(0.85)

def test_response_feedback_insert(session):
    sid = uuid.uuid4()
    session.add(ChatSession(session_id=sid, user_id="u", user_query="q"))
    session.commit()
    session.add(ResponseFeedback(session_id=sid, user_id="u", rating=1))
    session.commit()
    assert session.query(ResponseFeedback).first().rating == 1

def test_llm_judge_scores_insert(session):
    sid = uuid.uuid4()
    session.add(ChatSession(session_id=sid, user_id="u", user_query="q"))
    session.commit()
    session.add(LLMJudgeScore(session_id=sid, accuracy=8, safety=10))
    session.commit()
    assert session.query(LLMJudgeScore).first().accuracy == 8

def test_guardrail_event_insert(session):
    session.add(GuardrailEvent(guardrail_name="token_limit", triggered=True, detail="550 tokens"))
    session.commit()
    assert session.query(GuardrailEvent).first().triggered is True
