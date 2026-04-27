import pytest
import uuid
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from mao.db.models import (
    Base, ChatSession, ResponseMetrics,
    ResponseFeedback, LLMJudgeScore, GuardrailEvent,
)


@pytest.fixture(scope="module")
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture
def session(engine):
    connection = engine.connect()
    transaction = connection.begin()
    s = Session(bind=connection)
    yield s
    s.close()
    transaction.rollback()
    connection.close()


def test_chat_session_insert(session):
    sid = uuid.uuid4()
    row = ChatSession(session_id=sid, user_id="user-1", user_query="What is amyloid?")
    session.add(row)
    session.flush()
    result = session.get(ChatSession, sid)
    assert result.user_id == "user-1"
    assert result.uncertainty_flag is False


def test_response_metrics_insert(session):
    sid = uuid.uuid4()
    session.add(ChatSession(session_id=sid, user_id="u", user_query="q"))
    session.flush()
    row = ResponseMetrics(request_id="req-1", session_id=sid, user_id="u", faithfulness=0.85)
    session.add(row)
    session.flush()
    result = session.execute(select(ResponseMetrics).where(ResponseMetrics.request_id == "req-1")).scalar_one()
    assert result.faithfulness == pytest.approx(0.85)


def test_response_feedback_insert(session):
    sid = uuid.uuid4()
    session.add(ChatSession(session_id=sid, user_id="u", user_query="q"))
    session.flush()
    session.add(ResponseFeedback(session_id=sid, user_id="u", rating=1))
    session.flush()
    result = session.execute(select(ResponseFeedback)).scalar_one()
    assert result.rating == 1


def test_llm_judge_scores_insert(session):
    sid = uuid.uuid4()
    session.add(ChatSession(session_id=sid, user_id="u", user_query="q"))
    session.flush()
    session.add(LLMJudgeScore(session_id=sid, accuracy=8, safety=10))
    session.flush()
    result = session.execute(select(LLMJudgeScore)).scalar_one()
    assert result.accuracy == 8


def test_guardrail_event_insert(session):
    session.add(GuardrailEvent(guardrail_name="token_limit", triggered=True, detail="550 tokens"))
    session.flush()
    result = session.execute(select(GuardrailEvent)).scalar_one()
    assert result.triggered is True
