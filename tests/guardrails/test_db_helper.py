from unittest.mock import patch, MagicMock


def test_log_guardrail_event_does_not_raise() -> None:
    # get_db_session is imported inside the function from mao.db, patch it there
    with patch("mao.db.get_db_session", side_effect=Exception("DB unavailable")):
        from mao.guardrails.db_helper import log_guardrail_event
        log_guardrail_event("session-123", "prompt_injection", triggered=True, detail="test")


def test_log_guardrail_event_writes_row() -> None:
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    with patch("mao.db.get_db_session", return_value=mock_session):
        from mao.guardrails.db_helper import log_guardrail_event
        log_guardrail_event("session-abc", "token_limit", triggered=True)

    mock_session.add.assert_called_once()
    added_obj = mock_session.add.call_args[0][0]
    assert added_obj.guardrail_name == "token_limit"
    assert added_obj.triggered is True
