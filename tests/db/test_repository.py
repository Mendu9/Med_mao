"""Persistence must actually write the clinical audit trail, and feedback must work.

P1-7  /feedback imported asyncpg (in neither requirements file), omitted the
      NOT NULL user_id, and passed a str session_id against a Uuid FK column.
P1-8  /export/report could never succeed because report_card was never written.
P1-9  pii_scrubbed_query, council_verdict, nli_flags, report_card were never
      populated, and the streaming path wrote no ChatSession row at all.
P1-10 three different DB config sources could point at different databases.
"""
from __future__ import annotations

import uuid

import pytest

from mao.db import repository


@pytest.fixture
def db(tmp_path):
    """A real SQLite-backed session factory — no mocks, no external service."""
    from mao.db import configure_engine, reset_engine

    configure_engine(f"sqlite:///{tmp_path / 'test.db'}")
    yield
    reset_engine()


class TestChatSessionPersistence:
    def test_saving_returns_the_session_id(self, db) -> None:
        sid = repository.save_chat_session(
            user_id="u1", pii_scrubbed_query="q", response="r", agent_used="graphrag"
        )
        assert isinstance(sid, uuid.UUID)

    def test_clinical_audit_columns_are_written(self, db) -> None:
        """P1-9 — these four columns existed but nothing ever populated them."""
        sid = repository.save_chat_session(
            user_id="u1",
            pii_scrubbed_query="raw query with [SSN]",
            response="r",
            agent_used="clinical",
            council_verdict={"passed": True, "blocked_by": None},
            nli_flags=[{"claim": "c", "entailed": True}],
            report_card={"stage": "CN"},
            domain="alzheimer",
            uncertainty_flag=True,
        )
        row = repository.get_chat_session(sid)
        assert row is not None
        assert row.pii_scrubbed_query == "raw query with [SSN]"
        assert row.council_verdict == {"passed": True, "blocked_by": None}
        assert row.nli_flags == [{"claim": "c", "entailed": True}]
        assert row.report_card == {"stage": "CN"}
        assert row.uncertainty_flag is True

    def test_only_the_de_identified_query_is_stored(self, db) -> None:
        """The raw query used to be stored alongside the scrubbed one.

        That put patient-identifying text in plaintext in the operational
        database with no encryption, access control or retention policy. Both
        text columns now receive the de-identified value; retaining raw clinical
        text for audit would need a protected design of its own.
        """
        sid = repository.save_chat_session(
            user_id="u1", pii_scrubbed_query="scrubbed", response="r",
            agent_used="graphrag",
        )
        row = repository.get_chat_session(sid)
        assert row.user_query == "scrubbed"
        assert row.pii_scrubbed_query == "scrubbed"


class TestReportCardExport:
    def test_a_saved_report_card_is_retrievable(self, db) -> None:
        """P1-8 — export was a permanent 404 because nothing wrote this column."""
        sid = repository.save_chat_session(
            user_id="u1", pii_scrubbed_query="q", response="r",
            agent_used="clinical", report_card={"stage": "AD", "confidence_score": 0.9},
        )
        assert repository.get_report_card(sid) == {"stage": "AD", "confidence_score": 0.9}

    def test_missing_report_card_returns_none(self, db) -> None:
        sid = repository.save_chat_session(
            user_id="u1", pii_scrubbed_query="q", response="r", agent_used="graphrag"
        )
        assert repository.get_report_card(sid) is None

    def test_unknown_session_returns_none_rather_than_raising(self, db) -> None:
        assert repository.get_report_card(uuid.uuid4()) is None


class TestFeedback:
    def test_feedback_persists_against_a_real_session(self, db) -> None:
        """P1-7 — the insert omitted user_id (NOT NULL) and used a str FK."""
        sid = repository.save_chat_session(
            user_id="u1", pii_scrubbed_query="q", response="r", agent_used="graphrag"
        )
        assert repository.save_feedback(session_id=str(sid), thumbs_up=True, comment="good") is True

    def test_feedback_accepts_a_string_session_id(self, db) -> None:
        sid = repository.save_chat_session(
            user_id="u1", pii_scrubbed_query="q", response="r", agent_used="graphrag"
        )
        assert repository.save_feedback(session_id=str(sid), thumbs_up=False) is True

    def test_feedback_for_an_unknown_session_is_rejected_not_crashed(self, db) -> None:
        assert repository.save_feedback(session_id=str(uuid.uuid4()), thumbs_up=True) is False

    def test_feedback_for_a_malformed_session_id_is_rejected(self, db) -> None:
        assert repository.save_feedback(session_id="not-a-uuid", thumbs_up=True) is False

    def test_feedback_inherits_user_id_from_the_session(self, db) -> None:
        sid = repository.save_chat_session(
            user_id="alice", pii_scrubbed_query="q", response="r",
            agent_used="graphrag",
        )
        repository.save_feedback(session_id=str(sid), thumbs_up=True)
        assert repository.get_feedback_user_id(sid) == "alice"

    def test_repository_does_not_import_asyncpg(self) -> None:
        """P1-7 — asyncpg is in neither requirements file."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(repository))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "asyncpg" not in imported


class TestSingleDatabaseConfigSource:
    def test_database_url_has_one_resolver(self) -> None:
        """P1-10 — db/__init__, core/config, and the raw endpoints disagreed."""
        from mao.db import resolve_database_url
        from mao.core.config import cfg

        assert resolve_database_url() == cfg.postgres_url

    def test_mao_database_url_overrides_when_set(self, monkeypatch) -> None:
        from mao.db import resolve_database_url

        monkeypatch.setenv("MAO_DATABASE_URL", "postgresql://override/db")
        assert resolve_database_url() == "postgresql://override/db"

    def test_resolver_does_not_silently_fall_back_to_localhost(self, monkeypatch) -> None:
        """The audited bug: .env set DATABASE_URL but init_db read MAO_DATABASE_URL."""
        monkeypatch.delenv("MAO_DATABASE_URL", raising=False)
        monkeypatch.setenv("DATABASE_URL", "postgresql://real-host/db")
        import importlib

        import mao.core.config as config

        importlib.reload(config)
        from mao.db import resolve_database_url

        assert "localhost" not in resolve_database_url()
