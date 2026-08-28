"""M5 — the operational database stores de-identified queries only.

The adversarial review found `request.query` (raw) persisted to Postgres as
`user_query` alongside `pii_scrubbed_query`. It looked deliberate — a clinical
audit trail — but it left PHI in plaintext in the operational DB with no
documented encryption, access control or retention policy, and (given the
scrubber findings) the "scrubbed" column was not reliably de-identified either.

The decision recorded for Phase 1 is the safer default: persist the
de-identified query and nothing else. Retaining raw clinical text for audit is
not forbidden, but it has to be an explicit protected design — encryption,
access control, a retention policy and auditability — and that is not Phase 1
scope. Until such a design exists, raw PHI does not go to rest.
"""
from __future__ import annotations

import inspect

from mao.db import repository


class TestTheRepositoryCannotBeHandedRawPhi:
    def test_save_chat_session_takes_no_raw_query_parameter(self) -> None:
        params = inspect.signature(repository.save_chat_session).parameters
        assert "user_query" not in params

    def test_it_still_takes_the_scrubbed_query(self) -> None:
        params = inspect.signature(repository.save_chat_session).parameters
        assert "pii_scrubbed_query" in params

    def test_the_scrubbed_value_is_what_reaches_both_columns(self) -> None:
        """`user_query` is a schema column, not a licence to store raw text."""
        source = inspect.getsource(repository.save_chat_session)
        assert "user_query=pii_scrubbed_query" in source


class TestTheApiNeverPassesRawText:
    def test_main_does_not_hand_the_raw_query_to_the_repository(self) -> None:
        from mao.api import main

        source = inspect.getsource(main)
        start = source.index("save_chat_session(")
        call = source[start : start + 600]
        assert "raw_query" not in call

    def test_the_scrubbed_query_is_what_is_persisted(self) -> None:
        from mao.api import main

        source = inspect.getsource(main)
        start = source.index("save_chat_session(")
        call = source[start : start + 600]
        assert "pii_scrubbed_query=safe_query" in call


class TestTheDecisionIsRecordedInCode:
    def test_the_repository_documents_why_raw_text_is_not_stored(self) -> None:
        doc = repository.save_chat_session.__doc__ or ""
        assert "de-identified" in doc.lower()
