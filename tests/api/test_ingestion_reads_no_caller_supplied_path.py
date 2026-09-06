"""Wave 9 / B6 — `/ingest/alzheimers` was an arbitrary server-side file read.

`AlzheimersIngestRequest.data_dir` was a caller-supplied string that
`ingest_alzheimers_pdfs` used verbatim as a `Path`. No allowlist, no
containment, no normalisation, no authentication. Proven over HTTP at f757375
with a canary PDF outside the corpus root.

It is read *and* exfiltrate: the text is chunked, embedded, and then retrievable
through `/chat` as "evidence", so it is equally a corpus-poisoning primitive.

`00_RULES` names this directly — "Do not expose unrestricted SQL, shell,
filesystem, secrets, or operational DB access." Phase 1 removed the SQL route
and left this one.

The fix is not validation. A path allowlist is a thing that can be got wrong,
and the endpoint has no legitimate need for the parameter: the corpus directory
is a property of the deployment, not of the request. So the field is removed
from the HTTP surface, and the API can no longer express the attack.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


_KEY = "b6-regression-key"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """An AUTHORIZED client.

    ADV15-11 put the whole ingestion router behind a shared-secret dependency,
    so every request here now carries the key. That is not a weakening of these
    assertions: B6 is about what an authorized caller can make the server read,
    which is a strictly harder property than what an anonymous one can. The
    authorization boundary itself is tested in
    `test_ingest_requires_authorization.py`.
    """
    from mao.api.main import app
    from mao.core.config import INGEST_API_KEY_ENV

    monkeypatch.setenv(INGEST_API_KEY_ENV, _KEY)
    with TestClient(app, headers={"X-MAO-Ingest-Key": _KEY}) as test_client:
        yield test_client


class TestTheEndpointCannotBeToldWhereToRead:
    def test_the_request_model_has_no_path_field(self) -> None:
        """Structural: the attack must be inexpressible, not merely rejected."""
        from mao.api.routes.ingestion import AlzheimersIngestRequest

        assert "data_dir" not in AlzheimersIngestRequest.model_fields

    def test_a_caller_supplied_path_is_refused(self, client) -> None:
        """Silently ignoring the field would work, but an explicit refusal makes
        the attempt visible instead of discarding it without trace."""
        response = client.post(
            "/ingest/alzheimers",
            json={"data_dir": "C:/Windows/System32", "chunk_size": 512},
        )
        assert response.status_code == 422

    def test_the_canary_directory_is_never_opened(
        self, client, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """The end-to-end property: whatever the caller sends, ingestion runs
        against the configured corpus and nothing else."""
        from mao.api.routes import ingestion

        canary = tmp_path / "secrets"
        canary.mkdir()
        (canary / "confidential.pdf").write_bytes(b"%PDF-1.4 canary")

        seen: list[object] = []

        def _record(*args, **kwargs):  # noqa: ANN002, ANN003
            seen.append(kwargs.get("data_dir", "<not passed>"))
            return 0

        monkeypatch.setattr(
            "mao.data.ingest_alzheimers.ingest_alzheimers_pdfs", _record
        )

        client.post("/ingest/alzheimers", json={"chunk_size": 512})
        ingestion._run_alzheimers_ingestion(chunk_size=512, chunk_overlap=50)

        assert seen, "ingestion never ran — the probe is vacuous"
        for value in seen:
            assert str(canary) not in str(value)
            assert value in (None, "<not passed>"), (
                f"ingestion was handed a directory from outside configuration: {value!r}"
            )

    def test_the_background_runner_takes_no_directory(self) -> None:
        """The runner is the last place the parameter could be reintroduced
        without touching the request model."""
        import inspect

        from mao.api.routes.ingestion import _run_alzheimers_ingestion

        assert "data_dir" not in inspect.signature(_run_alzheimers_ingestion).parameters


class TestTheEndpointStillWorks:
    def test_ingestion_still_starts(self, client) -> None:
        response = client.post("/ingest/alzheimers", json={"chunk_size": 512})
        assert response.status_code == 200
        assert response.json()["status"] == "started"

    def test_chunking_parameters_are_still_accepted(self, client) -> None:
        """Removing the path must not remove the endpoint's real controls."""
        response = client.post(
            "/ingest/alzheimers", json={"chunk_size": 1024, "chunk_overlap": 128}
        )
        assert response.status_code == 200
