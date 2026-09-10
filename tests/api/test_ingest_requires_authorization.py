"""ADV15-11 — `/ingest` was an unauthenticated write into the evidence corpus.

`POST /ingest` took caller-supplied Wikipedia article titles and a
`max_articles` up to 100, with no authentication, no `Depends` and no key. Those
articles are fetched, chunked, embedded and then served back through `/chat` as
retrieved evidence — which is the safety council's and the judge's `context`,
the NLI gate's premise, and the citation list the clinician reads. So the public
surface that decides what the safety chain believes was writable by anyone,
which is the reachability half of ADV15-10.

The module had already reasoned its way to this conclusion once, about
`data_dir`: "a corpus-poisoning primitive", and "an allowlist is a thing that
can be got wrong". The identical primitive stayed open through `topics`.

These tests are over HTTP against the real app, because a dependency that is
declared but not wired into the router is exactly the failure this closes.

The case that matters most is `no key configured`. A control that allows when it
has not been set up is not a boundary, it is a default — so an unconfigured
deployment must refuse the write rather than serve it.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mao.api.routes.ingestion import INGEST_KEY_HEADER
from mao.core.config import INGEST_API_KEY_ENV

_KEY = "s3cret-ingest-key"

# Every write route on the ingestion router. `/ingest` is the blocker — it is
# the one taking caller-controlled content — but the control hangs off the
# router, so the parametrisation is what proves a future endpoint cannot be
# born unprotected.
_WRITE_ROUTES = [
    ("/ingest", {"topics": ["Alzheimer's disease"], "max_articles": 1}),
    ("/ingest/alzheimers", {"chunk_size": 512}),
    ("/ingest/knowledge-bases", {}),
    ("/ingest/pubmed", {}),
]


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch):
    """A deployment that has configured an ingestion key."""
    monkeypatch.setenv(INGEST_API_KEY_ENV, _KEY)
    from mao.api.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def unconfigured(monkeypatch: pytest.MonkeyPatch):
    """A deployment that has not."""
    monkeypatch.delenv(INGEST_API_KEY_ENV, raising=False)
    from mao.api.main import app

    with TestClient(app) as client:
        yield client


def _ingestion_never_ran(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Trip-wires on the background runners. Empty is the assertion."""
    ran: list[str] = []
    from mao.api.routes import ingestion

    for name in (
        "_run_ingestion",
        "_run_alzheimers_ingestion",
        "_run_knowledge_base_ingestion",
        "_run_pubmed_ingestion",
    ):
        monkeypatch.setattr(
            ingestion, name, (lambda n: lambda *a, **k: ran.append(n))(name)
        )
    return ran


class TestAnUnauthorizedCallerCannotWriteEvidence:
    @pytest.mark.parametrize(("path", "body"), _WRITE_ROUTES)
    def test_no_key_presented_is_refused(self, configured, path, body) -> None:
        assert configured.post(path, json=body).status_code == 401

    @pytest.mark.parametrize(("path", "body"), _WRITE_ROUTES)
    def test_a_wrong_key_is_refused(self, configured, path, body) -> None:
        response = configured.post(
            path, json=body, headers={INGEST_KEY_HEADER: "not-the-key"}
        )
        assert response.status_code == 401

    def test_a_key_that_is_a_prefix_of_the_real_one_is_refused(
        self, configured
    ) -> None:
        response = configured.post(
            "/ingest",
            json={"topics": ["x"]},
            headers={INGEST_KEY_HEADER: _KEY[:-1]},
        )
        assert response.status_code == 401

    def test_a_non_ascii_key_is_refused_rather_than_crashing(self, configured) -> None:
        """`hmac.compare_digest` raises TypeError on non-ASCII `str`, which would
        be a 500 and a stack trace where a 401 belongs.

        Sent as raw bytes: HTTP header values are latin-1 on the wire, and an
        httpx `str` header would be rejected client-side before the server ever
        saw it — which would make this a test of the test client.
        """
        response = configured.post(
            "/ingest",
            json={"topics": ["x"]},
            headers={INGEST_KEY_HEADER: "ké".encode("latin-1")},
        )
        assert response.status_code == 401

    def test_a_refused_request_starts_no_ingestion(
        self, configured, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The property that matters: refusal must happen before the corpus is
        touched, not after the background task is queued."""
        ran = _ingestion_never_ran(monkeypatch)

        for path, body in _WRITE_ROUTES:
            configured.post(path, json=body)
            configured.post(path, json=body, headers={INGEST_KEY_HEADER: "wrong"})

        assert ran == [], f"an unauthorized request still ran ingestion: {ran}"


class TestAnUnconfiguredDeploymentFailsClosed:
    @pytest.mark.parametrize(("path", "body"), _WRITE_ROUTES)
    def test_with_no_key_configured_the_write_route_refuses(
        self, unconfigured, path, body
    ) -> None:
        response = unconfigured.post(path, json=body)
        assert response.status_code == 503
        assert INGEST_API_KEY_ENV in response.json()["detail"]

    def test_no_key_configured_does_not_mean_any_key_works(
        self, unconfigured
    ) -> None:
        """The obvious wrong implementation — compare against "" and let an
        empty header through — would pass every test above."""
        for presented in ("", "anything", _KEY):
            response = unconfigured.post(
                "/ingest",
                json={"topics": ["x"]},
                headers={INGEST_KEY_HEADER: presented},
            )
            assert response.status_code == 503

    def test_an_unconfigured_deployment_ingests_nothing(
        self, unconfigured, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ran = _ingestion_never_ran(monkeypatch)

        for path, body in _WRITE_ROUTES:
            unconfigured.post(path, json=body)

        assert ran == []


class TestAnAuthorizedCallerStillWorks:
    @pytest.mark.parametrize(("path", "body"), _WRITE_ROUTES)
    def test_the_correct_key_is_accepted(self, configured, path, body) -> None:
        response = configured.post(
            path, json=body, headers={INGEST_KEY_HEADER: _KEY}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "started"

    def test_the_authorized_request_actually_queues_ingestion(
        self, configured, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-vacuity: a router that refused everything would pass the refusal
        tests and break the product."""
        ran = _ingestion_never_ran(monkeypatch)

        configured.post(
            "/ingest",
            json={"topics": ["Alzheimer's disease"], "max_articles": 1},
            headers={INGEST_KEY_HEADER: _KEY},
        )

        assert ran == ["_run_ingestion"]


class TestTheControlIsOnTheRouterNotOnOneRoute:
    """`topics` outlived the `data_dir` fix because the fix was per-route.

    A dependency declared on the router is inherited by whatever is added to
    this module next; a decorator on `/ingest` alone is a thing the next author
    has to remember.
    """

    def test_the_router_declares_the_dependency(self) -> None:
        from mao.api.routes.ingestion import require_ingest_key, router

        declared = [
            getattr(d.dependency, "__name__", "") for d in router.dependencies
        ]
        assert require_ingest_key.__name__ in declared

    def test_every_route_on_the_router_inherits_it(self) -> None:
        from mao.api.main import app
        from mao.api.routes.ingestion import require_ingest_key

        ingest_routes = [
            r for r in app.routes if str(getattr(r, "path", "")).startswith("/ingest")
        ]
        assert ingest_routes, "the ingestion routes are not mounted"
        for route in ingest_routes:
            names = [
                getattr(d.call, "__name__", "")
                for d in route.dependant.dependencies  # type: ignore[attr-defined]
            ]
            assert require_ingest_key.__name__ in names, (
                f"{route.path} is not behind the ingestion authorization boundary"
            )
