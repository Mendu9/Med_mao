"""A cache hit is still a clinical answer, and still has to leave a record.

Architecture review, MEDIUM: `/chat` returned from the cache branch *before*
`_persist_session` and `emit_trace`, so a cache-hit clinical answer produced no
`ChatSession` row and no trace. That is the same gap P1-9 was raised for on the
streaming path — the audit trail was complete only for requests that happened to
miss the cache, which is a property of load, not of clinical significance.

Also covers adversarial M-2: `mao/api/invocation.py` returned
`detail=f"Agent error: {exc}"` to the client. The graph handles PHI, so an
exception string is a channel out of the process; the request id is what a
clinician's support call actually needs, and the detail belongs in the log.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class _InlineExecutor:
    """Runs submitted work immediately, on the calling thread.

    `_persist_session` is dispatched fire-and-forget via `run_in_executor`, so
    asserting on it straight after the response is a race the test loses most of
    the time. Making the executor synchronous removes the race instead of
    papering over it with a sleep.
    """

    def submit(self, fn, *args, **kwargs):
        from concurrent.futures import Future

        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - mirror executor semantics
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True) -> None:
        return None


@pytest.fixture
def audited(monkeypatch: pytest.MonkeyPatch):
    """Capture what the request path persists and traces."""
    import mao.api.main as main

    persisted: list[dict] = []
    traced: list[dict] = []

    monkeypatch.setattr(main, "get_executor", lambda: _InlineExecutor())

    monkeypatch.setattr(
        main,
        "_persist_session",
        lambda safe_query, user_id, result, request_id: persisted.append(
            {"query": safe_query, "user_id": user_id, "request_id": request_id}
        ),
    )
    monkeypatch.setattr(
        main,
        "emit_trace",
        lambda *, trace_id, result, latency_ms: traced.append({"trace_id": trace_id}),
    )
    return {"persisted": persisted, "traced": traced}


@pytest.fixture
def cached_client(monkeypatch: pytest.MonkeyPatch, audited):
    """A client whose Redis always reports a hit."""
    import json

    import mao.api.main as main

    payload = json.dumps(
        {
            "response": "Donepezil is an acetylcholinesterase inhibitor.",
            "agent_used": "clinical",
            "intent": "clinical",
            "metadata": {"mode": "text_question", "sources": []},
        }
    )
    monkeypatch.setattr(main, "get_redis", lambda: object())
    monkeypatch.setattr(main, "safe_get", lambda client, key: payload)
    monkeypatch.setattr(main, "safe_set", lambda *a, **k: None)

    from mao.api.main import app

    with TestClient(app) as client:
        yield client, audited


class TestACacheHitLeavesAnAuditTrail:
    def test_the_cached_answer_is_returned(self, cached_client) -> None:
        client, _ = cached_client
        response = client.post(
            "/chat", json={"query": "what does donepezil do?", "user_id": "u1"}
        )
        assert response.status_code == 200
        assert "acetylcholinesterase" in response.json()["response"]

    def test_a_chat_session_row_is_written(self, cached_client) -> None:
        client, audited = cached_client
        client.post("/chat", json={"query": "what does donepezil do?", "user_id": "u1"})
        assert audited["persisted"], (
            "a cache-hit clinical answer left no ChatSession row — the audit "
            "trail is complete only for requests that miss the cache"
        )

    def test_a_trace_is_emitted(self, cached_client) -> None:
        client, audited = cached_client
        client.post("/chat", json={"query": "what does donepezil do?", "user_id": "u1"})
        assert audited["traced"], "a cache-hit answer emitted no trace"

    def test_the_persisted_query_is_the_de_identified_one(self, cached_client) -> None:
        client, audited = cached_client
        client.post(
            "/chat",
            json={"query": "Patient Name: John Smith needs review", "user_id": "u1"},
        )
        assert audited["persisted"]
        assert "John Smith" not in audited["persisted"][0]["query"]


class TestAnErrorDetailDoesNotLeaveTheProcess:
    """Adversarial M-2 — the 500 body carried the raw exception string."""

    def test_the_exception_text_is_not_returned_to_the_client(self) -> None:
        from mao.api.invocation import client_safe_detail

        detail = client_safe_detail(
            RuntimeError("Patient Name: John Smith failed at row 42"), "req-1"
        )
        assert "John Smith" not in detail
        assert "row 42" not in detail

    def test_the_request_id_is_returned_instead(self) -> None:
        """A support call needs a correlation handle, not a stack detail."""
        from mao.api.invocation import client_safe_detail

        assert "req-1" in client_safe_detail(RuntimeError("boom"), "req-1")

    def test_a_connectivity_failure_still_reads_as_one(self) -> None:
        from mao.api.invocation import client_safe_detail

        detail = client_safe_detail(
            RuntimeError("Connection error while reaching the provider"), "req-2"
        )
        assert "unreachable" in detail.lower()

    def test_the_graph_error_path_uses_it(self) -> None:
        """Bind the endpoint's behaviour, not just the helper's."""
        import asyncio

        import pytest as _pytest
        from fastapi import HTTPException

        import mao.api.invocation as invocation

        def _explode(graph, state):
            raise RuntimeError("Patient Name: John Smith failed at row 42")

        original = invocation.invoke_with_usage
        invocation.invoke_with_usage = _explode
        try:
            with _pytest.raises(HTTPException) as caught:
                asyncio.run(invocation.run_graph({}, "req-3"))
        finally:
            invocation.invoke_with_usage = original

        assert caught.value.status_code == 500
        assert "John Smith" not in str(caught.value.detail)
        assert "req-3" in str(caught.value.detail)
