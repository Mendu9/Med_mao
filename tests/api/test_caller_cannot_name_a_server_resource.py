"""Wave 9 / B6 cluster — the caller may send content, never a locator.

B6 was `/ingest/alzheimers` taking a `data_dir`. Closing only that endpoint
would have been cosmetic: remediating it surfaced three more caller-supplied
locators on the far busier `/chat` path, all reaching the same class of sink.

Proven by execution before this file was written:

    _extract_pdf_text({"report_path": <server file>})  -> canary text returned
    POST /chat metadata={"report_path": ...}           -> "Router: attachment
                                                          detected -> clinical"

so the request body reaches `PdfReader(path)`. Worse than B6, because the
extracted text is summarised into the response body — read *and* exfiltrate.

    report_path  -> pypdf reads any server file
    audio_path   -> whisper reads any server file
    image_url    -> requests.get to any host, including 169.254.169.254 and
                    anything else reachable from inside the deployment

The invariant, stated once so all four cannot drift apart again:

    A caller supplies attachment CONTENT. A caller never names a resource the
    server is to open, nor a host the server is to fetch from.

`report_b64`, `image_b64` and `audio_b64` already carry every legitimate use —
the client sends the bytes it has. The locator keys add no capability a caller
is entitled to, so they are refused at the boundary rather than sanitised.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from mao.api.main import app

    with TestClient(app) as test_client:
        yield test_client


class TestServerPathsAreRefusedAtTheBoundary:
    @pytest.mark.parametrize("key", ["report_path", "audio_path"])
    def test_a_server_path_is_refused(self, client, key: str, tmp_path) -> None:
        canary = tmp_path / "confidential.pdf"
        canary.write_bytes(b"%PDF-1.4 canary")

        response = client.post(
            "/chat",
            json={
                "query": "Summarise this report.",
                "user_id": "probe",
                "metadata": {key: str(canary)},
            },
        )
        assert response.status_code == 422, (
            f"{key} was accepted — the caller can still name a server file"
        )

    @pytest.mark.parametrize("key", ["report_path", "audio_path"])
    def test_the_refusal_names_the_offending_key(self, client, key: str) -> None:
        """A silent strip would also close the hole, but leaves an operator
        unable to tell an attack from a stale client."""
        response = client.post(
            "/chat",
            json={"query": "hello", "user_id": "probe", "metadata": {key: "/etc/passwd"}},
        )
        assert key in response.text

    def test_the_streaming_route_refuses_it_too(self, client) -> None:
        """Both routes take the same request model, so neither can drift."""
        response = client.post(
            "/chat/stream",
            json={
                "query": "Summarise this.",
                "user_id": "probe",
                "metadata": {"report_path": "/etc/passwd"},
            },
        )
        assert response.status_code == 422


class TestContentAttachmentsStillWork:
    """The fix must remove the locator, not the feature."""

    @pytest.mark.parametrize("key", ["report_b64", "image_b64", "audio_b64"])
    def test_base64_content_is_still_accepted(self, client, key: str) -> None:
        response = client.post(
            "/chat",
            json={"query": "What is this?", "user_id": "probe", "metadata": {key: ""}},
        )
        assert response.status_code != 422

    def test_ordinary_metadata_is_still_accepted(self, client) -> None:
        response = client.post(
            "/chat",
            json={
                "query": "What is the half-life of donepezil?",
                "user_id": "probe",
                "metadata": {"domain": "alzheimer"},
            },
        )
        assert response.status_code != 422


class TestOutboundFetchesAreContained:
    """`image_url` is fetched server-side with `requests.get`, so an unvalidated
    URL is a request forgery primitive with the deployment's own network
    position — and the fetched bytes are then handed to the vision model."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            "http://127.0.0.1:8000/records",
            "http://localhost/admin",
            "http://10.0.0.5/internal",
            "http://192.168.1.1/router",
            "http://[::1]/loopback",
            "file:///etc/passwd",
            "gopher://127.0.0.1:6379/_INFO",
        ],
    )
    def test_a_non_public_target_is_refused(self, url: str) -> None:
        from mao.safety.fetch import UnsafeFetchError, validate_fetch_url

        with pytest.raises(UnsafeFetchError):
            validate_fetch_url(url)

    @pytest.mark.parametrize(
        "url",
        ["https://example.org/scan.jpg", "http://example.com/a/b.png"],
    )
    def test_an_ordinary_public_image_url_is_allowed(self, url: str) -> None:
        from mao.safety.fetch import validate_fetch_url

        validate_fetch_url(url)

    def test_the_only_fetching_agent_routes_through_the_guard(self) -> None:
        """A guard only some callers use is the B7 shape - a control living in
        an agent instead of at the boundary.

        This used to loop over both agents, keyed on whether the module
        mentioned `image_url` at all. That key stopped discriminating at M-3:
        `clinical_agent` still READS `image_url` to detect an attachment, but
        no longer fetches it, and it names `fetch_image_bytes` only in the
        comment recording ADV17-5 - so the old assertion would have passed on a
        comment. It is rebound to the call itself.
        """
        import ast
        import inspect

        from mao.agents import clinical_agent, multimodal_agent

        def _called_names(module) -> set[str]:
            """Function names actually CALLED in the module - not names that
            merely appear in its text."""
            tree = ast.parse(inspect.getsource(module))
            names: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name):
                        names.add(func.id)
                    elif isinstance(func, ast.Attribute):
                        names.add(func.attr)
            return names

        assert "fetch_image_bytes" in _called_names(multimodal_agent), (
            "the one agent that fetches image_url stopped using the shared guard"
        )
        # ADV17-5: the clinical agent's fetch was the unguarded-by-position
        # one - it ran before any image gate. It must now make no fetch at all.
        clinical_calls = _called_names(clinical_agent)
        for forbidden in ("fetch_image_bytes", "urlopen", "urlretrieve"):
            assert forbidden not in clinical_calls, (
                f"clinical_agent calls {forbidden!r}; since M-3 it must perform "
                "no outbound fetch of a caller-supplied URL at all"
            )
