"""
Tests for clinical_agent.py — PDF report extraction and routing.
"""
from __future__ import annotations

import base64
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_pdf() -> bytes:
    """Return a minimal valid single-page PDF as bytes."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type /Catalog /Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type /Pages /Kids[3 0 R] /Count 1>>endobj\n"
        b"3 0 obj<</Type /Page /Parent 2 0 R /MediaBox[0 0 612 792]"
        b" /Contents 4 0 R /Resources<<>>>>endobj\n"
        b"4 0 obj<</Length 44>>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Patient: Test Report) Tj ET\n"
        b"endstream endobj\n"
        b"xref\n0 5\n0000000000 65535 f \n"
        b"0000000009 00000 n \n0000000058 00000 n \n"
        b"0000000115 00000 n \n0000000266 00000 n \n"
        b"trailer<</Size 5 /Root 1 0 R>>\nstartxref\n359\n%%EOF"
    )


# ---------------------------------------------------------------------------
# _extract_pdf_text
# ---------------------------------------------------------------------------

def test_extract_pdf_text_from_base64():
    """PDF bytes base64-encoded in metadata['report_b64'] are extracted without crash."""
    from mao.agents.clinical_agent import _extract_pdf_text

    pdf_bytes = _make_minimal_pdf()
    encoded = base64.b64encode(pdf_bytes).decode()
    result = _extract_pdf_text({"report_b64": encoded})
    assert isinstance(result, str)


def test_extract_pdf_text_missing_keys_returns_empty():
    from mao.agents.clinical_agent import _extract_pdf_text

    assert _extract_pdf_text({}) == ""


def test_extract_pdf_text_from_path(tmp_path):
    """PDF at report_path is read without crash."""
    from mao.agents.clinical_agent import _extract_pdf_text

    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_bytes(_make_minimal_pdf())
    result = _extract_pdf_text({"report_path": str(pdf_file)})
    assert isinstance(result, str)


def test_extract_pdf_text_invalid_b64_returns_empty():
    from mao.agents.clinical_agent import _extract_pdf_text

    assert _extract_pdf_text({"report_b64": "not-valid-base64!!!"}) == ""


# ---------------------------------------------------------------------------
# clinical_node — sub-mode routing
# ---------------------------------------------------------------------------

def _minimal_state(metadata: dict) -> dict:
    return {
        "user_query": "Analyse this report",
        "user_id": "test-user",
        "metadata": metadata,
        "memory_context": "",
        "chat_history": [],
    }


@patch("mao.agents.clinical_agent.retrieve", return_value=[])
@patch("mao.agents.clinical_agent._call_llm", return_value="Test LLM response.")
@patch("mao.agents.clinical_agent._extract_structured_fields", return_value={})
@patch("mao.agents.clinical_agent._summarize_report", return_value="Report summary.")
@patch("mao.agents.clinical_agent._web_search_clinical", return_value="No web results.")
def test_clinical_node_routes_to_pdf_mode(
    mock_web, mock_summary, mock_extract,
    mock_llm, mock_retrieve,
):
    """When report_b64 is present, clinical_node uses pdf_report mode."""
    from mao.agents.clinical_agent import clinical_node

    encoded = base64.b64encode(_make_minimal_pdf()).decode()
    state = _minimal_state({"report_b64": encoded})
    result = clinical_node(state)

    assert result["agent_used"] == "clinical"
    assert result["metadata"].get("mode") == "pdf_report"
    assert "IMPORTANT" in result["response"]  # disclaimer appended


@patch("mao.agents.clinical_agent.retrieve", return_value=[])
@patch("mao.agents.clinical_agent._call_llm", return_value="Clinical text answer.")
def test_clinical_node_routes_to_text_mode(
    mock_llm, mock_retrieve,
):
    """No image or report in metadata → text_question mode."""
    from mao.agents.clinical_agent import clinical_node

    state = _minimal_state({})
    result = clinical_node(state)

    assert result["agent_used"] == "clinical"
    assert result["metadata"].get("mode") == "text_question"


@patch("mao.agents.clinical_agent.retrieve", return_value=[])
@patch("mao.agents.clinical_agent._call_llm", return_value="Fallback LLM response.")
@patch("mao.agents.clinical_agent._extract_structured_fields", return_value={})
@patch("mao.agents.clinical_agent._summarize_report", return_value="")
@patch("mao.agents.clinical_agent._web_search_clinical", return_value="")
def test_clinical_node_invalid_pdf_no_crash(
    mock_web, mock_summary, mock_extract,
    mock_llm, mock_retrieve,
):
    """Invalid PDF bytes → graceful response, no exception raised."""
    from mao.agents.clinical_agent import clinical_node

    state = _minimal_state({"report_b64": base64.b64encode(b"not a pdf").decode()})
    result = clinical_node(state)

    assert result["agent_used"] == "clinical"
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0


# ---------------------------------------------------------------------------
# Frontend PDF routing (unit test — no running server needed)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "app/frontend.py is the legacy abandoned Gradio frontend — "
        "PDF routing (report_b64) is intentionally not implemented there. "
        "The active Streamlit UI (app/streamlit_app.py) correctly uses report_b64. "
        "See test_streamlit_routes_pdf_as_report_b64 for the active test."
    ),
    strict=False,
)
def test_send_query_routes_pdf_as_report_b64(tmp_path):
    """_send_query must send PDFs as report_b64, not image_b64."""
    pdf_file = tmp_path / "patient_report.pdf"
    pdf_file.write_bytes(_make_minimal_pdf())

    captured: dict = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(json or {})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "response": "Clinical summary.",
            "intent": "clinical",
            "agent_used": "clinical",
            "request_id": "abc123",
            "latency_ms": 100,
            "metadata": {"mode": "pdf_report", "chunks_retrieved": 3, "top_rag_score": 0.25},
        }
        return mock_resp

    # Patch httpx.post before importing the module to avoid Gradio UI build side effects
    with patch("httpx.post", side_effect=fake_post):
        from app.frontend import _send_query

        class FakeFile:
            name = str(pdf_file)

        _send_query("Analyse this report", [], FakeFile())

    meta = captured.get("metadata", {})
    assert "report_b64" in meta, "PDF must be sent as report_b64"
    assert "image_b64" not in meta, "PDF must NOT be sent as image_b64"


@pytest.mark.xfail(
    reason="app/frontend.py is the legacy abandoned Gradio frontend — not importable.",
    strict=False,
)
def test_send_query_routes_image_as_image_b64(tmp_path):
    """PNG files must still be sent as image_b64."""
    png_file = tmp_path / "brain_mri.png"
    png_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    captured: dict = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(json or {})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "response": "MRI analysis.",
            "intent": "clinical",
            "agent_used": "clinical",
            "request_id": "def456",
            "latency_ms": 200,
            "metadata": {},
        }
        return mock_resp

    with patch("httpx.post", side_effect=fake_post):
        from app.frontend import _send_query

        class FakeFile:
            name = str(png_file)

        _send_query("Analyse the MRI", [], FakeFile())

    meta = captured.get("metadata", {})
    assert "image_b64" in meta, "PNG must be sent as image_b64"
    assert "report_b64" not in meta, "PNG must NOT be sent as report_b64"


# ---------------------------------------------------------------------------
# P1-2 — NLI is computed in exactly one place, and it is not here
# ---------------------------------------------------------------------------

def test_clinical_node_does_not_run_its_own_nli_check():
    """NLI moved to the agent-independent verification node. Computing it here
    too would mean two owners of one signal, and the gate would still be inert
    for every agent that is not clinical."""
    from mao.agents.clinical_agent import clinical_node

    with patch("mao.eval.nli_checker.check_all_claims") as mock_nli, \
         patch("mao.agents.clinical_agent.retrieve", return_value=[]), \
         patch("mao.agents.clinical_agent._call_llm", return_value="Clinical answer."):
        out = clinical_node(_minimal_state({}))

    mock_nli.assert_not_called()
    assert "nli_flags" not in out, "verification_node owns nli_flags"


def test_clinical_disclaimer_comes_from_the_shared_constant():
    """output_guardrails re-asserts this exact disclaimer; one definition only."""
    from mao.agents.clinical_agent import clinical_node
    from mao.safety.verification import DISCLAIMER_MARKER

    with patch("mao.agents.clinical_agent.retrieve", return_value=[]), \
         patch("mao.agents.clinical_agent._call_llm", return_value="Clinical answer."):
        out = clinical_node(_minimal_state({}))

    assert DISCLAIMER_MARKER in out["response"]


# ---------------------------------------------------------------------------
# P0-4 — de-identify attachments before any third-party call
# ---------------------------------------------------------------------------

_RAW_REPORT = (
    "MEMORY CLINIC REPORT\n"
    "Patient Name: John Smith\n"
    "MRN: 12345678\n"
    "Address: 10 Downing Street\n"
    "DOB: 01/02/1946\n"
    "Contact: john.smith@example.com\n"
    "Findings: moderate hippocampal atrophy, MMSE 21/30.\n"
)

_IDENTIFIERS = [
    "John Smith",
    "12345678",
    "10 Downing Street",
    "01/02/1946",
    "john.smith@example.com",
]


def test_pdf_report_text_is_scrubbed_before_any_provider_call():
    """Text lifted out of an uploaded PDF must be de-identified before it is
    sent to Groq — the provider is a third party and the report is patient data."""
    from mao.agents.clinical_agent import _handle_pdf_report

    sent: list[str] = []

    def _recording_chat(messages, **kwargs):
        sent.extend(str(m.get("content", "")) for m in messages)
        return "{}"

    with patch("mao.agents.clinical_agent._extract_pdf_text", return_value=_RAW_REPORT), \
         patch("mao.core.llm.chat", side_effect=_recording_chat), \
         patch("mao.agents.clinical_agent.retrieve", return_value=[]), \
         patch("mao.agents.clinical_agent._web_search_clinical", return_value=""):
        _handle_pdf_report("Summarise this report", {"report_b64": "x"}, "")

    assert sent, "expected at least one provider call to record"
    blob = "\n".join(sent)
    for identifier in _IDENTIFIERS:
        assert identifier not in blob, f"{identifier!r} leaked to the provider"
    assert "hippocampal atrophy" in blob, "clinical content must survive scrubbing"


def test_pdf_report_text_is_scrubbed_before_retrieval_seed():
    """The retrieval seed is built from the report text; it must be scrubbed too."""
    from mao.agents.clinical_agent import _handle_pdf_report

    seeds: list[str] = []

    def _recording_retrieve(query, **kwargs):
        seeds.append(query)
        return []

    with patch("mao.agents.clinical_agent._extract_pdf_text", return_value=_RAW_REPORT), \
         patch("mao.core.llm.chat", return_value="{}"), \
         patch("mao.agents.clinical_agent.retrieve", side_effect=_recording_retrieve), \
         patch("mao.agents.clinical_agent._web_search_clinical", return_value=""):
        _handle_pdf_report("Summarise this report", {"report_b64": "x"}, "")

    blob = "\n".join(seeds)
    for identifier in _IDENTIFIERS:
        assert identifier not in blob, f"{identifier!r} leaked into the retrieval seed"


@pytest.mark.parametrize("raw_key", ["image_b64", "report_b64"])
def test_clinical_node_does_not_echo_raw_attachment_payloads(raw_key: str):
    """clinical_node spreads inbound metadata into state['metadata'], which the
    API returns to the client. Raw attachment payloads must not ride along."""
    from mao.agents.clinical_agent import clinical_node

    state = _minimal_state({raw_key: "QkFTRTY0UEFZTE9BRA==", "trace_id": "keep-me"})

    with patch("mao.agents.clinical_agent._handle_mri_image",
               return_value=("MRI answer.", {"mode": "mri_image", "_ranked_chunks": []})), \
         patch("mao.agents.clinical_agent._handle_pdf_report",
               return_value=("PDF answer.", {"mode": "pdf_report", "_ranked_chunks": []})):
        out = clinical_node(state)

    assert raw_key not in out["metadata"], (
        f"{raw_key} must be stripped before the response is built; "
        f"got keys: {sorted(out['metadata'])}"
    )
    assert out["metadata"].get("trace_id") == "keep-me", "unrelated metadata must survive"


# ---------------------------------------------------------------------------
# uncertainty_flag propagation (patient-safety signal)
# ---------------------------------------------------------------------------

def test_low_confidence_mri_sets_uncertainty_flag_in_metadata():
    """A low-confidence MRI prediction must surface uncertainty_flag=True in
    state['metadata'] — that is where the API reads it (main.py:305,444)."""
    from mao.agents.clinical_agent import clinical_node

    low_conf_pred = {
        "prediction": "MildDemented",
        "full_label": "Mild Demented",
        "confidence": 0.40,  # below MRI_CONFIDENCE_GATE (0.60)
        "all_scores": {},
    }
    result_meta = {
        "mode": "mri_image",
        "prediction": low_conf_pred,
        "sources": [],
        "chunks_retrieved": 0,
        "top_rag_score": 0.0,
        "rag_sufficient": False,
        "_ranked_chunks": [],
    }

    state = {
        "user_query": "Analyse this MRI",
        "user_id": "u-test",
        "domain": "alzheimer",
        "metadata": {"image_b64": "ZmFrZQ=="},  # triggers the MRI path
        "memory_context": "ctx",
    }

    with patch("mao.agents.clinical_agent._handle_mri_image",
               return_value=("MRI shows mild changes.", result_meta)):
        out = clinical_node(state)

    assert out["metadata"].get("uncertainty_flag") is True, (
        "uncertainty_flag must be present in metadata (API reads it there), "
        f"got metadata keys: {sorted(out['metadata'].keys())}"
    )
