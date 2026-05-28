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


@patch("mao.agents.clinical_agent.search_memories", return_value="")
@patch("mao.agents.clinical_agent.save_memory")
@patch("mao.agents.clinical_agent.retrieve", return_value=[])
@patch("mao.agents.clinical_agent._call_llm", return_value="Test LLM response.")
@patch("mao.agents.clinical_agent._extract_structured_fields", return_value={})
@patch("mao.agents.clinical_agent._summarize_report", return_value="Report summary.")
@patch("mao.agents.clinical_agent._web_search_clinical", return_value="No web results.")
@patch("mao.agents.clinical_agent.check_all_claims", return_value=[])
def test_clinical_node_routes_to_pdf_mode(
    mock_nli, mock_web, mock_summary, mock_extract,
    mock_llm, mock_retrieve, mock_save, mock_search,
):
    """When report_b64 is present, clinical_node uses pdf_report mode."""
    from mao.agents.clinical_agent import clinical_node

    encoded = base64.b64encode(_make_minimal_pdf()).decode()
    state = _minimal_state({"report_b64": encoded})
    result = clinical_node(state)

    assert result["agent_used"] == "clinical"
    assert result["metadata"].get("mode") == "pdf_report"
    assert "IMPORTANT" in result["response"]  # disclaimer appended


@patch("mao.agents.clinical_agent.search_memories", return_value="")
@patch("mao.agents.clinical_agent.save_memory")
@patch("mao.agents.clinical_agent.retrieve", return_value=[])
@patch("mao.agents.clinical_agent._call_llm", return_value="Clinical text answer.")
@patch("mao.agents.clinical_agent.check_all_claims", return_value=[])
def test_clinical_node_routes_to_text_mode(
    mock_nli, mock_llm, mock_retrieve, mock_save, mock_search,
):
    """No image or report in metadata → text_question mode."""
    from mao.agents.clinical_agent import clinical_node

    state = _minimal_state({})
    result = clinical_node(state)

    assert result["agent_used"] == "clinical"
    assert result["metadata"].get("mode") == "text_question"


@patch("mao.agents.clinical_agent.search_memories", return_value="")
@patch("mao.agents.clinical_agent.save_memory")
@patch("mao.agents.clinical_agent.retrieve", return_value=[])
@patch("mao.agents.clinical_agent._call_llm", return_value="Fallback LLM response.")
@patch("mao.agents.clinical_agent._extract_structured_fields", return_value={})
@patch("mao.agents.clinical_agent._summarize_report", return_value="")
@patch("mao.agents.clinical_agent._web_search_clinical", return_value="")
@patch("mao.agents.clinical_agent.check_all_claims", return_value=[])
def test_clinical_node_invalid_pdf_no_crash(
    mock_nli, mock_web, mock_summary, mock_extract,
    mock_llm, mock_retrieve, mock_save, mock_search,
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
