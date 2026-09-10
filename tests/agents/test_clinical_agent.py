"""
Tests for clinical_agent.py — PDF report extraction and routing.
"""
from __future__ import annotations

import base64
import pytest
from unittest.mock import patch

from tests.agents.gateway_stub import _completion


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
@patch("mao.agents.clinical_agent._synthesise", return_value="Test LLM response.")
@patch("mao.agents.clinical_agent._web_search_clinical", return_value="No web results.")
@patch(
    "mao.agents.clinical_agent._extract_pdf_text",
    return_value="Aged 78\nDonepezil 10 mg once daily\nBradycardia 48 bpm untreated\n",
)
def test_clinical_node_routes_to_pdf_mode(
    mock_extract, mock_web, mock_synth, mock_retrieve,
):
    """When report_b64 is present, clinical_node uses pdf_report mode.

    `_summarize_report` and `_extract_structured_fields` are gone: both sent the
    de-identified report to an external model, which the approved M-1 policy
    does not permit. The stubs they needed are replaced by `_synthesise`, which
    is the one external call the migrated path makes.
    """
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
@patch("mao.agents.clinical_agent._synthesise", return_value="Fallback LLM response.")
@patch("mao.agents.clinical_agent._web_search_clinical", return_value="")
def test_clinical_node_invalid_pdf_no_crash(mock_web, mock_synth, mock_retrieve):
    """Invalid PDF bytes → graceful response, no exception raised."""
    from mao.agents.clinical_agent import clinical_node

    state = _minimal_state({"report_b64": base64.b64encode(b"not a pdf").decode()})
    result = clinical_node(state)

    assert result["agent_used"] == "clinical"
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0


# ---------------------------------------------------------------------------
# Frontend PDF routing
#
# The two tests that lived here drove `app/frontend.py`, the legacy Gradio UI.
# Both were xfail(strict=False) and one of them XPASSed, so they asserted
# nothing in either direction; the architecture review counted that pair as
# suite noise.
#
# The UI itself was removed in Wave 7. It was wired into no entrypoint under
# deploy/ (HF Spaces runs app/streamlit_app.py), it needed an opt-in
# dependency, and it was one of the two frontends sending metadata["filename"]
# — the PHI vector in adversarial H-2. 00_RULES: "Do not preserve dead
# functionality solely because it already exists."
#
# PDF routing for the SHIPPED UI is covered by
# test_streamlit_routes_pdf_as_report_b64.


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


def test_no_part_of_the_report_document_reaches_the_provider():
    """Strengthened for the approved M-1 policy.

    This test used to assert that the SCRUBBED REPORT reaching the provider
    carried no identifiers — that is, it asserted the scrubber was the wall.
    Under the approved policy a scrubbed free-text clinical report is not a
    payload any external model may receive at all, so the assertion is now the
    stronger one: no line of the document goes out, and what does go out is the
    typed projection, which still carries the clinical fact.

    Bound at the REAL provider seam rather than by patching `gateway.complete`.
    Patching `complete` would have replaced the egress authorisation this test
    depends on, and would have missed `synthesise_clinical` entirely — it does
    not go through `complete()`.
    """
    from mao.agents.clinical_agent import _handle_pdf_report
    from mao.providers import gateway

    from tests.trust.recorders import RecordingProvider

    recorder = RecordingProvider()
    gateway.set_provider(recorder)
    try:
        with patch("mao.agents.clinical_agent._extract_pdf_text", return_value=_RAW_REPORT), \
             patch("mao.agents.clinical_agent.retrieve", return_value=[]), \
             patch("mao.agents.clinical_agent._web_search_clinical", return_value=""):
            _handle_pdf_report("Summarise this report", {"report_b64": "x"}, "")
    finally:
        gateway.reset_provider()

    assert recorder.calls, "the provider was never called — the probe is vacuous"
    blob = recorder.text()

    assert not recorder.leaked(_IDENTIFIERS), (
        f"{recorder.leaked(_IDENTIFIERS)} reached the provider"
    )
    # ...and not the document's identifier structure either, redacted or not.
    #
    # The list is the LABELLED HEADER and the PLACEHOLDERS, not every line of
    # the source. A document title carrying clinical vocabulary — `MEMORY CLINIC
    # REPORT` — does survive into the findings, and that is the correct error to
    # make: the only rule that would drop it is "an all-capitals line is a
    # heading", and `BRADYCARDIA PRESENT` is an all-capitals line that is a
    # cardiac finding. Carrying a title costs a few tokens; dropping a finding
    # is the failure this whole phase is about. Recorded as a residual.
    #
    # A placeholder is the decisive one: `[NAME]` in an outgoing payload means
    # the redacted DOCUMENT is being sent, which is what the policy forbids.
    for structural in ("Patient Name:", "MRN:", "Address:", "[NAME]", "[MRN]", "[DOB]"):
        assert structural not in blob, (
            f"{structural!r} reached the provider: the report document is being "
            "sent, and the approved M-1 policy admits only a SafeSynthesisContext"
        )
    assert "hippocampal atrophy" in blob, (
        "the clinical fact must survive into the safe projection — excluding "
        "identifiers may not be bought by dropping the finding"
    )


def test_pdf_report_text_is_scrubbed_before_retrieval_seed():
    """The retrieval seed is built from the report text; it must be scrubbed too."""
    from mao.agents.clinical_agent import _handle_pdf_report

    seeds: list[str] = []

    def _recording_retrieve(query, **kwargs):
        seeds.append(query)
        return []

    with patch("mao.agents.clinical_agent._extract_pdf_text", return_value=_RAW_REPORT), \
         patch("mao.providers.gateway.complete", return_value=_completion("{}")), \
         patch("mao.agents.clinical_agent.retrieve", side_effect=_recording_retrieve), \
         patch("mao.agents.clinical_agent._web_search_clinical", return_value=""):
        _handle_pdf_report("Summarise this report", {"report_b64": "x"}, "")

    blob = "\n".join(seeds)
    for identifier in _IDENTIFIERS:
        assert identifier not in blob, f"{identifier!r} leaked into the retrieval seed"


@pytest.mark.parametrize("raw_key", ["image_b64", "report_b64"])
def test_clinical_node_does_not_echo_raw_attachment_payloads(raw_key: str):
    """The response metadata is what the agent PRODUCED — never the caller's input.

    Tightened in Wave 7. This test used to assert that unrelated caller keys
    ("trace_id") survived into the response, which is precisely adversarial
    finding H-2: both shipped frontends send `filename`, and
    `Doe_Jane_MRN4471023_1948-03-12.pdf` rode that echo into two Redis keys and
    back to the client. Stripping `ATTACHMENT_KEYS` was never sufficient,
    because that list says which keys mean patient data is *attached*, not which
    keys may *contain* it. The assertion is inverted rather than removed.
    """
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
    assert "trace_id" not in out["metadata"], (
        "caller-supplied metadata must not be echoed into the response — it is "
        "cached in Redis and returned to the client (adversarial H-2)"
    )
    assert out["metadata"]["mode"] in {"mri_image", "pdf_report"}, (
        "the agent's own output must still reach the client"
    )


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
