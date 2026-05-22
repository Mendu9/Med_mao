"""
app/streamlit_app.py
--------------------
Streamlit frontend for MAO — 6-tab interface.

Tabs:
  1. Chat           — streaming chat with intent/source/eval sidebar
  2. Graph Explorer — interactive PyVis knowledge graph
  3. MRI Scan       — MRI image upload + clinical assessment
  4. Patient Report — PDF report upload + clinical summary
  5. System Health  — service status dashboard
  6. Eval Dashboard — RAGAS scores and feedback

Run:
    streamlit run app/streamlit_app.py

Backend: FastAPI on http://localhost:8080 (configurable via MAO_API_URL env var)
"""
from __future__ import annotations

import base64
import os
import sys
import time
import uuid
from typing import Any, Generator

# Ensure the project root is on sys.path so `mao.*` imports work when
# Streamlit launches the app from a different working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import requests
import streamlit as st
from app.graph_explorer import render_graph_explorer

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MAO Clinical AI",
    layout="wide",
    page_icon="\U0001f9e0",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL: str = os.getenv("MAO_API_URL", "http://localhost:8080")

# Intent -> badge color mapping
_INTENT_COLORS: dict[str, str] = {
    "graphrag":   "blue",
    "clinical":   "red",
    "code":       "green",
    "tool":       "orange",
    "sql":        "violet",
    "summarize":  "gray",
    "multimodal": "blue",
    "critic":     "orange",
    "chitchat":   "green",
    "fallback":   "gray",
}

# Node-type color palette is owned by graph_explorer.py (_NODE_TYPE_PALETTE).
# streamlit_app.py no longer duplicates it.


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = f"st-{uuid.uuid4().hex[:8]}"
    if "last_metadata" not in st.session_state:
        st.session_state["last_metadata"] = {}
    if "session_start" not in st.session_state:
        st.session_state["session_start"] = time.time()
    if "request_count" not in st.session_state:
        st.session_state["request_count"] = 0
    if "session_tokens" not in st.session_state:
        st.session_state["session_tokens"] = {"input": 0, "output": 0}
    if "session_cost" not in st.session_state:
        st.session_state["session_cost"] = 0.0


_init_session_state()


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _stream_response(
    query: str,
    user_id: str,
    history: list[dict[str, str]],
    metadata: dict[str, Any] | None = None,
) -> Generator[str, None, None]:
    """Generator that yields tokens from /chat/stream SSE endpoint.

    Handles 400 domain guardrail rejections by yielding the detail message directly.
    Falls back to /chat (non-streaming) if the stream read times out.

    Timeout tuple (connect=10s, read=300s): the read timeout must be generous because
    graph traversal + cross-encoder reranking + Groq LLM can take 60-180 s on CPU.
    """
    payload = {
        "query": query,
        "user_id": user_id,
        "chat_history": history,
        "metadata": metadata or {},
    }
    try:
        with requests.post(
            f"{API_URL}/chat/stream",
            json=payload,
            stream=True,
            timeout=(10, 300),  # (connect_timeout_s, read_timeout_s)
        ) as resp:
            if resp.status_code == 400:
                try:
                    detail = resp.json().get("detail", "This query is outside my clinical scope.")
                except Exception:
                    detail = "This query is outside my clinical scope."
                yield detail
                return
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("data: "):
                    token = line[6:]
                    if token == "[DONE]":
                        break
                    if token.startswith("__meta__:"):
                        yield token  # pass through raw for caller to handle
                        continue
                    yield token + " "
    except requests.exceptions.ConnectionError:
        yield (
            "\n\n_[Error: Cannot connect to backend. "
            f"Is the API running at {API_URL}?]_"
        )
    except requests.exceptions.ReadTimeout:
        # Stream read timed out — fall back to blocking /chat with a longer timeout
        yield "\n\n_[Stream timed out — retrying via direct endpoint...]_\n\n"
        text, _ = _chat_fallback(query, user_id, history, metadata)
        yield text
    except Exception as exc:
        yield f"\n\n_[Streaming error: {exc}]_"


def _chat_fallback(
    query: str,
    user_id: str,
    history: list[dict[str, str]],
    metadata: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Fallback to non-streaming /chat endpoint. Handles 400 guardrail rejections."""
    payload = {
        "query": query,
        "user_id": user_id,
        "chat_history": history,
        "metadata": metadata or {},
    }
    try:
        resp = requests.post(f"{API_URL}/chat", json=payload, timeout=(10, 300))
        if resp.status_code == 400:
            detail = resp.json().get("detail", "This query is outside my clinical scope.")
            return detail, {}
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", ""), data
    except requests.exceptions.ConnectionError:
        return (
            f"Cannot connect to backend at {API_URL}. "
            "Is the API server running?",
            {},
        )
    except Exception as exc:
        return f"Error: {exc}", {}


def _fetch_eval_dashboard() -> dict[str, Any]:
    try:
        resp = requests.get(f"{API_URL}/eval/dashboard", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


def _fetch_health() -> dict[str, Any]:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def _fetch_usage() -> dict[str, Any]:
    """Fetch Groq token usage from /usage endpoint. Returns {} if unavailable."""
    try:
        resp = requests.get(f"{API_URL}/usage", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def _fetch_graph_topology() -> dict[str, Any]:
    try:
        resp = requests.get(f"{API_URL}/graph", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Tab 1: Chat
# ---------------------------------------------------------------------------

def _render_chat_tab() -> None:
    """Render the Chat tab with streaming responses and metadata sidebar."""
    col_main, col_meta = st.columns([3, 1])

    with col_main:
        st.markdown("### Chat with MAO")
        st.markdown(
            "_Ask about Alzheimer's disease, stroke, brain health, or request clinical summaries._"
        )

        # Render conversation history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input(
            "Ask about Alzheimer's, stroke, or brain health..."
        )

        if user_input:
            st.session_state["request_count"] += 1

            # Append user message
            st.session_state["messages"].append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            # Build history for API (exclude the message just appended)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["messages"][:-1]
            ]

            full_response = ""
            resp_data: dict[str, Any] = {}

            with st.chat_message("assistant"):
                placeholder = st.empty()

                # Stream response word-by-word; capture trailing __meta__: event for metadata.
                for token in _stream_response(
                    user_input, st.session_state["user_id"], history
                ):
                    if token.startswith("__meta__:"):
                        try:
                            import json as _json
                            meta_parsed = _json.loads(token[9:].strip())
                            resp_data = meta_parsed
                        except Exception:
                            pass
                        continue
                    full_response += token
                    placeholder.markdown(full_response + " ▌")
                placeholder.markdown(full_response)

            st.session_state["messages"].append(
                {"role": "assistant", "content": full_response}
            )
            if resp_data:
                st.session_state["last_metadata"] = resp_data
            st.rerun()

    with col_meta:
        st.markdown("### Response Info")
        meta = st.session_state.get("last_metadata", {})

        # Uncertainty warning banner
        uncertainty = meta.get("uncertainty_flag", False)
        nli_val = None
        nested_meta_warn = meta.get("metadata") or {}
        try:
            nli_val = float(nested_meta_warn.get("nli_score", 1.0))
        except (TypeError, ValueError):
            nli_val = None

        if uncertainty or (nli_val is not None and nli_val < 0.7):
            st.warning(
                "⚠️ **Low confidence** — This response may contain unverified claims. "
                "Always verify with a qualified clinician.",
                icon="⚠️",
            )

        # Intent badge
        intent = meta.get("intent", "")
        if intent:
            color = _INTENT_COLORS.get(intent, "gray")
            st.markdown(f"**Intent:** :{color}[{intent}]")
        else:
            st.markdown("**Intent:** _waiting..._")

        agent = meta.get("agent_used", "")
        if agent:
            st.markdown(f"**Agent:** `{agent}`")

        latency = meta.get("latency_ms")
        if latency is not None:
            st.markdown(f"**Latency:** {latency:.0f} ms")

        st.divider()

        # Sources
        sources = meta.get("sources", [])
        if not sources:
            sources = (meta.get("metadata") or {}).get("sources", [])
        web_sources = meta.get("web_sources", [])
        if not web_sources:
            web_sources = (meta.get("metadata") or {}).get("web_sources", [])

        if sources:
            st.markdown("**Sources:**")
            for src in sources[:5]:
                title = src.get("title") or src.get("source", "Unknown")
                score = src.get("score")
                score_str = f" `{score:.3f}`" if score else ""
                st.markdown(f"- {title}{score_str}")

        if web_sources:
            st.markdown("**Web:**")
            for ws in web_sources[:3]:
                url = ws.get("url", "")
                title = ws.get("title", url)
                if url:
                    st.markdown(f"- [{title}]({url})")

        st.divider()

        # Eval scores
        nested_meta = meta.get("metadata") or {}
        faithfulness      = nested_meta.get("faithfulness")
        nli_score         = nested_meta.get("nli_score")
        answer_relevancy  = nested_meta.get("answer_relevancy")

        if faithfulness is not None:
            try:
                val = float(faithfulness)
                st.markdown(f"**Faithfulness:** {int(val * 100)}%")
                st.progress(min(1.0, val))
            except (TypeError, ValueError):
                pass

        if answer_relevancy is not None:
            try:
                val = float(answer_relevancy)
                st.markdown(f"**Answer relevancy:** {int(val * 100)}%")
                st.progress(min(1.0, val))
            except (TypeError, ValueError):
                pass

        if nli_score is not None:
            try:
                val = float(nli_score)
                st.markdown(f"**NLI score:** {int(val * 100)}%")
                st.progress(min(1.0, val))
            except (TypeError, ValueError):
                pass

        # Memory context
        memory_ctx = nested_meta.get("memory_context", [])
        if memory_ctx:
            st.divider()
            st.markdown("**Memory context:**")
            for item in memory_ctx[:3]:
                st.markdown(f"- _{str(item)[:80]}_")

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["last_metadata"] = {}
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 2: Graph Explorer
# ---------------------------------------------------------------------------

def _render_graph_tab() -> None:
    """Render the Graph Explorer tab — delegates to the Streamlit-native module."""
    render_graph_explorer()


# ---------------------------------------------------------------------------
# Tab 3: MRI Scan
# ---------------------------------------------------------------------------

def _render_mri_tab() -> None:
    """Render the MRI Scan tab — brain MRI image upload and clinical assessment."""
    st.markdown("### MRI Scan Analysis")
    st.info("Upload a brain MRI scan for AI-assisted clinical assessment.")

    st.warning(
        "**Disclaimer:** For research purposes only. "
        "This is NOT a clinical diagnosis tool. "
        "Always consult a qualified radiologist and neurologist."
    )

    uploaded_file = st.file_uploader(
        "Upload MRI scan",
        type=["jpg", "jpeg", "png", "nii"],
        help="JPEG/PNG for standard images, NIfTI (.nii) for volumetric scans",
    )

    clinical_notes = st.text_area(
        "Additional clinical context (optional)",
        placeholder="e.g. 65-year-old patient with progressive memory loss, MMSE score 22...",
        height=100,
        key="mri_clinical_notes",
    )

    if uploaded_file is not None:
        if uploaded_file.type in ("image/jpeg", "image/png"):
            st.image(uploaded_file, caption="Uploaded scan", use_container_width=False, width=400)

        if st.button("Analyse MRI", type="primary"):
            with st.spinner("Analysing MRI scan... This may take 30-60 seconds."):
                file_bytes = uploaded_file.read()
                b64 = base64.b64encode(file_bytes).decode("utf-8")

                query = (
                    "Analyse this MRI scan and provide a clinical assessment including "
                    "findings, differential diagnosis, and recommended next steps."
                )
                meta_payload: dict[str, Any] = {
                    "image_b64": b64,
                    "modality": "image",
                    "filename": uploaded_file.name,
                }

                if clinical_notes.strip():
                    query += f"\n\nClinical context: {clinical_notes.strip()}"

                payload = {
                    "query": query,
                    "user_id": st.session_state["user_id"],
                    "chat_history": [],
                    "metadata": meta_payload,
                }

                try:
                    resp = requests.post(f"{API_URL}/chat", json=payload, timeout=180)
                    if resp.status_code == 400:
                        st.warning(resp.json().get("detail", "Request rejected by guardrails."))
                    else:
                        resp.raise_for_status()
                        data = resp.json()
                        response_text = data.get("response", "No response received.")

                        st.markdown("### Clinical Assessment")
                        st.markdown(response_text)

                        nested_meta = data.get("metadata") or {}
                        mode = nested_meta.get("mode", "")
                        agent = data.get("agent_used", "")
                        intent = data.get("intent", "")
                        latency = data.get("latency_ms", 0)
                        if agent or intent:
                            st.caption(
                                f"Agent: `{agent}` | Intent: `{intent}` | "
                                f"Mode: `{mode}` | Latency: {latency:.0f} ms"
                            )

                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot connect to backend at {API_URL}. Is the API server running?")
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

    st.divider()
    st.markdown(
        "**MRI modalities:** T1-weighted, T2-weighted, FLAIR, DWI, SWI\n\n"
        "**Output includes:** Findings, differential diagnosis, recommended next steps"
    )


# ---------------------------------------------------------------------------
# Tab 4: Patient Report
# ---------------------------------------------------------------------------

def _render_report_tab() -> None:
    """Render the Patient Report tab — PDF upload and clinical summary."""
    st.markdown("### Patient Report Analysis")
    st.info("Upload a patient report PDF for AI-assisted clinical summary.")

    st.warning(
        "**Disclaimer:** For research purposes only. "
        "This is NOT a clinical diagnosis tool. "
        "Always consult a qualified clinician."
    )

    # --- Demo mode: let user try without uploading a real PDF ---
    use_demo = st.checkbox(
        "Use sample clinical report (demo — no PDF required)",
        value=False,
        key="report_use_demo",
    )
    _DEMO_REPORT = """NEUROLOGY OUTPATIENT CLINIC — DISCHARGE SUMMARY
Patient: [REDACTED]  |  Age: 72  |  Sex: M  |  Date: 2025-11-14

PRESENTING COMPLAINT
Progressive memory loss over 18 months. Difficulty with word-finding, getting lost in familiar areas, and managing finances.

HISTORY
MMSE score: 18/30. Mild-to-moderate cognitive impairment. Family history of Alzheimer's disease (mother). Non-smoker.

INVESTIGATIONS
MRI Brain: Hippocampal atrophy bilaterally, more prominent on the left. White matter changes (Fazekas grade 1).
CSF Biomarkers: Abeta42 reduced (450 pg/mL), total-tau elevated (640 pg/mL), phospho-tau elevated (88 pg/mL). Consistent with AD pathology.
ApoE Genotype: APOE ε4/ε3 heterozygous.
FDG-PET: Hypometabolism in bilateral parietal and temporal regions.

DIAGNOSIS
Mild-to-moderate Alzheimer's disease (ICD-10: G30.9). APOE ε4 carrier.

MEDICATIONS
- Donepezil 10 mg once daily (escalated from 5 mg after 4 weeks)
- Memantine 20 mg once daily
- Atorvastatin 20 mg once daily (cardiovascular risk management)
- Vitamin D3 1000 IU once daily

PLAN
1. Reassess in 6 months with repeat MMSE and ADL questionnaire.
2. Refer to occupational therapy for home-safety assessment.
3. Memory clinic support group referral.
4. Family counselling regarding prognosis and advance care planning.
5. Review driving capacity — refer to DVLA if MMSE falls below 16.

CLINICIAN: Dr. A. Patel, Consultant Neurologist
"""

    uploaded_file = st.file_uploader(
        "Upload patient report (PDF)",
        type=["pdf"],
        help="Discharge summaries, neuropsychological assessments, radiology reports, neurology letters.",
        disabled=use_demo,
    )

    clinical_notes = st.text_area(
        "Additional clinical context (optional)",
        placeholder="e.g. 65-year-old patient with progressive memory loss, MMSE score 22...",
        height=90,
        key="report_clinical_notes",
    )

    report_ready = use_demo or (uploaded_file is not None)

    if report_ready:
        if st.button("Analyse Report", type="primary"):
            with st.spinner("Extracting and analysing report... (30-90 s)"):
                if use_demo:
                    # Send demo text directly as query without PDF
                    query = (
                        "Analyse this medical report and provide a structured clinical summary "
                        "with key findings, diagnosis, medications, and recommended next steps.\n\n"
                        f"REPORT:\n{_DEMO_REPORT}"
                    )
                    meta_payload: dict[str, Any] = {"filename": "demo_neurology_report.txt"}
                else:
                    file_bytes = uploaded_file.read()  # type: ignore[union-attr]
                    b64 = base64.b64encode(file_bytes).decode("utf-8")
                    query = (
                        "Analyse this medical report and provide a structured clinical summary "
                        "with key findings, diagnosis, medications, and recommended next steps."
                    )
                    meta_payload = {
                        "report_b64": b64,
                        "filename": uploaded_file.name,  # type: ignore[union-attr]
                    }

                if clinical_notes.strip():
                    query += f"\n\nAdditional clinical context: {clinical_notes.strip()}"

                payload = {
                    "query": query,
                    "user_id": st.session_state["user_id"],
                    "chat_history": [],
                    "metadata": meta_payload,
                }

                try:
                    resp = requests.post(f"{API_URL}/chat", json=payload, timeout=(10, 300))
                    if resp.status_code == 400:
                        st.warning(resp.json().get("detail", "Request rejected by guardrails."))
                    else:
                        resp.raise_for_status()
                        data = resp.json()
                        response_text = data.get("response", "No response received.")

                        st.markdown("### Clinical Summary")
                        st.markdown(response_text)

                        # Show structured extraction if present in metadata
                        nested_meta = data.get("metadata") or {}
                        structured = nested_meta.get("structured_fields") or {}
                        if structured:
                            st.divider()
                            st.markdown("#### Extracted Fields")
                            for key, val in structured.items():
                                if val:
                                    st.markdown(f"**{key.title()}:** {val}")

                        # Show agent/latency caption
                        agent = data.get("agent_used", "")
                        intent = data.get("intent", "")
                        latency = data.get("latency_ms", 0)
                        mode = nested_meta.get("mode", "")
                        if agent or intent:
                            st.caption(
                                f"Agent: `{agent}` | Intent: `{intent}` | "
                                f"Mode: `{mode}` | Latency: {latency:.0f} ms"
                            )

                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot connect to backend at {API_URL}. Is the API server running?")
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

    st.divider()
    st.markdown(
        "**Accepted report types:** discharge summaries, neuropsychological assessments, "
        "radiology reports, neurology letters, neuropsychiatric evaluations\n\n"
        "**Output includes:** Diagnosis summary, key findings, medication list, next steps\n\n"
        "**Sample reports online:**\n"
        "- [NIH NIA Alzheimer's case vignettes](https://www.nia.nih.gov/health/alzheimers-and-dementia/alzheimers-disease-fact-sheet)\n"
        "- [ClinicalCases.org neurology cases](https://www.clinicalcases.org)\n"
        "- Tick **'Use sample clinical report'** above to run the demo without uploading."
    )


# ---------------------------------------------------------------------------
# Tab 5: System Health
# ---------------------------------------------------------------------------

def _status_badge(status: str) -> str:
    """Return a Streamlit colored-text badge for a service status string."""
    if status == "ok":
        return ":green[ok]"
    return f":red[{status}]"


def _render_health_tab() -> None:
    """Render the System Health dashboard."""
    st.markdown("### System Health")

    col_btn1, col_btn2 = st.columns([2, 8])
    with col_btn1:
        refresh_clicked = st.button("Refresh now", type="primary")
    with col_btn2:
        auto_refresh = st.toggle("Auto-refresh every 30 s", value=False)

    if refresh_clicked or auto_refresh:
        with st.spinner("Checking services..."):
            health = _fetch_health()

        overall = health.get("status", "unknown")
        st.markdown(f"**Overall status:** {_status_badge(overall)}")
        st.divider()

        hcol1, hcol2, hcol3, hcol4 = st.columns(4)
        with hcol1:
            groq_s = health.get("groq", "unknown")
            st.markdown(f"**Groq LLM**\n\n{_status_badge(groq_s)}")
        with hcol2:
            chroma = health.get("chromadb", "unknown")
            st.markdown(f"**ChromaDB**\n\n{_status_badge(chroma)}")
        with hcol3:
            postgres = health.get("postgres", "unknown")
            st.markdown(f"**PostgreSQL**\n\n{_status_badge(postgres)}")
        with hcol4:
            redis_s = health.get("redis", "unknown")
            st.markdown(f"**Redis**\n\n{_status_badge(redis_s)}")

        # LangSmith status row
        ls_key = os.getenv("LANGSMITH_API_KEY", "")
        ls_badge = ":green[enabled]" if ls_key else ":gray[disabled]"
        st.markdown(f"**LangSmith:** {ls_badge}")

        with st.expander("Raw health response"):
            st.json(health)

        if auto_refresh:
            time.sleep(30)
            st.rerun()
    else:
        st.info("Click 'Refresh now' to check service status.")

    # Session metrics (always shown)
    st.divider()
    st.markdown("### Session Info")
    uptime_s = time.time() - st.session_state.get("session_start", time.time())
    uptime_min = uptime_s / 60

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("API URL", API_URL)
    mcol2.metric("Uptime", f"{uptime_min:.1f} min")
    mcol3.metric("Requests this session", st.session_state.get("request_count", 0))

    st.caption(f"Session ID: `{st.session_state.get('user_id', 'unknown')}`")

    # ------------------------------------------------------------------
    # Groq API Usage section
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### Groq API Usage")

    usage_col1, usage_col2 = st.columns([1, 9])
    with usage_col1:
        refresh_usage = st.button("Refresh", key="usage_refresh")

    usage_data: dict[str, Any] = {}
    if refresh_usage:
        usage_data = _fetch_usage()

    if usage_data:
        total_in   = usage_data.get("total_input_tokens", 0)
        total_out  = usage_data.get("total_output_tokens", 0)
        total_tok  = usage_data.get("total_tokens", 0)
        cost_usd   = usage_data.get("estimated_cost_usd", 0.0)
        per_model  = usage_data.get("per_model", {})
        rl_info    = usage_data.get("rate_limit_info", {})

        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Input tokens",  f"{total_in:,}")
        u2.metric("Output tokens", f"{total_out:,}")
        u3.metric("Total tokens",  f"{total_tok:,}")
        u4.metric("Est. cost",     f"${cost_usd:.4f}")

        if per_model:
            st.markdown("**Per-model breakdown:**")
            for mdl, stats in per_model.items():
                st.caption(
                    f"`{mdl}` — {stats.get('requests', 0)} requests | "
                    f"in: {stats.get('input_tokens', 0):,} | "
                    f"out: {stats.get('output_tokens', 0):,} | "
                    f"cost: ${stats.get('cost_usd', 0):.4f}"
                )

        # Rate-limit progress bars
        if rl_info:
            st.markdown("**Rate-limit status (last observed):**")
            limit_req = rl_info.get("limit_requests")
            rem_req   = rl_info.get("remaining_requests")
            limit_tok = rl_info.get("limit_tokens")
            rem_tok   = rl_info.get("remaining_tokens")

            if limit_req and limit_req > 0 and rem_req is not None:
                used_req_frac = 1.0 - (rem_req / limit_req)
                used_pct      = int(used_req_frac * 100)
                color         = "green" if used_pct < 50 else ("orange" if used_pct < 80 else "red")
                st.markdown(
                    f"Requests/min: {limit_req - rem_req}/{limit_req} used "
                    f"(:{color}[{used_pct}%])"
                )
                st.progress(min(1.0, used_req_frac))
                if used_pct >= 80:
                    st.warning(
                        f"Approaching Groq RPM limit — {used_pct}% used. "
                        "Slow down or wait for reset.",
                        icon="⚠️",
                    )

            if limit_tok and limit_tok > 0 and rem_tok is not None:
                used_tok_frac = 1.0 - (rem_tok / limit_tok)
                used_pct      = int(used_tok_frac * 100)
                color         = "green" if used_pct < 50 else ("orange" if used_pct < 80 else "red")
                st.markdown(
                    f"Tokens/min: {limit_tok - rem_tok:,}/{limit_tok:,} used "
                    f"(:{color}[{used_pct}%])"
                )
                st.progress(min(1.0, used_tok_frac))
                if used_pct >= 80:
                    st.warning(
                        f"Approaching Groq TPM limit — {used_pct}% used.",
                        icon="⚠️",
                    )
    else:
        st.caption("Click 'Refresh' to load usage data (requires API to be running).")


# ---------------------------------------------------------------------------
# Tab 6: Eval Dashboard
# ---------------------------------------------------------------------------

def _render_eval_tab() -> None:
    """Render the Eval & Observability dashboard tab."""
    st.markdown("### Evaluation & Observability")

    langsmith_project = os.getenv("LANGSMITH_PROJECT", "MED")
    ls_enabled = bool(os.getenv("LANGSMITH_API_KEY"))

    # Observability links panel
    with st.expander("📍 Where to view each system", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**RAGAS scores**")
            st.caption("This tab (table below) or `GET /eval/dashboard`")
            st.markdown("**Prometheus metrics**")
            st.caption("`http://localhost:8080/metrics` — scrape with Grafana for charts")
            st.markdown("**NLI checker**")
            st.caption("Inline in API logs (`unentailed_ratio=X`). ⚠️ appended to responses when >30%")
        with col_b:
            st.markdown("**LangSmith traces**")
            if ls_enabled:
                st.caption(f"Enabled ✅ — project: `{langsmith_project}`")
                st.markdown("[Open smith.langchain.com →](https://smith.langchain.com)")
            else:
                st.caption("Not enabled — add `LANGSMITH_API_KEY` to .env")
            st.markdown("**pgAdmin tables**")
            st.caption(
                "Host: `localhost:5432` | DB: `mao` | User: `mao` | PW: `mao`\n\n"
                "Tables: `response_metrics` (RAGAS), `chat_sessions` (history), "
                "`guardrail_events` (blocks), `response_feedback` (thumbs)"
            )

    st.divider()

    if st.button("🔄 Refresh RAGAS scores", type="secondary"):
        data = _fetch_eval_dashboard()
        if "error" in data:
            st.error(f"Dashboard unavailable: {data['error']}")
        else:
            metrics = data.get("metrics", [])
            feedback = data.get("feedback", {})

            # Feedback summary
            thumbs_up = int(feedback.get("True", 0))
            thumbs_down = int(feedback.get("False", 0))
            total = thumbs_up + thumbs_down
            satisfaction = f"{thumbs_up / total * 100:.0f}%" if total else "—"

            fcol1, fcol2, fcol3 = st.columns(3)
            fcol1.metric("👍 Helpful", thumbs_up)
            fcol2.metric("👎 Not helpful", thumbs_down)
            fcol3.metric("Satisfaction", satisfaction)

            st.divider()

            if not metrics:
                st.info("No RAGAS metrics yet. Send some queries to populate.")
            else:
                import pandas as pd
                df = pd.DataFrame(metrics)
                cols = [c for c in [
                    "faithfulness", "answer_relevancy", "context_precision", "context_recall",
                    "coherence", "fluency", "helpfulness", "perplexity_proxy", "judge_safety",
                ] if c in df.columns]

                # Score averages
                if cols:
                    st.markdown("**Score averages (all time):**")
                    avg_cols = st.columns(len(cols))
                    for i, col in enumerate(cols):
                        avg = df[col].dropna().mean()
                        delta = "⚠️ hallucination risk" if col == "faithfulness" and avg < 0.7 else None
                        avg_cols[i].metric(col, f"{avg:.3f}", delta=delta, delta_color="inverse")

                st.markdown("**Last 20 turns:**")
                st.dataframe(df[cols].tail(20), use_container_width=True)

    _render_retrieval_metrics_section()


# ---------------------------------------------------------------------------
# Retrieval metrics helpers
# ---------------------------------------------------------------------------

def _fetch_retrieval_metrics(k: int = 5, regenerate: bool = False, samples: int = 50) -> dict:
    params = {"k": k, "regenerate": str(regenerate).lower(), "samples": samples}
    try:
        resp = requests.get(f"{API_URL}/eval/retrieval", params=params, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


def _render_retrieval_metrics_section() -> None:
    """Retrieval quality section inside the Eval tab: MRR, P@K, R@K, F1@K."""
    st.divider()
    st.markdown("### Retrieval Quality Metrics")
    st.caption(
        "MRR and P/R/F1@K measure how well the retriever ranks relevant chunks. "
        "Requires a golden dataset (auto-generated via Groq on first run, ~2 min)."
    )

    col_k, col_s, col_btn, col_regen = st.columns([1, 1, 1, 1])
    with col_k:
        k_val = st.selectbox("K cutoff", [3, 5, 10], index=1, key="retrieval_k")
    with col_s:
        samples_val = st.number_input("Golden samples", min_value=10, max_value=200, value=50, step=10, key="retrieval_samples")
    with col_btn:
        st.write("")
        run_eval = st.button("Run Retrieval Eval", type="secondary")
    with col_regen:
        st.write("")
        regen = st.checkbox("Regenerate golden dataset", value=False, key="retrieval_regen")

    if run_eval:
        with st.spinner("Running retrieval evaluation... (may take 1-3 min)"):
            data = _fetch_retrieval_metrics(k=k_val, regenerate=regen, samples=int(samples_val))

        if "error" in data:
            st.error(f"Retrieval eval failed: {data['error']}")
            return

        if data.get("status") == "no_golden_dataset":
            st.warning(data.get("message", "No golden dataset. Check 'Regenerate' and re-run."))
            return

        current = data.get("current", {})
        if current:
            st.markdown(f"**Results — K={current.get('k', k_val)}, {current.get('samples', 0)} queries**")
            m1, m2, m3, m4 = st.columns(4)
            mrr = current.get("mrr", 0)
            m1.metric("MRR", f"{mrr:.3f}", help="Mean Reciprocal Rank — higher is better (max 1.0)")
            m2.metric(f"Precision@{k_val}", f"{current.get('mean_precision_at_k', 0):.3f}",
                      help="Fraction of top-K results that are relevant")
            m3.metric(f"Recall@{k_val}", f"{current.get('mean_recall_at_k', 0):.3f}",
                      help="Fraction of relevant chunks found in top-K")
            m4.metric(f"F1@{k_val}", f"{current.get('mean_f1_at_k', 0):.3f}",
                      help="Harmonic mean of P@K and R@K")

            if mrr < 0.3:
                st.warning("MRR < 0.3 — retriever is not ranking relevant chunks highly. "
                           "Consider rerunning ingestion or tuning the reranker.")
            elif mrr >= 0.7:
                st.success("MRR >= 0.7 — good retrieval quality.")

            st.caption(f"Golden dataset size: {data.get('golden_dataset_size', '?')} samples")

        details = data.get("details", [])
        if details:
            with st.expander(f"Per-query breakdown (first {len(details)})"):
                import pandas as pd
                df = pd.DataFrame(details)[["question", "precision_at_k", "recall_at_k", "f1_at_k", "reciprocal_rank"]]
                df.columns = ["Question", f"P@{k_val}", f"R@{k_val}", f"F1@{k_val}", "RR"]
                st.dataframe(df, use_container_width=True)

        history = data.get("history", [])
        if len(history) > 1:
            with st.expander("Historical eval runs"):
                import pandas as pd
                hdf = pd.DataFrame(history)
                hdf = hdf.rename(columns={
                    "mrr": "MRR",
                    "mean_precision_at_k": "P@K",
                    "mean_recall_at_k": "R@K",
                    "mean_f1_at_k": "F1@K",
                    "n_samples": "Samples",
                    "created_at": "Run at",
                })
                st.dataframe(hdf[["Run at", "k", "Samples", "MRR", "P@K", "R@K", "F1@K"]],
                             use_container_width=True)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("MAO Clinical AI")
    st.caption(
        "Multi-Agent Orchestrator — Biomedical AI for Alzheimer's & Stroke Research"
    )

    tab_chat, tab_graph, tab_mri, tab_report, tab_health, tab_eval = st.tabs([
        "Chat",
        "Graph Explorer",
        "MRI Scan",
        "Patient Report",
        "System Health",
        "Eval Dashboard",
    ])

    with tab_chat:
        _render_chat_tab()

    with tab_graph:
        _render_graph_tab()

    with tab_mri:
        _render_mri_tab()

    with tab_report:
        _render_report_tab()

    with tab_health:
        _render_health_tab()

    with tab_eval:
        _render_eval_tab()

    # Subtle footer
    ls_key = os.getenv("LANGSMITH_API_KEY", "")
    ls_icon = "✅" if ls_key else "❌"
    st.caption(
        f"MAO Clinical AI — powered by LangGraph + GraphRAG + Groq | LangSmith: {ls_icon}"
    )


if __name__ == "__main__":
    main()
