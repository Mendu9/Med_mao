"""
app/frontend.py
---------------
Gradio 5 clinical AI assistant UI for MAO.
Calls the MAO FastAPI backend at http://localhost:8080.

Usage:
    python app/frontend.py
    # Opens at http://127.0.0.1:7860

Requirements:
    pip install "gradio>=5.0" requests
"""

from __future__ import annotations

import base64
import logging
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAO_API_URL = "http://localhost:8080"

_DISCLAIMER_HTML = """
<div style="background:#fff3cd;border:1px solid #ffc107;border-radius:6px;
            padding:10px 14px;margin-bottom:8px;font-size:0.88em;color:#333;">
    <strong>⚠️ AI Decision Support Only</strong> — Not a medical diagnosis.
    All results must be reviewed by a licensed healthcare professional.
</div>
"""

_WELCOME = (
    "Hello! I'm the MAO Clinical AI Assistant.\n\n"
    "I can help with:\n"
    "- **MRI Analysis** — upload a brain scan for Alzheimer's stage prediction\n"
    "- **Report Analysis** — upload a PDF medical report for structured review\n"
    "- **Clinical Questions** — ask about stages, biomarkers, or treatments\n"
    "- **Code / Visualization** — generate NIfTI viewing code\n\n"
    "Upload a file or type your question below to get started."
)


# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

def call_mao_api(
    query: str,
    user_id: str,
    chat_history: list[dict],
    image_path: str | None,
    report_path: str | None,
    nifti_path: str | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    if image_path:
        metadata["image_b64"] = base64.b64encode(Path(image_path).read_bytes()).decode()
        if not query.strip():
            query = "What does this MRI show? Predict the Alzheimer's stage and explain the clinical implications."

    if report_path:
        metadata["report_b64"] = base64.b64encode(Path(report_path).read_bytes()).decode()
        if not query.strip():
            query = "Analyse this medical report: summarise findings, match to research, and suggest clinical guidance."

    if nifti_path:
        metadata["nifti_b64"] = base64.b64encode(Path(nifti_path).read_bytes()).decode()
        if not query.strip():
            query = "Generate an interactive 3D Plotly visualization for this NIfTI brain scan."

    payload = {
        "query":        query.strip() or "Hello",
        "user_id":      user_id,
        "chat_history": chat_history[-6:],
        "metadata":     metadata,
    }

    try:
        resp = requests.post(f"{MAO_API_URL}/chat", json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {
            "response": (
                "Cannot connect to MAO API at http://localhost:8080.\n\n"
                "Start the API first:\n```\nuvicorn mao.api.main:app --port 8080\n```"
            ),
            "agent_used": "error", "intent": "error", "metadata": {}, "latency_ms": 0,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "response": f"Request failed: {exc}",
            "agent_used": "error", "intent": "error", "metadata": {}, "latency_ms": 0,
        }


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_prediction(pred: dict) -> str:
    if not pred or pred.get("error"):
        return ""
    label      = pred.get("prediction", "?")
    full_label = pred.get("full_label", label)
    confidence = pred.get("confidence", 0.0)
    scores     = pred.get("all_scores", {})
    lines = [
        f"### MRI Prediction\n**{full_label}** (`{label}`) — {confidence*100:.1f}% confidence\n",
        "| Stage | Probability |",
        "|-------|-------------|",
    ]
    for lbl, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {lbl} | {score*100:.1f}% |")
    return "\n".join(lines)


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "_No research sources for this response._"
    lines = ["### Research Sources\n"]
    for i, s in enumerate(sources, 1):
        src      = s.get("source", "unknown")
        chunk_id = s.get("chunk_id", "—")
        score    = s.get("score", 0.0)
        snippet  = s.get("snippet", "")[:280]
        lines.append(
            f"**[{i}] {src}**  \n"
            f"Chunk: `{chunk_id}` · Relevance: `{score:.3f}`  \n"
            f"> {snippet}...\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_interface():
    with gr.Blocks(title="MAO Clinical AI", theme=gr.themes.Soft()) as demo:

        # --- per-session state (plain string defaults — no lambdas) ---
        session_id    = gr.State(value="")   # filled on first send
        history_state = gr.State(value=[])

        gr.HTML("<h1 style='text-align:center'>🧠 MAO Clinical AI Assistant</h1>")
        gr.HTML("<p style='text-align:center;color:#666'>Alzheimer's Disease Decision Support</p>")
        gr.HTML(_DISCLAIMER_HTML)

        with gr.Row(equal_height=False):

            # ── Left: uploads ──────────────────────────────────────────
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### Upload")
                image_upload  = gr.Image(label="MRI Image (PNG/JPG)", type="filepath", sources=["upload"])
                report_upload = gr.File(label="Medical Report (PDF)", file_types=[".pdf"])
                nifti_upload  = gr.File(label="NIfTI (.nii / .nii.gz)", file_types=[".nii", ".gz"])
                gr.Markdown("---")
                prediction_md = gr.Markdown(value="", visible=False)
                clear_btn     = gr.Button("🗑 Clear", variant="secondary", size="sm")

            # ── Right: chat ────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[{"role": "assistant", "content": _WELCOME}],
                    type="messages",
                    height=460,
                    show_label=False,
                )

                with gr.Accordion("📄 Research Sources", open=False):
                    sources_md = gr.Markdown("_Sources will appear here._")

                with gr.Row():
                    agent_box   = gr.Textbox(label="Agent",   interactive=False, scale=1, max_lines=1)
                    intent_box  = gr.Textbox(label="Intent",  interactive=False, scale=1, max_lines=1)
                    latency_box = gr.Textbox(label="Latency", interactive=False, scale=1, max_lines=1)

                with gr.Row():
                    msg_box  = gr.Textbox(placeholder="Type your question...", show_label=False, scale=5, lines=1)
                    send_btn = gr.Button("Send →", variant="primary", scale=1)

        # ── Respond handler ────────────────────────────────────────────

        def respond(message, chat, hist, uid, image, report, nifti):
            # Generate session ID on first message
            if not uid:
                uid = f"user-{uuid.uuid4().hex[:8]}"

            report_path = report if isinstance(report, str) else (report.name if report else None)
            nifti_path  = nifti  if isinstance(nifti,  str) else (nifti.name  if nifti  else None)

            result = call_mao_api(
                query=message,
                user_id=uid,
                chat_history=hist,
                image_path=image,
                report_path=report_path,
                nifti_path=nifti_path,
            )

            response    = result.get("response", "")
            agent_used  = result.get("agent_used", "unknown")
            intent      = result.get("intent",     "unknown")
            latency     = result.get("latency_ms",  0)
            meta        = result.get("metadata",    {})

            display_msg = message if message.strip() else "(file uploaded)"
            new_chat = chat + [
                {"role": "user",      "content": display_msg},
                {"role": "assistant", "content": response},
            ]
            new_hist = hist + [
                {"role": "user",      "content": display_msg},
                {"role": "assistant", "content": response},
            ]

            sources   = meta.get("sources", [])
            pred_dict = meta.get("prediction", {})
            pred_text = _format_prediction(pred_dict)

            return (
                new_chat,                                         # chatbot
                new_hist,                                         # history_state
                uid,                                              # session_id
                "",                                               # clear msg_box
                _format_sources(sources),                         # sources_md
                gr.update(value=pred_text, visible=bool(pred_text)),  # prediction_md
                agent_used,
                intent,
                f"{latency:.0f} ms",
            )

        inputs  = [msg_box, chatbot, history_state, session_id, image_upload, report_upload, nifti_upload]
        outputs = [chatbot, history_state, session_id, msg_box, sources_md, prediction_md, agent_box, intent_box, latency_box]

        send_btn.click(respond, inputs=inputs, outputs=outputs)
        msg_box.submit(respond,  inputs=inputs, outputs=outputs)

        # ── Clear handler ──────────────────────────────────────────────

        def clear_all():
            return (
                [{"role": "assistant", "content": _WELCOME}],
                [],
                "",
                "",
                "_Sources will appear here._",
                gr.update(value="", visible=False),
                "", "", "",
            )

        clear_btn.click(
            clear_all,
            outputs=[chatbot, history_state, session_id, msg_box, sources_md, prediction_md, agent_box, intent_box, latency_box],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
