import gradio as gr
import httpx
import json
import os
try:
    from app.graph_explorer import build_graph_explorer_tab
except ModuleNotFoundError:
    # Running as `python app/frontend.py` — project root not in sys.path yet
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from app.graph_explorer import build_graph_explorer_tab

API_BASE = os.getenv("MAO_API_BASE", "http://localhost:8080")


def _send_query(query: str, history: list, file_obj) -> tuple:
    if not query or not query.strip():
        query = "Analyse the uploaded MRI scan and provide a clinical assessment."
    try:
        import base64
        metadata = {}
        if file_obj is not None:
            try:
                with open(file_obj.name, "rb") as f:
                    metadata["image_b64"] = base64.b64encode(f.read()).decode()
                    metadata["filename"] = os.path.basename(file_obj.name)
            except Exception:
                pass
        payload = {"query": query, "metadata": metadata}
        resp = httpx.post(f"{API_BASE}/chat", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", data.get("response", ""))
        session_id = data.get("session_id", "")
        report_card = data.get("report_card")

        history = list(history or [])
        history.append((query, answer))

        report_html = _render_report_card_html(report_card) if report_card else ""
        return history, report_html, session_id
    except Exception as e:
        history = list(history or [])
        history.append((query, f"Error: {e}"))
        return history, "", ""


def _render_report_card_html(card: dict) -> str:
    if not card:
        return ""
    uncertainty_banner = ""
    if card.get("uncertainty_flag"):
        uncertainty_banner = (
            '<div style="background:#fee;border:1px solid red;padding:8px;'
            'border-radius:4px;margin-bottom:12px;">'
            "<b>LOW CONFIDENCE</b> — MRI model confidence below threshold.</div>"
        )
    meds = card.get("medications", [])
    meds_html = ""
    if meds:
        rows = "".join(
            f"<tr><td>{m.get('name','')}</td><td>{m.get('dose','')}</td>"
            f"<td>{m.get('evidence','')}</td><td>{m.get('notes','')}</td></tr>"
            for m in meds
        )
        meds_html = (
            "<h4>Medications</h4><table border='1' cellpadding='4'>"
            "<tr><th>Drug</th><th>Dose</th><th>Evidence</th><th>Notes</th></tr>"
            f"{rows}</table>"
        )
    steps = "".join(f"<li>{s}</li>" for s in card.get("recommended_next_steps", []))
    refs = "".join(f"<li>{r}</li>" for r in card.get("literature_evidence", []))
    conf = card.get("confidence_score", 0)
    return (
        f"{uncertainty_banner}"
        f'<div style="font-family:sans-serif;max-width:800px">'
        f"<h3>Stage: {card.get('stage', '—')}</h3>"
        f"<p>{card.get('stage_interpretation', '')}</p>"
        f"<h4>Clinical Significance</h4><p>{card.get('clinical_significance', '')}</p>"
        f"{meds_html}"
        f"<h4>Recommended Next Steps</h4><ul>{steps}</ul>"
        f"<h4>Literature Evidence</h4><ul>{refs}</ul>"
        f"<p><b>Confidence:</b> {conf:.0%}</p>"
        f"<p><i>{card.get('disclaimer', '')}</i></p>"
        f"</div>"
    )


def _send_feedback(session_id: str, thumbs_up: bool, comment: str) -> str:
    if not session_id:
        return "No session to rate."
    try:
        httpx.post(
            f"{API_BASE}/feedback",
            json={"session_id": session_id, "thumbs_up": thumbs_up, "comment": comment},
            timeout=10,
        )
        return "Feedback submitted. Thank you."
    except Exception as e:
        return f"Feedback failed: {e}"


def _download_report(session_id: str):
    if not session_id:
        return None
    try:
        resp = httpx.get(f"{API_BASE}/export/report/{session_id}", timeout=30)
        resp.raise_for_status()
        import tempfile, pathlib
        tmp = pathlib.Path(tempfile.gettempdir()) / f"report_{session_id}.pdf"
        tmp.write_bytes(resp.content)
        return str(tmp)
    except Exception:
        return None


def _load_dashboard():
    try:
        resp = httpx.get(f"{API_BASE}/eval/dashboard", timeout=10)
        data = resp.json()
        metrics = data.get("metrics", [])
        feedback = data.get("feedback", {})
        fb_text = (
            f"Thumbs up: {feedback.get('True', 0)} | "
            f"Thumbs down: {feedback.get('False', 0)}"
        )
        if not metrics:
            return None, fb_text
        import pandas as pd
        df = pd.DataFrame(metrics)
        cols = [c for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] if c in df.columns]
        return df[cols].tail(20) if cols else None, fb_text
    except Exception as e:
        return None, f"Dashboard error: {e}"


with gr.Blocks(title="MAO Clinical AI") as demo:
    session_id_state = gr.State("")

    with gr.Tabs():
        # ── Tab 1: Clinical ──────────────────────────────────────────────
        with gr.Tab("Clinical"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="MAO Clinical Chat", height=500
                    )
                    with gr.Row():
                        query_input = gr.Textbox(
                            placeholder="Enter your clinical query...",
                            scale=4,
                            show_label=False,
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Column(scale=2):
                    file_upload = gr.File(
                        label="Upload MRI / PDF Report",
                        file_types=[".nii", ".nii.gz", ".png", ".jpg", ".pdf"],
                    )
                    report_display = gr.HTML(label="Report Card")
                    with gr.Row():
                        thumb_up_btn = gr.Button("👍 Helpful", size="sm")
                        thumb_down_btn = gr.Button("👎 Not helpful", size="sm")
                    feedback_comment = gr.Textbox(
                        placeholder="Optional comment...", show_label=False
                    )
                    feedback_status = gr.Textbox(show_label=False, interactive=False)
                    download_btn = gr.Button("📄 Download PDF Report", size="sm")
                    pdf_file = gr.File(label="PDF", interactive=False)

            submit_btn.click(
                _send_query,
                inputs=[query_input, chatbot, file_upload],
                outputs=[chatbot, report_display, session_id_state],
            ).then(lambda: "", outputs=query_input)

            query_input.submit(
                _send_query,
                inputs=[query_input, chatbot, file_upload],
                outputs=[chatbot, report_display, session_id_state],
            )

            thumb_up_btn.click(
                lambda sid, c: _send_feedback(sid, True, c),
                inputs=[session_id_state, feedback_comment],
                outputs=feedback_status,
            )
            thumb_down_btn.click(
                lambda sid, c: _send_feedback(sid, False, c),
                inputs=[session_id_state, feedback_comment],
                outputs=feedback_status,
            )
            download_btn.click(
                _download_report, inputs=session_id_state, outputs=pdf_file
            )

        # ── Tab 2: History ───────────────────────────────────────────────
        with gr.Tab("History"):
            gr.Markdown("## Session History\nPrevious queries and answers will appear here.")
            history_display = gr.JSON(label="Session Log", value=[])

        # ── Tab 3: Eval Dashboard ────────────────────────────────────────
        with gr.Tab("Eval Dashboard"):
            refresh_btn = gr.Button("Refresh Dashboard")
            metrics_df = gr.Dataframe(
                label="RAGAS Metrics (last 20 turns)",
                headers=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
            )
            feedback_summary = gr.Textbox(label="User Feedback Summary", interactive=False)

            refresh_btn.click(
                _load_dashboard,
                outputs=[metrics_df, feedback_summary],
            )

        # ── Tab 4: Graph Explorer ────────────────────────────────────────
        build_graph_explorer_tab()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
