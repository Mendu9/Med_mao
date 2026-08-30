"""Clinical AI agent: MRI stage prediction (EfficientNetB3), PDF report analysis, and AD research retrieval."""

from __future__ import annotations

import base64
import json
import logging


from mao.agents.multimodal_agent import handle_audio, handle_image
from mao.core.config import MRI_CONFIDENCE_GATE
from mao.core.pii_scrubber import scrub_pii
from mao.core.state import MAOState
from mao.memory.mem0_handler import build_system_prompt
from mao.prompts import get_prompt
from mao.providers import gateway
from mao.providers.registry import ModelRole
from mao.rag.retriever import retrieve
from mao.report.report_card import SourceEntry, build_report_card
from mao.safety.verification import CLINICAL_DISCLAIMER
from mao.schemas.evidence import from_chunk as evidence_from_chunk

logger = logging.getLogger(__name__)

# Medical disclaimer — always appended, never omitted. Defined in
# mao.safety.verification so output_guardrails can re-assert it without
# importing this agent (P0-2).
_DISCLAIMER = CLINICAL_DISCLAIMER

# Answer budgets, named so the live call-site probe can assert against the
# values this module really uses. The bound model's analysis channel is paid for
# on top of these by the gateway — see mao/providers/registry.py.
_REPORT_SUMMARY_MAX_TOKENS = 300
_EXTRACTION_MAX_TOKENS = 512
_SYNTHESIS_MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# System prompts
#
# Read from the registry on every call, never copied into module constants.
# This agent used to carry inline prompt text, and the registered
# `clinical.extraction` spec had already diverged from it: the registry copy
# carries a de-identification instruction the inline copy lacked. The prompt
# test asserted registration, not consumption, so the registry stayed green
# while the highest-risk agent in the system ran unversioned, untraceable text.
# ---------------------------------------------------------------------------

def _clinical_system() -> str:
    return get_prompt("clinical.synthesis").template


def _extraction_system() -> str:
    return get_prompt("clinical.extraction").template

# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def clinical_node(state: MAOState) -> MAOState:
    """LangGraph node: clinical decision support."""
    user_query: str = state.get("pii_scrubbed_query") or state["user_query"]
    domain: str     = state.get("domain", "alzheimer")
    metadata: dict  = state.get("metadata", {})
    memory_context: str = state.get("memory_context", "")


    # --- Detect sub-mode ---
    #
    # The router sends *every* attachment here, so every attachment type needs a
    # branch. Audio had none: an uploaded voice sample was accepted, priced HIGH
    # risk, and then fell through to the plain-text path, which answered the
    # caption and never mentioned that the recording had been discarded. Voice is
    # a recognised Alzheimer's biomarker modality, so silently dropping it is a
    # clinical failure, not a missing nicety.
    has_image  = bool(metadata.get("image_b64") or metadata.get("image_url"))
    has_report = bool(metadata.get("report_b64") or metadata.get("report_path"))
    has_audio  = bool(metadata.get("audio_b64") or metadata.get("audio_path"))

    if has_image:
        response, result_meta = _handle_mri_image(user_query, metadata, memory_context, domain=domain)
    elif has_report:
        response, result_meta = _handle_pdf_report(user_query, metadata, memory_context, domain=domain)
    elif has_audio:
        # One shared transcription implementation, owned by multimodal_agent.
        response, result_meta = handle_audio(user_query, metadata, memory_context)
        result_meta = {**result_meta, "mode": result_meta.get("mode", "audio")}
    else:
        response, result_meta = _handle_text_question(user_query, memory_context, domain=domain)

    # NOTE: the NLI entailment check used to live here. It now runs in
    # mao.safety.verification.verification_node, which sees every agent's output
    # rather than only this one's (P1-2). Do not reintroduce it here.

    # --- Uncertainty flag from MRI confidence ---
    prediction = result_meta.get("prediction", {})
    mri_confidence = float(prediction.get("confidence", 1.0))
    uncertainty_flag = mri_confidence < MRI_CONFIDENCE_GATE if prediction else False

    # --- Build ReportCard ---
    sources = [
        SourceEntry(tool="retriever", snippet=str(s.get("snippet", ""))[:500], query=user_query)
        for s in result_meta.get("sources", [])[:3]
    ]
    report = build_report_card(
        stage=prediction.get("prediction", "N/A") if prediction else "N/A",
        stage_interpretation=_interpret_stage(prediction.get("prediction", "") if prediction else ""),
        clinical_significance=response[:300],
        medications=[],
        literature_evidence=[s.get("source", "") for s in result_meta.get("sources", [])[:3]],
        recommended_next_steps=["Consult a specialist for comprehensive evaluation."],
        confidence_score=mri_confidence if prediction else 1.0,
        sources=sources,
        uncertainty_flag=uncertainty_flag,
    )

    # An empty synthesis must not become a response whose entire body is the
    # disclaimer (adversarial L-3). That reads to a clinician as "the system
    # considered your question and had nothing to say", when what happened is
    # that generation produced nothing — a different fact, and one they can act
    # on by retrying.
    if not response.strip():
        logger.warning("Clinical synthesis produced no text for mode=%s",
                       result_meta.get("mode", "unknown"))
        response = (
            "I could not generate a clinical assessment for this request. "
            "This is a system failure, not a clinical finding. Please retry, "
            "and consult a licensed clinician directly if the problem persists."
        )

    response = response + _DISCLAIMER

    state["response"]     = response
    state["agent_used"]   = "clinical"
    # The response carries what this agent PRODUCED. Nothing the caller sent is
    # echoed back.
    #
    # Stripping `ATTACHMENT_KEYS` and echoing the rest was not enough, because
    # that list describes which keys mean "patient data is attached" — not which
    # keys may contain it. Both shipped frontends send `filename`, and
    # `Doe_Jane_MRN4471023_1948-03-12.pdf` reached Redis intact, where it is
    # cached under two keys and returned to the client. M5's "persist only
    # de-identified text" held for Postgres and for the provider; it did not
    # hold here.
    #
    # Scrubbing the echo instead would not have worked: that filename has no
    # field labels for the labelled rules, and its underscores suppress the word
    # boundaries the shape rules need. There is no general way to de-identify an
    # arbitrary caller-chosen string, so the fix is to not carry it.
    state["metadata"]     = {
        **{k: v for k, v in result_meta.items() if not k.startswith("_")},
        "uncertainty_flag": uncertainty_flag,  # API reads this from metadata
    }
    state["report_card"]  = report.to_dict()
    state["uncertainty_flag"] = uncertainty_flag
    # Publish the evidence this answer was built from.
    #
    # This route used to publish none: the chunks stayed under `_ranked_chunks`
    # in `result_meta`, which the `_`-prefix filter above strips. So on the
    # HIGHEST-risk route in the system, `state["retrieved_docs"]` was empty and
    # three controls quietly degraded at once — the NLI gate had no premise and
    # returned no flags, the judge scored groundedness against nothing, and
    # `_members_for("")` seated only the safety member, so the accuracy and
    # hallucination vetoes never ran on a clinical answer.
    state["retrieved_docs"] = [
        evidence_from_chunk(c) for c in result_meta.get("_ranked_chunks", []) or []
    ]
    return state


def _rag_is_sufficient(ranked_chunks: list) -> bool:
    """Whether retrieval covered the question. See graphrag_agent.rag_is_sufficient.

    One implementation, imported — the 0.20 literal here and the 0.10 literal
    there were two uncalibrated thresholds applied to the same ambiguous score,
    which is exactly the drift the safety policy centralisation exists to stop.
    """
    from mao.agents.graphrag_agent import rag_is_sufficient

    return rag_is_sufficient(ranked_chunks)


def _interpret_stage(stage: str) -> str:
    return {
        "CN":   "Cognitively normal — no significant impairment detected.",
        "EMCI": "Early mild cognitive impairment — early intervention recommended.",
        "LMCI": "Late mild cognitive impairment — closer monitoring advised.",
        "AD":   "Alzheimer's disease — comprehensive care planning recommended.",
    }.get(stage, "Stage information unavailable.")


# ---------------------------------------------------------------------------
# Mode 1 — MRI image
# ---------------------------------------------------------------------------

def _handle_mri_image(
    user_query: str,
    metadata: dict,
    memory_context: str,
    domain: str = "alzheimer",
) -> tuple[str, dict]:
    """Predict → retrieve → web search → synthesize."""

    # Step 1: Run EfficientNetB3
    prediction = _run_mri_prediction(metadata)
    vision_used = False

    if prediction.get("error"):
        logger.warning("MRI prediction failed: %s", prediction["error"])
        # The stage predictor only understands brain MRI. When it cannot answer —
        # because the model is unavailable, or because this is a photograph of a
        # pill bottle rather than a scan — the image still has to be *looked at*.
        # This branch used to synthesise from retrieval alone, so an upload the
        # predictor did not recognise was answered as though no image had been
        # sent, with nothing telling the clinician it had been ignored.
        described, vision_used = _describe_image(user_query, metadata, memory_context)
        pred_block = (
            f"MRI stage prediction did not apply to this image.\n\n"
            f"Image description:\n{described}"
            if vision_used
            else "MRI prediction could not be completed (model unavailable)."
        )
        augmented_query = f"{described} {user_query}" if vision_used else user_query
    else:
        label      = prediction["prediction"]
        full_label = prediction["full_label"]
        confidence = prediction["confidence"]
        scores     = prediction["all_scores"]
        pred_block = _format_prediction(label, full_label, confidence, scores)
        augmented_query = (
            f"Clinical implications and treatment considerations for "
            f"{full_label} ({label}) in Alzheimer's disease. {user_query}"
        )

    # Step 2: GraphRAG retrieval
    try:
        ranked_chunks = retrieve(augmented_query, domain=domain)
    except Exception as exc:  # noqa: BLE001
        logger.error("GraphRAG retrieval failed: %s", exc)
        ranked_chunks = []

    research_block = _format_sources(ranked_chunks)

    # Step 3: Web search for latest evidence
    web_block = _web_search_clinical(
        f"{prediction.get('prediction', '')} Alzheimer's disease treatment 2024"
        if not prediction.get("error") else user_query
    )

    # Step 4: Synthesize
    system_prompt = build_system_prompt(_clinical_system(), memory_context)
    user_prompt = (
        f"User question: {user_query}\n\n"
        f"## MRI Prediction Result\n{pred_block}\n\n"
        f"## Relevant Research Papers\n{research_block}\n\n"
        f"## Current Clinical Evidence (Web)\n{web_block}\n\n"
        "Please synthesize the above into a structured clinical decision-support response. "
        "Include: interpretation, clinical significance, recommended next steps, and cited sources."
    )
    response = _call_llm(system_prompt, user_prompt)

    top_score = ranked_chunks[0].score if ranked_chunks else 0.0
    sources = [
        {"source": c.metadata.get("source", ""), "chunk_id": c.metadata.get("chunk_id", ""),
         "doc_id": c.metadata.get("title", c.metadata.get("source", "")),
         "score": round(c.score, 4), "snippet": c.text[:300]}
        for c in ranked_chunks
    ]
    return response, {
        "mode": "mri_image",
        "prediction": prediction,
        "vision_fallback": vision_used,
        "sources": sources,
        "chunks_retrieved": len(ranked_chunks),
        "top_rag_score": round(top_score, 4),
        "rag_sufficient": _rag_is_sufficient(ranked_chunks),
        "score_scorer": getattr(ranked_chunks[0], "scorer", "") if ranked_chunks else "",
        "_ranked_chunks": ranked_chunks,
    }


def _describe_image(
    user_query: str, metadata: dict, memory_context: str
) -> tuple[str, bool]:
    """Describe an image the stage predictor could not classify.

    Returns (description, succeeded). Never raises: a vision outage should cost
    the description, not the whole clinical request.
    """
    try:
        described, meta = handle_image(user_query, metadata, memory_context)
    except Exception as exc:  # noqa: BLE001
        logger.error("Vision fallback failed: %s", exc)
        return "", False
    if meta.get("error") or not described.strip():
        logger.warning("Vision fallback produced nothing: %s", meta.get("error", ""))
        return "", False
    return described, True


def _run_mri_prediction(metadata: dict) -> dict:
    """Call MRIPredictor with image from metadata."""
    try:
        from mao.models.mri_predictor import get_predictor
        predictor = get_predictor()
        if metadata.get("image_b64"):
            return predictor.predict(metadata["image_b64"])
        if metadata.get("image_url"):
            # Through the shared guard, never `requests.get` directly: the URL
            # is caller-supplied and the fetch runs with the deployment's own
            # network position. See `mao/safety/fetch.py`.
            from mao.safety.fetch import fetch_image_bytes

            return predictor.predict(fetch_image_bytes(metadata["image_url"]))
        return {"error": "no image data found in metadata"}
    except Exception as exc:  # noqa: BLE001
        logger.error("MRI prediction call failed: %s", exc)
        return {"error": str(exc)}


def _format_prediction(label: str, full_label: str, confidence: float, scores: dict) -> str:
    lines = [
        f"**Predicted Stage: {full_label} ({label})**",
        f"Confidence: {confidence * 100:.1f}%",
        "",
        "All class probabilities:",
    ]
    for lbl, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(score * 20)
        lines.append(f"  {lbl:4s}: {score * 100:5.1f}%  {bar}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mode 2 — PDF report
# ---------------------------------------------------------------------------

def _handle_pdf_report(
    user_query: str,
    metadata: dict,
    memory_context: str,
    domain: str = "alzheimer",
) -> tuple[str, dict]:
    """Extract PDF text → summarize → structured extraction → research → web → synthesize."""

    # Step 1: Get PDF text
    raw_report_text = _extract_pdf_text(metadata)
    if not raw_report_text:
        return (
            "Could not extract text from the provided PDF. "
            "Please ensure the file is a readable PDF.",
            {"mode": "pdf_report", "error": "pdf_extraction_failed"},
        )

    # P0-4: de-identify BEFORE anything leaves the process. Every downstream use
    # of the report — summarisation, structured extraction, the retrieval seed,
    # the final synthesis prompt — reads the scrubbed text, so there is no path
    # from an uploaded report to a third party carrying direct identifiers.
    report_text = scrub_pii(raw_report_text)

    # Step 2: Structured extraction
    extracted = _extract_structured_fields(report_text)

    # Step 3: Summarize
    summary = _summarize_report(report_text, memory_context)

    # Step 4: GraphRAG retrieval on report content
    retrieval_query = report_text[:500]  # first 500 chars as retrieval seed
    if extracted.get("diagnosis"):
        retrieval_query = f"{extracted['diagnosis']} {retrieval_query}"
    try:
        ranked_chunks = retrieve(retrieval_query, domain=domain)
    except Exception as exc:  # noqa: BLE001
        logger.error("GraphRAG retrieval failed: %s", exc)
        ranked_chunks = []

    research_block = _format_sources(ranked_chunks)

    # Step 5: Web search for guidelines
    search_query = f"{extracted.get('diagnosis', 'Alzheimer disease')} clinical guidelines treatment 2024"
    web_block = _web_search_clinical(search_query)

    # Step 6: Synthesize
    system_prompt = build_system_prompt(_clinical_system(), memory_context)

    extracted_text = json.dumps(extracted, indent=2) if extracted else "(extraction failed)"
    user_prompt = (
        f"User request: {user_query}\n\n"
        f"## Report Summary\n{summary}\n\n"
        f"## Extracted Clinical Data\n```json\n{extracted_text}\n```\n\n"
        f"## Relevant Research Papers\n{research_block}\n\n"
        f"## Clinical Guidelines (Web)\n{web_block}\n\n"
        "Provide a structured clinical decision-support response with sections:\n"
        "## Summary\n## Key Findings\n## Relevant Research\n## Clinical Guidance\n"
        "Always cite research papers by their filename and chunk ID."
    )
    response = _call_llm(system_prompt, user_prompt)

    top_score = ranked_chunks[0].score if ranked_chunks else 0.0
    sources = [
        {"source": c.metadata.get("source", ""), "chunk_id": c.metadata.get("chunk_id", ""),
         "doc_id": c.metadata.get("title", c.metadata.get("source", "")),
         "score": round(c.score, 4), "snippet": c.text[:300]}
        for c in ranked_chunks
    ]
    return response, {
        "mode": "pdf_report",
        "extracted": extracted,
        "sources": sources,
        "chunks_retrieved": len(ranked_chunks),
        "top_rag_score": round(top_score, 4),
        "rag_sufficient": _rag_is_sufficient(ranked_chunks),
        "score_scorer": getattr(ranked_chunks[0], "scorer", "") if ranked_chunks else "",
        "report_length": len(report_text),
        "_ranked_chunks": ranked_chunks,
    }


def _extract_pdf_text(metadata: dict) -> str:
    """Extract text from PDF in metadata (base64 or file path)."""
    try:
        from pypdf import PdfReader
        import io
    except ImportError:
        logger.error("pypdf not installed. Run: pip install pypdf")
        return ""

    try:
        if metadata.get("report_b64"):
            pdf_bytes = base64.b64decode(metadata["report_b64"])
            reader = PdfReader(io.BytesIO(pdf_bytes))
        elif metadata.get("report_path"):
            reader = PdfReader(metadata["report_path"])
        else:
            return ""

        pages = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)
            except Exception:  # noqa: BLE001
                continue
        return "\n\n".join(pages)
    except Exception as exc:  # noqa: BLE001
        logger.error("PDF extraction failed: %s", exc)
        return ""


def _summarize_report(report_text: str, memory_context: str) -> str:
    """Map-reduce summarization of a medical report."""
    if not report_text.strip():
        return ""
    text = report_text[:3000]
    prompt = (
        "Summarise the following medical report in 3-5 sentences, "
        "focusing on the primary diagnosis, key findings, and current treatment.\n\n"
        f"Report:\n{text}"
    )
    try:
        return gateway.complete(
            role=ModelRole.CLINICAL_SYNTHESIS,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=_REPORT_SUMMARY_MAX_TOKENS,
        ).text.strip()
    except Exception as exc:
        logger.error("Report summarization failed: %s", exc)
        return text[:500]


def _extract_structured_fields(report_text: str) -> dict:
    """Extract structured clinical data from report text.

    The system prompt is the registered `clinical.extraction` spec. This used to
    build its own inline instruction, so the registered version — the one
    carrying "The report has already been de-identified; do not attempt to infer
    patient identity." — never reached a model at all, and the prompt test that
    asserted its registration passed regardless.
    """
    try:
        raw = gateway.complete(
            role=ModelRole.CLINICAL_SYNTHESIS,
            messages=[
                {"role": "system", "content": _extraction_system()},
                {"role": "user", "content": f"Report:\n{report_text[:2000]}"},
            ],
            temperature=0.0,
            max_tokens=_EXTRACTION_MAX_TOKENS,
        ).text.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end]) if start >= 0 else {}
    except Exception as exc:
        logger.error("Structured extraction failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Mode 3 — Text question only
# ---------------------------------------------------------------------------

def _handle_text_question(
    user_query: str,
    memory_context: str,
    domain: str = "alzheimer",
) -> tuple[str, dict]:
    """GraphRAG retrieval + synthesis with clinical framing."""
    try:
        ranked_chunks = retrieve(user_query, domain=domain)
    except Exception as exc:  # noqa: BLE001
        logger.error("GraphRAG retrieval failed: %s", exc)
        ranked_chunks = []

    research_block = _format_sources(ranked_chunks)

    system_prompt = build_system_prompt(_clinical_system(), memory_context)
    user_prompt = (
        f"## Relevant Research\n{research_block}\n\n"
        f"User question: {user_query}\n\n"
        "Provide a clear, evidence-based answer citing the research sources above."
    )
    response = _call_llm(system_prompt, user_prompt)

    top_score = ranked_chunks[0].score if ranked_chunks else 0.0
    sources = [
        {"source": c.metadata.get("source", ""), "chunk_id": c.metadata.get("chunk_id", ""),
         "doc_id": c.metadata.get("title", c.metadata.get("source", "")),
         "score": round(c.score, 4), "snippet": c.text[:300]}
        for c in ranked_chunks
    ]
    return response, {
        "mode": "text_question",
        "sources": sources,
        "chunks_retrieved": len(ranked_chunks),
        "top_rag_score": round(top_score, 4),
        "rag_sufficient": _rag_is_sufficient(ranked_chunks),
        "score_scorer": getattr(ranked_chunks[0], "scorer", "") if ranked_chunks else "",
        "_ranked_chunks": ranked_chunks,
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _format_sources(ranked_chunks) -> str:
    """Format retrieved chunks with source filename, chunk_id, score, and snippet."""
    if not ranked_chunks:
        return "No relevant research found in knowledge base."
    lines = []
    for i, chunk in enumerate(ranked_chunks, 1):
        source   = chunk.metadata.get("source", "unknown")
        chunk_id = chunk.metadata.get("chunk_id", "—")
        score    = chunk.score
        snippet  = chunk.text[:400].replace("\n", " ")
        lines.append(
            f"[{i}] **{source}** (chunk: {chunk_id}, relevance: {score:.3f})\n"
            f"    {snippet}..."
        )
    return "\n\n".join(lines)


def _web_search_clinical(query: str) -> str:
    """Web search for clinical evidence — uses multi-provider fallback chain."""
    from mao.core.web_search import web_search

    results = web_search(query, num_results=4)
    if not results:
        return "No web results found."
    lines = []
    for r in results:
        title = r.get("title", "")
        body  = r.get("body", "")[:300]
        lines.append(f"- **{title}**\n  {body}")
    return "\n\n".join(lines)


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Clinical synthesis, through the gateway on the clinical role."""
    try:
        return gateway.complete(
            role=ModelRole.CLINICAL_SYNTHESIS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=_SYNTHESIS_MAX_TOKENS,
        ).text.strip()
    except Exception as exc:
        logger.error("Clinical LLM call failed: %s", exc)
        return f"Response generation failed: {exc}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from mao.core.state import make_initial_state
    state = make_initial_state(
        "My patient has LMCI. What treatment options exist?",
        "user-test",
    )
    state = clinical_node(state)
    print(state["response"][:500])
    print("\nAgent:", state["agent_used"])
    print("Mode:", state["metadata"].get("mode"))
    print("Chunks:", state["metadata"].get("chunks_retrieved"))
