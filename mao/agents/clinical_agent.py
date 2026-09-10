"""Clinical AI agent: MRI stage prediction (EfficientNetB3), PDF report analysis, and AD research retrieval."""

from __future__ import annotations

import base64
import logging


from mao.agents.multimodal_agent import handle_audio, handle_image
from mao.core.config import MRI_CONFIDENCE_GATE
from mao.core.state import MAOState
from mao.memory.mem0_handler import build_system_prompt
from mao.prompts import get_prompt
from mao.providers import gateway
from mao.providers.registry import ModelRole
from mao.rag.retriever import retrieve
from mao.report.report_card import SourceEntry, build_report_card
from mao.safety.verification import CLINICAL_DISCLAIMER
from mao.schemas.evidence import from_chunk as evidence_from_chunk
from mao.trust.classes import (
    InputChannel,
    PublicEvidence,
    SafeDerivedText,
    SafeSynthesisContext,
)
from mao.trust.egress.gateway import current_protection
from mao.trust.egress.policy import EgressPurpose
from mao.trust.handoff.compiler import compile_handoff
from mao.trust.inputs import limits
from mao.trust.inputs.boundary import protect_channel

logger = logging.getLogger(__name__)

# Medical disclaimer — always appended, never omitted. Defined in
# mao.safety.verification so output_guardrails can re-assert it without
# importing this agent (P0-2).
_DISCLAIMER = CLINICAL_DISCLAIMER

# Answer budgets, named so the live call-site probe can assert against the
# values this module really uses. The bound model's analysis channel is paid for
# on top of these by the gateway — see mao/providers/registry.py.
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
    # disclaimer (adversarial L-3). The substitution itself now lives at the API
    # boundary in `mao/api/finalize.py`, because this copy was the whole defect:
    # the guard went into this agent only, and `graphrag_agent` — the route real
    # clinical traffic takes — kept returning an empty string.
    #
    # What has to stay here is *not appending the disclaimer to nothing*. An
    # empty body plus the mandatory disclaimer is not an answer, and it is
    # non-empty, so it would sail past the boundary guard looking like content.
    if response.strip():
        response = response + _DISCLAIMER
    else:
        logger.warning(
            "Clinical synthesis produced no text for mode=%s — leaving the body "
            "empty for the boundary guard to report as a failure",
            result_meta.get("mode", "unknown"),
        )

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
    r"""Protected text -> typed facts -> SafeEvidenceQuery -> SafeSynthesisContext.

    The shape of this function is the M-1 migration. It used to be:

        extract PDF -> scrub -> send the scrubbed note to a model three times

    and the scrubbed note WAS the payload — for the summary, for the structured
    extraction, and for the final synthesis prompt. Under the approved policy a
    scrubbed free-text report is not a payload any external model may receive,
    so the note now stops here. What crosses the boundary is a typed projection
    the Safe Handoff Compiler built field by field, and if it cannot build one
    the request is refused with the fields that would resolve it.
    """
    protected = _protected_report(metadata)
    if protected is None:
        return (
            "Could not extract text from the provided PDF. "
            "Please ensure the file is a readable PDF.",
            {"mode": "pdf_report", "error": "pdf_extraction_failed"},
        )
    report_text = protected.text

    # The safe projections. `HandoffRefused` is a deliberate outcome, not a
    # failure: it means no payload could be built that both excludes the
    # identifiers and carries enough of the case to answer, and the caller is
    # told which structured fields would resolve that.
    handoff = compile_handoff(
        report_text,
        question=user_query,
        structured=_structured_fields(metadata),
        provenance=("report",),
    )

    # Retrieval is seeded from the SAFE EVIDENCE QUERY, not from the first 500
    # characters of the note. The old seed was a slice of a patient document
    # sent to the vector store; this one is the concepts the case is about.
    retrieval_query = " ".join(
        [
            handoff.evidence_query.search_intent,
            *handoff.evidence_query.condition,
            *handoff.evidence_query.intervention,
            *handoff.evidence_query.findings,
        ]
    ).strip()
    try:
        ranked_chunks = retrieve(retrieval_query, domain=domain)
    except Exception as exc:  # noqa: BLE001
        logger.error("GraphRAG retrieval failed: %s", exc)
        ranked_chunks = []

    web_results = _web_search_clinical(
        " ".join(
            [
                *handoff.evidence_query.condition[:2],
                *handoff.evidence_query.intervention[:2],
                "clinical guidelines",
            ]
        ).strip()
        or "Alzheimer disease clinical guidelines"
    )

    evidence = _public_evidence(ranked_chunks, web_results)
    system_prompt = build_system_prompt(_clinical_system(), memory_context)
    response = _synthesise(system_prompt, handoff.synthesis_context, evidence)

    top_score = ranked_chunks[0].score if ranked_chunks else 0.0
    sources = [
        {"source": c.metadata.get("source", ""), "chunk_id": c.metadata.get("chunk_id", ""),
         "doc_id": c.metadata.get("title", c.metadata.get("source", "")),
         "score": round(c.score, 4), "snippet": c.text[:300]}
        for c in ranked_chunks
    ]
    return response, {
        "mode": "pdf_report",
        # The safe projection's own fields, never the note. This dictionary is
        # echoed into the response metadata, cached in Redis and written to a
        # trace, so what goes in it is an egress decision like any other.
        "safe_case_fields": sorted(handoff.evidence_query.provenance),
        "clinical_coverage": round(handoff.facts.coverage(), 3),
        "sources": sources,
        "chunks_retrieved": len(ranked_chunks),
        "top_rag_score": round(top_score, 4),
        "rag_sufficient": _rag_is_sufficient(ranked_chunks),
        "score_scorer": getattr(ranked_chunks[0], "scorer", "") if ranked_chunks else "",
        "report_length": len(report_text),
        "_ranked_chunks": ranked_chunks,
    }


def _extract_pdf_text(metadata: dict) -> str:
    """Extract text from PDF in metadata (base64 or file path), size-capped.

    Two caps, both before any expensive work, both refusing rather than
    truncating. A 194 KB request extracted to 520,318 characters and held one of
    eight worker threads for 34 seconds inside `find_ambiguities` and
    `scrub_pii`, and doubling the page count roughly quadrupled the cost;
    `ChatRequest.query` was capped at 8,000 characters while `metadata` was an
    unvalidated dict, so the cap was on the one channel that did not need it.

    The remaining quadratic in `layout._pair_by_type` is deferred. The cap is
    what makes deferring it safe, which is why the review called it the minimum
    viable fix and why it is Phase 1 scope while the rewrite is not.
    """
    try:
        from pypdf import PdfReader
        import io
    except ImportError:
        logger.error("pypdf not installed. Run: pip install pypdf")
        return ""

    try:
        if metadata.get("report_b64"):
            pdf_bytes = base64.b64decode(metadata["report_b64"])
            limits.check(
                "the decoded report",
                len(pdf_bytes),
                limits.MAX_DECODED_ATTACHMENT_BYTES,
            )
            reader = PdfReader(io.BytesIO(pdf_bytes))
        elif metadata.get("report_path"):
            reader = PdfReader(metadata["report_path"])
        else:
            return ""

        pages: list[str] = []
        extracted = 0
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)
                    extracted += len(text)
                    # Checked per page rather than once at the end, so an
                    # oversized document costs the pages read so far and not the
                    # whole extraction.
                    limits.check(
                        "the extracted report text",
                        extracted,
                        limits.MAX_EXTRACTED_TEXT_CHARS,
                    )
            except limits.InputTooLarge:
                raise
            except Exception:  # noqa: BLE001
                continue
        return "\n\n".join(pages)
    except limits.InputTooLarge:
        # A refusal, not a failure. It must reach the route, which turns it into
        # a 413 telling the caller what to do; swallowing it here would answer
        # the request from an empty report and say nothing about why.
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("PDF extraction failed: %s", exc)
        return ""


def _protected_report(metadata: dict) -> SafeDerivedText | None:
    """The report, de-identified exactly once, recorded against this request.

    Returns None when there was no readable report. Raises `AmbiguousDocument`
    when the patient header cannot be resolved — the upload path refuses rather
    than guessing, because the document is processed unseen and a wrong guess is
    silent either way it goes wrong.

    The caller's structured patient fields go in FIRST, by exact match. That is
    the remedy the refusal itself names, and it is what makes the 422 an
    actionable request: a header the caller has already stated needs no boundary
    to be established, so the same document processes correctly once the fields
    are supplied.
    """
    raw = _extract_pdf_text(metadata)
    if not raw:
        return None
    return protect_channel(
        raw,
        InputChannel.REPORT,
        refuse_ambiguity=True,
        structured=_structured_fields(metadata),
    )


#: Metadata keys carrying caller-supplied structured patient fields.
#:
#: This is the pathway a refusal points at. A caller told "send the patient
#: identifiers as structured metadata fields" needs somewhere to send them, and
#: without this key the advice was unactionable.
_STRUCTURED_KEY = "patient_fields"


def _structured_fields(metadata: dict) -> dict[str, str]:
    """The caller's structured patient fields, if any, as strings."""
    fields = metadata.get(_STRUCTURED_KEY)
    if not isinstance(fields, dict):
        return {}
    return {str(k): str(v) for k, v in fields.items() if v is not None}


def _public_evidence(ranked_chunks, web_results: str) -> list[PublicEvidence]:
    """Retrieved evidence as the trust class it is.

    Registered with this request's protection as it is built, so the egress
    assertion masks it out before looking for identifiers. A memory clinic's
    corpus says `Parkinson` on almost every page, and refusing an answer because
    the evidence mentions Parkinson's disease while the patient is also called
    Parkinson would be a false refusal on the clinical path — a patient-safety
    cost paid for no privacy gain.
    """
    protection = current_protection()
    evidence: list[PublicEvidence] = []
    for index, chunk in enumerate(ranked_chunks or [], 1):
        item = PublicEvidence(
            evidence_id=str(chunk.metadata.get("chunk_id", f"chunk-{index}")),
            text=chunk.text,
            source_id=str(chunk.metadata.get("source", "")),
            title=str(chunk.metadata.get("title", "")),
        )
        evidence.append(item)
        if protection is not None:
            protection.register_evidence(item.text)
    if web_results and web_results != "No web results found.":
        item = PublicEvidence(evidence_id="web", text=web_results, source_id="web_search")
        evidence.append(item)
        if protection is not None:
            protection.register_evidence(item.text)
    return evidence


def _synthesise(
    system_prompt: str,
    context: SafeSynthesisContext,
    evidence: list[PublicEvidence],
) -> str:
    """External clinical synthesis, from the typed projection only."""
    try:
        return gateway.synthesise_clinical(
            system_prompt=system_prompt,
            context=context,
            evidence=evidence,
            max_tokens=_SYNTHESIS_MAX_TOKENS,
        ).text.strip()
    except Exception as exc:
        logger.error("Clinical synthesis failed: %s", exc)
        return f"Response generation failed: {exc}"


# ---------------------------------------------------------------------------
# `_summarize_report` and `_extract_structured_fields` are GONE, not moved.
#
# Both sent the de-identified report text to an external `CLINICAL_SYNTHESIS`
# model — one to summarise it, one to ask it for JSON. The approved M-1 policy
# names "raw/scrubbed free-text clinical reports" among the payloads no external
# model may receive, so migrating only the FINAL synthesis call would have left
# the whole document leaving the process one call earlier, twice.
#
# The extraction they performed still happens; it happens in-process, in
# `mao.trust.handoff.extract`, where sending the document nowhere is the whole
# point. That extractor is deliberately conservative: it carries what has a
# shape and REPORTS what it could not carry, and the compiler refuses rather
# than letting an answer be built on a minority of a clinical document.
#
# The summary is not replaced. It existed to compress the report for a prompt
# that no longer receives the report — `SafeSynthesisContext` is already the
# minimum-necessary projection, so summarising it would be compressing a
# compression, and the thing lost would be clinical detail.
# ---------------------------------------------------------------------------


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
    """Synthesis for the paths whose payload is a de-identified QUESTION.

    The MODEL role stays `CLINICAL_SYNTHESIS` — that is a statement about which
    model is capable of the reasoning. The EGRESS purpose is
    `GENERAL_SYNTHESIS`, which is a statement about what is being sent, and the
    two are deliberately separate: the payload here is the caller's own question
    after the protected boundary, plus public evidence, and neither is a patient
    document.

    The report path does not come through here. It carries a patient document,
    so it goes through `gateway.synthesise_clinical` with a typed projection,
    and `complete()` refuses `CLINICAL_SYNTHESIS` outright so the two cannot be
    confused by a later edit.
    """
    try:
        return gateway.complete(
            role=ModelRole.CLINICAL_SYNTHESIS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            purpose=EgressPurpose.GENERAL_SYNTHESIS,
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
