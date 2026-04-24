"""
mao/api/main.py
---------------
FastAPI entrypoint for the MAO system.

Endpoints:
  POST /chat        — Main conversational endpoint
  POST /ingest      — Trigger Wikipedia ingestion into ChromaDB
  GET  /health      — Liveness check (Ollama + ChromaDB + Postgres)
  GET  /graph       — Return graph topology as Mermaid string

Design:
  - Async endpoints throughout
  - Pydantic models for all request/response schemas
  - Structured logging on every request
  - The LangGraph graph is a module-level singleton (built once at startup)
  - Heavy LangGraph/LLM calls run in a thread executor to avoid blocking
    the event loop (LangGraph's .invoke() is synchronous)

Usage:
  uvicorn mao.api.main:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests as http_requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mao.core.config import cfg
from mao.core.state import make_initial_state
from mao.graph import get_graph
from mao.monitoring.metrics import (
    active_requests_gauge,
    record_request,
    record_reranker_score,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

app = FastAPI(
    title="MAO — Multi-Agent Orchestrator",
    version="1.0.0",
    description="LangGraph + GraphRAG + Mem0 + Ollama multi-agent system",
)

# Mount Prometheus /metrics endpoint
try:
    from prometheus_client import make_asgi_app as _make_metrics_app
    app.mount("/metrics", _make_metrics_app())
    logger.info("Prometheus /metrics endpoint mounted.")
except ImportError:
    logger.warning("prometheus-client not installed — /metrics endpoint unavailable.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for running synchronous LangGraph calls without blocking asyncio
_executor = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    user_id: str = Field(default_factory=lambda: f"anon-{uuid.uuid4().hex[:8]}")
    chat_history: list[ChatMessage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    response: str
    agent_used: str
    intent: str
    metadata: dict[str, Any]
    request_id: str
    latency_ms: float


class IngestRequest(BaseModel):
    topics: list[str] = Field(
        default=["Machine learning", "Natural language processing", "Knowledge graph"],
        description="Wikipedia article titles to ingest",
    )
    max_articles: int = Field(default=10, ge=1, le=100)


class IngestResponse(BaseModel):
    status: str
    articles_ingested: int
    message: str


class HealthResponse(BaseModel):
    status: str
    ollama: str
    chromadb: str
    postgres: str


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    """Pre-warm graph + MRI model before accepting any requests."""
    logger.info("MAO API starting up — downloading/loading models...")
    loop = asyncio.get_event_loop()

    def _warm_all():
        # MRI model first (may download 127MB on first run)
        try:
            from mao.models.mri_predictor import get_predictor
            get_predictor()._load()  # force download + load now
            logger.info("MRI predictor ready.")
        except Exception as exc:
            logger.warning("MRI predictor warm-up failed (non-fatal): %s", exc)
        # Then build the LangGraph (fast)
        get_graph()

    # AWAIT so uvicorn holds off accepting requests until done
    await loop.run_in_executor(_executor, _warm_all)
    logger.info("MAO API ready — accepting requests.")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    _executor.shutdown(wait=False)
    logger.info("MAO API shutting down.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request) -> ChatResponse:
    """
    Main chat endpoint.

    Constructs a MAOState, invokes the LangGraph graph in a thread pool
    (because LangGraph .invoke() is synchronous), and returns the result.
    """
    request_id = uuid.uuid4().hex
    start_time = time.perf_counter()

    logger.info(
        "request_id=%s user_id=%s query=%r",
        request_id,
        request.user_id,
        request.query[:80],
    )

    # Build initial state
    state = make_initial_state(
        user_query=request.query,
        user_id=request.user_id,
        chat_history=[m.model_dump() for m in request.chat_history],
    )
    # Attach any extra metadata (e.g., image_b64 for multimodal)
    state["metadata"] = request.metadata

    # Run graph in thread pool to keep async loop unblocked
    try:
        loop = asyncio.get_event_loop()
        graph = get_graph()
        result = await loop.run_in_executor(_executor, graph.invoke, state)
    except Exception as exc:
        logger.error("Graph invocation failed request_id=%s: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    latency_ms = (time.perf_counter() - start_time) * 1000
    agent_used = result.get("agent_used", "unknown")
    intent     = result.get("intent", "unknown")
    metadata   = result.get("metadata", {})

    logger.info(
        "request_id=%s agent=%s intent=%s latency=%.0fms",
        request_id, agent_used, intent, latency_ms,
    )

    # Record Prometheus metrics
    record_request(agent=agent_used, intent=intent, latency_seconds=latency_ms / 1000)
    top_scores = metadata.get("top_scores", [])
    if top_scores:
        record_reranker_score(agent=agent_used, top_score=top_scores[0])

    # Fire RAGAS hallucination scoring as a background task (non-blocking)
    contexts = [s.get("snippet", "") for s in metadata.get("sources", []) if s.get("snippet")]
    if contexts:
        asyncio.create_task(
            _score_response_async(
                question=request.query,
                answer=result.get("response", ""),
                contexts=contexts,
                user_id=request.user_id,
                request_id=request_id,
                agent_used=agent_used,
                latency_ms=latency_ms,
            )
        )

    return ChatResponse(
        response=result.get("response", ""),
        agent_used=agent_used,
        intent=intent,
        metadata=metadata,
        request_id=request_id,
        latency_ms=round(latency_ms, 1),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    """
    Trigger Wikipedia ingestion in the background.

    Returns immediately; ingestion runs asynchronously.
    Check /health for ChromaDB document count to monitor progress.
    """
    background_tasks.add_task(
        _run_ingestion,
        topics=request.topics,
        max_articles=request.max_articles,
    )
    return IngestResponse(
        status="started",
        articles_ingested=0,
        message=f"Ingestion started for {len(request.topics)} topics in background.",
    )


class AlzheimersIngestRequest(BaseModel):
    data_dir: str | None = Field(
        default=None,
        description="Path to PDF directory. Defaults to ad/rag/data/",
    )
    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=50, ge=0, le=256)


@app.post("/ingest/alzheimers", response_model=IngestResponse)
async def ingest_alzheimers(
    request: AlzheimersIngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """
    Trigger ingestion of Alzheimer's research PDFs into ChromaDB.

    PDFs are loaded from ad/rag/data/ (or a custom path), chunked,
    embedded with nomic-embed-text, and stored in ChromaDB alongside
    the Wikipedia knowledge base. No router changes needed — graphrag_agent
    will automatically retrieve from these documents after ingestion.
    """
    background_tasks.add_task(
        _run_alzheimers_ingestion,
        data_dir=request.data_dir,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    return IngestResponse(
        status="started",
        articles_ingested=0,
        message="Alzheimer's PDF ingestion started in background. Check logs for progress.",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Check connectivity to all downstream services.
    Returns {"status": "ok"} only if all three pass.
    """
    ollama_status   = await _check_ollama()
    chromadb_status = await _check_chromadb()
    postgres_status = await _check_postgres()

    overall = (
        "ok"
        if all(s == "ok" for s in [ollama_status, chromadb_status, postgres_status])
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        ollama=ollama_status,
        chromadb=chromadb_status,
        postgres=postgres_status,
    )


@app.get("/graph")
async def graph_topology() -> dict[str, Any]:
    """Return graph topology for debugging/documentation."""
    return {
        "nodes": [
            "router_node",
            "summarizer_node",
            "graphrag_node",
            "tool_node",
            "sql_node",
            "multimodal_node",
            "code_node",
            "critic_node",
            "clinical_node",
        ],
        "entry": "router_node",
        "routing": "conditional on state.intent",
        "intents": [
            "summarize", "graphrag", "tool", "sql",
            "multimodal", "code", "critic", "clinical", "fallback",
        ],
    }


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

async def _score_response_async(
    question: str,
    answer: str,
    contexts: list[str],
    user_id: str,
    request_id: str,
    agent_used: str,
    latency_ms: float,
) -> None:
    """Fire-and-forget RAGAS scoring — never raises, never blocks the caller."""
    try:
        from mao.eval.ragas_evaluator import score_response
        await score_response(
            question=question,
            answer=answer,
            contexts=contexts,
            user_id=user_id,
            request_id=request_id,
            agent_used=agent_used,
            latency_ms=latency_ms,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Background RAGAS scoring failed: %s", exc)


def _run_ingestion(topics: list[str], max_articles: int) -> None:
    """Synchronous ingestion wrapper — runs in BackgroundTasks thread."""
    try:
        from mao.data.ingest_wikipedia import ingest_wikipedia_topics
        count = ingest_wikipedia_topics(topics, max_articles=max_articles)
        logger.info("Ingestion complete: %d articles", count)
    except Exception as exc:  # noqa: BLE001
        logger.error("Background ingestion failed: %s", exc)


def _run_alzheimers_ingestion(
    data_dir: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Synchronous Alzheimer's PDF ingestion — runs in BackgroundTasks thread."""
    try:
        from mao.data.ingest_alzheimers import ingest_alzheimers_pdfs
        count = ingest_alzheimers_pdfs(
            data_dir=data_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.info("Alzheimer's ingestion complete: %d PDFs", count)
    except Exception as exc:  # noqa: BLE001
        logger.error("Alzheimer's ingestion failed: %s", exc)


# ---------------------------------------------------------------------------
# Health check helpers
# ---------------------------------------------------------------------------

async def _check_ollama() -> str:
    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: http_requests.get(f"{cfg.ollama_base_url}/api/tags", timeout=5),
        )
        return "ok" if resp.status_code == 200 else f"http_{resp.status_code}"
    except Exception as exc:
        return f"error: {exc}"


async def _check_chromadb() -> str:
    try:
        import chromadb
        loop = asyncio.get_event_loop()

        def _ping():
            client = chromadb.HttpClient(host=cfg.chroma_host, port=cfg.chroma_port)
            client.heartbeat()

        await loop.run_in_executor(None, _ping)
        return "ok"
    except Exception as exc:
        return f"error: {exc}"


async def _check_postgres() -> str:
    try:
        from mao.agents.sql_agent import _get_engine
        from sqlalchemy import text
        loop = asyncio.get_event_loop()

        def _ping():
            engine = _get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

        await loop.run_in_executor(None, _ping)
        return "ok"
    except Exception as exc:
        return f"error: {exc}"


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    session_id: str
    thumbs_up: bool
    comment: str = ""


@app.post("/feedback")
async def post_feedback(req: FeedbackRequest):
    import asyncpg
    import os
    try:
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        await conn.execute(
            "INSERT INTO response_feedback (session_id, thumbs_up, comment, created_at) "
            "VALUES ($1, $2, $3, NOW())",
            req.session_id, req.thumbs_up, req.comment,
        )
        await conn.close()
        return {"status": "ok"}
    except Exception as e:
        logger.error("Feedback insert failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save feedback")


# ---------------------------------------------------------------------------
# PDF Export
# ---------------------------------------------------------------------------

@app.get("/export/report/{session_id}")
async def export_report(session_id: str):
    import asyncpg
    import json as _json
    import os
    from fastapi import Response
    from mao.report.report_card import build_report_card
    try:
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        row = await conn.fetchrow(
            "SELECT report_card FROM chat_sessions WHERE session_id = $1", session_id
        )
        await conn.close()
        if not row or not row["report_card"]:
            raise HTTPException(status_code=404, detail="Report not found")
        card = build_report_card(**_json.loads(row["report_card"]))
        pdf_bytes = card.to_pdf()
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=report_{session_id}.pdf"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("PDF export failed: %s", e)
        raise HTTPException(status_code=500, detail="PDF export failed")


# ---------------------------------------------------------------------------
# Eval Dashboard
# ---------------------------------------------------------------------------

@app.get("/eval/dashboard")
async def eval_dashboard():
    import asyncpg
    import os
    try:
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        rows = await conn.fetch(
            "SELECT faithfulness, answer_relevancy, context_precision, context_recall, "
            "created_at FROM response_metrics ORDER BY created_at DESC LIMIT 100"
        )
        feedback_rows = await conn.fetch(
            "SELECT thumbs_up, COUNT(*) as cnt FROM response_feedback GROUP BY thumbs_up"
        )
        await conn.close()
        metrics = [dict(r) for r in rows]
        feedback = {str(r["thumbs_up"]): r["cnt"] for r in feedback_rows}
        return {"metrics": metrics, "feedback": feedback}
    except Exception as e:
        logger.error("Dashboard query failed: %s", e)
        raise HTTPException(status_code=500, detail="Dashboard unavailable")
