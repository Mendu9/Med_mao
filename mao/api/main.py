"""FastAPI entrypoint for MAO: /chat, /chat/stream (SSE), /ingest, /health, /graph endpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import requests as http_requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mao.core.config import cfg
from mao.core.logging_config import configure_logging, set_trace_id
from mao.core.rate_limiter import check_rate_limit
from mao.core.redis_client import get_redis, safe_get, safe_set
from mao.core.state import make_initial_state
from mao.db import get_db_session, init_db
from mao.db.models import ChatSession
from mao.graph import get_graph
from mao.guardrails import apply_input_guardrails, apply_output_guardrails
from mao.monitoring.metrics import (
    active_requests_gauge,
    record_request,
    record_reranker_score,
)

logger = logging.getLogger(__name__)
# configure_logging is called inside lifespan startup (after uvicorn installs its handlers)

# Thread pool for running synchronous LangGraph calls without blocking asyncio.
# Each LangGraph invocation can take 60-180 s; 8 workers allows concurrent requests
# without queuing (which would make subsequent requests appear to hang/timeout).
_executor = ThreadPoolExecutor(max_workers=8)

# Keep strong references to fire-and-forget asyncio tasks to prevent GC cancellation.
_bg_tasks: set[asyncio.Task] = set()


# ---------------------------------------------------------------------------
# Lifespan: replaces deprecated @app.on_event("startup"/"shutdown")
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm graph + models before accepting requests; clean up on shutdown."""
    configure_logging(level=cfg.log_level)  # after uvicorn installs its own handlers
    logger.info("MAO API starting up — downloading/loading models...")
    loop = asyncio.get_running_loop()

    def _warm_all():
        # Initialise DB first (idempotent, raises on failure)
        try:
            init_db()
            logger.info("Database ready.")
        except Exception as exc:
            logger.warning("Database init failed (non-fatal at startup): %s", exc)
        # MRI model (may download 127MB on first run)
        try:
            from mao.models.mri_predictor import get_predictor
            get_predictor()._load()
            logger.info("MRI predictor ready.")
        except Exception as exc:
            logger.warning("MRI predictor warm-up failed (non-fatal): %s", exc)
        # Reranker (~568 MB) — skip warm-up if MAO_DISABLE_RERANKER is set
        import os as _os
        if _os.getenv("MAO_DISABLE_RERANKER", "").lower() not in ("1", "true", "yes"):
            try:
                from mao.rag.reranker import _get_reranker
                _get_reranker()
                logger.info("Reranker ready.")
            except Exception as exc:
                logger.warning("Reranker warm-up failed (non-fatal): %s", exc)
        # Build the LangGraph (fast)
        get_graph()

    # AWAIT so uvicorn holds off accepting requests until done
    await loop.run_in_executor(_executor, _warm_all)
    logger.info("MAO API ready — accepting requests.")

    yield  # application runs here

    # Shutdown
    _executor.shutdown(wait=False)
    logger.info("MAO API shutting down.")


app = FastAPI(
    title="MAO — Multi-Agent Orchestrator",
    version="1.0.0",
    description="LangGraph + GraphRAG + Mem0 + Groq multi-agent system",
    lifespan=lifespan,
)

# Mount Prometheus /metrics endpoint
try:
    from prometheus_client import make_asgi_app as _make_metrics_app
    app.mount("/metrics", _make_metrics_app())
    logger.info("Prometheus /metrics endpoint mounted.")
except ImportError:
    logger.warning("prometheus-client not installed — /metrics endpoint unavailable.")

_CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:7860,http://localhost:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.middleware("http")
async def _trace_id_middleware(request: Request, call_next):
    """Inject trace_id into logging context for every request."""
    trace_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    set_trace_id(trace_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = trace_id
    return response


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
    sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="RAG chunk citations: [{chunk_id, source, doc_id, score}]",
    )
    web_sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Web search citations: [{title, url}]",
    )


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
    groq: str
    vector_store: str
    postgres: str


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
    _client_ip = req.client.host if req.client else None
    if not check_rate_limit(request.user_id, client_ip=_client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in 60 seconds.")

    request_id = uuid.uuid4().hex
    start_time = time.perf_counter()

    # Cache lookup — return immediately for identical recent queries (TTL 5 min)
    _query_cache_key = (
        f"query:{hashlib.md5(f'{request.user_id}:{request.query}'.encode()).hexdigest()}"
    )
    _redis = get_redis()
    _cached = safe_get(_redis, _query_cache_key)
    if _cached:
        try:
            _cached_data = json.loads(_cached)
            logger.info("Cache HIT request_id=%s", request_id)
            return ChatResponse(
                response=_cached_data.get("response", ""),
                agent_used=_cached_data.get("agent_used", "cache"),
                intent=_cached_data.get("intent", ""),
                metadata={},
                request_id=request_id,
                latency_ms=0.0,
                sources=[],
                web_sources=[],
            )
        except Exception as _cache_exc:
            logger.debug("Cache entry malformed, ignoring: %s", _cache_exc)

    logger.info(
        "request_id=%s user_id=%s query=%r",
        request_id,
        request.user_id,
        request.query[:80],
    )

    safe_query = await apply_input_guardrails(request.query, request_id)

    # Build initial state using PII-scrubbed query
    state = make_initial_state(
        user_query=safe_query,
        user_id=request.user_id,
        chat_history=[m.model_dump() for m in request.chat_history],
    )
    # Attach any extra metadata (e.g., image_b64 for multimodal)
    state["metadata"] = request.metadata

    # Run graph in thread pool to keep async loop unblocked
    try:
        loop = asyncio.get_running_loop()
        graph = get_graph()
        result = await loop.run_in_executor(_executor, graph.invoke, state)
    except Exception as exc:
        logger.error("Graph invocation failed request_id=%s: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    result = await apply_output_guardrails(result, request_id)

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

    # Persist chat session (fire-and-forget, never blocks response)
    def _save_session() -> None:
        try:
            with get_db_session() as db:
                db.add(ChatSession(
                    user_id=request.user_id,
                    user_query=safe_query,
                    response=result.get("response", ""),
                    agent_used=agent_used,
                    domain=metadata.get("domain"),
                    uncertainty_flag=bool(metadata.get("uncertainty_flag", False)),
                ))
        except Exception as exc:
            logger.warning("ChatSession persist failed request_id=%s: %s", request_id, exc)

    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, _save_session)

    # Cache session response in Redis for fast repeated lookups (TTL 1h)
    def _cache_session() -> None:
        try:
            redis_client = get_redis()
            _payload = json.dumps({
                "response": result.get("response", ""),
                "agent_used": agent_used,
                "intent": intent,
            })
            # Per-request session cache (TTL 1h)
            safe_set(redis_client, f"session:{request_id}", _payload, ex=3600)
            # Query-hash cache for deduplication (TTL 5 min)
            safe_set(redis_client, _query_cache_key, _payload, ex=300)
        except Exception as exc:
            logger.debug("Session cache write failed: %s", exc)

    loop.run_in_executor(_executor, _cache_session)

    # Fire RAGAS hallucination scoring as a background task (non-blocking)
    contexts = [s.get("snippet", "") for s in metadata.get("sources", []) if s.get("snippet")]
    if contexts:
        _task = asyncio.create_task(
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
        _bg_tasks.add(_task)
        _task.add_done_callback(_bg_tasks.discard)

    return ChatResponse(
        response=result.get("response", ""),
        agent_used=agent_used,
        intent=intent,
        metadata=metadata,
        request_id=request_id,
        latency_ms=round(latency_ms, 1),
        sources=metadata.get("sources", []),
        web_sources=metadata.get("web_sources", []),
    )


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, req: Request) -> StreamingResponse:  # noqa: C901
    """
    SSE streaming chat endpoint — true token-by-token streaming from Groq.

    Architecture:
      1. Input guardrails applied synchronously (PII scrub etc.)
      2. LangGraph graph runs in executor to build state (router, retrieval, guardrails)
         up to the point where the final LLM call is needed.
      3. If the resolved agent supports streaming (graphrag / clinical / summarizer),
         the final answer is generated via chat_stream() with stream=True and tokens
         are yielded directly to the SSE response.
      4. If streaming is not possible (e.g. structured JSON agent like sql/tool),
         we fall back to word-splitting the pre-computed response — still fast since
         the heavy work (retrieval, reranking) was already done before this point.

    SSE format per token: "data: {token}\\n\\n"
    Metadata event:       "data: __meta__:{json}\\n\\n"
    Terminator:           "data: [DONE]\\n\\n"
    """
    _client_ip = req.client.host if req.client else None
    if not check_rate_limit(request.user_id, client_ip=_client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in 60 seconds.")

    request_id = uuid.uuid4().hex
    start_time = time.perf_counter()

    logger.info(
        "stream request_id=%s user_id=%s query=%r",
        request_id,
        request.user_id,
        request.query[:80],
    )

    safe_query = await apply_input_guardrails(request.query, request_id)

    state = make_initial_state(
        user_query=safe_query,
        user_id=request.user_id,
        chat_history=[m.model_dump() for m in request.chat_history],
    )
    state["metadata"] = request.metadata
    # Signal to streaming-capable agents to defer their final LLM call
    state["_want_stream"] = True

    # Run graph in executor — retrieval, graph traversal, reranking happen here.
    # The agents that support streaming (graphrag/clinical/summarizer) store their
    # LLM prompt in state["_stream_messages"] and their model in state["_stream_model"]
    # instead of making the final LLM call themselves, so we can stream it below.
    try:
        loop = asyncio.get_running_loop()
        graph = get_graph()
        result = await loop.run_in_executor(_executor, graph.invoke, state)
    except Exception as exc:
        logger.error("Graph invocation failed request_id=%s: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    result = await apply_output_guardrails(result, request_id)

    latency_ms = (time.perf_counter() - start_time) * 1000
    agent_used = result.get("agent_used", "unknown")

    logger.info(
        "stream request_id=%s agent=%s latency=%.0fms",
        request_id, agent_used, latency_ms,
    )

    # Build SSE metadata payload (sent once after all tokens)
    def _meta_payload(final_latency: float) -> str:
        payload = {
            "intent": result.get("intent", ""),
            "agent_used": agent_used,
            "latency_ms": round(final_latency, 1),
            "sources": result.get("metadata", {}).get("sources", []),
            "web_sources": result.get("metadata", {}).get("web_sources", []),
            "uncertainty_flag": result.get("metadata", {}).get("uncertainty_flag", False),
        }
        return f"data: __meta__:{json.dumps(payload)}\n\n"

    # ----------------------------------------------------------------
    # True streaming path: if the graph stored _stream_messages, call
    # Groq with stream=True and yield each delta token directly.
    # ----------------------------------------------------------------
    stream_messages: list[dict] | None = result.get("_stream_messages")
    stream_model: str | None = result.get("_stream_model")

    def _sse_encode(token: str) -> str:
        """Encode a token as a safe SSE data line.

        JSON-encodes the token so newlines, colons, and other SSE-special
        characters cannot corrupt the frame or be misinterpreted by EventSource.
        Client must JSON.parse each data payload.
        """
        return f"data: {json.dumps(token)}\n\n"

    if stream_messages:
        from mao.core.llm import chat_stream as _chat_stream
        from mao.core.config import FAST_MODEL

        _model = stream_model or FAST_MODEL
        _running_loop = asyncio.get_running_loop()

        async def _true_token_generator():
            # Bridge sync generator to async via queue — yields tokens as they arrive
            # rather than buffering the entire response first.
            queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)

            def _produce():
                try:
                    for tok in _chat_stream(stream_messages, model=_model):
                        if tok:
                            _running_loop.call_soon_threadsafe(queue.put_nowait, tok)
                except Exception as exc:
                    logger.error("Groq stream failed request_id=%s: %s", request_id, exc)
                finally:
                    _running_loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

            _executor.submit(_produce)

            while True:
                tok = await queue.get()
                if tok is None:
                    break
                yield _sse_encode(tok)
                await asyncio.sleep(0)

            final_ms = (time.perf_counter() - start_time) * 1000
            yield _meta_payload(final_ms)
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _true_token_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Request-ID": request_id},
        )

    # ----------------------------------------------------------------
    # Fallback path: word-split the pre-computed response.
    # Used for structured agents (sql, tool, multimodal) whose output
    # is fully computed before streaming begins.
    # ----------------------------------------------------------------
    response_text: str = result.get("response", "")

    async def _word_token_generator():
        for word in response_text.split(" "):
            if word:
                yield _sse_encode(word + " ")
                await asyncio.sleep(0)
        final_ms = (time.perf_counter() - start_time) * 1000
        yield _meta_payload(final_ms)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _word_token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Request-ID": request_id},
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


@app.post("/ingest/knowledge-bases")
async def ingest_knowledge_bases(background_tasks: BackgroundTasks):
    """Trigger download and loading of pre-built biomedical KGs (PrimeKG, HPO, MONDO)."""
    def _run():
        try:
            from mao.data.ingest_knowledge_bases import run as run_kg_ingest
            stats = run_kg_ingest(skip_download=False, ontology_only=False)
            logger.info("KG ingestion complete: %s", stats)
        except Exception as exc:  # noqa: BLE001
            logger.error("KG ingestion failed: %s", exc)
    background_tasks.add_task(_run)
    return {
        "status": "started",
        "message": "Knowledge base ingestion started in background. Check logs for progress.",
    }


@app.post("/ingest/pubmed", response_model=IngestResponse)
async def ingest_pubmed(background_tasks: BackgroundTasks) -> IngestResponse:
    """Trigger PubMed abstract ingestion as a background task."""
    def _run():
        from mao.data.ingest_pubmed import ingest_pubmed_abstracts
        ingest_pubmed_abstracts()
    background_tasks.add_task(_run)
    return IngestResponse(
        status="started",
        articles_ingested=0,
        message="PubMed ingestion started in background. Check logs for progress.",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Check connectivity to all downstream services.
    Returns {"status": "ok"} only if all three pass.
    """
    groq_status          = await _check_groq()
    vector_store_status  = await _check_vector_store()
    postgres_status      = await _check_postgres()

    overall = (
        "ok"
        if all(s == "ok" for s in [groq_status, vector_store_status, postgres_status])
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        groq=groq_status,
        vector_store=vector_store_status,
        postgres=postgres_status,
    )


@app.get("/usage")
async def get_usage() -> dict[str, Any]:
    """
    Return accumulated Groq token usage and estimated cost for this process.

    Response fields:
      total_input_tokens   — int
      total_output_tokens  — int
      total_tokens         — int
      estimated_cost_usd   — float (6 decimal places)
      per_model            — {model: {input_tokens, output_tokens, requests, cost_usd}}
      rate_limit_info      — last observed Groq rate-limit headers (may be {})
    """
    from mao.core.groq_usage import tracker
    return tracker.get_summary()


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

async def _check_groq() -> str:
    try:
        from mao.core import llm as groq_llm
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: groq_llm.chat([{"role": "user", "content": "ping"}], max_tokens=3),
        )
        return "ok"
    except Exception as exc:
        return f"error: {exc}"


async def _check_vector_store() -> str:
    try:
        loop = asyncio.get_running_loop()
        if cfg.vector_backend == "qdrant":
            def _ping():
                from qdrant_client import QdrantClient
                client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
                client.get_collections()
            await loop.run_in_executor(None, _ping)
        else:
            def _ping():  # type: ignore[misc]
                import chromadb
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
        loop = asyncio.get_running_loop()

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
    try:
        async with await asyncpg.connect(os.environ["DATABASE_URL"]) as conn:
            await conn.execute(
                "INSERT INTO response_feedback (session_id, rating, comment, created_at) "
                "VALUES ($1, $2, $3, NOW())",
                req.session_id, 1 if req.thumbs_up else -1, req.comment,
            )
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
    import os
    from fastapi import Response
    from mao.report.report_card import build_report_card
    try:
        async with await asyncpg.connect(os.environ["DATABASE_URL"]) as conn:
            row = await conn.fetchrow(
                "SELECT report_card FROM chat_sessions WHERE session_id = $1", session_id
            )
        if not row or not row["report_card"]:
            raise HTTPException(status_code=404, detail="Report not found")
        card = build_report_card(**json.loads(row["report_card"]))
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
    """Return RAGAS metrics and feedback summary using the existing SQLAlchemy session."""
    try:
        from sqlalchemy import text

        def _query():
            with get_db_session() as db:
                metrics_rows = db.execute(text(
                    "SELECT faithfulness, answer_relevancy, context_precision, context_recall,"
                    " latency_ms, agent_used, created_at"
                    " FROM response_metrics ORDER BY created_at DESC LIMIT 100"
                )).fetchall()
                # rating: 1 = helpful, -1 = not helpful
                feedback_rows = db.execute(text(
                    "SELECT rating, COUNT(*) AS cnt FROM response_feedback GROUP BY rating"
                )).fetchall()
            metrics = [
                {
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                    "latency_ms": r.latency_ms,
                    "agent_used": r.agent_used,
                }
                for r in metrics_rows
            ]
            # Map to True/False keys so the Streamlit tab can display 👍/👎
            feedback: dict[str, int] = {}
            for r in feedback_rows:
                key = "True" if r.rating == 1 else "False"
                feedback[key] = int(r.cnt)
            return {"metrics": metrics, "feedback": feedback}

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, _query)
    except Exception as exc:
        logger.error("Dashboard query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Dashboard unavailable: {exc}") from exc


@app.get("/eval/retrieval")
async def eval_retrieval(k: int = 5, regenerate: bool = False, samples: int = 50):
    """
    Return retrieval evaluation metrics (MRR, P@K, R@K, F1@K).

    Query params:
      k          — Top-K cutoff (default 5)
      regenerate — If true, regenerate golden dataset from ChromaDB first
      samples    — Number of golden samples (only used when regenerate=true)
    """
    def _run():
        from mao.eval.retrieval_metrics import (
            generate_golden_dataset,
            load_golden_dataset,
            run_retrieval_eval,
        )
        from sqlalchemy import text

        if regenerate:
            generate_golden_dataset(n_samples=samples)

        dataset = load_golden_dataset()
        if not dataset:
            return {
                "status": "no_golden_dataset",
                "message": (
                    "No golden dataset found. Call with ?regenerate=true&samples=50 "
                    "to generate one (~2 min, costs ~50 Groq API calls)."
                ),
                "current": None,
                "history": [],
            }

        current = run_retrieval_eval(k=k)

        history = []
        try:
            with get_db_session() as db:
                rows = db.execute(text(
                    "SELECT k, n_samples, mrr, mean_precision_at_k, mean_recall_at_k,"
                    " mean_f1_at_k, created_at"
                    " FROM retrieval_eval_results ORDER BY created_at DESC LIMIT 20"
                )).fetchall()
                history = [
                    {
                        "k": r.k,
                        "n_samples": r.n_samples,
                        "mrr": r.mrr,
                        "mean_precision_at_k": r.mean_precision_at_k,
                        "mean_recall_at_k": r.mean_recall_at_k,
                        "mean_f1_at_k": r.mean_f1_at_k,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    }
                    for r in rows
                ]
        except Exception as exc:
            logger.warning("Could not load retrieval eval history: %s", exc)

        return {
            "status": "ok",
            "golden_dataset_size": len(dataset),
            "current": {kk: vv for kk, vv in current.items() if kk != "details"},
            "details": current.get("details", [])[:20],
            "history": history,
        }

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_executor, _run)
    except Exception as exc:
        logger.error("Retrieval eval failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Direct execution — reads port from MAO_API_PORT env var (default 8080)
# If 8080 is blocked (WinError 10013), set MAO_API_PORT=8081 in .env
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MAO_API_PORT", "8080"))
    uvicorn.run("mao.api.main:app", host="0.0.0.0", port=port, reload=False)
