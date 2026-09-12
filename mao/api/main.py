"""MAO Clinical AI — REST API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from mao.core.config import cfg
from mao.core.logging_config import configure_logging, set_trace_id
from mao.core.rate_limiter import check_rate_limit
from mao.core.deadline import request_deadline
from mao.core.redis_client import get_redis, safe_get, safe_set
from mao.api.cache_key import CacheKeyInputs, build_chat_cache_key
from mao.api.protected_input import (
    payload_too_large,
    protect_chat_request,
    redaction_notice,
)
from mao.api.streaming import may_stream_raw_tokens
from mao.api.tracing import emit_trace
from mao.core.state import initial_state_from_protected
from mao.db import init_db
from mao.db.repository import save_chat_session
from mao.graph import get_graph
from mao.api.executor import get_executor, restart_executor, shutdown_executor
from mao.api.finalize import finalize_response
from mao.api.invocation import run_graph
from mao.core.deident.ambiguity import AmbiguousDocument
from mao.api.routes import ALL_ROUTERS
from mao.api import sse
from mao.guardrails import apply_input_guardrails
from mao.safety.policy import SERVER_LOCATOR_KEYS
from mao.trust.classes import TrustClass
from mao.trust.egress.gateway import RequestProtection, protected_request
from mao.trust.egress.policy import EgressPurpose
from mao.trust.inputs import limits
from mao.monitoring.metrics import (
    record_request,
    record_reranker_score,
)

logger = logging.getLogger(__name__)

# Ceiling on how long one request may hold an executor thread.
#
# A clinical /chat makes seven sequential gateway calls. Each could wait out
# rate limits independently and only the council leg was bounded, so a single
# request could occupy one of eight worker threads for roughly half an hour —
# eight of them take the pool. The rate limiter in front fails OPEN when Redis
# is down, and `user_id` is caller-supplied, so nothing upstream bounds this.
#
# Generous enough that a genuine burst limit is still waited out (the whole
# point of the retry policy), far below the ~28 minutes measured at f757375.
REQUEST_DEADLINE_SECONDS = 180.0
# configure_logging is called inside lifespan startup (after uvicorn installs its handlers)

# The thread pool lives in `mao.api.executor` so the route modules can share it
# without importing this one.
#  held strong references to fire-and-forget asyncio tasks so the GC
# could not cancel them. The only such task was the RAGAS scorer, which is off
# the request path under ADV16-3, so there is nothing left to hold.


# ---------------------------------------------------------------------------
# Lifespan: replaces deprecated @app.on_event("startup"/"shutdown")
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm graph + models before accepting requests; clean up on shutdown."""
    # A previous uvicorn lifecycle may have shut the pool down.
    restart_executor()
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
        # Embedding model — warm so first query doesn't pay load penalty (~5s)
        try:
            from mao.rag.embedder import embed_query as _embed_query
            _embed_query("warm-up")
            logger.info("Embedder ready.")
        except Exception as exc:
            logger.warning("Embedder warm-up failed (non-fatal): %s", exc)
        # Build the LangGraph (fast)
        get_graph()

    # AWAIT so uvicorn holds off accepting requests until done
    await loop.run_in_executor(get_executor(), _warm_all)
    logger.info("MAO API ready — accepting requests.")

    yield  # application runs here

    # Shutdown
    shutdown_executor()
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

    @field_validator("metadata")
    @classmethod
    def _no_server_locators(cls, metadata: dict[str, Any]) -> dict[str, Any]:
        """A caller supplies attachment content, never a server-side locator.

        `report_path` reached `PdfReader(path)` and `audio_path` reached
        whisper, so either one turned `/chat` into an unauthenticated arbitrary
        file read whose text was summarised straight back into the response.
        The `_b64` variants carry every legitimate use.

        Validated on the model rather than in each route handler, so `/chat` and
        `/chat/stream` cannot drift apart, and refused rather than stripped, so
        an operator can tell an attack from a stale client.
        """
        offending = sorted(SERVER_LOCATOR_KEYS.intersection(metadata))
        if offending:
            raise ValueError(
                f"metadata may not name server-side resources: {', '.join(offending)}. "
                "Send the file content as report_b64 or audio_b64 instead."
            )
        return metadata


def _may_cache_response(result: dict[str, Any]) -> bool:
    """Whether a result may be stored and served to a later identical query.

    A withheld answer must not be. `_cache_session` stored `result["response"]`
    unconditionally, so a transient 429 refusal was written under the query key
    for 300 seconds — and the message tells the clinician to "try again in a
    moment", while the retry was served the same cached refusal with
    `served_from_cache=True`. The honesty fix that produced that message is
    correct; caching it undid its only actionable half. A council safety veto
    cached the same way.

    Fails closed. A result whose `output_blocked` flag is missing or not a bool
    has unknown provenance, and unknown provenance is not something to serve
    twice.
    """
    blocked = result.get("output_blocked")
    if blocked is not False:
        return False
    return bool((result.get("response") or "").strip())


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
    redaction_notice: list[str] = Field(
        default_factory=list,
        description=(
            "What the protected input boundary removed from the query or the "
            "history, by line and field. Never the value."
        ),
    )


# ---------------------------------------------------------------------------
# Mounted routers
#
# Ingestion, health, records and evaluation each own their own module. This file
# keeps only the request path — chat and streaming — plus app wiring.
# ---------------------------------------------------------------------------

for _router in ALL_ROUTERS:
    app.include_router(_router)


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

    # Guardrails run BEFORE the cache is consulted. A cache lookup that precedes
    # input validation lets a request that would have been rejected be answered
    # from a previous, accepted one. They no longer de-identify anything: they
    # decide whether the request is processed, and the boundary below decides
    # what it is processed as. See mao/guardrails/input_guardrails.py.
    checked_query = await apply_input_guardrails(request.query, request_id)

    # EVERY channel through ONE boundary (ADV15-8). `chat_history` used to be
    # passed to `make_initial_state` verbatim, and `router`, `graphrag_agent`
    # and `critic_agent` each hand it straight to the provider — so a history
    # turn carrying a name, an MRN, an NHS number and a date of birth delivered
    # all four to a third party, with the de-identifier never called on it.
    try:
        protected = protect_chat_request(
            query=checked_query,
            chat_history=[m.model_dump() for m in request.chat_history],
            metadata=request.metadata,
            trace_id=request_id,
        )
    except limits.InputTooLarge as exc:
        raise payload_too_large(exc, request_id) from exc

    # A-2. The chat posture redacts rather than refusing, and at b63311d it
    # also said nothing - so a clinician whose instrument name was taken with
    # the patient's surname got an answer built without it and no indication
    # that anything had gone. Computed once here and carried by both routes
    # from the same formatter the 422 uses, because `/chat` and `/chat/stream`
    # formatting their own text is how they drifted apart before.
    _redaction_notice = redaction_notice(protected)

    safe_query = protected.query.text
    safe_history = protected.history_as_messages()

    # Bound for the whole request, so the run-scoped egress assertion covers
    # every external call it makes. The sink is several frames below here,
    # inside an executor thread; `run_graph` copies this context into the worker
    # explicitly, which is what makes a ContextVar bound at the route reach it.
    with protected_request(protected.protection):
        # Logged AFTER de-identification, not before. This call used to sit
        # above the line computing `safe_query` and logged `request.query`,
        # putting raw PHI in the application log on every request — an
        # independent sink from the provider and the database, and one the M5
        # decision to "persist the de-identified query only" already ruled out.
        logger.info(
            "request_id=%s user_id=%s query=%r",
            request_id,
            request.user_id,
            safe_query[:80],
        )

        # Cache key folds in every input that can change the answer — history,
        # attachment content, and the model/policy/index versions (P1-4). The
        # history that participates is the DE-IDENTIFIED one, because that is
        # what the graph actually sees; keying on text no consumer reads would
        # make two identical requests miss each other over a difference nothing
        # downstream could act on.
        _query_cache_key = build_chat_cache_key(
            CacheKeyInputs(
                user_id=request.user_id,
                query=safe_query,
                chat_history=safe_history,
                metadata=request.metadata,
            )
        )
        _redis = get_redis()
        _cached = safe_get(_redis, _query_cache_key)
        if _cached:
            try:
                _cached_data = json.loads(_cached)
                logger.info("Cache HIT request_id=%s", request_id)
                # Provenance is part of the answer for an evidence-grounded product.
                # The previous cache-hit branch returned empty sources and metadata,
                # so a cached clinical answer shipped with zero citations (P1-5).
                _cached_meta = _cached_data.get("metadata", {})
                _cached_latency = (time.perf_counter() - start_time) * 1000

                # A cache hit is still a clinical answer delivered to a clinician,
                # so it still has to leave a record. This branch used to `return`
                # above both of these, which made the audit trail a function of
                # cache state rather than of clinical significance — the same gap
                # P1-9 was raised for on the streaming path. `served_from_cache`
                # distinguishes the two in the trace rather than hiding one.
                _cached_result = {
                    **_cached_data,
                    "metadata": {**_cached_meta, "served_from_cache": True},
                }
                emit_trace(
                    trace_id=request_id, result=_cached_result, latency_ms=_cached_latency
                )
                _cache_loop = asyncio.get_running_loop()
                _cache_loop.run_in_executor(
                    get_executor(),
                    _persist_session,
                    safe_query,
                    request.user_id,
                    _cached_result,
                    request_id,
                )

                return ChatResponse(
                    response=_cached_data.get("response", ""),
                    agent_used=_cached_data.get("agent_used", "cache"),
                    intent=_cached_data.get("intent", ""),
                    metadata=_cached_result["metadata"],
                    request_id=request_id,
                    latency_ms=round(_cached_latency, 1),
                    sources=_cached_meta.get("sources", []),
                    web_sources=_cached_meta.get("web_sources", []),
                    # From THIS request's boundary, not from the cached entry.
                    # The notice is a property of what was removed from the
                    # text this caller sent, and a cache hit still removed it.
                    redaction_notice=_redaction_notice,
                )
            except Exception as _cache_exc:
                logger.debug("Cache entry malformed, ignoring: %s", _cache_exc)

        # Built from the boundary's OUTPUT, never from strings. The constructor
        # takes a `ProtectedInput`, so there is no spelling of this line that
        # leaves a channel un-de-identified — which is what ADV15-8 was.
        state = initial_state_from_protected(protected, request.user_id)
        # Attach any extra metadata (e.g., image_b64 for multimodal)
        state["metadata"] = request.metadata

        # Run graph in thread pool to keep async loop unblocked
        try:
            with request_deadline(REQUEST_DEADLINE_SECONDS):
                result = await run_graph(state, request_id)
        except limits.InputTooLarge as exc:
            # Raised where the bytes are read — `_extract_pdf_text` reads the
            # attachment and the extracted text, `handle_audio` the transcript —
            # so it surfaces from inside the graph rather than from the boundary
            # above. Translated here, identically on both routes, because a
            # refusal that reaches one route as a 413 and the other as a 500 is
            # the drift that let `/chat/stream` answer a bare Internal Server
            # Error where `/chat` answered 422.
            raise payload_too_large(exc, request_id) from exc
        except AmbiguousDocument as exc:
            # An uploaded report whose patient header cannot be de-identified
            # confidently is REFUSED, not guessed at. Processing it either leaks a
            # surname to a third party or deletes a contraindication before the
            # model reads it, and the caller never sees which — the document is
            # processed unseen. Ask for the patient fields instead of inferring
            # them. See mao/core/deident/ambiguity.py.
            logger.warning(
                "request_id=%s refusing an ambiguous report header: %s",
                request_id, exc.report.describe(),
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    "The patient header in this report could not be de-identified "
                    "unambiguously, so it was not processed. Send the patient "
                    "identifiers as structured metadata fields instead of relying "
                    "on them being detected in the document text. Ambiguous fields: "
                    + exc.report.describe()
                ),
            ) from exc

        result = await finalize_response(result, request_id)

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

        emit_trace(trace_id=request_id, result=result, latency_ms=latency_ms)

        # Persist chat session (fire-and-forget, never blocks response)
        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            get_executor(), _persist_session, safe_query, request.user_id, result, request_id
        )

        # Cache session response in Redis for fast repeated lookups (TTL 1h)
        def _cache_session() -> None:
            if not _may_cache_response(result):
                logger.info(
                    "request_id=%s not cached (blocked_by=%s)",
                    request_id,
                    result.get("output_blocked_by", "empty_or_unknown"),
                )
                return
            try:
                redis_client = get_redis()
                # Provenance is cached with the answer so a cache hit can return
                # the same citations the live path would have (P1-5).
                _payload = json.dumps({
                    "response": result.get("response", ""),
                    "agent_used": agent_used,
                    "intent": intent,
                    "metadata": metadata,
                })
                # Per-request session cache (TTL 1h)
                safe_set(redis_client, f"session:{request_id}", _payload, ex=3600)
                # Query-hash cache for deduplication (TTL 5 min)
                safe_set(redis_client, _query_cache_key, _payload, ex=300)
            except Exception as exc:
                logger.debug("Session cache write failed: %s", exc)

        loop.run_in_executor(get_executor(), _cache_session)

        # RAGAS scoring was fired here as a background task ON THE REQUEST
        # PATH. `mao/eval/ragas_evaluator.py` builds a `langchain_groq.ChatGroq`,
        # whose `validate_environment` constructs its own `groq.Groq` and
        # `groq.AsyncGroq` - clients the EgressGateway cannot see. Measured at
        # b63311d: 45 SDK create() calls against 8 authorise() calls on one
        # ordinary /chat, every async one unauthorised, carrying the question
        # and the FULL GENERATED CLINICAL ANSWER to a third-party model with no
        # destination, no purpose, no trust class and - the part that matters
        # most - no run-scoped identifier assertion (ADV16-3).
        #
        # The evaluator's own source already said it was eval-only and not on
        # the request path (ADV16-4). That is now true. Scoring belongs to the
        # offline evaluation lane, which reads the trace.
        #
        # Removing it also takes out the last reachable double-scrub site and
        # the 89-138 s per request that both PROJECT_STATE's latency
        # observation and the reviewer's measurement attribute to these jobs.

        return ChatResponse(
            response=result.get("response", ""),
            agent_used=agent_used,
            intent=intent,
            metadata=metadata,
            request_id=request_id,
            latency_ms=round(latency_ms, 1),
            sources=metadata.get("sources", []),
            web_sources=metadata.get("web_sources", []),
            redaction_notice=_redaction_notice,
        )


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, req: Request) -> StreamingResponse:
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

    checked_query = await apply_input_guardrails(request.query, request_id)

    # The SAME boundary as `/chat`, with the same channels and the same
    # refusals. Streaming and non-streaming must carry equivalent privacy
    # contracts (01_ARCHITECTURE.md), and the way these two routes have drifted
    # before is by each spelling out its own input handling.
    try:
        protected = protect_chat_request(
            query=checked_query,
            chat_history=[m.model_dump() for m in request.chat_history],
            metadata=request.metadata,
            trace_id=request_id,
        )
    except limits.InputTooLarge as exc:
        raise payload_too_large(exc, request_id) from exc

    # A-2. The chat posture redacts rather than refusing, and at b63311d it
    # also said nothing - so a clinician whose instrument name was taken with
    # the patient's surname got an answer built without it and no indication
    # that anything had gone. Computed once here and carried by both routes
    # from the same formatter the 422 uses, because `/chat` and `/chat/stream`
    # formatting their own text is how they drifted apart before.
    _redaction_notice = redaction_notice(protected)

    safe_query = protected.query.text

    with protected_request(protected.protection):
        # Logged after de-identification — see the note on the /chat route.
        logger.info(
            "stream request_id=%s user_id=%s query=%r",
            request_id,
            request.user_id,
            safe_query[:80],
        )

        state = initial_state_from_protected(protected, request.user_id)
        state["metadata"] = request.metadata
        # Signal to streaming-capable agents to defer their final LLM call
        state["_want_stream"] = True

        # Run graph in executor — retrieval, graph traversal, reranking happen here.
        # The agents that support streaming (graphrag/clinical/summarizer) store their
        # LLM prompt in state["_stream_messages"] and their model in state["_stream_model"]
        # instead of making the final LLM call themselves, so we can stream it below.
        try:
            with request_deadline(REQUEST_DEADLINE_SECONDS):
                result = await run_graph(state, request_id)
        except limits.InputTooLarge as exc:
            # The SAME 413 as `/chat`, from the same constructor. See the note
            # on the non-streaming route.
            raise payload_too_large(exc, request_id) from exc
        except AmbiguousDocument as exc:
            # The SAME refusal as `/chat`. Without this the stream route answered a
            # bare `Internal Server Error` — no 422, no structured-fields guidance,
            # and not even the correlation id, so it was strictly worse than the
            # generic handler it bypassed. Two routes reaching the same graph must
            # not disagree about what a refusal means.
            logger.warning(
                "stream request_id=%s refusing an ambiguous report header: %s",
                request_id, exc.report.describe(),
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    "The patient header in this report could not be de-identified "
                    "unambiguously, so it was not processed. Send the patient "
                    "identifiers as structured metadata fields instead of relying "
                    "on them being detected in the document text. Ambiguous fields: "
                    + exc.report.describe()
                ),
            ) from exc

        result = await finalize_response(result, request_id)

        latency_ms = (time.perf_counter() - start_time) * 1000
        agent_used = result.get("agent_used", "unknown")

        logger.info(
            "stream request_id=%s agent=%s latency=%.0fms",
            request_id, agent_used, latency_ms,
        )

        # The streaming path used to record nothing: no metrics, no audit row, no
        # trace — so observability and the clinical audit trail were systematically
        # missing for the primary user-facing path (P1-9, P1-22).
        record_request(
            agent=agent_used,
            intent=result.get("intent", "unknown"),
            latency_seconds=latency_ms / 1000,
        )
        emit_trace(trace_id=request_id, result=result, latency_ms=latency_ms)
        _loop = asyncio.get_running_loop()
        _loop.run_in_executor(
            get_executor(), _persist_session, safe_query, request.user_id, result, request_id
        )

        # Onto the metadata frame the stream already emits, so the streamed
        # route carries exactly what the buffered one does.
        result.setdefault("metadata", {})["redaction_notice"] = _redaction_notice

        def _trailer() -> str:
            return sse.meta_payload(result, agent_used, sse.elapsed_ms(start_time))

        # ----------------------------------------------------------------
        # Raw token streaming — permitted ONLY where the safety policy exempts the
        # request from verification. Raw tokens come straight from the provider and
        # have passed through neither the council, the NLI gate, the judge, nor the
        # output guardrails; streaming them for a route that requires verification
        # is exactly the audited P0-1 bypass. `may_stream_raw_tokens` fails closed.
        # ----------------------------------------------------------------
        stream_messages = result.get("_stream_messages") if may_stream_raw_tokens(result) else None
        headers = {"Cache-Control": "no-cache", "X-Request-ID": request_id}

        if stream_messages:
            # Through the gateway on the synthesis role, not `mao.core.llm` with a
            # `FAST_MODEL` config alias. That import was a non-agent gateway bypass
            # on the request path: it skipped role resolution, so a retired id could
            # reach the provider, and it skipped the reasoning-overhead budgeting
            # every other call site now gets. The call itself is made by
            # `_protected_token_stream`, which re-binds this request's
            # protection around it.
            from mao.providers.registry import ModelRole

            # The role the agent deferred with, not a role chosen here. Hardcoding
            # one would silently stream on a different capability than the request
            # was routed to — the same class of drift as naming a literal model id.
            _stream_role = ModelRole(result.get("_stream_role") or ModelRole.GENERAL_SYNTHESIS.value)

            # ...and likewise the PURPOSE. The egress policy is the same on both
            # transports: a purpose that may not send free text when the answer is
            # buffered may not send it when the answer is streamed. Deferring the
            # purpose alongside the role is what stops the streaming route
            # authorising something the non-streaming route would refuse — the
            # class of drift that let `/chat/stream` answer 500 where `/chat`
            # answered 422.
            _stream_purpose = EgressPurpose(
                result.get("_stream_purpose") or EgressPurpose.GENERAL_SYNTHESIS.value
            )

            return StreamingResponse(
                sse.raw_token_stream(
                    produce_tokens=lambda: _protected_token_stream(
                        protected.protection,
                        role=_stream_role,
                        messages=stream_messages,
                        purpose=_stream_purpose,
                        # Stated, not defaulted. The streamed path carries the
                        # same de-identified text the buffered one does, and
                        # transport is not a safety input.
                        trust_class=TrustClass.SAFE_DERIVED_TEXT,
                        temperature=0.1,
                        max_tokens=768,
                    ),
                    submit=get_executor().submit,
                    trailer=_trailer,
                    request_id=request_id,
                ),
                media_type="text/event-stream",
                headers=headers,
            )

        # The default path: stream the text that came out of the verification chain
        # and the output guardrails. Slower to first token, deliberately.
        return StreamingResponse(
            sse.chunked_text_stream(text=result.get("response", ""), trailer=_trailer),
            media_type="text/event-stream",
            headers=headers,
        )


def _protected_token_stream(protection: RequestProtection, **call: Any):
    """Re-bind this request's protection around the streamed provider call.

    The `StreamingResponse` body is iterated AFTER the route function returns,
    and `raw_token_stream` runs its producer on an executor thread — neither
    carries the ContextVar the route bound. Without this the one external call
    the streaming route makes would be the one call the run-scoped egress
    assertion does not cover, which is exactly the `/chat` versus `/chat/stream`
    asymmetry this phase has had to close twice already.
    """
    from mao.providers import gateway

    with protected_request(protection):
        yield from gateway.stream(**call)


def _persist_session(
    safe_query: str,
    user_id: str,
    result: dict[str, Any],
    request_id: str,
) -> None:
    """Write one turn's full clinical audit trail. Never raises.

    Used by BOTH /chat and /chat/stream. The streaming path previously wrote no
    ChatSession row at all, so streamed clinical answers — the majority of real
    traffic — left no audit record whatsoever (P1-9).

    Takes only the de-identified query. The raw one is deliberately not a
    parameter, so raw PHI is not carried into a background task, a thread pool,
    or an exception traceback on its way to a database that has no encryption or
    retention controls attached to it.
    """
    metadata = result.get("metadata") or {}
    try:
        save_chat_session(
            user_id=user_id,
            pii_scrubbed_query=safe_query,
            response=result.get("response", ""),
            agent_used=result.get("agent_used", "unknown"),
            domain=metadata.get("domain"),
            uncertainty_flag=bool(metadata.get("uncertainty_flag", False)),
            council_verdict=result.get("council_verdict"),
            nli_flags=result.get("nli_flags"),
            report_card=result.get("report_card"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ChatSession persist failed request_id=%s: %s", request_id, exc)




# ---------------------------------------------------------------------------
# Direct execution — reads port from MAO_API_PORT env var (default 8080)
# If 8080 is blocked (WinError 10013), set MAO_API_PORT=8081 in .env
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MAO_API_PORT", "8080"))
    uvicorn.run("mao.api.main:app", host="0.0.0.0", port=port, reload=False)
