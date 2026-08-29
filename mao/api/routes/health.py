"""Liveness, usage accounting, and graph topology.

Each downstream probe is independent and returns a string rather than raising,
so one dead dependency reports as degraded instead of taking the whole endpoint
down with it.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from mao.core.config import cfg

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    groq: str
    vector_store: str
    postgres: str
    # Redis was a real dependency with no probe: it backs the query cache and the
    # rate limiter, and the health panel had a Redis tile that could only ever
    # render "unknown" because nothing reported on it.
    redis: str


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Check connectivity to all downstream services.

    Reports "ok" only when every probe passes; any failure degrades the whole.
    """
    groq_status, vector_store_status, postgres_status, redis_status = await asyncio.gather(
        _check_groq(),
        _check_vector_store(),
        _check_postgres(),
        _check_redis(),
    )

    overall = (
        "ok"
        if all(
            s == "ok"
            for s in (groq_status, vector_store_status, postgres_status, redis_status)
        )
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        groq=groq_status,
        vector_store=vector_store_status,
        postgres=postgres_status,
        redis=redis_status,
    )


@router.get("/usage")
async def get_usage() -> dict[str, Any]:
    """Accumulated token usage and estimated cost for this process.

    Fields: total_input_tokens, total_output_tokens, total_tokens,
    estimated_cost_usd, per_model, rate_limit_info.
    """
    from mao.core.groq_usage import tracker

    return tracker.get_summary()


@router.get("/graph")
async def graph_topology_endpoint() -> dict[str, Any]:
    """The topology of the graph that actually runs.

    Derived, never hand-maintained: the previous literal advertised `code_node`
    and `sql_node` long after both were deleted, omitted eight real nodes, and
    named the wrong entry point — and the UI consumes this (P1-6).
    """
    from mao.api.topology import graph_topology

    return graph_topology()


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

async def _check_groq() -> str:
    """Probe the provider through the gateway, on the role a router call uses.

    Going through the gateway rather than the raw SDK means the probe fails for
    the same reasons production does — including a role bound to a model the
    provider no longer serves.
    """
    try:
        from mao.providers import gateway
        from mao.providers.registry import ModelRole

        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(
            None,
            lambda: gateway.complete(
                role=ModelRole.ROUTER_FAST,
                messages=[{"role": "user", "content": "Reply with: ok"}],
                temperature=0.0,
                max_tokens=64,
            ),
        )
        if not completion.text.strip():
            return f"error: {completion.model_id} returned an empty completion"
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


async def _check_vector_store() -> str:
    try:
        loop = asyncio.get_running_loop()
        if cfg.vector_backend == "qdrant":
            def _ping() -> None:
                from qdrant_client import QdrantClient

                client = QdrantClient(
                    url=cfg.qdrant_url, api_key=cfg.qdrant_api_key, prefer_grpc=False
                )
                client.get_collections()
        else:
            def _ping() -> None:
                import chromadb

                client = chromadb.HttpClient(host=cfg.chroma_host, port=cfg.chroma_port)
                client.heartbeat()

        await loop.run_in_executor(None, _ping)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


async def _check_redis() -> str:
    """Probe Redis through the shared client, which fails soft by design.

    `get_redis()` returns None when Redis is unreachable or unconfigured, so
    that a cache outage degrades performance rather than the request path. That
    is the right runtime behaviour and exactly why it needs a probe: without one
    the system runs cacheless indefinitely with nothing saying so.
    """
    try:
        from mao.core.redis_client import get_redis

        loop = asyncio.get_running_loop()

        def _ping() -> str:
            client = get_redis()
            if client is None:
                return "error: redis unavailable"
            client.ping()
            return "ok"

        return await loop.run_in_executor(None, _ping)
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


async def _check_postgres() -> str:
    """Probe Postgres over the shared session factory.

    This used to reach into `mao.agents.sql_agent` for a second, separately
    configured engine. That module was removed with the SQL route (P0-3), and
    the second engine was part of the three-config-source problem (P1-10).
    """
    try:
        from sqlalchemy import text

        from mao.db import get_db_session, init_db, is_initialised

        loop = asyncio.get_running_loop()

        def _ping() -> None:
            if not is_initialised():
                init_db()
            with get_db_session() as db:
                db.execute(text("SELECT 1"))

        await loop.run_in_executor(None, _ping)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"
