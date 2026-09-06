"""Corpus ingestion endpoints.

Ingestion is not really an API concern — it is a data-plane job the API happens
to be able to trigger. Keeping it in its own module is the first step to moving
it behind a proper job runner; until then it at least stops inflating the
request-path module.
"""
from __future__ import annotations

import hmac
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from mao.core.config import INGEST_API_KEY_ENV, ingest_api_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ADV15-11 — `/ingest` was an unauthenticated write into the evidence corpus.
#
# It took caller-supplied Wikipedia article titles and a `max_articles` up to
# 100 with no authentication, no dependency and no key. Those articles are
# fetched, chunked, embedded and then served back through `/chat` as retrieved
# evidence — which is the council's and the judge's context, the NLI gate's
# premise, and the citation list the clinician reads. So the surface that
# decides what the safety chain believes was writable by anyone, which is the
# reachability half of ADV15-10.
#
# This module already reasoned its way to the same conclusion once, about
# `data_dir`: "a corpus-poisoning primitive", and "an allowlist is a thing that
# can be got wrong". The identical primitive stayed open through `topics`.
#
# The control hangs off the ROUTER, not off the one blocking route. `/ingest` is
# the poisoning primitive today; the next endpoint added to this module would
# otherwise be born unprotected, which is exactly how `topics` outlived the
# `data_dir` fix. The other three routes take no caller-controlled content, so
# authorizing them costs nothing and removes the judgement call.
#
# Fail closed: with no key configured the write surface refuses rather than
# opens. A control that allows when unconfigured is not a boundary, it is a
# default. Full identity and RBAC remain Phase 6; this is the "narrow explicit
# authorization control" the Phase 1 minimum asks for.
# ---------------------------------------------------------------------------

INGEST_KEY_HEADER = "X-MAO-Ingest-Key"


def require_ingest_key(
    presented: str | None = Header(default=None, alias=INGEST_KEY_HEADER),
) -> None:
    """Authorize one write into the evidence corpus, or refuse it."""
    configured = ingest_api_key()
    if not configured:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Corpus ingestion is disabled: no {INGEST_API_KEY_ENV} is "
                "configured, so no caller can be authorized to write evidence."
            ),
        )
    # compare_digest on bytes rather than str: a header can legitimately carry
    # non-ASCII, and the str form raises TypeError on it — a 500, and an
    # exception trace, where a 401 belongs.
    if presented is None or not hmac.compare_digest(
        presented.encode("utf-8"), configured.encode("utf-8")
    ):
        logger.warning("Rejected an unauthorized corpus ingestion request")
        raise HTTPException(
            status_code=401, detail="Missing or invalid corpus ingestion key."
        )


router = APIRouter(tags=["ingestion"], dependencies=[Depends(require_ingest_key)])


class IngestRequest(BaseModel):
    """Wikipedia titles to fetch into the evidence corpus.

    `topics` is still caller-supplied, because naming an article is the whole
    point of the endpoint. What changed is who may call it — see
    `require_ingest_key`. ADV15-11 was never that the parameter existed; it was
    that anyone on the internet could use it to choose what the safety council
    and the judge would later read as evidence.
    """

    topics: list[str] = Field(
        default=["Machine learning", "Natural language processing", "Knowledge graph"],
        description="Wikipedia article titles to ingest",
    )
    max_articles: int = Field(default=10, ge=1, le=100)


class IngestResponse(BaseModel):
    status: str
    articles_ingested: int
    message: str


class AlzheimersIngestRequest(BaseModel):
    """Chunking parameters only.

    This model used to carry `data_dir`, a caller-supplied string that the
    ingester used verbatim as a `Path` — an unauthenticated arbitrary
    server-side file read, and, because the text is chunked, embedded and then
    retrievable through `/chat` as evidence, a corpus-poisoning primitive too.
    `00_RULES` forbids exposing unrestricted filesystem access.

    The corpus directory is a property of the deployment, not of the request, so
    the field is gone rather than validated: an allowlist is a thing that can be
    got wrong, and nothing legitimate needed the parameter. `extra="forbid"`
    turns a stale or hostile caller still sending it into a visible 422 rather
    than a silent discard.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=50, ge=0, le=256)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    """
    Trigger Wikipedia ingestion in the background.

    Returns immediately; ingestion runs asynchronously.
    Check /health for vector store status to monitor progress.
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


@router.post("/ingest/alzheimers", response_model=IngestResponse)
async def ingest_alzheimers(
    request: AlzheimersIngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """
    Trigger ingestion of Alzheimer's research PDFs into the vector store.

    PDFs are loaded from the configured corpus directory, chunked, embedded, and
    stored alongside the Wikipedia knowledge base. No router changes needed —
    graphrag_agent retrieves from these documents automatically after ingestion.

    The directory is not a request parameter. See `AlzheimersIngestRequest`.
    """
    background_tasks.add_task(
        _run_alzheimers_ingestion,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    return IngestResponse(
        status="started",
        articles_ingested=0,
        message="Alzheimer's PDF ingestion started in background. Check logs for progress.",
    )


@router.post("/ingest/knowledge-bases")
async def ingest_knowledge_bases(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Trigger download and loading of pre-built biomedical KGs (PrimeKG, HPO, MONDO)."""
    background_tasks.add_task(_run_knowledge_base_ingestion)
    return {
        "status": "started",
        "message": "Knowledge base ingestion started in background. Check logs for progress.",
    }


@router.post("/ingest/pubmed", response_model=IngestResponse)
async def ingest_pubmed(background_tasks: BackgroundTasks) -> IngestResponse:
    """Trigger PubMed abstract ingestion as a background task."""
    background_tasks.add_task(_run_pubmed_ingestion)
    return IngestResponse(
        status="started",
        articles_ingested=0,
        message="PubMed ingestion started in background. Check logs for progress.",
    )


# ---------------------------------------------------------------------------
# Background runners — synchronous, executed in the BackgroundTasks thread.
# Every one swallows its exception: a failed ingestion must be visible in the
# log, not raised into a request that has already been answered.
# ---------------------------------------------------------------------------

def _run_ingestion(topics: list[str], max_articles: int) -> None:
    try:
        from mao.data.ingest_wikipedia import ingest_wikipedia_topics

        count = ingest_wikipedia_topics(topics, max_articles=max_articles)
        logger.info("Ingestion complete: %d articles", count)
    except Exception as exc:  # noqa: BLE001
        logger.error("Background ingestion failed: %s", exc)


def _run_alzheimers_ingestion(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Takes no directory, so none can be reintroduced without also changing
    the request model — the two places B6 had to be fixed in."""
    try:
        from mao.data.ingest_alzheimers import ingest_alzheimers_pdfs

        count = ingest_alzheimers_pdfs(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.info("Alzheimer's ingestion complete: %d PDFs", count)
    except Exception as exc:  # noqa: BLE001
        logger.error("Alzheimer's ingestion failed: %s", exc)


def _run_knowledge_base_ingestion() -> None:
    try:
        from mao.data.ingest_knowledge_bases import run as run_kg_ingest

        stats = run_kg_ingest(skip_download=False, ontology_only=False)
        logger.info("KG ingestion complete: %s", stats)
    except Exception as exc:  # noqa: BLE001
        logger.error("KG ingestion failed: %s", exc)


def _run_pubmed_ingestion() -> None:
    try:
        from mao.data.ingest_pubmed import ingest_pubmed_abstracts

        ingest_pubmed_abstracts()
        logger.info("PubMed ingestion complete")
    except Exception as exc:  # noqa: BLE001
        logger.error("PubMed ingestion failed: %s", exc)
