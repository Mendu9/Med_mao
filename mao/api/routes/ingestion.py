"""Corpus ingestion endpoints.

Ingestion is not really an API concern — it is a data-plane job the API happens
to be able to trigger. Keeping it in its own module is the first step to moving
it behind a proper job runner; until then it at least stops inflating the
request-path module.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingestion"])


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
