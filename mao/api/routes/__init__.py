"""Route modules, one per responsibility.

`mao/api/main.py` had accumulated about twelve of these — chat, streaming, four
ingestion endpoints, three health probes, usage, topology, feedback, export and
two eval endpoints — in a single file that grew during the very phase meant to
make the system coherent. Ingestion orchestration and evaluation do not belong
under `api/` at all; splitting them out is the first step to moving them.

`main` keeps the request path (chat and streaming) and the app wiring.
"""
from __future__ import annotations

from mao.api.routes.evaluation import router as evaluation_router
from mao.api.routes.health import router as health_router
from mao.api.routes.ingestion import router as ingestion_router
from mao.api.routes.records import router as records_router

ALL_ROUTERS = (
    ingestion_router,
    health_router,
    records_router,
    evaluation_router,
)

__all__ = [
    "ALL_ROUTERS",
    "evaluation_router",
    "health_router",
    "ingestion_router",
    "records_router",
]
