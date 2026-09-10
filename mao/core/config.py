"""Central configuration loaded from environment variables.

Model selection is NOT owned here. `mao.providers.registry` binds capability
roles to concrete models; the module-level aliases at the bottom of this file
are thin, backwards-compatible views onto those bindings. Safety thresholds are
likewise owned by `mao.safety.policy`.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

from mao.providers.gateway import model_id_for
from mao.providers.registry import ModelRole
from mao.safety.policy import get_policy

load_dotenv(Path(__file__).parent.parent.parent / ".env")


@dataclass(frozen=True)
class MAOConfig:
    # Provider credential. Model *ids* come from the ModelRegistry, not from here.
    groq_api_key: str   = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str     = field(default_factory=lambda: model_id_for(ModelRole.GENERAL_SYNTHESIS))
    groq_judge_model: str = field(default_factory=lambda: model_id_for(ModelRole.SAFETY_JUDGE))

    # Embeddings (sentence-transformers, local, no API key)
    # NeuML/pubmedbert-base-embeddings: 768-dim, trained on PubMed, far better biomedical recall
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "NeuML/pubmedbert-base-embeddings"))
    embed_dim: int   = field(default_factory=lambda: int(os.getenv("EMBED_DIM", "768")))

    # Reranker
    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
    reranker_top_n: int = field(default_factory=lambda: int(os.getenv("RERANKER_TOP_N", "80")))
    reranker_top_k: int = field(default_factory=lambda: int(os.getenv("RERANKER_TOP_K", "10")))

    # ChromaDB
    chroma_host: str       = field(default_factory=lambda: os.getenv("CHROMA_HOST", "localhost"))
    chroma_port: int       = field(default_factory=lambda: int(os.getenv("CHROMA_PORT", "8000")))
    chroma_collection: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "mao_knowledge"))

    # Qdrant (cloud mirror for deployment)
    qdrant_url: str | None      = field(default_factory=lambda: os.getenv("QDRANT_CLUSTER_ENDPOINT") or None)
    qdrant_api_key: str | None  = field(default_factory=lambda: os.getenv("QDRANT_API_KEY") or None)
    qdrant_collection: str      = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "mao_knowledge"))

    # Vector backend: "qdrant" (default) | "chromadb" (local fallback)
    vector_backend: str = field(default_factory=lambda: os.getenv("VECTOR_BACKEND", "qdrant"))

    # Postgres (optional — for metrics only)
    postgres_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://mao:mao@localhost:5432/mao"))

    # API
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8080")))

    # Paths
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", str(Path(__file__).parent.parent / "data"))))

    # Misc
    log_level: str          = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


cfg = MAOConfig()

# ---------------------------------------------------------------------------
# Write-ingestion authorization — ADV15-11
#
# Deliberately NOT a field on `cfg`. `cfg` is a frozen dataclass built once at
# import, so a key configured or rotated after startup would never be seen —
# and the refusal branch that matters most, "no key configured", could not be
# exercised against the running app at all. A control whose fail-closed path is
# untestable is the same class of thing as a control that is not there.
# ---------------------------------------------------------------------------
INGEST_API_KEY_ENV = "MAO_INGEST_API_KEY"


def ingest_api_key() -> str:
    """The configured shared secret for corpus writes; empty when unconfigured."""
    return os.getenv(INGEST_API_KEY_ENV, "").strip()


# Token budget (per clinical call)
TOKEN_BUDGET: int = 32000

# Query cache TTL (seconds)
CACHE_TTL: int = 300

# ---------------------------------------------------------------------------
# Legacy model aliases.
#
# Kept so existing imports keep working, but they are now *views* onto role
# bindings in mao.providers.registry. Prefer `model_id_for(ModelRole.X)` in new
# code. Overriding a role uses that role's own env var (MAO_MODEL_<ROLE>);
# there is no nested getenv chain a generic GROQ_MODEL could hijack (P1-17).
# ---------------------------------------------------------------------------
FAST_MODEL: str = model_id_for(ModelRole.GENERAL_SYNTHESIS)
CLINICAL_MODEL: str = model_id_for(ModelRole.CLINICAL_SYNTHESIS)

# Council
#
# COUNCIL_MAX_TOKENS is the budget for the VERDICT — "VERDICT: PASS" plus one
# sentence, measured at 38-52 tokens. The bound model's analysis channel is paid
# for separately by the gateway from the model record's declared reasoning
# overhead, so this number no longer has to guess at how much a model thinks.
COUNCIL_MAX_TOKENS: int = 200

# Must exceed `retry.MAX_RATE_LIMIT_WAIT_SECONDS` plus the call itself, or the
# council's own timeout cancels a member that is correctly waiting out a burst
# rate limit — and a cancelled member fails closed, so the timeout would turn a
# recoverable 429 into a withheld clinical answer.
COUNCIL_TIMEOUT_SECONDS: float = 120.0

# NLI — a local cross-encoder, not a provider-routed role
NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"
NLI_ENTAILMENT_THRESHOLD: float = 0.75

# ---------------------------------------------------------------------------
# Safety thresholds are owned by mao.safety.policy. Re-exported for existing
# imports only — change them in the policy, never here.
# ---------------------------------------------------------------------------
MRI_CONFIDENCE_GATE: float = get_policy().mri_confidence_gate
