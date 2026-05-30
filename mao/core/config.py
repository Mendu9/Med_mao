"""
mao/core/config.py — central config from .env
All modules import `cfg` from here.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")


@dataclass(frozen=True)
class MAOConfig:
    # LLM backend — model name used by both Groq and Ollama
    groq_api_key: str   = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str     = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    groq_judge_model: str = field(default_factory=lambda: os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile"))
    ollama_base_url: str  = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

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
    code_exec_timeout: int  = field(default_factory=lambda: int(os.getenv("CODE_EXEC_TIMEOUT", "10")))


cfg = MAOConfig()

# Token budget (per clinical call)
TOKEN_BUDGET: int = 32000

# Query cache TTL (seconds)
CACHE_TTL: int = 300

# LLM tiers — default to Groq fast model; set FAST_MODEL/CLINICAL_MODEL in .env to override
FAST_MODEL: str = os.getenv("FAST_MODEL", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
CLINICAL_MODEL: str = os.getenv("CLINICAL_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

# Council
COUNCIL_MAX_TOKENS: int = 200
COUNCIL_TIMEOUT_SECONDS: float = 30.0

# NLI
NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"
NLI_ENTAILMENT_THRESHOLD: float = 0.75

# EfficientNetB3 confidence gate
MRI_CONFIDENCE_GATE: float = 0.60
