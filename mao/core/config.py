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
    # Groq LLM
    groq_api_key: str   = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str     = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))

    # Pinecone
    pinecone_api_key: str    = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    pinecone_index: str      = field(default_factory=lambda: os.getenv("PINECONE_INDEX", "mao-knowledge-base"))
    pinecone_mem0_index: str = field(default_factory=lambda: os.getenv("PINECONE_MEM0_INDEX", "mao-mem0"))
    pinecone_region: str     = field(default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1"))

    # Embeddings (sentence-transformers, local, no API key)
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    embed_dim: int   = field(default_factory=lambda: int(os.getenv("EMBED_DIM", "384")))

    # Reranker
    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
    reranker_top_n: int = field(default_factory=lambda: int(os.getenv("RERANKER_TOP_N", "20")))
    reranker_top_k: int = field(default_factory=lambda: int(os.getenv("RERANKER_TOP_K", "5")))

    # Postgres (optional — for metrics only)
    postgres_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://mao:mao@localhost:5432/mao"))

    # API
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8080")))

    # Paths
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "d:/project/mao/data")))

    # Misc
    log_level: str          = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    code_exec_timeout: int  = field(default_factory=lambda: int(os.getenv("CODE_EXEC_TIMEOUT", "10")))


cfg = MAOConfig()

# Token budget (per clinical call)
TOKEN_BUDGET: int = 7050

# Query cache TTL (seconds)
CACHE_TTL: int = 300

# LLM tiers
FAST_MODEL: str = "llama-3.1-8b-instant"
CLINICAL_MODEL: str = "llama-3.3-70b-versatile"

# Pinecone domain indexes
PINECONE_INDEX_ALZHEIMER: str = "mao-knowledge-alzheimer"
PINECONE_INDEX_STROKE: str = "mao-knowledge-stroke"

# Council
COUNCIL_MAX_TOKENS: int = 200
COUNCIL_TIMEOUT_SECONDS: float = 30.0

# NLI
NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"
NLI_ENTAILMENT_THRESHOLD: float = 0.5

# EfficientNetB3 confidence gate
MRI_CONFIDENCE_GATE: float = 0.60
