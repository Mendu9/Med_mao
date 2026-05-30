# Deployment & Branching Strategy

## Branches

| Branch | Purpose | Description |
|--------|---------|-------------|
| `master` | Local production | Stable local dev with D-drive models, ChromaDB |
| `main` | GitHub default | Clean public-facing codebase, no local paths |
| `hf-spaces` | HF Spaces deploy | Docker-based, Qdrant backend, secrets via HF env |

## Workflow

1. Develop locally on `master`
2. Before pushing to GitHub: merge `master` → `main`, strip sensitive configs
3. For HF Spaces: merge `main` → `hf-spaces`, set `VECTOR_BACKEND=qdrant`

## Key differences between local and HF Spaces

| Config | Local (master) | HF Spaces (hf-spaces) |
|--------|---------------|----------------------|
| Vector store | ChromaDB (localhost:8000) | Qdrant (cloud) |
| Reranker | Enabled, D:/hf_cache | Disabled (2.2GB RAM limit) |
| Graph | Optional | Disabled (MAO_DISABLE_GRAPH=1) |
| LLM | Groq | Groq (same) |
| Port | 8080 API + Streamlit | 7860 (combined) |

## HF Spaces secrets to set
- GROQ_API_KEY
- QDRANT_API_KEY
- QDRANT_CLUSTER_ENDPOINT
- LANGSMITH_API_KEY (optional, for tracing)
- MAO_DISABLE_GRAPH=1
- MAO_DISABLE_RERANKER=1 (RAM limited)
- MAO_DISABLE_MEM0=1
- VECTOR_BACKEND=qdrant
