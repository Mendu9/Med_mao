# MAO Commands Reference

**Quick reference for starting, testing, and operating the MAO system.**

---

## Prerequisites

Install Python dependencies:
```bash
pip install -r requirements.txt
```

No Ollama needed — all LLM calls go through Groq (cloud API, free tier).

---

## Starting the Full Stack

Open **3 terminal tabs** in this order:

### Tab 1 — Vector Store

**Option A — Qdrant (default, recommended):**

Qdrant runs as a cloud service (no local process needed). Set these in `.env`:
```bash
QDRANT_CLUSTER_ENDPOINT=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
VECTOR_BACKEND=qdrant
```

Or run Qdrant locally via Docker:
```bash
docker run -d --name mao-qdrant -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
# Set QDRANT_CLUSTER_ENDPOINT=http://localhost:6333 (no API key needed for local)
```

**Option B — ChromaDB (local fallback):**

```bash
chroma run --path ./chroma_data --port 8000
# Set VECTOR_BACKEND=chromadb in .env
```

Ready when you see: `Started server process`

### Tab 2 — Infrastructure (Redis + PostgreSQL)

**First time only** (creates the containers):
```bash
docker run -d --name mao-redis    -p 6379:6379 redis:alpine
docker run -d --name mao-postgres -p 5432:5432 \
  -e POSTGRES_USER=mao -e POSTGRES_PASSWORD=mao -e POSTGRES_DB=mao \
  postgres:16
```

**Every subsequent time** (containers already exist):
```bash
docker start mao-redis mao-postgres
```

Apply DB migrations (after PostgreSQL is running, one-time):
```bash
alembic upgrade head
```

> Both Redis and PostgreSQL are optional — the API degrades gracefully if they are unavailable.

### Tab 3 — MAO Backend API (port 8080)

```bash
uvicorn mao.api.main:app --host 0.0.0.0 --port 8080 --reload
```

Ready when you see:
```
INFO:     MAO API ready — accepting requests.
INFO:     Uvicorn running on http://0.0.0.0:8080
```

### Tab 4 — Frontend UI

**Streamlit (recommended — new):**
```bash
streamlit run app/streamlit_app.py
```
Opens at: http://localhost:8501

**Gradio (legacy — still works):**
```bash
python app/graph_explorer.py
```
Opens at: http://localhost:7860

---

## Key URLs

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| Gradio UI (legacy) | http://localhost:7860 |
| FastAPI backend | http://localhost:8080 |
| API docs (Swagger) | http://localhost:8080/docs |
| API docs (ReDoc) | http://localhost:8080/redoc |
| ChromaDB (fallback) | http://localhost:8000 |
| Qdrant (local Docker) | http://localhost:6333 |
| Prometheus metrics | http://localhost:8080/metrics |

---

## Environment Variables

Minimum required `.env` (copy from `.env.example`):

```bash
# LLM — Groq cloud API (required)
GROQ_API_KEY=your_groq_api_key_here

# Vector store — Qdrant (default)
VECTOR_BACKEND=qdrant
QDRANT_CLUSTER_ENDPOINT=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=mao_knowledge

# Vector store — ChromaDB (local fallback, set VECTOR_BACKEND=chromadb to use)
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION=mao_alzheimers

# Database (optional — enables persistence)
DATABASE_URL=postgresql://mao:mao@localhost:5432/mao

# Redis cache (optional — enables query caching)
REDIS_URL=redis://localhost:6379/0

# Web search — set at least one
BRAVE_API_KEY=your_key_here      # Free: 2000 req/month
SERPAPI_KEY=your_key_here        # Free: 100 req/month

# HuggingFace cache — point to D: if C: is nearly full
HF_HOME=D:/hf_cache
TRANSFORMERS_CACHE=D:/hf_cache/transformers

# PubMed email — required for Entrez API (use your real email)
PUBMED_EMAIL=your_email@example.com
```

---

## Data Ingestion

Run these **once** after first setup to populate the knowledge base:

### Alzheimer's PDFs (primary corpus — ~22 research papers)

```bash
python -m mao.data.ingest_alzheimers
```

### Wikipedia articles

```bash
# Via API (backend must be running)
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{"topics": ["Alzheimer disease", "Stroke", "Amyloid beta", "Tau protein"], "max_articles": 20}'
```

### PubMed abstracts (~900 recent abstracts, requires PUBMED_EMAIL in .env)

```bash
# CLI
python -m mao.data.ingest_pubmed

# Or via API (runs in background)
curl -X POST http://localhost:8080/ingest/pubmed
```

### PrimeKG + AlzKB knowledge graphs (large — PrimeKG is ~370 MB download)

```bash
# CLI — downloads PrimeKG if not present, loads AlzKB CSVs if placed in mao/data/alzkb/
python -m mao.data.ingest_knowledge_bases

# Or via API (runs in background, check logs for progress)
curl -X POST http://localhost:8080/ingest/knowledge-bases
```

### To reingest all data

```bash
python -m mao.data.ingest_all
```
python.exe -m mao.data.reingest_all --skip-drop --only pubmed wikipedia pm
---

## API Usage

### Health check

```bash
curl http://localhost:8080/health
# Returns: {"status":"ok","groq":"ok","chromadb":"ok","postgres":"ok"}
```

### Chat (blocking)

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What drugs are approved for Alzheimer treatment?", "user_id": "test"}'
```

### Chat (streaming SSE)

```bash
curl -X POST http://localhost:8080/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain tau tangles", "user_id": "test"}'
# Streams: data: {word}\n\n ... data: [DONE]\n\n
```

### Entity graph (Mermaid format)

```bash
curl http://localhost:8080/graph
```

---

## Running Tests

### Default (fast, no external services needed, ~90 seconds)

```bash
pytest
# or
pytest -v
```

### Specific test groups

```bash
pytest tests/agents/ -v          # Agent routing + decomposition
pytest tests/core/ -v            # PII, retry, token counter
pytest tests/eval/ -v            # RAG recall, routing, NLI, council
pytest tests/guardrails/ -v      # Input/output guardrails
pytest tests/e2e/ -v             # Playwright Gradio E2E (requires running app)
```

### With coverage

```bash
pytest --cov=mao --cov-report=term-missing
```

### Skip the known flaky Gradio E2E timing test

```bash
pytest --deselect tests/e2e/test_gradio_frontend.py::test_graph_explorer_refresh_renders_content
```

---

## Database Operations

```bash
# Apply all pending migrations
alembic upgrade head

# Check current migration state
alembic current

# Roll back one migration
alembic downgrade -1

# Connect directly
psql postgresql://mao:mao@localhost:5432/mao

# View recent sessions
SELECT * FROM chat_sessions ORDER BY created_at DESC LIMIT 10;
```

---

## Troubleshooting

### Qdrant collection empty or unreachable

```bash
# Check collection stats (cloud)
python -c "
from qdrant_client import QdrantClient
import os
client = QdrantClient(url=os.environ['QDRANT_CLUSTER_ENDPOINT'], api_key=os.environ['QDRANT_API_KEY'])
info = client.get_collection('mao_knowledge')
print(f'Vectors: {info.vectors_count}')
"
# If 0 or error: check QDRANT_CLUSTER_ENDPOINT and QDRANT_API_KEY in .env, then reingest
```

### ChromaDB collection empty

```bash
python -c "
import chromadb
client = chromadb.HttpClient(host='localhost', port=8000)
col = client.get_collection('mao_alzheimers')
print(f'Documents: {col.count()}')
"
# If 0: python -m mao.data.ingest_alzheimers
```

### Redis not running (non-fatal — caching just disabled)

```bash
docker start mao-redis
# or create fresh:
docker run -d --name mao-redis -p 6379:6379 redis:alpine
```

### PostgreSQL not running (non-fatal — persistence disabled)

```bash
docker start mao-postgres
```

### HuggingFace models downloading to wrong drive

```bash
# Add to .env:
HF_HOME=D:/hf_cache
TRANSFORMERS_CACHE=D:/hf_cache/transformers
```

### pytest import errors

```bash
cd d:/project && pytest   # always run from project root
```

---

## Development Workflow

### Minimal startup with Qdrant (default — no local vector store needed)

```
Tab 1: uvicorn mao.api.main:app --host 0.0.0.0 --port 8080 --reload
Tab 2: streamlit run app/streamlit_app.py
# Requires QDRANT_CLUSTER_ENDPOINT + QDRANT_API_KEY in .env
```

### Minimal startup with ChromaDB (local fallback)

```
Tab 1: chroma run --path ./chroma_data --port 8000
Tab 2: uvicorn mao.api.main:app --host 0.0.0.0 --port 8080 --reload   (VECTOR_BACKEND=chromadb)
Tab 3: streamlit run app/streamlit_app.py
```

### Full stack startup

```
Tab 1: docker start mao-redis mao-postgres  (then close tab)
Tab 2: uvicorn mao.api.main:app --host 0.0.0.0 --port 8080 --reload
Tab 3: streamlit run app/streamlit_app.py
# Uses Qdrant cloud by default; add chroma run if VECTOR_BACKEND=chromadb
```

### Run tests before committing

```bash
pytest -m "not slow and not integration" -v
```

### Code quality

```bash
ruff check mao/ tests/
mypy mao/ --ignore-missing-imports
bandit -r mao/
```

# Generate golden dataset (one-time, ~2 min)
python -m mao.eval.retrieval_metrics --generate --samples 50

# Then run evaluation
python -m mao.eval.retrieval_metrics --eval --k 5

python -m mao.eval.retrieval_metrics --generate --samples 100 --eval --k 5
