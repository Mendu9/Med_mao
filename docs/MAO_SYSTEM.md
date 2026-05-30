# MAO — Multi-Agent Orchestrator: System Documentation

## Overview

MAO is a clinical AI system designed for Alzheimer's and stroke domains. It combines a LangGraph multi-agent pipeline, GraphRAG retrieval, persistent memory, safety guardrails, and a Streamlit web UI. LLM inference runs via Groq cloud API (free tier). Frontend: `app/streamlit_app.py` (Streamlit, port 8501). Legacy Gradio frontend (`app/frontend.py`) is abandoned — do not edit.

---

## Architecture

```
User Query (Gradio UI / FastAPI)
         │
         ▼
  ┌─────────────┐
  │ PII Scrubber│  removes names, DOB, identifiers before any LLM call
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │   Router    │  domain_classifier → intent classification
  └──────┬──────┘
         │ intent: summarize / graphrag / clinical / tool / sql /
         │         multimodal / code / critic / fallback
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │                     LangGraph Graph                      │
  │                                                          │
  │  summarizer_node   graphrag_node   clinical_node         │
  │  tool_node         sql_node        multimodal_node       │
  │  code_node         critic_node                           │
  │                          │                               │
  │              domain_supervisor_node                      │
  │              llm_council_node  (parallel safety votes)   │
  │              senior_supervisor_node                      │
  └──────────────────────────┬───────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        NLI Checker    RAGAS Eval     LLM Judge
     (hallucination) (faithfulness) (accuracy 1-10)
              │
              ▼
       Response → User
              │
              ▼
     ┌────────────────┐
     │   PostgreSQL   │  ChatSession, ResponseMetrics,
     │   (persist)    │  ResponseFeedback, LLMJudgeScore,
     └────────────────┘  GuardrailEvent
              │
              ▼
     ┌────────────────┐
     │     Redis      │  session cache (1h TTL)
     └────────────────┘  rate limit counter (60s window)
```

---

## Modules

### `mao/api/main.py` — FastAPI Entrypoint
- `POST /chat` — main conversational endpoint; runs LangGraph in thread pool
- `POST /ingest` — Wikipedia ingestion background task
- `POST /ingest/alzheimers` — PDF ingestion for Alzheimer's domain
- `GET /health` — checks Ollama + ChromaDB + PostgreSQL
- `GET /graph` — returns LangGraph topology
- `POST /feedback` — user thumbs-up/down feedback → DB
- `GET /export/report/{session_id}` — export PDF report card
- `GET /eval/dashboard` — RAGAS metrics summary
- Rate limiting: 20 req/min per user via Redis (fails open)
- Startup: `init_db()` + MRI model pre-warm + LangGraph build

### `mao/graph.py` — LangGraph Graph
- Builds the directed graph with conditional routing on `state.intent`
- Module-level singleton (`get_graph()`) built once at startup
- Nodes: router → domain agent → domain_supervisor → llm_council → senior_supervisor

### `mao/agents/` — Agent Nodes

| File | Role |
|------|------|
| `router.py` | Classifies intent + domain using Ollama |
| `graphrag_agent.py` | ChromaDB retrieval + reranking + Ollama generation |
| `clinical_agent.py` | Alzheimer's/stroke specialist — Pinecone + NLI + report card |
| `summarizer_agent.py` | Summarisation tasks |
| `tool_agent.py` | Calculator, unit converter, date tools |
| `sql_agent.py` | Natural language → SQL → PostgreSQL |
| `multimodal_agent.py` | MRI image classification (ResNet-18) |
| `code_agent.py` | Code generation/explanation |
| `critic_agent.py` | Self-critique and answer refinement |
| `domain_supervisor.py` | Domain-level quality gating |
| `llm_council.py` | Parallel safety votes (3 LLM judges) |
| `senior_supervisor.py` | Final verdict aggregation |
| `query_decomposer.py` | Breaks complex queries into sub-questions |
| `domain_classifier.py` | alzheimer / stroke / general classification |

### `mao/rag/` — Retrieval-Augmented Generation

| File | Role |
|------|------|
| `embedder.py` | nomic-embed-text via Ollama → float vectors |
| `retriever.py` | Pinecone vector search (alzheimer-index, stroke-index); `_vector_search()` |
| `graph_builder.py` | Builds NetworkX entity graph from ChromaDB chunks; `load_graph()` |
| `reranker.py` | BAAI/bge-reranker-v2-m3 CrossEncoder — re-scores top-k chunks |

**Flow:** query → embed → Pinecone search (top-20) → CrossEncoder rerank (top-5) → LLM

### `mao/memory/mem0_handler.py` — Persistent Memory
- Mem0 library: stores and retrieves per-user conversation facts
- Injected into clinical_agent context at each turn
- Keys by `user_id`

### `mao/core/` — Core Utilities

| File | Role |
|------|------|
| `config.py` | Pydantic Settings — all env vars in one place |
| `state.py` | `MAOState` TypedDict — shared state across all graph nodes |
| `llm.py` | `chat()` + `chat_with_budget()` — Ollama call with token budget + retry |
| `retry.py` | Exponential backoff decorator for LLM/network calls |
| `token_counter.py` | tiktoken-based token counting + truncation |
| `pii_scrubber.py` | Regex + NER-based PII removal (names, DOB, NHS/SSN numbers) |
| `query_cache.py` | Redis-backed RAG cache (TTL 300s, key `qcache:<sha256>`); in-memory fallback |
| `redis_client.py` | Singleton Redis client; `safe_get` / `safe_set` never raise |
| `rate_limiter.py` | Redis INCR sliding window — 20 req/min; fails open if Redis down |

### `mao/db/` — Database Layer

| File | Role |
|------|------|
| `models.py` | 5 SQLAlchemy ORM models (see schema below) |
| `__init__.py` | `init_db()` (thread-safe, idempotent) + `get_db_session()` context manager |
| `migrations/` | Alembic — `alembic upgrade head` applies all migrations |

**Schema (PostgreSQL):**
```
chat_sessions       — one row per /chat request
response_metrics    — RAGAS scores per session
response_feedback   — user thumbs-up/down
llm_judge_scores    — accuracy/completeness/safety/clarity (1-10)
guardrail_events    — PII / safety guardrail triggers
```

### `mao/eval/` — Evaluation

| File | Role |
|------|------|
| `ragas_evaluator.py` | Async RAGAS scoring (faithfulness, answer_relevancy, context_precision, context_recall) → stored to `response_metrics` |
| `nli_checker.py` | CrossEncoder entailment — detects hallucination (score < 0.5 flagged) |
| `llm_judge.py` | LLM-as-judge: scores response on 4 dimensions (1-10) → stored to `llm_judge_scores` |

### `mao/report/report_card.py` — PDF Export
- Builds structured clinical report card (patient summary, drug interactions, risk scores)
- `GET /export/report/{session_id}` triggers PDF generation

### `mao/monitoring/metrics.py` — Prometheus
- Exposes `/metrics` endpoint
- Gauges: `faithfulness_gauge`, `hallucination_rate_gauge`, `active_requests`, `request_latency`

### `mao/models/mri_predictor.py` — MRI Classification
- ResNet-18 for MRI scan classification
- Pre-warmed at startup (downloads ~127MB on first run)
- Called by `multimodal_agent` for image-based queries

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **LLM inference** | Ollama (local) — llama3 / mistral |
| **Embeddings** | nomic-embed-text via Ollama |
| **Vector DB (RAG)** | Pinecone — `alzheimer-index`, `stroke-index` |
| **Vector DB (Wiki)** | ChromaDB — Wikipedia article chunks |
| **Graph** | NetworkX — entity co-occurrence graph |
| **Graph viz** | pyvis — interactive HTML graph |
| **Reranker** | BAAI/bge-reranker-v2-m3 (HuggingFace CrossEncoder) |
| **Memory** | Mem0 — per-user persistent facts |
| **Orchestration** | LangGraph (StateGraph) |
| **API** | FastAPI + Uvicorn |
| **UI** | Gradio 5 |
| **Database** | PostgreSQL 16 + SQLAlchemy 2.x + Alembic |
| **Cache / Rate limit** | Redis 7 |
| **Eval** | RAGAS, NLI CrossEncoder, LLM-as-judge |
| **Metrics** | Prometheus + prometheus-client |
| **PDF export** | ReportLab / WeasyPrint |
| **MRI model** | PyTorch ResNet-18 |
| **PII scrubbing** | spaCy NER + regex |

---

## Prerequisites

```
Python 3.10+
PostgreSQL 16   (localhost:5432)
Redis 7         (localhost:6379)
Ollama          (localhost:11434)
Pinecone account + API key
```

**Pull Ollama models:**
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

---

## Environment Variables

Create a `.env` file in the project root:

```bash
# Required
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENV=us-east-1-aws

# Database (defaults shown)
MAO_DATABASE_URL=postgresql://mao:mao@localhost:5432/mao

# Redis (default shown)
REDIS_URL=redis://localhost:6379/0

# Ollama (default shown)
OLLAMA_BASE_URL=http://localhost:11434

# Optional
LOG_LEVEL=INFO
MEM0_API_KEY=your-mem0-key
```

---

## Local Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start infrastructure
```bash
# PostgreSQL (Docker)
docker run -e POSTGRES_USER=mao -e POSTGRES_PASSWORD=mao \
           -e POSTGRES_DB=mao -p 5432:5432 -d postgres:16

# Redis (Docker)
docker run -p 6379:6379 -d redis:7

# Ollama
ollama serve
ollama pull llama3
ollama pull nomic-embed-text
```

### 3. Apply database migrations
```bash
alembic -c mao/db/migrations/alembic.ini upgrade head
```

### 4. Ingest data
```bash
# Wikipedia baseline into ChromaDB
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{"topics": ["Alzheimer disease", "Stroke", "Donepezil"], "max_articles": 20}'

# Alzheimer PDFs — drop PDFs into mao/rag/data/ then:
curl -X POST http://localhost:8080/ingest/alzheimers
```

### 5. Start the API
```bash
uvicorn mao.api.main:app --host 0.0.0.0 --port 8080 --reload
```

### 6. Start the Gradio UI
```bash
python app/frontend.py
# Opens at http://localhost:7860
```

---

## Request Flow (Step-by-Step)

```
1.  User sends query via Gradio or POST /chat

2.  Rate limiter checks Redis (20 req/min per user_id)
    → 429 if exceeded, else continue

3.  PII Scrubber strips names, DOB, NHS/SSN numbers from query

4.  Router node classifies:
    - domain: alzheimer / stroke / general
    - intent: graphrag / clinical / summarize / tool / sql / ...

5.  Domain agent executes:
    a. Embed query via nomic-embed-text (Ollama)
    b. Check Redis query cache (TTL 300s) — return early on hit
    c. Pinecone vector search → top-20 chunks
    d. CrossEncoder rerank (BAAI/bge-reranker-v2-m3) → top-5 chunks
    e. Inject Mem0 user memory + chunks into prompt
    f. LLM call (Ollama llama3) → response

6.  Domain Supervisor checks response quality

7.  LLM Council: 3 parallel safety votes → aggregate verdict

8.  Senior Supervisor: final pass/fail gate

9.  NLI Checker: entailment score on response vs. chunks
    → uncertainty_flag = True if score < 0.5

10. Response returned to user

11. Background tasks (fire-and-forget, non-blocking):
    a. ChatSession written to PostgreSQL
    b. Session JSON cached in Redis (TTL 3600s)
    c. RAGAS scoring: faithfulness, answer_relevancy, etc. → response_metrics
    d. LLM Judge: accuracy/completeness/safety/clarity → llm_judge_scores

12. Prometheus metrics updated (/metrics endpoint)
```

---

## Running Tests

```bash
pytest tests/ -v                  # all tests
pytest tests/core/ -v             # core utilities
pytest tests/db/ -v               # ORM + migrations
pytest tests/eval/ -v             # RAGAS, NLI
pytest tests/agents/ -v           # agent nodes
```

---

## Key Design Decisions

- **Fails open everywhere** — Redis down, Pinecone down, RAGAS unavailable: the API keeps serving. Only optional features degrade.
- **Fire-and-forget persistence** — DB writes and eval scoring run in background threads via `asyncio.create_task` + `run_in_executor`, never blocking the user response.
- **Local-first LLMs** — Ollama means no API keys or costs for inference. Pinecone is the only external paid service required.
- **PII before LLM** — PII scrubbing runs before any data reaches the LLM or is persisted.
- **Idempotent DB init** — `init_db()` is thread-safe and safe to call multiple times.
- **Rate limiting fails open** — Redis unavailable → all requests pass through, preventing a Redis outage from taking down the API.
