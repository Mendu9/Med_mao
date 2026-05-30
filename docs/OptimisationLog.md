# MAO Optimisation Log

**Last Updated:** 2026-05-30

---

## Tier 1 — Completed (2026-05-30, commit a67d1f0)

### 1. True Token-by-Token Streaming from Groq
**File:** `mao/api/main.py`

Previous implementation collected all tokens before yielding. Replaced with an `asyncio.Queue` bridge:
- Sync `_chat_stream()` runs in `ThreadPoolExecutor`, pushes tokens via `loop.call_soon_threadsafe(queue.put_nowait, tok)`
- Async generator awaits from queue, yields `data: {json_encoded_token}\n\n`
- JSON encoding prevents SSE frame corruption from tokens containing `\n`
- Graphrag/clinical agents set `_want_stream=True` in state to defer their LLM call

### 2. BM25 Safe Serialization (pickle to JSON)
**File:** `mao/rag/retriever.py`, `mao/data/bm25_corpus.json`

Pickle is unsafe and version-fragile. Replaced with JSON:
- 154,250 documents stored as `mao/data/bm25_corpus.json` (95 MB)
- Index rebuilt from corpus in memory on load
- Thread-safe: `_bm25_loaded = True` set LAST inside lock, after all globals populated
- Legacy `bm25_index.pkl` auto-removed after first migration

### 3. IP-Based Rate Limiting
**File:** `mao/core/rate_limiter.py`

Atomic Redis Lua script (INCR+EXPIRE in one round-trip):
- `ratelimit:user:{id}` bucket: 20 req/min
- `ratelimit:ip:{ip}` bucket: 60 req/min
- Degrades gracefully when Redis unavailable

### 4. Pinecone Config Fields Removed
**Files:** `mao/core/config.py`, `mao/memory/mem0_handler.py`

Removed `pinecone_api_key`, `pinecone_index`, `pinecone_mem0_index`, `pinecone_region` from `MAOConfig` and 13 dead comment lines from `mem0_handler.py`.

---

## Tier 2 — Completed (2026-05-30, commit a67d1f0)

### 5. Postgres Connection Pool
**File:** `mao/db/__init__.py`

`pool_size=10, max_overflow=20, pool_timeout=30, pool_recycle=1800` — prevents connection exhaustion under concurrent load.

### 6. Qdrant Payload Indexing
**File:** `mao/rag/retriever.py`

At startup, two idempotent indexes created:
- `entities` field: TEXT index (WORD tokenizer) — enables `MatchText` filtered scroll
- `domain` field: KEYWORD index — enables exact domain filtering

### 7. Semantic Query Cache
**File:** `mao/core/query_cache.py`

`make_semantic_key(query, domain, top_k)` normalizes query (lowercase + strip punctuation + collapse whitespace) before SHA-256 hashing. Cache hit fires before embedding is computed.

### 8. Async NLI Checker
**File:** `mao/eval/nli_checker.py`

`async_check_all_claims()` runs the cross-encoder in a thread executor with `threading.Lock()` around lazy model init.

### 9. Groq Async Singleton
**File:** `mao/core/llm.py`

`_async_groq_client` singleton — avoids creating a new `AsyncGroq()` per council call. `achat()` uses singleton.

---

## Tier 3 — Completed (2026-05-30, commit a67d1f0)

### 10. Async LLM Council
**File:** `mao/agents/llm_council.py`

Three council agents run in parallel via `asyncio.gather`:
- `run_council_async()` — async, 30s timeout per agent
- `run_council()` — sync wrapper via `asyncio.run()` (correct for executor threads)
- Fail-safe: any error returns `{"passed": False, "blocked_by": "council_error"}`
- Clinical threshold: `fail_count == 0` — any single FAIL blocks

### 11. Structured JSON Logging + Request Tracing
**File:** `mao/core/logging_config.py`

- `_JSONFormatter`: single-line JSON with `ts`, `level`, `logger`, `msg`, `trace_id`
- Per-request `trace_id` via `contextvars.ContextVar` — propagates through async tasks
- `_trace_id_middleware` injects UUID trace_id into every request
- Auto-detects container via `/.dockerenv`

### 12. Eval Feedback Loop
**Files:** `mao/eval/ragas_evaluator.py`, `mao/db/models.py`

`RetrainingCandidate` DB table — stores responses where `faithfulness < 0.6` for offline analysis:
- Fields: `request_id, user_id, agent_used, question, answer, contexts (JSON), faithfulness, answer_relevancy, trigger_reason`

### 13. PubMed Live Search as Fallback
**Files:** `mao/data/ingest_pubmed.py`, `mao/agents/graphrag_agent.py`

When RAG confidence is below threshold, queries Entrez API for live PubMed abstracts:
- `live_pubmed_search(query, max_results=5)` — returns `{title, abstract, year, journal, pmid}` dicts
- Inserted as `[PUBMED N]` context before web results
- Requires `PUBMED_EMAIL` in `.env` and `pip install biopython`

### 14. Multi-Worker Deployment
**Files:** `gunicorn.conf.py`, `Dockerfile`

`gunicorn.conf.py` with `workers = cpu_count * 2 + 1`, `UvicornWorker`, `max_requests=1000`, `preload_app=True`.

Start: `gunicorn mao.api.main:app --config gunicorn.conf.py --bind 0.0.0.0:8080`

---

## 4-Agent Audit Fixes (2026-05-30, commit a67d1f0)

### Critical

- **BM25 lock gap**: `_bm25_loaded = True` now set LAST after all globals populated — prevents concurrent thread seeing partial state
- **`ResponseMetrics.session_id`**: was `nullable=False` — every metrics insert silently failed; fixed to `nullable=True`
- **Council fail-safe**: exception previously returned `{"passed": True}`; fixed to `{"passed": False, "blocked_by": "council_error"}`

### High

- `asyncio.get_event_loop()` replaced with `asyncio.get_running_loop()` in all 9 occurrences (main.py, ragas_evaluator.py, nli_checker.py)
- SSE frame corruption: `json.dumps(token)` encoding prevents `\n` in tokens splitting frames
- Council threshold: `fail_count < 2` → `fail_count == 0`
- Domain supervisor + senior supervisor: markdown JSON fence stripping before `json.loads()`
- Router: added `image_url` and `report_b64` to clinical routing deterministic check

### Medium

- Reranker scalar guard: `isinstance(raw, (int, float))` wraps scalar in list before scoring
- `embed_query()`: `.flatten()` for guaranteed 1-D output
- `ingest_pubmed.py`: removed top-level `logging.basicConfig()` (was corrupting app log format)
- `ingest_alzheimers._chunk_text`: default domain `"alzheimers"` → `"alzheimer"`
- `db_helper.py`: removed redundant `session.commit()` (context manager commits on exit)
- NLI checker: `threading.Lock()` around lazy init

---

## Post-Audit Fixes (2026-05-30, current session)

### Mem0 API Compatibility
**File:** `mao/memory/mem0_handler.py`

`mem0 >= 0.1.40` moved `user_id` from top-level kwarg to `filters={"user_id": ...}`. Fixed with try/except graceful fallback for both `search()` and `add()`.

### Mem0 TPM Rate Limit
**File:** `mao/memory/mem0_handler.py`

Full RAG responses (2000+ tokens) exceeded Groq's 6000 TPM limit for `llama-3.1-8b-instant`. Truncated user query to 400 chars and assistant response to 800 chars before saving to Mem0.

### BM25 Pickle Migration
`bm25_index.pkl` (154,250 docs) migrated to `bm25_corpus.json` (95 MB). Pickle deleted.

### Background Task GC Fix
**File:** `mao/api/main.py`

`asyncio.create_task()` for RAGAS background scoring had no strong reference. Fixed with `_bg_tasks` set:
```python
_task = asyncio.create_task(...)
_bg_tasks.add(_task)
_task.add_done_callback(_bg_tasks.discard)
```

### ragas_evaluator get_event_loop
**File:** `mao/eval/ragas_evaluator.py`

`run_full_ragas_eval()` used deprecated `asyncio.get_event_loop()`. Fixed to `asyncio.get_running_loop()`.

---

## Current Baseline (2026-05-30)

| Metric | Value |
|--------|-------|
| Tests | 143 passed, 29 skipped, 1 xfailed, 1 xpassed |
| 30-question flow | 30/30 responses, 29/30 intent, 30/30 RAG hits |
| Graph nodes | 1766 (697 CHEMICAL + 1069 DISEASE) |
| Graph edges | 5713 (all `co-occurs_with`) |
| BM25 corpus | 154,250 documents |
| Vector backend | Qdrant cloud (`mao_knowledge`) |

---

## Remaining / Future Optimisations

- **Asyncpg native pool** for `/feedback` and `/export/report` endpoints
- **Qdrant for Mem0** (replacing ChromaDB for user memory)
- **Semantic triple extraction** — run `triple_extractor.py` on ingested corpus to add typed edges (treats/causes/inhibits) replacing all `co-occurs_with`
- **Streaming metadata SSE event** — trailing `data: __meta__:{json}\n\n` for Streamlit sidebar real-time update
- **`asyncio.gather` for parallel sub-query retrieval** in `graphrag_agent`
