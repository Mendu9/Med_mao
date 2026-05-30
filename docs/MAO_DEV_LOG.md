# MAO Development Log

**Last Updated:** 2026-05-30

---

## What Was Built (in chronological order)

### Phase 1 — Core Architecture

**Commits:** `00c3277`, `297dce4`, `d1847a7`

**Database & Persistence**
- 5-table PostgreSQL schema via SQLAlchemy + Alembic:
  - `chat_sessions` — conversation history per user
  - `response_metrics` — latency, token count, model used
  - `response_feedback` — user ratings (1-5 stars)
  - `llm_judge_scores` — automatic quality evaluation
  - `guardrail_events` — input/output validation log
- Graceful degradation: app works without Postgres (queries/feedback not persisted)

**Vector Store & RAG**
- ChromaDB vector store (localhost:8000, `mao_knowledge` collection)
- GraphRAG 5-step pipeline:
  1. Vector search: top-20 candidates from ChromaDB
  2. Entity extraction: spaCy NER on retrieved chunks
  3. Graph traversal: NetworkX BFS (2 hops) to expand entity neighbors
  4. Merge & deduplicate: combine vector + graph-expanded chunks
  5. Rerank: BAAI/bge-reranker-v2-m3 cross-encoder, top-5 final results
- Entity graph: 222 nodes, 368 edges (built from ingested knowledge)

**LangGraph Clinical AI**
- 15-node directed graph:
  - `router` (mistral) → intent classification (8 labels)
  - 7 specialist agents: graphrag, summarizer, tool, sql, code, critic, multimodal
  - Memory, safety gates, and chain-of-thought nodes
- Per-agent memory pattern: search_memories → process → save_memory
- Mem0 memory backend with Pinecone + Ollama + sentence-transformers

**API & UI**
- FastAPI backend (port 8080): `/chat`, `/ingest`, `/ingest/alzheimers`, `/health`, `/graph`
- Gradio UI frontend (port 7860): 4 tabs (Chat, Session History, Graph Explorer, Settings)
- All multimodal support: text, image (llava), audio (Whisper)

**LLM & Embeddings**
- Ollama local LLM (http://localhost:11434):
  - `gemma2:2b` (lightweight routing)
  - `llama3.1:8b` (deep reasoning)
  - `nomic-embed-text` (768-dim vectors)
  - `llava` (image understanding)
- HuggingFace local models:
  - `BAAI/bge-reranker-v2-m3` (reranking, 600MB)
  - `cross-encoder/nli-deberta-v3-small` (entailment checking)
  - `openai/whisper-base` (audio transcription)

**Caching & Rate Limiting**
- Redis caching: query results, embedding cache, memory summaries (ex=300s)
- Rate limiting: 100 req/min per user_id
- Graceful degradation: code handles Redis absence (caching disabled)

---

### Phase 2 — Bug Fixes

**Commit:** `8f06761`

| Error | Root Cause | Fix |
|-------|-----------|-----|
| Decomposer returned non-JSON with markdown fences | `gemma2:2b` wraps JSON output in ` ```json [...] ``` ` | Added `_strip_fences()` helper to strip backticks before `json.loads()` |
| `Mem0 search failed: "At least one of 'user_id' required"` | Used wrong Mem0 API param: `filters={"user_id":...}` | Changed to `user_id=user_id` direct kwarg in `search()` call |
| `timed out` error in UI for MRI analysis | Reranker (568MB) downloading on first request, 120s timeout | Pre-warm reranker at API startup + raised frontend httpx timeout from 120s to 300s |
| Redis unavailable warning spammed every request | `_sync_client` stayed `None`, `get_redis()` retried every call | Added `_redis_unavailable` boolean sentinel: set on first failure, suppresses retries |

**Files Modified:**
- `mao/agents/query_decomposer.py` — added `_strip_fences()`
- `mao/memory/mem0_handler.py` — fixed Mem0 API call signature
- `mao/api/main.py` — added reranker pre-warm at startup
- `mao/core/redis_client.py` — added `_redis_unavailable` sentinel
- `app/frontend.py` — raised httpx timeout to 300s

---

### Phase 3 — Eval Harness, Guardrails, Graph Explorer, Web Search

**Commits:** `5ef693f`, `6721a6b`

**Eval Harness**
- Golden dataset: 20 test cases (Q&A pairs) covering Alzheimer's domain
- Chunk fixtures: 75 pre-extracted knowledge base chunks for reproducible testing
- 5 test suites:
  - `test_routing.py` — intent classification accuracy
  - `test_rag_recall.py` — RAG top-5 includes correct sources
  - `test_hallucination.py` — NLI entailment checker (cross-encoder/nli-deberta-v3-small)
  - `test_council_veto.py` — LLM council majority vote blocks unsafe answers
  - `test_ragas_scores.py` — RAGAS metrics (faithfulness, relevance) validation

**Guardrails Package**
- Input guardrails:
  - Prompt injection regex patterns (SQL, jailbreak, code execution attempts)
  - Token limit: 500 max before rejection
  - PII logging: detect SSN, email, credit card (logged to DB, never sent to LLM)
- Output guardrails:
  - NLI confidence gate: reject answers if entailment score < 0.7
  - Council safety gate: require 2/3 votes to approve unsafe-flagged answers
  - LLM judge gate: automatic rubric scoring (code clarity, SQL safety, relevance)
- Wired into `POST /chat` response pipeline

**Graph Explorer Tab**
- Tab 4 in Gradio UI
- Pyvis interactive visualization of entity graph (top 200 nodes by betweenness)
- ChromaDB chunk browser: side-by-side entity nodes + source chunks
- Toggleable node categories (Alzheimer's, Stroke, Dementia, etc.)

**Multi-Provider Web Search**
- Fallback chain: Brave Search → SerpAPI → DuckDuckGo (never raises)
- Handles rate limits gracefully (Brave free tier: 2000 req/month, prevents DDG 429 errors)
- Used by `tool_agent` for latest news + current facts

**Test Infrastructure**
- `conftest.py` at project root: fixes sys.path for pytest import mode
- `pytest.ini`: `addopts = --import-mode=importlib` (avoids package shadowing)
- All 42 tests fixed: proper mock patching, keyword fixtures, dynamic web_search dispatch

---

## Errors Faced & Fixes

| Error | Root Cause | Fix | Commit |
|-------|-----------|-----|--------|
| `json.JSONDecodeError: Decomposer returned non-JSON: \`\`\`json [...] \`\`\`` | `gemma2:2b` wraps JSON in markdown fences instead of plain output | Added `_strip_fences()` before `json.loads()` in query_decomposer.py | 8f06761 |
| `Mem0 search failed: At least one of 'user_id', 'agent_id', or 'run_id' must be provided` | Wrong Mem0 API: tried `filters={"user_id":...}` instead of direct kwarg | Changed to `user_id=user_id` direct parameter | 8f06761 |
| `Error: timed out` in Gradio UI for MRI analysis | Reranker (568MB BERT model) downloading on first request; 120s httpx timeout too short | Pre-warm reranker at API startup + raise frontend timeout to 300s | 8f06761 |
| `Redis unavailable` warning printed on every request | `_sync_client` remained `None` after first failure, retried connect every request | Added `_redis_unavailable` boolean sentinel to suppress retries after first failure | 8f06761 |
| `DuckDuckGo 202 Ratelimit` crashes in clinical_agent | DuckDuckGo aggressively rate-limits; single provider insufficient | Implemented multi-provider fallback: Brave → SerpAPI → DDG (all wrapped in try/except) | 5ef693f |
| `ModuleNotFoundError: No module named 'app.graph_explorer'` in pytest | pytest with `--import-mode=prepend` shadowed `app` package with current dir | Added `conftest.py` + `pytest.ini` with `--import-mode=importlib` | 6721a6b |
| Web search mock tests hitting real HTTP endpoints | `_PROVIDERS` dict built at import time with static function refs; mocking failed | Made `web_search()` look up providers dynamically via `import mao.core.web_search as _self` | 6721a6b |
| `AttributeError: 'Signature' object has no attribute 'bind'` in guardrails tests | `get_db_session` imported inside function from `mao.db`; patch targeted wrong namespace | Changed patch target to `mao.db.get_db_session` instead of guardrails module | 6721a6b |
| RAG recall tests fail: keyword mismatches between golden dataset + fixture chunks | Golden dataset keywords didn't match actual chunk text (e.g., "thrombolysis" vs "thrombolytic") | Updated 5 keywords in fixtures: "thrombolysis"→"thrombolytic", "bleeding"→"hemorrhage", etc. | 6721a6b |

---

## What Is Yet To Implement

- [ ] **Redis server** — not running in dev environment; code gracefully disables caching but it remains disabled until Redis starts
  - Start: `docker run -d -p 6379:6379 redis:alpine`
  - Recommended: Redis cloud (upstash.com) for production

- [ ] **Brave Search API key** — free tier gives 2000 queries/month, eliminates DuckDuckGo rate-limiting completely
  - Get key: https://api.search.brave.com
  - Set: `BRAVE_API_KEY=your_key` in `.env`

- [ ] **PostgreSQL database** — chat history, feedback, metrics logged to in-memory dicts; not persisted across restarts
  - Connection string: `postgresql://mao:mao@localhost:5432/mao`
  - Start: `docker run -d --name mao-postgres -e POSTGRES_USER=mao -e POSTGRES_PASSWORD=mao -e POSTGRES_DB=mao -p 5432:5432 postgres:16`
  - Migrate: `alembic upgrade head`

- [ ] **Session History tab** — UI shows placeholder; not wired to backend session retrieval
  - Needs: `GET /sessions/{user_id}` endpoint + Gradio component update

- [ ] **PDF report export** — `/export/report/{session_id}` endpoint skeleton exists but not fully wired
  - Needs: ReportCard generation + file download handler

- [ ] **Slow test suite optimization** — NLI tests and RAGAS faithfulness tests require real cross-encoder + RAGAS libraries
  - Marked: `@pytest.mark.slow`
  - Run with: `pytest -m slow` (takes minutes)
  - Improvement: Mock NLI scorer for faster CI (currently hits real model)

---

## Suggestions for Improvement

### Immediate Wins

1. **Add `.env` file** — Currently all config is hardcoded defaults (Ollama URL, ChromaDB host, etc.)
   - Create: `cp .env.example .env` + `python-dotenv.load_dotenv()` in config.py
   - Benefits: Easy env switching (dev/staging/prod), no code changes

2. **Clean up git repo** — Remove unnecessary committed files
   - Add to `.gitignore`:
     ```
     __pycache__/
     *.pyc
     .pytest_cache/
     .ruff_cache/
     chroma_data/
     .env.local
     *.db
     ```
   - Current bloat: 100+ MB of `__pycache__` and ChromaDB data committed

3. **Get Brave Search API key** — Free tier (2000 req/month) eliminates all DuckDuckGo rate-limiting
   - Cost: Free
   - Setup time: 2 minutes
   - Impact: Web search 100% reliable

4. **Start Redis** — One-liner to enable caching + rate limiting
   - Start: `docker run -d -p 6379:6379 redis:alpine`
   - Cost: Free (local) or $5/month (upstash cloud)
   - Impact: 10x faster repeated queries, rate limiting active

5. **Start PostgreSQL** — Keep chat history + feedback across restarts
   - Start: `docker run -d --name mao-postgres -e POSTGRES_USER=mao -e POSTGRES_PASSWORD=mao -e POSTGRES_DB=mao -p 5432:5432 postgres:16`
   - Migrate: `alembic upgrade head`
   - Cost: Free (local)
   - Impact: Session replay, analytics, feedback loop

### Medium-Effort Improvements

6. **Mock LLM calls in test suite** — Routing tests call real Ollama; slow for CI
   - Current: `classify_domain()` queries real LLM (adds 2-3s per test)
   - Improvement: Mock to return fixed intent classifications
   - Benefit: CI/CD faster, less dependent on Ollama availability

7. **Graph Explorer domain filter** — Currently loads all 222 nodes; rendering slow on older machines
   - Improvement: Add dropdown filter (Alzheimer's / Stroke / Dementia)
   - Benefit: Faster pyvis rendering, focused visualization

8. **Session History tab wiring** — Backend ready, frontend stub only
   - Add: `GET /sessions/{user_id}` endpoint
   - Wire: Gradio component to populate chat history from DB
   - Benefit: Full conversation replay, multi-session support

9. **Test coverage for slow tests** — Mock RAGAS scorer to avoid real inference
   - Current: `test_ragas_scores.py` marked `@pytest.mark.slow`
   - Improvement: Mock `evaluate()` to return fixed scores, keep one slow variant
   - Benefit: Fast CI (5s instead of 2+ minutes), real tests available via `pytest -m slow`

10. **Add GitHub Actions CI** — Run tests on push
    - Add: `.github/workflows/test.yml`
    - Include: Fast suite (no slow tests), coverage report, lint check
    - Benefit: Catch regressions before merge

### Architecture Considerations

11. **Guardrails severity levels** — Currently all gates are hard blocks; no soft warnings
    - Improvement: Add `confidence` field to guardrail responses (0-1 scale)
    - Benefit: UI can show warnings vs errors, better UX feedback

12. **Web search provider metrics** — No tracking of which provider succeeded
    - Improvement: Log provider + latency to `response_metrics` table
    - Benefit: Monitor which providers are most reliable

13. **Conversation context limit** — No sliding window for long multi-turn chats
    - Current: Entire chat_history sent to every agent
    - Improvement: Keep only last 5 turns + system context
    - Benefit: Unbounded growth prevented, token cost capped

---

## Known Limitations

1. **Ollama model inference time** — No GPU; all models run on CPU
   - llama3.1:8b first token latency: ~500ms
   - Workaround: Run on machine with CUDA GPU

2. **HuggingFace cache size** — Redirected to D: drive because C: was 98% full
   - Current cache: ~8 GB
   - Clean: `rm -rf D:/hf_cache` and let it rebuild (automatic)

3. **Mem0 ChromaDB backend** — Mem0 uses ChromaDB for user memory vectors (local or remote)

4. **No authentication** — API endpoints have no auth; assume trusted environment
   - Improvement needed before production: Add JWT token validation to POST /chat

5. **Guardrails regex patterns** — Basic pattern matching; no ML-based injection detection
   - Example: Can't detect semantic SQL injection (`SELECT * FROM users WHERE id = 1 OR 1=1`)
   - Improvement: Use dedicated NLI-based SQL validator

---

## Project Statistics (as of 2026-05-30)

- **Total commits:** 40+ (since initial commit)
- **Test suite:** 143 passed, 29 skipped, 1 xfailed, 1 xpassed
- **Total lines of code:** ~12,000 (mao/ + tests/)
- **External dependencies:** 30+ (FastAPI, LangGraph, Mem0, Groq, Qdrant, etc.)
- **Data ingested:** 22 Alzheimer's research PDFs + PubMed abstracts + Wikipedia articles
- **Entity graph:** 1,766 nodes (697 CHEMICAL + 1,069 DISEASE), 5,713 edges
- **BM25 corpus:** 154,250 documents
- **Vector backend:** Qdrant cloud (`mao_knowledge` collection)
- **30-question flow test:** 30/30 responses correct, 29/30 intents correct

---

## Phase 5 — Enterprise Optimisations (2026-05-30)

**Commits:** `a67d1f0`

### Tier 1 — Streaming + Safety + Cleanup
- True token-by-token streaming via `asyncio.Queue` bridge (not batch)
- BM25 safe serialization: pickle → JSON corpus (154,250 docs, 95 MB)
- IP rate limiting: Redis Lua atomic INCR+EXPIRE (two buckets: user + IP)
- Pinecone fully removed from config and mem0_handler

### Tier 2 — Backend Quality
- Postgres connection pool: `pool_size=10, max_overflow=20`
- Qdrant payload indexes at startup: `entities` (TEXT) + `domain` (KEYWORD)
- Semantic query cache: normalize → SHA-256 → cache hit before embedding
- Async NLI checker with threading.Lock for lazy init

### Tier 3 — Scale + Intelligence
- Async LLM council: 3 parallel Groq calls via `asyncio.gather`
- Structured JSON logging with per-request `trace_id` via `contextvars`
- Eval feedback loop: `RetrainingCandidate` table for `faithfulness < 0.6`
- PubMed live search as fallback when RAG confidence low
- Multi-worker deployment: `gunicorn.conf.py` with `UvicornWorker`

### 4-Agent Audit — 30+ Bugs Fixed
Critical: BM25 lock race, ResponseMetrics silent insert failure, council fail-safe.
High: `get_event_loop()` → `get_running_loop()` (9 occurrences), SSE JSON encoding, council threshold.
Medium: reranker scalar guard, embed flatten, ingest log corruption, domain fix, NLI lock.

### Post-Audit Session Fixes
- Mem0 API: `user_id=` → `filters={"user_id":}` (mem0 >= 0.1.40 API change)
- Mem0 TPM: truncate response to 800 chars before saving (prevents Groq 413 errors)
- BM25 pickle migration: migrated 154,250-doc pickle to JSON
- Background task GC: `_bg_tasks` set prevents RAGAS task being garbage collected
- ragas_evaluator: `get_event_loop()` → `get_running_loop()` in `run_full_ragas_eval`
