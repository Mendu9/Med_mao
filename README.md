# MAO — Multi-Agent Orchestrator

A locally-running, production-grade AI system that routes user queries to 7 specialized agents. Zero external API costs — everything runs on your machine via Ollama.

---

## Architecture

```
User
  │
  ▼
POST /chat  (FastAPI :8080)
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LangGraph StateGraph                                                        │
│                                                                              │
│   START                                                                      │
│     │                                                                        │
│     ▼                                                                        │
│  router_node  ──── 1. search_memories (Mem0)                                 │
│   (mistral)   ──── 2. classify intent (8 labels)                             │
│     │                                                                        │
│     └──[state["intent"]]──────────────────────────────────────────────┐     │
│                                                                        │     │
│   graphrag  summarize   tool    sql    code   critic  multimodal      │     │
│      │          │        │       │       │       │         │          │     │
│      ▼          ▼        ▼       ▼       ▼       ▼         ▼          │     │
│  [all agents: search_memories → process → save_memory]                │     │
│      │                                                                 │     │
│      └──────────────────────────────────────────────────────────► END │     │
└─────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
ChatResponse  { response, agent_used, intent, metadata, latency_ms }
```

### GraphRAG retrieval pipeline (inside graphrag_node)

```
user query
    │
    ▼  1. Vector search     → ChromaDB, top-20 candidates
    │
    ▼  2. Entity extraction → spaCy NER on retrieved chunks
    │
    ▼  3. Graph traversal   → NetworkX BFS, 2 hops, expand entity neighbours
    │
    ▼  4. Merge + dedup     → combine vector hits + graph-expanded chunks
    │
    ▼  5. Rerank            → BAAI/bge-reranker-v2-m3, top-5
    │
    ▼  LLM generation       → llama3.1:8b with context + memory
```

---

## Project structure

```
project/
├── README.md                       ← this file
├── SETUP.md                        ← step-by-step setup + all commands
├── requirements.txt                ← Python dependencies
├── docker-compose.yml              ← Ollama + ChromaDB + Postgres + MAO API
├── Dockerfile                      ← MAO API container image
│
├── mao/                            ← main application package
│   │
│   ├── core/
│   │   ├── config.py               ← MAOConfig dataclass — all settings from .env
│   │   └── state.py                ← MAOState TypedDict — single data contract for all nodes
│   │
│   ├── memory/
│   │   └── mem0_handler.py         ← search_memories / save_memory / build_system_prompt
│   │
│   ├── rag/
│   │   ├── embedder.py             ← embed_query / embed_texts via nomic-embed-text (Ollama)
│   │   ├── graph_builder.py        ← spaCy NER → NetworkX co-occurrence entity graph
│   │   ├── reranker.py             ← BAAI/bge-reranker-v2-m3 cross-encoder, lazy-loaded
│   │   └── retriever.py            ← retrieve() — orchestrates all 5 GraphRAG steps
│   │
│   ├── agents/
│   │   ├── router.py               ← entry node: Mem0 pre-load + mistral intent classification
│   │   ├── graphrag_agent.py       ← factual Q&A: 5-step retrieval → llama3.1:8b
│   │   ├── summarizer_agent.py     ← map-reduce summarization (inline text or KB retrieval)
│   │   ├── tool_agent.py           ← ReAct loop: DuckDuckGo + Wikipedia API + numexpr calc
│   │   ├── sql_agent.py            ← NL → SQL → execute (Postgres) → NL answer
│   │   ├── code_agent.py           ← generate / explain / debug / sandboxed execution
│   │   ├── critic_agent.py         ← rubric review (code/prose/SQL/plan) with 1-10 score
│   │   └── multimodal_agent.py     ← image via llava + audio transcription via Whisper
│   │
│   ├── api/
│   │   └── main.py                 ← FastAPI: /chat /ingest /ingest/alzheimers /health /graph
│   │
│   ├── data/
│   │   ├── ingest_wikipedia.py     ← Wikipedia articles + SQuAD 2.0 → ChromaDB + entity graph
│   │   └── ingest_alzheimers.py    ← 22 Alzheimer's research PDFs → ChromaDB + entity graph
│   │
│   └── graph.py                    ← builds + compiles LangGraph StateGraph (singleton)
│
└── ad/                             ← Alzheimer's Streamlit app (standalone)
    ├── app.py                      ← Streamlit UI: 3D MRI viewer, EfficientNetB3, biomarker plots
    └── rag/data/                   ← 22 research PDFs ingested by mao/data/ingest_alzheimers.py
```

---

## Agents reference

| Agent | Intent | Model | What it does |
|-------|--------|-------|-------------|
| **router** | *(entry)* | mistral | Classifies intent; loads Mem0 memory for all downstream agents |
| **graphrag** | `graphrag`, `fallback` | llama3.1:8b | Answers factual questions — 5-step GraphRAG over Wikipedia + AD papers |
| **summarizer** | `summarize` | mistral | Condenses pasted text or KB content; map-reduce for texts > 6000 chars |
| **tool** | `tool` | mistral | ReAct loop: DuckDuckGo web search, Wikipedia API, safe calculator |
| **sql** | `sql` | mistral | Text → SQL → Postgres → natural language answer; SELECT-only guard |
| **code** | `code` | llama3.1:8b | Generate / explain / debug / sandboxed-execute Python |
| **critic** | `critic` | llama3.1:8b | Rubric review of code, prose, SQL, or plans with an Overall score X/10 |
| **multimodal** | `multimodal` | llava / Whisper | Image description + OCR via llava; audio transcription via Whisper |

### Intent trigger examples

```
graphrag:    "Who invented the transformer architecture?"
             "What causes tau tangles in Alzheimer's disease?"
summarize:   "Summarize this: [paste article text]"
             "Give me a TL;DR of the above"
tool:        "What's the latest news on GPT-5?"
             "What is 15% of 3840?"
sql:         "How many Wikipedia articles do we have per category?"
code:        "Write a Python function to merge two sorted lists"
             "Run this and tell me the output: [paste]"
critic:      "Review my code: [paste]"
             "Score this SQL query for performance: [paste]"
multimodal:  query = "What do you see?" + metadata.image_b64 = "<base64>"
```

---

## API endpoints

### POST /chat

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the role of amyloid beta in Alzheimer disease?",
    "user_id": "alice"
  }'
```

Response:
```json
{
  "response": "Amyloid beta (Aβ) is a peptide that aggregates...",
  "agent_used": "graphrag",
  "intent": "graphrag",
  "metadata": {
    "chunks_retrieved": 5,
    "top_scores": [0.92, 0.87, 0.81, 0.79, 0.74],
    "sources": ["nihms-893438.pdf", "Alzheimerdisease.pdf"]
  },
  "request_id": "a1b2c3d4",
  "latency_ms": 1840.5
}
```

With chat history (multi-turn):
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What about tau protein?",
    "user_id": "alice",
    "chat_history": [
      {"role": "user",      "content": "What is amyloid beta?"},
      {"role": "assistant", "content": "Amyloid beta is a peptide..."}
    ]
  }'
```

### POST /ingest — Wikipedia + SQuAD

```bash
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "topics": ["Machine learning", "Knowledge graph", "BERT (language model)"],
    "max_articles": 10
  }'
```

### POST /ingest/alzheimers — Research PDFs

```bash
# Default: reads from ad/rag/data/ (22 PDFs)
curl -X POST http://localhost:8080/ingest/alzheimers \
  -H "Content-Type: application/json" \
  -d '{}'

# Custom path
curl -X POST http://localhost:8080/ingest/alzheimers \
  -H "Content-Type: application/json" \
  -d '{"data_dir": "/absolute/path/to/pdfs", "chunk_size": 512}'
```

### GET /health

```bash
curl http://localhost:8080/health
# {"status":"ok","ollama":"ok","chromadb":"ok","postgres":"ok"}
```

### GET /graph

```bash
curl http://localhost:8080/graph
# Returns node list + intent routing map as JSON
```

### Multimodal — image

```bash
IMAGE_B64=$(base64 -w 0 /path/to/image.png)

curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"What do you see in this brain MRI?\",
    \"user_id\": \"alice\",
    \"metadata\": {\"modality\": \"image\", \"image_b64\": \"$IMAGE_B64\"}
  }"
```

### Multimodal — audio

```bash
AUDIO_B64=$(base64 -w 0 /path/to/recording.wav)

curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"Summarize this recording\",
    \"user_id\": \"alice\",
    \"metadata\": {\"modality\": \"audio\", \"audio_b64\": \"$AUDIO_B64\"}
  }"
```

---

## State contract

Every LangGraph node receives and writes to the same `MAOState` dict:

```python
class MAOState(TypedDict, total=False):
    user_query:     str           # raw user message
    user_id:        str           # Mem0 scoping key (per-user memory)
    intent:         str           # router writes: graphrag/summarize/tool/sql/code/critic/multimodal
    memory_context: str           # Mem0 formatted bullet list — loaded by router, used by all agents
    chat_history:   list[dict]    # [{"role": "user/assistant", "content": "..."}]
    response:       str           # agent writes the final answer here
    agent_used:     str           # which agent ran: "graphrag", "code", etc.
    metadata:       dict          # agent-specific extras (scores, SQL, tool trace, sources)
    error:          str           # non-empty if something failed
```

---

## Memory pattern

Every agent follows this exact pattern — no exceptions:

```python
# BEFORE LLM call
memory_context = search_memories(user_query, user_id)   # Mem0 semantic search
system_prompt  = build_system_prompt(BASE_PROMPT, memory_context)

# ... call Ollama LLM ...

# AFTER LLM response
save_memory(user_query, response, user_id)              # Mem0 auto-compresses
```

Mem0 uses ChromaDB + nomic-embed-text as backend — zero extra services.

---

## Alzheimer's domain integration

The 22 research PDFs in `ad/rag/data/` are ingested into the **same ChromaDB collection** as the Wikipedia knowledge base. After ingestion:

- No new agent needed
- No router changes needed
- Questions about Alzheimer's automatically route to `graphrag_agent`
- Retrieved chunks come from both Wikipedia AND the research papers
- `metadata.sources` in the response shows which PDF(s) were cited

```bash
# Ingest once
curl -X POST http://localhost:8080/ingest/alzheimers -d '{}'

# Then ask
curl -X POST http://localhost:8080/chat \
  -d '{"query": "What biomarkers indicate early Alzheimer disease?", "user_id": "test"}'
```

### What changed vs. the naive RAG in ad/rag/

| Component | Naive RAG (ad/rag/) | MAO |
|-----------|---------------------|-----|
| Vector store | FAISS | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 (384-dim) | nomic-embed-text via Ollama (768-dim) |
| LLM | Groq API (paid, cloud) | llama3.1:8b via Ollama (local, free) |
| Retrieval | Simple vector similarity | 5-step GraphRAG + reranker |
| External tools | Wikipedia/arXiv/PubMed/Tavily | tool_agent (DuckDuckGo + Wikipedia) |
| API keys required | GROQ_API_KEY, TAVILY_API_KEY, HF_TOKEN | none |

---

## Models

| Model | Provider | Purpose | Approx. RAM |
|-------|----------|---------|------------|
| `mistral` | Ollama | Routing, SQL, tools, summarization | ~5 GB |
| `llama3.1:8b` | Ollama | Factual QA, code, critique | ~5 GB |
| `nomic-embed-text` | Ollama | Text embeddings (768-dim) | ~300 MB |
| `llava` | Ollama | Image understanding | ~4 GB |
| `BAAI/bge-reranker-v2-m3` | HuggingFace (local) | Reranking | ~600 MB |
| `whisper-base` | OpenAI Whisper (local) | Audio transcription | ~150 MB |

---

## Infrastructure

| Service | Port | Purpose |
|---------|------|---------|
| Ollama | 11434 | LLM + embedding inference |
| ChromaDB | 8000 | Vector store + Mem0 backend |
| PostgreSQL | 5432 | Structured data (SQL agent) |
| MAO API | 8080 | FastAPI application |

---

## Tech stack rationale

| Choice | Why |
|--------|-----|
| NetworkX over Neo4j | Zero infra, pure Python, serializable to JSON |
| bge-reranker-v2-m3 | MTEB reranking SOTA, runs on CPU, no GPU needed |
| Mem0 over custom memory | Auto-compression + dedup; open source, production-tested |
| numexpr over eval() | `eval()` is a code injection risk; numexpr is math-only |
| subprocess for code exec | Isolated process, OS-enforced timeout, no shared interpreter |
| Separate critic agent | Prevents sycophancy — mirrors Constitutional AI / RLHF design |
| mistral for routing | Fastest local model; classification needs speed not depth |
| llama3.1:8b for reasoning | +15% on HumanEval; better multi-hop QA vs mistral |

---

## See also

- [SETUP.md](SETUP.md) — full setup instructions with every command
- [mao/core/state.py](mao/core/state.py) — state contract
- [mao/core/config.py](mao/core/config.py) — all configuration options
- [mao/graph.py](mao/graph.py) — LangGraph assembly
