---
title: MAO Clinical AI Assistant
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.40.0"
app_file: app.py
pinned: false
license: mit
---

# MAO — Medical Multi-Agent Orchestrator

An evidence-grounded clinical AI system that routes biomedical queries to specialised agents.
Synthesis runs on hosted Groq models addressed by capability role; embeddings, reranking and
NLI run locally.

> **Note:** parts of this README below the architecture section still describe an older
> Ollama-based topology and are being reconciled. Code is authoritative.

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
│   graphrag  summarize   tool   critic  multimodal  clinical           │     │
│      │          │        │       │         │          │              │     │
│      ▼          ▼        ▼       ▼         ▼          ▼              │     │
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
    ▼  1. Query expansion    → synonym/ontology expansion
    │
    ▼  2. BM25 sparse search → rank-bm25 keyword retrieval
    │
    ▼  3. Vector search      → ChromaDB, top-20 candidates
    │
    ▼  4. RRF merge          → Reciprocal Rank Fusion of BM25 + vector
    │
    ▼  5. Entity extraction  → spaCy NER on merged chunks
    │
    ▼  6. Graph traversal    → NetworkX BFS, 2 hops, expand entity neighbours
    │
    ▼  7. Rerank             → BAAI/bge-reranker-v2-m3, top-5
    │
    ▼  LLM generation        → llama3.1:8b with context + memory
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
│   │   └── retriever.py            ← retrieve() — 9-step GraphRAG pipeline
│   │
│   ├── agents/
│   │   ├── router.py               ← entry node: Mem0 pre-load + mistral intent classification
│   │   ├── graphrag_agent.py       ← factual Q&A: full GraphRAG pipeline
│   │   ├── summarizer_agent.py     ← map-reduce summarization
│   │   ├── tool_agent.py           ← ReAct loop: DuckDuckGo + Wikipedia API + calculator
│   │   ├── critic_agent.py         ← evidence-oriented critique of a draft answer
│   │   ├── clinical_agent.py       ← MRI analysis + Alzheimer's stage prediction
│   │   └── multimodal_agent.py     ← image via llava + audio transcription via Whisper
│   │
│   ├── api/
│   │   └── main.py                 ← FastAPI: /chat /ingest /health /graph
│   │
│   ├── data/
│   │   ├── ingest_wikipedia.py     ← Wikipedia articles → ChromaDB + entity graph
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
| **router** | *(entry)* | `ROUTER_FAST` | Classifies intent; loads Mem0 memory for all downstream agents |
| **graphrag** | `graphrag`, `fallback` | `GENERAL_SYNTHESIS` | Answers biomedical questions — 9-step GraphRAG over research papers + Wikipedia |
| **summarizer** | `summarize` | `GENERAL_SYNTHESIS` | Condenses pasted text or KB content; map-reduce for long texts |
| **tool** | `tool` | `GENERAL_SYNTHESIS` | ReAct loop: DuckDuckGo web search, Wikipedia API, safe calculator |
| **critic** | `critic` | `GENERAL_SYNTHESIS` | Evidence-oriented critique: unsupported claims, missing evidence, overstated certainty |
| **clinical** | `clinical` | `CLINICAL_SYNTHESIS` | MRI scan analysis + Alzheimer's stage prediction + structured report |
| **multimodal** | `multimodal` | `VISION` | Image description; audio transcription via Whisper |

### Intent trigger examples

```
graphrag:    "What causes tau tangles in Alzheimer's disease?"
             "How does neuroinflammation contribute to neurodegeneration?"
             "What are the risk factors for ischemic stroke?"
summarize:   "Summarize this: [paste article text]"
tool:        "What's the latest research on APOE4?"
critic:      "Review this clinical note: [paste]"
clinical:    query = "Analyse this MRI" + metadata.image_b64 = "<base64>"
multimodal:  query = "What do you see?" + metadata.image_b64 = "<base64>"
```

---

## Models

| Model | Provider | Purpose | Approx. RAM |
|-------|----------|---------|------------|
| `ROUTER_FAST` / `GENERAL_SYNTHESIS` | Groq | Routing, extraction, general synthesis | hosted |
| `CLINICAL_SYNTHESIS` / `SAFETY_JUDGE` | Groq | Clinical synthesis, safety council, judge | hosted |
| `VISION` | Groq | Image understanding | hosted |
| `NeuML/pubmedbert-base-embeddings` | local | Text embeddings (768-dim) | ~400 MB |
| `BAAI/bge-reranker-v2-m3` | HuggingFace (local) | Reranking | ~600 MB |
| `whisper-base` | OpenAI Whisper (local) | Audio transcription | ~150 MB |

Business logic requests a **role**, never a literal model id. Roles are bound to concrete
models in `mao/providers/registry.py` and overridden per role via `MAO_MODEL_<ROLE>`.

---

## Infrastructure

| Service | Port | Purpose |
|---------|------|---------|
| ChromaDB | 8000 | Vector store + Mem0 backend |
| PostgreSQL | 5432 | Chat sessions, feedback, audit trail |
| MAO API | 8080 | FastAPI application |

---

## Tech stack rationale

| Choice | Why |
|--------|-----|
| NetworkX over Neo4j | Zero infra, pure Python, serializable to JSON |
| bge-reranker-v2-m3 | MTEB reranking SOTA, runs on CPU, no GPU needed |
| Mem0 over custom memory | Auto-compression + dedup; open source, production-tested |
| numexpr over eval() | `eval()` is a code injection risk; numexpr is math-only |
| Separate critic agent | Prevents sycophancy — mirrors Constitutional AI / RLHF design |
| mistral for routing | Fastest local model; classification needs speed not depth |
| llama3.1:8b for reasoning | Better multi-hop biomedical QA vs mistral |
| BM25 + RRF | Hybrid retrieval improves recall for rare clinical terms and gene names |

---

## See also

- [SETUP.md](SETUP.md) — full setup instructions with every command
- [mao/core/state.py](mao/core/state.py) — state contract
- [mao/core/config.py](mao/core/config.py) — all configuration options
- [mao/graph.py](mao/graph.py) — LangGraph assembly
