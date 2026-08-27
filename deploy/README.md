# Deployment targets

Deployment differences live here, one directory per target. Core application
logic lives in `mao/` and `app/` and is identical for every target — nothing in
this tree may fork behaviour per environment, only wiring (P1-16).

Before this directory existed there were four divergent entrypoints with no
statement of which one was real: `app.py`, `start.sh`, `Dockerfile` and an
untracked `Dockerfile.hf`. The table below is now the single answer.

## Which entrypoint is authoritative

| Target | Entrypoint | Processes | Ports | Notes |
|---|---|---|---|---|
| **HuggingFace Space (live)** | `app.py` → `deploy/huggingface/space_entrypoint.py` | one: Streamlit, with FastAPI on a background thread | UI 7860 (HF-managed), API 8080 (in-process) | **This is what actually runs today.** HF reads `app_file: app.py` and `sdk: streamlit` from the README front-matter, installs `requirements.txt` and the apt packages in `packages.txt`. |
| **HuggingFace Space (Docker SDK)** | `deploy/huggingface/Dockerfile` → `deploy/huggingface/start_hf.sh` | two: uvicorn + Streamlit | UI 7860, API 8080 | Alternative. Requires switching the Space to `sdk: docker`; HF's Docker SDK expects the Dockerfile at the repo root, so this file must be copied or symlinked there when the switch is made. |
| **Self-hosted API** | root `Dockerfile` → `gunicorn` + `gunicorn.conf.py` | N gunicorn workers | API 8080 | API only, no UI. Workers default to `2 * CPU + 1`; override with `MAO_WORKERS`. Set `MAO_DISABLE_RERANKER=1` to drop ~568 MB per worker. |
| **Local dev services** | `docker-compose.yml` | Postgres | 5432 | Backing services only; run the app from your own shell. |

`app.py` must stay at the repo root — HF resolves `app_file` relative to the
repo. It is a shim and nothing else; the HF-specific toggles, the FastAPI
thread and the Qdrant keep-alive all live in `space_entrypoint.py`.

## Dependency manifests

`requirements.txt` is authoritative for **every** target. There is no per-target
requirements file: `requirements-hf.txt` was retired because it had drifted from
reality (it claimed "FastAPI server not running on HF Spaces", which stopped
being true once `app.py` started it on 8080) and nothing ever installed it
(P1-14).

| File | Installed by | Contents |
|---|---|---|
| `requirements.txt` | every target | the full serving path: API, Streamlit UI, GraphRAG retrieval, spaCy NER + model, DB drivers, metrics |
| `requirements-optional.txt` | nobody by default | opt-in extras: TensorFlow/OpenCV MRI prediction, Whisper audio, ChromaDB backend, corpus-ingestion tooling, the legacy Gradio UI |
| `requirements-dev.txt` | developers and CI | pytest, ruff, playwright |

`tests/deploy/test_deployment_manifests.py` reconciles these against an AST
import scan of the repository and fails if a runtime import has no declaration,
if a declaration has no import, or if a Dockerfile `COPY`s an untracked path.

## Environment variables that change deployment behaviour

| Variable | Default | Effect |
|---|---|---|
| `MAO_API_PORT` | `8080` | Port the FastAPI backend binds |
| `PORT` | `7860` | Streamlit UI port (Docker SDK target) |
| `HF_HOME` | `/tmp/hf_cache` on Spaces, `<repo>/hf_cache` otherwise | Model download cache, honoured by the embedder, the reranker and the MRI predictor |
| `MAO_WORKERS` | `2 * CPU + 1` | gunicorn worker count (self-hosted target) |
| `MAO_DISABLE_RERANKER` | `0` | Skip the reranker. Part of the retrieval cache key, so toggling it cannot serve stale results |
| `MAO_DISABLE_BM25` | `1` on Spaces | BM25 over 154K docs costs 4+ min/query on free CPU |
| `MAO_DISABLE_MEM0` | `1` on Spaces | Mem0 needs a local ChromaDB service |
| `MAO_INDEX_VERSION` | `v1` | Bump after re-ingesting the corpus to invalidate every cached retrieval |
| `VECTOR_BACKEND` | `qdrant` | `chromadb` requires the optional extra |
