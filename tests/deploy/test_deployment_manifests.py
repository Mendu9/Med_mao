"""Deployment manifests must match the code and be buildable from a clean clone.

Covers P1-13 (requirements vs real imports), P1-14 (one authoritative manifest),
P1-15 (Dockerfiles reference only tracked files) and P1-16 (deploy/<target>/).
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO_ROOT / "requirements.txt"
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    ".pytest_cache", ".ruff_cache", "hf_cache", "chroma_data", "build", "dist",
    # AD/ is a separate, independently-versioned repository that happens to sit
    # inside this working tree. Its imports are not this project's dependencies.
    "AD",
    # Superseded interpreter kept for recovery, not part of the source tree.
    ".venv.broken-py312",
    "site-packages",
    # .claude/worktrees/agent-* hold complete duplicate copies of an OLDER
    # codebase (pre-Groq: ollama, duckduckgo_search, pinecone, pyvis). Scanning
    # them would make this project appear to depend on packages no current
    # module imports.
    ".claude",
}
FIRST_PARTY = {"mao", "app", "tests", "ad", "conftest", "setup", "deploy", "scripts"}

# Third-party import name -> PyPI distribution that provides it.
IMPORT_TO_DISTRIBUTION = {
    "Bio": "biopython",
    "alembic": "alembic",
    "FlagEmbedding": "FlagEmbedding",
    "asyncpg": "asyncpg",
    "chromadb": "chromadb",
    "cv2": "opencv-python-headless",
    "datasets": "datasets",
    "dotenv": "python-dotenv",
    "fastapi": "fastapi",
    "groq": "groq",
    "httpx": "httpx",
    "huggingface_hub": "huggingface_hub",
    "langchain_community": "langchain-community",
    "langchain_groq": "langchain-groq",
    "langchain_huggingface": "langchain-huggingface",
    "langgraph": "langgraph",
    "langsmith": "langsmith",
    "lxml": "lxml",
    "mem0": "mem0ai",
    "networkx": "networkx",
    "nltk": "nltk",
    "numexpr": "numexpr",
    "numpy": "numpy",
    "pandas": "pandas",
    "playwright": "playwright",
    "plotly": "plotly",
    "prometheus_client": "prometheus-client",
    "pronto": "pronto",
    "psutil": "psutil",
    "pydantic": "pydantic",
    "pypdf": "pypdf",
    "pytest": "pytest",
    "qdrant_client": "qdrant-client",
    "ragas": "ragas",
    "rank_bm25": "rank-bm25",
    "redis": "redis",
    "reportlab": "reportlab",
    "requests": "requests",
    "sentence_transformers": "sentence-transformers",
    "setuptools": "setuptools",
    "spacy": "spacy",
    "sqlalchemy": "SQLAlchemy",
    "streamlit": "streamlit",
    "tensorflow": "tensorflow",
    "tiktoken": "tiktoken",
    "tqdm": "tqdm",
    "uvicorn": "uvicorn",
    "whisper": "openai-whisper",
    "wikipediaapi": "Wikipedia-API",
}

# The serving surface: everything HF Spaces and the API container actually run.
RUNTIME_ROOTS = ("mao", "app/streamlit_app.py", "app/graph_explorer.py", "app.py", "deploy")
# Modules that are deliberately NOT part of the runtime manifest.
NOT_RUNTIME = {
    "pytest", "playwright", "setuptools",      # dev/test tooling
    "tensorflow", "cv2", "whisper",            # heavyweight optional models
    "chromadb", "datasets", "Bio", "lxml",     # ingestion / non-default backend
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _python_files(roots: tuple[str, ...] | None = None) -> list[Path]:
    if roots is None:
        candidates = sorted(REPO_ROOT.rglob("*.py"))
    else:
        candidates = []
        for root in roots:
            target = REPO_ROOT / root
            if target.is_dir():
                candidates.extend(sorted(target.rglob("*.py")))
            elif target.is_file():
                candidates.append(target)
    return [p for p in candidates if not any(part in SKIP_DIRS for part in p.parts)]


class _ImportVisitor(ast.NodeVisitor):
    """Collect top-level module names, split into hard (module scope) and lazy."""

    def __init__(self) -> None:
        self.hard: set[str] = set()
        self.lazy: set[str] = set()
        self._depth = 0
        self._guarded = 0

    def _record(self, dotted: str) -> None:
        top = dotted.split(".")[0]
        if top in sys.stdlib_module_names or top in FIRST_PARTY:
            return
        (self.lazy if (self._depth or self._guarded) else self.hard).add(top)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not node.level and node.module:
            self._record(node.module)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Try(self, node: ast.Try) -> None:
        self._guarded += 1
        self.generic_visit(node)
        self._guarded -= 1


def _scan_imports(roots: tuple[str, ...] | None = None) -> tuple[set[str], set[str]]:
    visitor = _ImportVisitor()
    for path in _python_files(roots):
        try:
            visitor.visit(ast.parse(path.read_text(encoding="utf-8", errors="replace")))
        except SyntaxError:  # pragma: no cover - repo is expected to parse
            continue
    return visitor.hard, visitor.lazy


_REQ_LINE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")


def _declared(path: Path) -> set[str]:
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _REQ_LINE.match(line)
        if match:
            names.add(match.group(1).lower().replace("_", "-"))
    return names


def _tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    return set(out.stdout.splitlines())


# ---------------------------------------------------------------------------
# P1-13 — requirements.txt must cover every import the runtime actually makes
# ---------------------------------------------------------------------------

def _normalise(distribution: str) -> str:
    return distribution.lower().replace("_", "-")


def _all_manifests() -> set[str]:
    return (
        _declared(REQUIREMENTS)
        | _declared(REPO_ROOT / "requirements-dev.txt")
        | _declared(REPO_ROOT / "requirements-optional.txt")
    )


def test_every_third_party_import_is_declared_somewhere() -> None:
    """No import in the repo may be satisfied by luck (P1-13)."""
    hard, lazy = _scan_imports()
    declared = _all_manifests()
    unmapped = sorted(m for m in (hard | lazy) if m not in IMPORT_TO_DISTRIBUTION)
    assert not unmapped, f"import scan found unmapped third-party modules: {unmapped}"
    missing = sorted(
        IMPORT_TO_DISTRIBUTION[m]
        for m in (hard | lazy)
        if _normalise(IMPORT_TO_DISTRIBUTION[m]) not in declared
    )
    assert not missing, f"imported but declared in no manifest: {missing}"


def test_runtime_imports_are_in_the_authoritative_manifest() -> None:
    """Anything the served app imports must be in requirements.txt itself (P1-13)."""
    hard, lazy = _scan_imports(RUNTIME_ROOTS)
    declared = _declared(REQUIREMENTS)
    missing = sorted(
        IMPORT_TO_DISTRIBUTION[m]
        for m in (hard | lazy)
        if m not in NOT_RUNTIME and _normalise(IMPORT_TO_DISTRIBUTION[m]) not in declared
    )
    assert not missing, f"runtime imports missing from requirements.txt: {missing}"


@pytest.mark.parametrize(
    "distribution",
    [
        "psycopg2-binary",  # SQLAlchemy DBAPI for postgresql:// — /health depends on it
        # asyncpg was required only by the raw-connection /feedback and /export
        # endpoints. Both now go through mao/db/repository.py on the shared
        # SQLAlchemy engine (P1-7, P1-10), so nothing imports it any more.
        "streamlit",        # app/streamlit_app.py — the HF Space UI
        "nltk",             # mao/rag/chunker.py sentence splitting
        "spacy",            # mao/rag/graph_builder.py NER (retrieval steps 5-7)
        "huggingface-hub",  # mao/rag/retriever.py, mao/models/mri_predictor.py
        "prometheus-client",  # mao/monitoring/metrics.py, /metrics endpoint
        "tiktoken",         # mao/core/token_counter.py (module-level import)
    ],
)
def test_named_runtime_dependency_is_declared(distribution: str) -> None:
    assert distribution.lower() in _declared(REQUIREMENTS)


def test_spacy_model_is_installed_by_the_manifest() -> None:
    """spaCy NER is dead without a model; HF Spaces has no post-install hook."""
    text = REQUIREMENTS.read_text(encoding="utf-8")
    assert "en_core_web_sm" in text


def test_pytest_is_not_a_runtime_dependency() -> None:
    """Test tooling belongs in requirements-dev.txt, never in the deployed image."""
    declared = _declared(REQUIREMENTS)
    assert "pytest" not in declared
    assert "playwright" not in declared
    assert "pytest" in _declared(REPO_ROOT / "requirements-dev.txt")


def test_heavy_optional_extras_are_documented_not_silently_dropped() -> None:
    optional = REPO_ROOT / "requirements-optional.txt"
    assert optional.exists(), "heavyweight extras must live in a documented extras file"
    declared = _declared(optional)
    for distribution in ("tensorflow", "opencv-python-headless", "chromadb"):
        assert distribution in declared, f"{distribution} must be declared as an opt-in extra"


def test_requirements_declares_nothing_the_code_never_imports() -> None:
    hard, lazy = _scan_imports()
    used = {
        IMPORT_TO_DISTRIBUTION[m].lower().replace("_", "-")
        for m in (hard | lazy)
        if m in IMPORT_TO_DISTRIBUTION
    }
    # Distributions legitimately declared without a direct import: resolver
    # pins, DBAPI drivers, dynamically-imported names and spaCy model wheels.
    allowed_indirect = {
        "grpcio",           # resolver floor for cp312/cp313 wheels
        "protobuf",         # ditto (streamlit requires >=5)
        "transformers",     # version floor for sentence-transformers/FlagEmbedding
        "psycopg2-binary",  # SQLAlchemy DBAPI, selected by the postgresql:// URL
        "en-core-web-sm",   # spaCy model wheel, loaded by name via spacy.load()
        "langchain-core",   # version floor shared by langgraph and langchain-groq
        "ddgs",             # imported via importlib by mao/core/web_search.py
    }
    declared = _declared(REQUIREMENTS)
    unused = sorted(declared - used - allowed_indirect)
    assert not unused, f"declared but never imported: {unused}"


# ---------------------------------------------------------------------------
# P1-14 — exactly one authoritative runtime manifest
# ---------------------------------------------------------------------------

def test_stale_hf_manifest_is_retired() -> None:
    assert not (REPO_ROOT / "requirements-hf.txt").exists(), (
        "requirements-hf.txt was stale and unused; requirements.txt is authoritative"
    )
    assert "requirements-hf.txt" not in _tracked_files()


def test_requirements_states_that_it_is_authoritative() -> None:
    header = REQUIREMENTS.read_text(encoding="utf-8")[:1200]
    assert "authoritative" in header.lower()


# ---------------------------------------------------------------------------
# P1-15 — Dockerfiles must build from a clean clone
# ---------------------------------------------------------------------------

_COPY_RE = re.compile(r"^\s*COPY\s+(?:--[^\s]+\s+)*(.+)$", re.IGNORECASE)


def _copy_sources(dockerfile: Path) -> list[str]:
    sources: list[str] = []
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        match = _COPY_RE.match(line)
        if not match:
            continue
        parts = match.group(1).split()
        sources.extend(parts[:-1])  # last token is the destination
    return sources


@pytest.mark.parametrize("name", ["Dockerfile", "deploy/huggingface/Dockerfile"])
def test_dockerfile_copies_only_tracked_paths(name: str) -> None:
    dockerfile = REPO_ROOT / name
    assert dockerfile.exists(), f"{name} is missing"
    tracked = _tracked_files()
    for source in _copy_sources(dockerfile):
        if source in {".", "./"}:
            continue
        path = (REPO_ROOT / source).resolve()
        assert path.exists(), f"{name} COPYs {source!r} which does not exist"
        rel = path.relative_to(REPO_ROOT).as_posix()
        assert rel in tracked, f"{name} COPYs {source!r} which is not tracked by git"


def test_dockerignore_keeps_secrets_and_caches_out_of_images() -> None:
    """Both Dockerfiles `COPY . .`; without a .dockerignore that bakes .env in."""
    dockerignore = REPO_ROOT / ".dockerignore"
    assert dockerignore.exists(), "COPY . . needs a .dockerignore or .env ships in the image"
    patterns = {
        line.strip()
        for line in dockerignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for required in (".env", ".git", "hf_cache", ".venv"):
        assert required in patterns, f".dockerignore must exclude {required}"


def test_dockerfile_spacy_download_has_spacy_installed() -> None:
    """Dockerfile:20 ran `spacy download` after spacy was dropped from requirements (P1-15)."""
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    if "spacy download" in text:
        assert "spacy" in _declared(REQUIREMENTS)


def _gitignore_patterns() -> list[str]:
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def test_start_hf_script_is_tracked_not_ignored() -> None:
    """The HF launcher is deployment SOURCE — it must be in version control (P1-15)."""
    tracked = _tracked_files()
    assert "deploy/huggingface/start_hf.sh" in tracked
    assert "deploy/huggingface/Dockerfile" in tracked


def test_no_unanchored_pattern_can_swallow_the_deploy_glue() -> None:
    """A bare `start_hf.sh` line matches at every depth, including under deploy/.

    That is what excluded the deploy source in the first place (P1-15). A
    root-anchored `/scripts/start_hf.sh` is fine — it hides the superseded copy
    at the repo root and cannot reach `deploy/huggingface/`. So the rule is
    about anchoring, not about mentioning the filename.
    """
    risky = [
        p
        for p in _gitignore_patterns()
        if ("start_hf.sh" in p or "Dockerfile.hf" in p) and not p.startswith("/")
    ]
    assert not risky, f"unanchored deploy-glue patterns: {risky}"


def test_gitignore_does_not_exclude_the_deploy_tree() -> None:
    check = subprocess.run(
        ["git", "check-ignore", "-q", "deploy/huggingface/start_hf.sh"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert check.returncode == 1, "deploy/huggingface/start_hf.sh is gitignored"


# ---------------------------------------------------------------------------
# P1-16 — deployment glue lives under deploy/<target>/
# ---------------------------------------------------------------------------

def test_huggingface_deployment_assets_live_under_deploy() -> None:
    target = REPO_ROOT / "deploy" / "huggingface"
    assert (target / "Dockerfile").exists()
    assert (target / "start_hf.sh").exists()
    assert (target / "space_entrypoint.py").exists()


def test_deploy_documents_the_authoritative_entrypoint_per_target() -> None:
    readme = (REPO_ROOT / "deploy" / "README.md").read_text(encoding="utf-8")
    for token in ("app.py", "Dockerfile", "gunicorn", "huggingface"):
        assert token in readme, f"deploy/README.md must document {token}"


def test_hf_space_entrypoint_still_resolves() -> None:
    """HF reads app_file from README front-matter; that file must exist at the root."""
    front_matter = (REPO_ROOT / "README.md").read_text(encoding="utf-8").split("---")[1]
    match = re.search(r"^app_file:\s*(\S+)", front_matter, re.MULTILINE)
    assert match, "README front-matter must declare app_file for HF Spaces"
    assert (REPO_ROOT / match.group(1)).exists()


def test_root_app_py_delegates_to_the_deploy_target() -> None:
    """Root app.py stays a thin shim; the HF logic lives under deploy/huggingface/."""
    text = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    assert "deploy" in text and "huggingface" in text
    assert len(text.splitlines()) < 40, "app.py must stay a shim, not a second entrypoint"


def test_no_duplicate_two_process_launcher_at_the_root() -> None:
    """start.sh and scripts/start_hf.sh were two divergent copies of one script (P1-16).

    Asserted against *tracked* state, not the filesystem. The old copy was
    gitignored and untracked, so a fresh clone never had it; a stale leftover in
    one developer's working tree is not a repository defect, and a filesystem
    assertion would make this test pass or fail by accident of local history.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "scripts/start_hf.sh", "start.sh"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    ).stdout.split()
    assert tracked == [], f"duplicate launcher still tracked: {tracked}"
    assert (REPO_ROOT / "deploy" / "huggingface" / "start_hf.sh").exists()
