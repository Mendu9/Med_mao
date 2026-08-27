"""The MRI model cache must stay inside the repo and honour HF_HOME (P2-8)."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _reload(monkeypatch: pytest.MonkeyPatch, hf_home: str | None):
    if hf_home is None:
        monkeypatch.delenv("HF_HOME", raising=False)
    else:
        monkeypatch.setenv("HF_HOME", hf_home)
    module = importlib.import_module("mao.models.mri_predictor")
    return importlib.reload(module)


def test_cache_dir_stays_inside_the_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """parents[3] resolved one level ABOVE the repo root — an off-by-one (P2-8)."""
    module = _reload(monkeypatch, None)
    cache_dir = module.cache_dir()
    assert cache_dir == REPO_ROOT / "hf_cache", (
        f"cache dir {cache_dir} escaped the repo root {REPO_ROOT}"
    )
    assert REPO_ROOT in cache_dir.parents or cache_dir == REPO_ROOT


def test_cache_dir_honours_hf_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """HF Spaces sets HF_HOME=/tmp/hf_cache; the predictor must not ignore it (P2-8)."""
    module = _reload(monkeypatch, str(tmp_path / "hf"))
    assert module.cache_dir() == tmp_path / "hf"


def test_blank_hf_home_falls_back_to_the_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload(monkeypatch, "   ")
    assert module.cache_dir() == REPO_ROOT / "hf_cache"


def test_module_constant_matches_the_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload(monkeypatch, None)
    assert module._CACHE_DIR == module.cache_dir()
