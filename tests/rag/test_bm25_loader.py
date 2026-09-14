"""The BM25 loader must degrade safely and must never reach the network.

R6 recorded three defects in this loader: it called ``hf_hub_download`` with no
``revision=`` (silently tracking a floating ``main``), it verified no checksum,
and it cached into ``local_dir="/tmp"``, which does not exist on Windows.

The corpus it downloaded is the legacy 161 MB blob that the P2-0 audit found
unpublishable — 35% of its PMC documents carry no redistribution right. So the
fix is not to pin that download but to remove it: without a local corpus the
retriever degrades to dense-only retrieval, which it already supports.

These tests are the regression guard against any of that returning.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from mao.rag import retriever


@pytest.fixture()
def fresh_loader(monkeypatch):
    """Reset the module-level BM25 state so each test starts unloaded."""
    monkeypatch.setattr(retriever, "_bm25_loaded", False, raising=False)
    monkeypatch.setattr(retriever, "_bm25_index", None, raising=False)
    monkeypatch.setattr(retriever, "_bm25_corpus", [], raising=False)
    monkeypatch.setattr(retriever, "_bm25_chunk_ids", [], raising=False)
    monkeypatch.setattr(retriever, "_bm25_metadata", [], raising=False)
    monkeypatch.delenv("MAO_DISABLE_BM25", raising=False)
    return retriever


class TestDegradesWithoutACorpus:
    """V2: a fresh clone has no corpus on disk. That must not be fatal."""

    def test_loading_without_a_corpus_does_not_raise(self, fresh_loader, tmp_path) -> None:
        monkey_path = tmp_path / "absent.json"
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(retriever, "_BM25_JSON_PATH", monkey_path)
            mp.setattr(retriever, "_BM25_PICKLE_PATH", tmp_path / "absent.pkl")
            retriever._load_bm25_from_disk()

    def test_loading_without_a_corpus_marks_the_attempt_complete(
        self, fresh_loader, tmp_path
    ) -> None:
        """``_bm25_loaded`` must be set, or every query retries the load."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(retriever, "_BM25_JSON_PATH", tmp_path / "absent.json")
            mp.setattr(retriever, "_BM25_PICKLE_PATH", tmp_path / "absent.pkl")
            retriever._load_bm25_from_disk()
        assert retriever._bm25_loaded is True

    def test_bm25_stays_disabled_rather_than_half_built(self, fresh_loader, tmp_path) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(retriever, "_BM25_JSON_PATH", tmp_path / "absent.json")
            mp.setattr(retriever, "_BM25_PICKLE_PATH", tmp_path / "absent.pkl")
            retriever._load_bm25_from_disk()
        assert retriever._bm25_index is None
        assert retriever._bm25_corpus == []


class TestNoNetworkFromTheLoader:
    def test_a_missing_corpus_triggers_no_hub_download(self, fresh_loader, tmp_path) -> None:
        """The floating, unverified download must be gone, not merely unused."""
        import huggingface_hub

        # Record rather than raise: the loader catches broadly, so an exception
        # here would be swallowed and the test would pass without proving anything.
        calls: list[dict] = []

        def record(*args: object, **kwargs: object) -> str:
            calls.append({"args": args, "kwargs": kwargs})
            return ""

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(huggingface_hub, "hf_hub_download", record)
            mp.setattr(retriever, "_BM25_JSON_PATH", tmp_path / "absent.json")
            mp.setattr(retriever, "_BM25_PICKLE_PATH", tmp_path / "absent.pkl")
            retriever._load_bm25_from_disk()

        assert calls == [], "the BM25 loader fetched an artifact; R6 forbids an unpinned fetch here"


class TestNoPosixOnlyCachePath:
    def test_the_module_hardcodes_no_tmp_directory(self) -> None:
        """``local_dir="/tmp"`` was POSIX-only and broke on this Windows box."""
        source = Path(retriever.__file__).read_text(encoding="utf-8")
        assert '"/tmp"' not in source
        assert "'/tmp'" not in source

    def test_the_module_no_longer_references_a_floating_hub_repo(self) -> None:
        source = Path(retriever.__file__).read_text(encoding="utf-8")
        assert "BM25_HF_REPO" not in source


class TestExplicitDisable:
    def test_disable_flag_skips_loading_entirely(self, fresh_loader, monkeypatch) -> None:
        monkeypatch.setenv("MAO_DISABLE_BM25", "1")
        retriever._load_bm25_from_disk()
        assert retriever._bm25_loaded is True
        assert retriever._bm25_index is None


class TestLoadsARealCorpusFromDisk:
    def test_a_local_corpus_is_still_loaded(self, fresh_loader, tmp_path) -> None:
        """Removing the download must not break the supported local path."""
        pytest.importorskip("rank_bm25")
        import json

        corpus_path = tmp_path / "bm25_corpus.json"
        corpus_path.write_text(
            json.dumps(
                {
                    "corpus": ["alpha beta gamma", "delta epsilon"],
                    "chunk_ids": ["c1", "c2"],
                    "metadata": [{"source": "a"}, {"source": "b"}],
                }
            ),
            encoding="utf-8",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(retriever, "_BM25_JSON_PATH", corpus_path)
            mp.setattr(retriever, "_BM25_PICKLE_PATH", tmp_path / "absent.pkl")
            retriever._load_bm25_from_disk()
        assert retriever._bm25_corpus == ["alpha beta gamma", "delta epsilon"]
        assert retriever._bm25_index is not None


def test_env_is_not_leaked_between_tests() -> None:
    assert os.getenv("MAO_DISABLE_BM25") in (None, "0")
