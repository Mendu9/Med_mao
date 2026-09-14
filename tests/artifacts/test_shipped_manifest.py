"""The manifests that actually ship, and proof that CI needs no network.

Two separate concerns share this file because they share a subject: the pins in
Git. The first group checks the production pin is well-formed and points where
it should. The second proves the whole materialization path can run offline on a
committed fixture, so unit CI never downloads a production artifact.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mao.artifacts import load_manifest, resolve

_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_MANIFEST = _ROOT / "manifests" / "corpus.manifest.json"
FIXTURE_DIR = _ROOT / "tests" / "fixtures" / "corpus"
FIXTURE_MANIFEST = FIXTURE_DIR / "corpus_v1_sample.manifest.json"


class TestShippedCorpusPin:
    def test_the_shipped_manifest_parses(self) -> None:
        assert load_manifest(SHIPPED_MANIFEST).artifact_id == "mao-corpus-source"

    def test_it_pins_an_immutable_revision(self) -> None:
        """Parsing enforces this; asserting it here states the intent outright."""
        revision = load_manifest(SHIPPED_MANIFEST).revision
        assert len(revision) == 40 and int(revision, 16) >= 0

    def test_it_points_at_a_dataset_repo_not_the_space(self) -> None:
        manifest = load_manifest(SHIPPED_MANIFEST)
        assert manifest.repo_type == "dataset"
        assert "spaces" not in manifest.repo

    def test_it_declares_the_584_document_corpus(self) -> None:
        manifest = load_manifest(SHIPPED_MANIFEST)
        documents = next(f for f in manifest.files if f.path == "documents.parquet")
        assert documents.rows == 584

    def test_it_does_not_publish_the_legacy_blob(self) -> None:
        """The 161 MB unlicensed chunk corpus must never appear in a pin."""
        manifest = load_manifest(SHIPPED_MANIFEST)
        assert not any("bm25_corpus" in path for path in manifest.paths)

    def test_every_declared_file_carries_a_hash_and_a_size(self) -> None:
        for entry in load_manifest(SHIPPED_MANIFEST).files:
            assert len(entry.sha256) == 64
            assert entry.size_bytes > 0


class _LocalFetcher:
    """Serves files from a directory — the offline stand-in for the Hub."""

    def __init__(self, source: Path) -> None:
        self.source = source
        self.calls = 0

    def __call__(self, *, filename: str, dest: Path, **_kwargs: object) -> None:
        self.calls += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.source / filename, dest)


class TestOfflineFixture:
    def test_the_fixture_manifest_parses(self) -> None:
        assert load_manifest(FIXTURE_MANIFEST).artifact_id == "mao-corpus-sample"

    def test_the_fixture_artifact_resolves_with_no_network(self, tmp_path) -> None:
        manifest = load_manifest(FIXTURE_MANIFEST)
        resolved = resolve(manifest, cache_root=tmp_path, fetch=_LocalFetcher(FIXTURE_DIR))
        assert resolved.file("corpus_v1_sample.parquet").is_file()

    def test_the_fixture_hashes_match_the_committed_bytes(self, tmp_path) -> None:
        """If the committed sample changes, this fails — the pin is real."""
        manifest = load_manifest(FIXTURE_MANIFEST)
        resolve(manifest, cache_root=tmp_path, fetch=_LocalFetcher(FIXTURE_DIR))

    def test_the_fixture_is_small_enough_for_git(self) -> None:
        sample = FIXTURE_DIR / "corpus_v1_sample.parquet"
        assert sample.stat().st_size < 1_000_000

    def test_a_warm_cache_triggers_no_fetch_at_all(self, tmp_path) -> None:
        """Once materialized, resolution is pure verification — no egress."""
        manifest = load_manifest(FIXTURE_MANIFEST)
        fetcher = _LocalFetcher(FIXTURE_DIR)
        resolve(manifest, cache_root=tmp_path, fetch=fetcher)
        assert fetcher.calls == 1

        def refuse(**_kwargs: object) -> None:
            raise AssertionError("unit tests must not download a production artifact")

        resolve(manifest, cache_root=tmp_path, fetch=refuse)


class TestSampleIsUsableAsACorpus:
    def test_the_sample_carries_the_full_document_schema(self, tmp_path) -> None:
        pq = pytest.importorskip("pyarrow.parquet")
        manifest = load_manifest(FIXTURE_MANIFEST)
        resolved = resolve(manifest, cache_root=tmp_path, fetch=_LocalFetcher(FIXTURE_DIR))
        table = pq.read_table(resolved.file("corpus_v1_sample.parquet"))
        for column in ("pmcid", "title", "license_id", "text", "sha256"):
            assert column in table.column_names

    def test_no_sample_document_is_unlicensed(self, tmp_path) -> None:
        pq = pytest.importorskip("pyarrow.parquet")
        manifest = load_manifest(FIXTURE_MANIFEST)
        resolved = resolve(manifest, cache_root=tmp_path, fetch=_LocalFetcher(FIXTURE_DIR))
        table = pq.read_table(resolved.file("corpus_v1_sample.parquet"))
        assert "UNKNOWN" not in set(table.column("license_id").to_pylist())
