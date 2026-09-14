"""Materialization is fail-closed: nothing is handed back unverified.

These tests never touch the network. The fetcher is injected, so what is
exercised is the real verification, the real cache layout and the real refusal
paths — only the HTTP call is substituted.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mao.artifacts import (
    ArtifactFetchError,
    ArtifactIntegrityError,
    artifact_cache_root,
    artifact_dir,
    parse_manifest,
    resolve,
)

REVISION = "3f6b1c2d4e5a60718293a4b5c6d7e8f901234567"
OTHER_REVISION = "0123456789abcdef0123456789abcdef01234567"
PAYLOAD = b"corpus-v1-document-bytes"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


def _manifest(revision: str = REVISION, sha: str = PAYLOAD_SHA, size: int = len(PAYLOAD)):
    return parse_manifest(
        {
            "artifact_id": "mao-corpus-source",
            "repo": "ArunMendu/mao-corpus-source",
            "repo_type": "dataset",
            "revision": revision,
            "files": [{"path": "documents.parquet", "sha256": sha, "bytes": size}],
        }
    )


class _RecordingFetcher:
    """Writes a fixed payload and records exactly what was asked for."""

    def __init__(self, payload: bytes = PAYLOAD) -> None:
        self.payload = payload
        self.calls: list[dict[str, str]] = []

    def __call__(self, *, repo: str, repo_type: str, revision: str, filename: str, dest: Path) -> None:
        self.calls.append(
            {"repo": repo, "repo_type": repo_type, "revision": revision, "filename": filename}
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.payload)


class TestCacheLocation:
    def test_never_defaults_to_tmp(self) -> None:
        """R6: ``local_dir="/tmp"`` is POSIX-only and broken on Windows."""
        root = artifact_cache_root(env={})
        assert "/tmp" not in root.as_posix()

    def test_default_root_is_absolute(self) -> None:
        assert artifact_cache_root(env={}).is_absolute()

    def test_honours_an_explicit_cache_env(self, tmp_path) -> None:
        root = artifact_cache_root(env={"MAO_ARTIFACT_CACHE": str(tmp_path / "c")})
        assert root == (tmp_path / "c").resolve()

    def test_honours_hf_home(self, tmp_path) -> None:
        root = artifact_cache_root(env={"HF_HOME": str(tmp_path / "hf")})
        assert (tmp_path / "hf").resolve() in root.parents or root == (tmp_path / "hf").resolve()

    def test_explicit_cache_env_wins_over_hf_home(self, tmp_path) -> None:
        root = artifact_cache_root(
            env={"MAO_ARTIFACT_CACHE": str(tmp_path / "c"), "HF_HOME": str(tmp_path / "hf")}
        )
        assert root == (tmp_path / "c").resolve()

    def test_artifact_dir_is_deterministic(self, tmp_path) -> None:
        first = artifact_dir(_manifest(), cache_root=tmp_path)
        second = artifact_dir(_manifest(), cache_root=tmp_path)
        assert first == second

    def test_artifact_dir_separates_revisions(self, tmp_path) -> None:
        """Two pins of the same artifact must never share a directory."""
        a = artifact_dir(_manifest(revision=REVISION), cache_root=tmp_path)
        b = artifact_dir(_manifest(revision=OTHER_REVISION), cache_root=tmp_path)
        assert a != b

    def test_artifact_dir_uses_posix_style_segments(self, tmp_path) -> None:
        """Same relative layout on every platform, so caches are portable."""
        path = artifact_dir(_manifest(), cache_root=tmp_path)
        assert path.relative_to(tmp_path).as_posix() == f"mao-corpus-source/{REVISION}"


class TestSuccessfulResolution:
    def test_returns_a_path_to_the_materialized_file(self, tmp_path) -> None:
        resolved = resolve(_manifest(), cache_root=tmp_path, fetch=_RecordingFetcher())
        assert resolved.file("documents.parquet").read_bytes() == PAYLOAD

    def test_materializes_under_the_deterministic_cache_dir(self, tmp_path) -> None:
        resolved = resolve(_manifest(), cache_root=tmp_path, fetch=_RecordingFetcher())
        assert resolved.root == artifact_dir(_manifest(), cache_root=tmp_path)

    def test_requests_the_pinned_revision_never_a_branch(self, tmp_path) -> None:
        fetcher = _RecordingFetcher()
        resolve(_manifest(), cache_root=tmp_path, fetch=fetcher)
        assert fetcher.calls[0]["revision"] == REVISION

    def test_requests_the_declared_repo_and_type(self, tmp_path) -> None:
        fetcher = _RecordingFetcher()
        resolve(_manifest(), cache_root=tmp_path, fetch=fetcher)
        assert fetcher.calls[0]["repo"] == "ArunMendu/mao-corpus-source"
        assert fetcher.calls[0]["repo_type"] == "dataset"

    def test_a_second_resolve_reuses_the_cache(self, tmp_path) -> None:
        fetcher = _RecordingFetcher()
        resolve(_manifest(), cache_root=tmp_path, fetch=fetcher)
        resolve(_manifest(), cache_root=tmp_path, fetch=fetcher)
        assert len(fetcher.calls) == 1


class TestIntegrityIsEnforced:
    def test_refuses_a_checksum_mismatch(self, tmp_path) -> None:
        fetcher = _RecordingFetcher(payload=b"tampered bytes here!!!!!")
        with pytest.raises(ArtifactIntegrityError):
            resolve(_manifest(), cache_root=tmp_path, fetch=fetcher)

    def test_refuses_a_byte_size_mismatch(self, tmp_path) -> None:
        """Size is checked independently, so a length change is caught on its own."""
        manifest = _manifest(size=len(PAYLOAD) + 1)
        with pytest.raises(ArtifactIntegrityError, match="bytes"):
            resolve(manifest, cache_root=tmp_path, fetch=_RecordingFetcher())

    def test_integrity_error_names_artifact_expected_and_actual(self, tmp_path) -> None:
        fetcher = _RecordingFetcher(payload=b"tampered bytes here!!!!!")
        with pytest.raises(ArtifactIntegrityError) as exc:
            resolve(_manifest(), cache_root=tmp_path, fetch=fetcher)
        message = str(exc.value)
        assert "mao-corpus-source" in message
        assert PAYLOAD_SHA in message
        assert hashlib.sha256(b"tampered bytes here!!!!!").hexdigest() in message

    def test_integrity_error_states_a_remedy(self, tmp_path) -> None:
        fetcher = _RecordingFetcher(payload=b"tampered bytes here!!!!!")
        with pytest.raises(ArtifactIntegrityError, match="(?i)remedy|re-?download|delete"):
            resolve(_manifest(), cache_root=tmp_path, fetch=fetcher)

    def test_a_corrupt_download_is_not_left_in_the_cache(self, tmp_path) -> None:
        """A poisoned file must not survive to be served by the next call."""
        fetcher = _RecordingFetcher(payload=b"tampered bytes here!!!!!")
        with pytest.raises(ArtifactIntegrityError):
            resolve(_manifest(), cache_root=tmp_path, fetch=fetcher)
        cached = artifact_dir(_manifest(), cache_root=tmp_path) / "documents.parquet"
        assert not cached.exists()

    def test_a_cached_file_corrupted_on_disk_is_refused(self, tmp_path) -> None:
        """Verification is on every use, not only on download."""
        resolve(_manifest(), cache_root=tmp_path, fetch=_RecordingFetcher())
        cached = artifact_dir(_manifest(), cache_root=tmp_path) / "documents.parquet"
        cached.write_bytes(b"corrupted after the fact!")
        with pytest.raises(ArtifactIntegrityError):
            resolve(_manifest(), cache_root=tmp_path, fetch=_RecordingFetcher())


class TestFetchFailureIsActionable:
    def test_wraps_a_fetch_failure(self, tmp_path) -> None:
        def broken(**_kwargs: object) -> None:
            raise OSError("404 not found")

        with pytest.raises(ArtifactFetchError, match="mao-corpus-source"):
            resolve(_manifest(), cache_root=tmp_path, fetch=broken)

    def test_fetch_failure_names_the_repo_and_revision(self, tmp_path) -> None:
        def broken(**_kwargs: object) -> None:
            raise OSError("404 not found")

        with pytest.raises(ArtifactFetchError) as exc:
            resolve(_manifest(), cache_root=tmp_path, fetch=broken)
        assert "ArunMendu/mao-corpus-source" in str(exc.value)
        assert REVISION in str(exc.value)

    def test_fetch_failure_preserves_the_cause(self, tmp_path) -> None:
        def broken(**_kwargs: object) -> None:
            raise OSError("404 not found")

        with pytest.raises(ArtifactFetchError) as exc:
            resolve(_manifest(), cache_root=tmp_path, fetch=broken)
        assert isinstance(exc.value.__cause__, OSError)

    def test_a_fetcher_that_writes_nothing_is_refused(self, tmp_path) -> None:
        def silent(**_kwargs: object) -> None:
            return None

        with pytest.raises(ArtifactFetchError):
            resolve(_manifest(), cache_root=tmp_path, fetch=silent)


class TestUnknownFile:
    def test_asking_for_a_file_not_in_the_manifest_raises(self, tmp_path) -> None:
        resolved = resolve(_manifest(), cache_root=tmp_path, fetch=_RecordingFetcher())
        with pytest.raises(KeyError):
            resolved.file("not-in-manifest.parquet")
