"""Building the pin from real bytes on disk.

The manifest is only trustworthy if it was derived from the same bytes that were
uploaded. These tests pin that the builder hashes the actual file rather than
copying a number from anywhere else, and that whatever it emits is accepted by
the same validator the runtime uses.
"""
from __future__ import annotations

import hashlib

import pytest

from mao.artifacts import ManifestError, parse_manifest
from mao.artifacts.publish import build_manifest

REVISION = "3f6b1c2d4e5a60718293a4b5c6d7e8f901234567"


@pytest.fixture()
def artifact_files(tmp_path):
    (tmp_path / "documents.parquet").write_bytes(b"parquet-bytes")
    (tmp_path / "corpus_manifest.json").write_bytes(b"{}")
    return tmp_path


class TestBuildManifest:
    def test_hashes_the_actual_file_on_disk(self, artifact_files) -> None:
        manifest = build_manifest(
            artifact_id="mao-corpus-source",
            repo="ArunMendu/mao-corpus-source",
            revision=REVISION,
            root=artifact_files,
            paths=["documents.parquet"],
        )
        entry = manifest["files"][0]
        assert entry["sha256"] == hashlib.sha256(b"parquet-bytes").hexdigest()

    def test_records_the_actual_byte_size(self, artifact_files) -> None:
        manifest = build_manifest(
            artifact_id="mao-corpus-source",
            repo="ArunMendu/mao-corpus-source",
            revision=REVISION,
            root=artifact_files,
            paths=["documents.parquet"],
        )
        assert manifest["files"][0]["bytes"] == len(b"parquet-bytes")

    def test_includes_every_requested_path(self, artifact_files) -> None:
        manifest = build_manifest(
            artifact_id="mao-corpus-source",
            repo="ArunMendu/mao-corpus-source",
            revision=REVISION,
            root=artifact_files,
            paths=["documents.parquet", "corpus_manifest.json"],
        )
        assert [f["path"] for f in manifest["files"]] == [
            "corpus_manifest.json",
            "documents.parquet",
        ]

    def test_carries_row_counts_when_supplied(self, artifact_files) -> None:
        manifest = build_manifest(
            artifact_id="mao-corpus-source",
            repo="ArunMendu/mao-corpus-source",
            revision=REVISION,
            root=artifact_files,
            paths=["documents.parquet"],
            rows={"documents.parquet": 584},
        )
        assert manifest["files"][0]["rows"] == 584

    def test_output_is_accepted_by_the_runtime_validator(self, artifact_files) -> None:
        """Whatever the publisher emits, the resolver must be able to parse."""
        manifest = build_manifest(
            artifact_id="mao-corpus-source",
            repo="ArunMendu/mao-corpus-source",
            revision=REVISION,
            root=artifact_files,
            paths=["documents.parquet", "corpus_manifest.json"],
        )
        parsed = parse_manifest(manifest)
        assert parsed.revision == REVISION
        assert len(parsed.files) == 2

    def test_refuses_to_pin_a_floating_revision(self, artifact_files) -> None:
        """A publisher that could emit 'main' would reintroduce R6."""
        with pytest.raises(ManifestError, match="revision"):
            build_manifest(
                artifact_id="mao-corpus-source",
                repo="ArunMendu/mao-corpus-source",
                revision="main",
                root=artifact_files,
                paths=["documents.parquet"],
            )

    def test_refuses_a_missing_file(self, artifact_files) -> None:
        with pytest.raises(FileNotFoundError):
            build_manifest(
                artifact_id="mao-corpus-source",
                repo="ArunMendu/mao-corpus-source",
                revision=REVISION,
                root=artifact_files,
                paths=["absent.parquet"],
            )

    def test_is_deterministic_for_the_same_inputs(self, artifact_files) -> None:
        kwargs = {
            "artifact_id": "mao-corpus-source",
            "repo": "ArunMendu/mao-corpus-source",
            "revision": REVISION,
            "root": artifact_files,
            "paths": ["corpus_manifest.json", "documents.parquet"],
        }
        first = build_manifest(**kwargs)
        second = build_manifest(**kwargs)
        assert first == second

    def test_defaults_to_a_dataset_repo(self, artifact_files) -> None:
        manifest = build_manifest(
            artifact_id="mao-corpus-source",
            repo="ArunMendu/mao-corpus-source",
            revision=REVISION,
            root=artifact_files,
            paths=["documents.parquet"],
        )
        assert manifest["repo_type"] == "dataset"
