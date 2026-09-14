"""The artifact manifest is the pin. These tests pin what it refuses.

R6 recorded three defects in the legacy loader: it downloaded a floating
``main``, verified no checksum, and cached into a POSIX-only ``/tmp``. The
manifest is where the first two are made impossible — a manifest that does not
name an immutable revision and a SHA-256 for every file must not parse at all.
"""
from __future__ import annotations

import json

import pytest

from mao.artifacts import ManifestError, load_manifest, parse_manifest

REVISION = "3f6b1c2d4e5a60718293a4b5c6d7e8f901234567"
SHA256 = "1a06ca1b6a22ae28b320c4462d337ce706337379dabbd80f5b91287c2ea60290"


def _manifest(**overrides: object) -> dict:
    data = {
        "artifact_id": "mao-corpus-source",
        "repo": "ArunMendu/mao-corpus-source",
        "repo_type": "dataset",
        "revision": REVISION,
        "files": [{"path": "documents.parquet", "sha256": SHA256, "bytes": 7991287, "rows": 584}],
    }
    data.update(overrides)
    return data


class TestWellFormedManifest:
    def test_parses_the_pinned_fields(self) -> None:
        manifest = parse_manifest(_manifest())
        assert manifest.artifact_id == "mao-corpus-source"
        assert manifest.repo == "ArunMendu/mao-corpus-source"
        assert manifest.repo_type == "dataset"
        assert manifest.revision == REVISION

    def test_parses_every_file_entry(self) -> None:
        manifest = parse_manifest(_manifest())
        entry = manifest.files[0]
        assert entry.path == "documents.parquet"
        assert entry.sha256 == SHA256
        assert entry.size_bytes == 7991287
        assert entry.rows == 584

    def test_loads_from_a_json_file(self, tmp_path) -> None:
        path = tmp_path / "corpus.manifest.json"
        path.write_text(json.dumps(_manifest()), encoding="utf-8")
        assert load_manifest(path).revision == REVISION


class TestRevisionMustBeImmutable:
    """The R6 defect: ``hf_hub_download`` with no ``revision=`` tracks a branch."""

    @pytest.mark.parametrize("floating", ["main", "HEAD", "refs/heads/main", "v1.0.0"])
    def test_refuses_a_floating_revision(self, floating: str) -> None:
        with pytest.raises(ManifestError, match="revision"):
            parse_manifest(_manifest(revision=floating))

    def test_refuses_a_short_revision(self) -> None:
        with pytest.raises(ManifestError, match="revision"):
            parse_manifest(_manifest(revision=REVISION[:7]))

    def test_refuses_a_non_hex_revision(self) -> None:
        with pytest.raises(ManifestError, match="revision"):
            parse_manifest(_manifest(revision="z" * 40))

    def test_refuses_a_missing_revision(self) -> None:
        data = _manifest()
        del data["revision"]
        with pytest.raises(ManifestError, match="revision"):
            parse_manifest(data)

    def test_error_names_the_artifact_so_it_is_actionable(self) -> None:
        with pytest.raises(ManifestError, match="mao-corpus-source"):
            parse_manifest(_manifest(revision="main"))


class TestEveryFileMustCarryIntegrityFields:
    def test_refuses_a_manifest_with_no_files(self) -> None:
        with pytest.raises(ManifestError, match="files"):
            parse_manifest(_manifest(files=[]))

    def test_refuses_a_file_with_no_sha256(self) -> None:
        with pytest.raises(ManifestError, match="sha256"):
            parse_manifest(_manifest(files=[{"path": "d.parquet", "bytes": 1}]))

    def test_refuses_a_malformed_sha256(self) -> None:
        with pytest.raises(ManifestError, match="sha256"):
            parse_manifest(_manifest(files=[{"path": "d.parquet", "sha256": "abc", "bytes": 1}]))

    def test_refuses_a_file_with_no_byte_size(self) -> None:
        with pytest.raises(ManifestError, match="bytes"):
            parse_manifest(_manifest(files=[{"path": "d.parquet", "sha256": SHA256}]))

    def test_refuses_a_negative_byte_size(self) -> None:
        with pytest.raises(ManifestError, match="bytes"):
            parse_manifest(_manifest(files=[{"path": "d.parquet", "sha256": SHA256, "bytes": -1}]))

    def test_refuses_a_file_with_no_path(self) -> None:
        with pytest.raises(ManifestError, match="path"):
            parse_manifest(_manifest(files=[{"sha256": SHA256, "bytes": 1}]))


class TestPathsStayInsideTheArtifact:
    """A manifest is data; a traversing path in it must not escape the cache."""

    @pytest.mark.parametrize("hostile", ["../escape.parquet", "/etc/passwd", "a/../../b"])
    def test_refuses_a_path_that_escapes(self, hostile: str) -> None:
        with pytest.raises(ManifestError, match="path"):
            parse_manifest(_manifest(files=[{"path": hostile, "sha256": SHA256, "bytes": 1}]))

    def test_refuses_a_backslash_path_that_escapes_on_windows(self) -> None:
        with pytest.raises(ManifestError, match="path"):
            parse_manifest(_manifest(files=[{"path": r"..\escape", "sha256": SHA256, "bytes": 1}]))


class TestRepoType:
    def test_refuses_an_unknown_repo_type(self) -> None:
        with pytest.raises(ManifestError, match="repo_type"):
            parse_manifest(_manifest(repo_type="space"))

    def test_refuses_a_missing_repo(self) -> None:
        data = _manifest()
        del data["repo"]
        with pytest.raises(ManifestError, match="repo"):
            parse_manifest(data)
