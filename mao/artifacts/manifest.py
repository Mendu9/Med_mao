"""The pinned description of an external artifact.

A manifest is small, reviewable and lives in Git. The artifact it names lives in
an artifact store and does not. Everything needed to prove that the bytes handed
back are the bytes that were reviewed is here: an immutable revision, and a
SHA-256 plus byte size for every file.

The validation is deliberately unforgiving. R6 recorded a loader that tracked a
floating ``main`` with no checksum at all; a manifest that could express that
would reintroduce it, so such a manifest does not parse.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import re

#: A Git commit SHA-1, lowercase and complete. Branch names, tags and short
#: SHAs are all mutable or ambiguous and are refused.
_REVISION_RE = re.compile(r"\A[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")

#: An artifact id becomes a cache directory name, so it is restricted to
#: characters that are safe on every filesystem and cannot traverse.
_ARTIFACT_ID_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")

#: Artifact stores we address. The live Space is deliberately absent: artifacts
#: are never served from the deployment target.
REPO_TYPES: frozenset[str] = frozenset({"dataset", "model"})


class ArtifactError(Exception):
    """Base class for every artifact failure. Always fail closed."""


class ManifestError(ArtifactError):
    """A manifest does not describe an artifact that can be verified."""


@dataclass(frozen=True)
class ArtifactFile:
    """One file inside an artifact, with the integrity fields that pin it."""

    path: str
    sha256: str
    size_bytes: int
    rows: int | None = None


@dataclass(frozen=True)
class ArtifactManifest:
    """An immutable pin: this artifact, at this revision, with these hashes."""

    artifact_id: str
    repo: str
    repo_type: str
    revision: str
    files: tuple[ArtifactFile, ...]
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(entry.path for entry in self.files)


def artifact_dir_name(artifact_id: str) -> str:
    """The cache directory name for an artifact id, refusing anything unsafe.

    :func:`parse_manifest` already enforces this, so a manifest that parsed can
    always be cached. The check is repeated here because a manifest can also be
    constructed directly in code.
    """
    if not _ARTIFACT_ID_RE.match(artifact_id or ""):
        raise ManifestError(
            f"artifact_id {artifact_id!r} is not a safe directory name; "
            "expected letters, digits, '.', '_' or '-'"
        )
    return artifact_id


def _require_str(data: Mapping[str, Any], key: str, artifact_id: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{artifact_id}: manifest field '{key}' is missing or not a string")
    return value.strip()


def _validate_member_path(raw: Any, artifact_id: str) -> str:
    """Refuse any path that could escape the artifact's cache directory."""
    if not isinstance(raw, str) or not raw.strip():
        raise ManifestError(f"{artifact_id}: every file entry needs a non-empty 'path'")
    candidate = raw.strip()
    # Normalize both separators before inspecting segments: a manifest is data
    # from outside, and Windows honours '\' as a separator too.
    segments = candidate.replace("\\", "/").split("/")
    if candidate.startswith(("/", "\\")) or Path(candidate).is_absolute():
        raise ManifestError(f"{artifact_id}: file 'path' must be relative, got {candidate!r}")
    if any(segment in ("", ".", "..") for segment in segments):
        raise ManifestError(
            f"{artifact_id}: file 'path' must not traverse directories, got {candidate!r}"
        )
    return candidate


def _parse_file(raw: Any, artifact_id: str) -> ArtifactFile:
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{artifact_id}: every entry in 'files' must be an object")

    path = _validate_member_path(raw.get("path"), artifact_id)

    sha256 = raw.get("sha256")
    if not isinstance(sha256, str) or not _SHA256_RE.match(sha256.strip().lower()):
        raise ManifestError(
            f"{artifact_id}: file {path!r} needs a 64-character hex 'sha256', got {sha256!r}"
        )

    size_bytes = raw.get("bytes")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        raise ManifestError(
            f"{artifact_id}: file {path!r} needs a non-negative integer 'bytes', got {size_bytes!r}"
        )

    rows = raw.get("rows")
    if rows is not None and (not isinstance(rows, int) or isinstance(rows, bool)):
        raise ManifestError(f"{artifact_id}: file {path!r} has a non-integer 'rows'")

    return ArtifactFile(path=path, sha256=sha256.strip().lower(), size_bytes=size_bytes, rows=rows)


def parse_manifest(data: Mapping[str, Any]) -> ArtifactManifest:
    """Validate a manifest mapping, or raise :class:`ManifestError`."""
    if not isinstance(data, Mapping):
        raise ManifestError("manifest must be a JSON object")

    raw_id = data.get("artifact_id")
    artifact_id = raw_id.strip() if isinstance(raw_id, str) and raw_id.strip() else "<unknown artifact>"
    if artifact_id == "<unknown artifact>":
        raise ManifestError("manifest field 'artifact_id' is missing or not a string")
    artifact_dir_name(artifact_id)  # refuse ids that cannot become a cache directory

    repo = _require_str(data, "repo", artifact_id)
    repo_type = _require_str(data, "repo_type", artifact_id)
    if repo_type not in REPO_TYPES:
        raise ManifestError(
            f"{artifact_id}: 'repo_type' must be one of {sorted(REPO_TYPES)}, got {repo_type!r}"
        )

    revision = _require_str(data, "revision", artifact_id)
    if not _REVISION_RE.match(revision.lower()):
        raise ManifestError(
            f"{artifact_id}: 'revision' must be a full 40-character commit sha, got {revision!r}. "
            "A branch or tag is mutable and cannot pin an artifact."
        )

    raw_files = data.get("files")
    if not isinstance(raw_files, (list, tuple)) or not raw_files:
        raise ManifestError(f"{artifact_id}: manifest needs a non-empty 'files' list")

    files = tuple(_parse_file(entry, artifact_id) for entry in raw_files)
    seen: set[str] = set()
    for entry in files:
        if entry.path in seen:
            raise ManifestError(f"{artifact_id}: duplicate file 'path' {entry.path!r}")
        seen.add(entry.path)

    extra = {
        key: value
        for key, value in data.items()
        if key not in {"artifact_id", "repo", "repo_type", "revision", "files"}
    }
    return ArtifactManifest(
        artifact_id=artifact_id,
        repo=repo,
        repo_type=repo_type,
        revision=revision.lower(),
        files=files,
        extra=extra,
    )


def load_manifest(path: str | Path) -> ArtifactManifest:
    """Read and validate a manifest from disk."""
    location = Path(path)
    try:
        data = json.loads(location.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"artifact manifest not found: {location}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"artifact manifest {location} is not valid JSON: {exc}") from exc
    return parse_manifest(data)
