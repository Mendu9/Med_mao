"""Derive a manifest from the bytes that are actually being published.

Kept separate from :mod:`mao.artifacts.resolve` because publishing is a build-time
concern and resolving is a runtime one — the runtime must never be able to write
a pin for itself.

The builder validates its own output through :func:`~mao.artifacts.manifest.parse_manifest`,
so a manifest that this module emits is always one the resolver will accept.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mao.artifacts.manifest import parse_manifest

_CHUNK = 1024 * 1024


def _digest(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def build_manifest(
    *,
    artifact_id: str,
    repo: str,
    revision: str,
    root: str | Path,
    paths: Sequence[str],
    repo_type: str = "dataset",
    rows: Mapping[str, int] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash every file under ``root`` and return a validated manifest mapping.

    ``revision`` must already be the immutable commit sha the files were
    uploaded as; this function does not contact the store.
    """
    base = Path(root)
    row_counts = dict(rows or {})

    entries: list[dict[str, Any]] = []
    for relative in sorted(paths):
        location = base / relative
        if not location.is_file():
            raise FileNotFoundError(f"cannot pin {relative!r}: no such file under {base}")
        sha256, size_bytes = _digest(location)
        entry: dict[str, Any] = {"path": relative, "sha256": sha256, "bytes": size_bytes}
        if relative in row_counts:
            entry["rows"] = int(row_counts[relative])
        entries.append(entry)

    manifest: dict[str, Any] = {
        "artifact_id": artifact_id,
        "repo": repo,
        "repo_type": repo_type,
        "revision": revision,
        "files": entries,
    }
    if extra:
        manifest.update(dict(extra))

    # Fail here rather than at runtime: an unparseable pin must never ship.
    parse_manifest(manifest)
    return manifest


def utc_now_iso() -> str:
    """Timestamp for manifest provenance, in UTC with an explicit offset."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
