"""Pinned external artifacts: manifests in Git, bytes in an artifact store.

A corpus is not source code. It lives in a versioned artifact store, addressed
by an immutable revision and verified by SHA-256 before anything reads it. What
lives in Git is the manifest — small, reviewable, and the only thing that
decides which bytes are legitimate.

Typical use::

    from mao.artifacts import load_manifest, resolve

    manifest = load_manifest("manifests/corpus.manifest.json")
    documents = resolve(manifest).file("documents.parquet")

Every failure path raises :class:`ArtifactError`; none returns unverified bytes.
"""
from __future__ import annotations

from mao.artifacts.manifest import (
    REPO_TYPES,
    ArtifactError,
    ArtifactFile,
    ArtifactManifest,
    ManifestError,
    artifact_dir_name,
    load_manifest,
    parse_manifest,
)
from mao.artifacts.resolve import (
    ArtifactFetchError,
    ArtifactIntegrityError,
    ResolvedArtifact,
    artifact_cache_root,
    artifact_dir,
    resolve,
)

__all__ = [
    "REPO_TYPES",
    "ArtifactError",
    "ArtifactFetchError",
    "ArtifactFile",
    "ArtifactIntegrityError",
    "ArtifactManifest",
    "ManifestError",
    "ResolvedArtifact",
    "artifact_cache_root",
    "artifact_dir",
    "artifact_dir_name",
    "load_manifest",
    "parse_manifest",
    "resolve",
]
