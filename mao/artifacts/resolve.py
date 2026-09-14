"""Materialize a pinned artifact into a deterministic local cache.

The contract is narrow on purpose: given a manifest, either every file is
present and its bytes hash to what the manifest says, or nothing is returned.
There is no partial success and no unverified path out of this module.

Three defects recorded as R6 are closed here by construction:

* the revision is always the manifest's pinned commit sha, never a branch;
* every file is verified by SHA-256 *and* byte size on every resolve, not only
  on download;
* the cache root is derived from ``MAO_ARTIFACT_CACHE`` / ``HF_HOME`` / the
  user's home directory — never a hardcoded ``/tmp``, which is POSIX-only.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Protocol

from mao.artifacts.manifest import ArtifactError, ArtifactFile, ArtifactManifest, artifact_dir_name

logger = logging.getLogger(__name__)

#: Read size for hashing. Artifacts are tens of megabytes; never read whole.
_CHUNK = 1024 * 1024


class ArtifactIntegrityError(ArtifactError):
    """Materialized bytes do not match the manifest. Never serve them."""


class ArtifactFetchError(ArtifactError):
    """The artifact could not be retrieved from its store."""


class Fetcher(Protocol):
    """Writes ``filename`` from ``repo`` at ``revision`` to ``dest``.

    Injectable so the verification path can be exercised without a network:
    tests supply a fetcher that copies from a committed fixture.
    """

    def __call__(
        self, *, repo: str, repo_type: str, revision: str, filename: str, dest: Path
    ) -> None: ...


@dataclass(frozen=True)
class ResolvedArtifact:
    """Verified local paths for every file a manifest declares."""

    root: Path
    files: Mapping[str, Path]

    def file(self, path: str) -> Path:
        """Return the verified local path for one declared file."""
        try:
            return self.files[path]
        except KeyError:
            raise KeyError(
                f"{path!r} is not declared in this artifact; declared: {sorted(self.files)}"
            ) from None


def artifact_cache_root(env: Mapping[str, str] | None = None) -> Path:
    """Where artifacts are materialized, resolved identically on every platform.

    Precedence: an explicit ``MAO_ARTIFACT_CACHE``, then a Hugging Face
    ``HF_HOME``, then ``~/.cache/mao/artifacts``. Never the system temp
    directory: that is world-writable, cleared unpredictably, and ``/tmp`` in
    particular does not exist on Windows.
    """
    environment = os.environ if env is None else env

    explicit = environment.get("MAO_ARTIFACT_CACHE", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()

    hf_home = environment.get("HF_HOME", "").strip()
    if hf_home:
        return (Path(hf_home).expanduser() / "mao-artifacts").resolve()

    return (Path.home() / ".cache" / "mao" / "artifacts").resolve()


def artifact_dir(manifest: ArtifactManifest, *, cache_root: str | Path | None = None) -> Path:
    """The deterministic directory for one artifact at one revision.

    The revision is part of the path, so two pins of the same artifact can
    coexist and a re-pin can never be served stale bytes from the old one.
    """
    root = Path(cache_root).expanduser() if cache_root is not None else artifact_cache_root()
    return root / artifact_dir_name(manifest.artifact_id) / manifest.revision


def _digest(path: Path) -> tuple[str, int]:
    """Return ``(sha256_hex, byte_count)`` without reading the file whole."""
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def _integrity_failure(
    manifest: ArtifactManifest, entry: ArtifactFile, actual_sha: str, actual_bytes: int
) -> ArtifactIntegrityError:
    return ArtifactIntegrityError(
        f"{manifest.artifact_id}: integrity check failed for {entry.path!r} from "
        f"{manifest.repo}@{manifest.revision} — expected sha256 {entry.sha256} "
        f"({entry.size_bytes} bytes), got {actual_sha} ({actual_bytes} bytes). "
        f"The unverified copy has been deleted. Remedy: re-download; if the artifact was "
        f"legitimately republished, update the manifest pin and review the new bytes."
    )


def _verify_or_discard(manifest: ArtifactManifest, entry: ArtifactFile, path: Path) -> None:
    """Hash ``path`` against the manifest; delete and raise if it disagrees."""
    actual_sha, actual_bytes = _digest(path)
    if actual_sha == entry.sha256 and actual_bytes == entry.size_bytes:
        return
    path.unlink(missing_ok=True)
    raise _integrity_failure(manifest, entry, actual_sha, actual_bytes)


def _fetch_verified(
    manifest: ArtifactManifest, entry: ArtifactFile, dest: Path, fetch: Fetcher
) -> None:
    """Download to a staging file, verify it, and only then publish it."""
    staged = dest.with_name(dest.name + ".part")
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.unlink(missing_ok=True)

    try:
        fetch(
            repo=manifest.repo,
            repo_type=manifest.repo_type,
            revision=manifest.revision,
            filename=entry.path,
            dest=staged,
        )
    except Exception as exc:  # noqa: BLE001 — every failure mode is reported the same way
        staged.unlink(missing_ok=True)
        raise ArtifactFetchError(
            f"{manifest.artifact_id}: could not fetch {entry.path!r} from "
            f"{manifest.repo}@{manifest.revision} ({manifest.repo_type}): {exc}"
        ) from exc

    if not staged.exists():
        raise ArtifactFetchError(
            f"{manifest.artifact_id}: fetching {entry.path!r} from "
            f"{manifest.repo}@{manifest.revision} produced no file at {staged}"
        )

    _verify_or_discard(manifest, entry, staged)
    staged.replace(dest)


def _hf_download(*, repo: str, repo_type: str, revision: str, filename: str, dest: Path) -> None:
    """Default fetcher: Hugging Face Hub, always at a pinned revision."""
    from huggingface_hub import hf_hub_download

    source = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type=repo_type,
        revision=revision,  # pinned: never a floating branch
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


def resolve(
    manifest: ArtifactManifest,
    *,
    cache_root: str | Path | None = None,
    fetch: Fetcher | None = None,
) -> ResolvedArtifact:
    """Materialize every file in ``manifest`` and verify it before returning.

    Raises :class:`ArtifactFetchError` if the store cannot serve a file and
    :class:`ArtifactIntegrityError` if the bytes disagree with the manifest.
    Nothing unverified is ever returned.
    """
    fetcher: Fetcher = _hf_download if fetch is None else fetch
    root = artifact_dir(manifest, cache_root=cache_root)

    materialized: dict[str, Path] = {}
    for entry in manifest.files:
        dest = root.joinpath(*PurePosixPath(entry.path.replace("\\", "/")).parts)
        if dest.exists():
            # Re-verify on every use: a cached file can rot, be edited, or be
            # written by something that is not this module.
            _verify_or_discard(manifest, entry, dest)
        else:
            logger.info(
                "materializing %s:%s from %s@%s",
                manifest.artifact_id,
                entry.path,
                manifest.repo,
                manifest.revision,
            )
            _fetch_verified(manifest, entry, dest, fetcher)
        materialized[entry.path] = dest

    return ResolvedArtifact(root=root, files=materialized)
