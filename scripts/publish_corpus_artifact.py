"""A4 runner — publish Corpus V1 to its dedicated artifact store and pin it.

Uploads an explicit allow-list of files to a Hugging Face **Dataset** repository
(never the live Space), reads back the immutable commit revision, writes the Git
manifest that pins it, and then verifies the pin by materializing the artifact
through the ordinary runtime resolver.

    python -m scripts.publish_corpus_artifact --dry-run
    python -m scripts.publish_corpus_artifact

The legacy ``mao/data/bm25_corpus.json`` is not publishable — it is unlicensed
chunk text — and cannot be uploaded by this script: only the paths in
:data:`ARTIFACT_FILES` are ever sent.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mao.artifacts import load_manifest, resolve
from mao.artifacts.publish import build_manifest, utc_now_iso

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("publish_corpus_artifact")

_ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_ID = "mao-corpus-source"
REPO_ID = "ArunMendu/mao-corpus-source"
REPO_TYPE = "dataset"

BUILD_DIR = _ROOT / "build" / "corpus_v1"
MANIFEST_PATH = _ROOT / "manifests" / "corpus.manifest.json"
DATASET_CARD = _ROOT / "mao" / "corpus" / "DATASET_CARD.md"

#: The complete set of files published. Nothing outside this list is uploaded.
ARTIFACT_FILES: tuple[str, ...] = (
    "documents.parquet",
    "corpus_manifest.json",
    "exclusions.json",
    "provenance.json",
)

#: Paths that must never reach an artifact store, checked explicitly so the
#: prohibition is enforced rather than merely documented.
FORBIDDEN = ("bm25_corpus.json",)


def _preflight() -> dict:
    """Confirm the local build is present and self-consistent before uploading."""
    build_manifest_path = BUILD_DIR / "corpus_manifest.json"
    if not build_manifest_path.is_file():
        raise SystemExit(f"corpus build not found at {BUILD_DIR}; run scripts/build_corpus_v1.py")

    recorded = json.loads(build_manifest_path.read_text(encoding="utf-8"))

    for name in ARTIFACT_FILES:
        if any(bad in name for bad in FORBIDDEN):
            raise SystemExit(f"refusing to publish forbidden artifact {name!r}")
        if not (BUILD_DIR / name).is_file():
            raise SystemExit(f"missing artifact file: {BUILD_DIR / name}")

    # The build manifest already records documents.parquet; re-derive the hash
    # from disk and refuse if the bytes have drifted since the build.
    local = build_manifest(
        artifact_id=ARTIFACT_ID,
        repo=REPO_ID,
        revision="0" * 40,
        root=BUILD_DIR,
        paths=["documents.parquet"],
    )
    derived = local["files"][0]
    for entry in recorded.get("files", []):
        if entry["path"] == "documents.parquet":
            if entry["sha256"] != derived["sha256"] or entry["bytes"] != derived["bytes"]:
                raise SystemExit(
                    "documents.parquet does not match the hash recorded at build time — "
                    f"build said {entry['sha256']} ({entry['bytes']} B), "
                    f"disk has {derived['sha256']} ({derived['bytes']} B)"
                )
            logger.info("preflight: documents.parquet matches its build-time hash")
    return recorded


def _upload(private: bool) -> str:
    """Create the dataset repo if needed, upload the allow-list, return the sha."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=private, exist_ok=True)
    logger.info("repo ready: %s (%s, private=%s)", REPO_ID, REPO_TYPE, private)

    operations = []
    from huggingface_hub import CommitOperationAdd

    for name in ARTIFACT_FILES:
        operations.append(
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(BUILD_DIR / name))
        )
    operations.append(
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(DATASET_CARD))
    )

    info = api.create_commit(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        operations=operations,
        commit_message="Corpus V1 — 584 license-verified PMC documents",
        commit_description=(
            "Document-level corpus. Every document carries a positively identified "
            "redistribution license; 0 UNKNOWN, 0 no-derivatives, retracted excluded."
        ),
    )
    revision = getattr(info, "oid", None)
    if not revision:
        raise SystemExit("upload succeeded but no commit sha was returned")
    logger.info("uploaded at immutable revision %s", revision)
    return str(revision)


def _write_pin(revision: str, recorded: dict) -> Path:
    """Write the Git-side manifest that pins the uploaded revision."""
    rows = {"documents.parquet": int(recorded.get("counts", {}).get("admitted", 0))}
    manifest = build_manifest(
        artifact_id=ARTIFACT_ID,
        repo=REPO_ID,
        revision=revision,
        root=BUILD_DIR,
        paths=list(ARTIFACT_FILES),
        rows=rows,
        extra={
            "corpus_version": recorded.get("corpus_version"),
            "granularity": recorded.get("granularity"),
            "license_summary": recorded.get("license_summary"),
            "counts": recorded.get("counts"),
            "pinned_utc": utc_now_iso(),
        },
    )
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote pin: %s", MANIFEST_PATH)
    return MANIFEST_PATH


def _verify_round_trip() -> None:
    """Materialize through the real resolver — the pin must actually work."""
    manifest = load_manifest(MANIFEST_PATH)
    resolved = resolve(manifest)
    for entry in manifest.files:
        logger.info("verified %s -> %s", entry.path, resolved.file(entry.path))
    logger.info("round-trip verification passed at revision %s", manifest.revision)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="preflight only; upload nothing")
    parser.add_argument("--private", action="store_true", help="create the dataset repo private")
    args = parser.parse_args(argv)

    recorded = _preflight()
    if args.dry_run:
        logger.info("dry run: would publish %s to %s", list(ARTIFACT_FILES), REPO_ID)
        return 0

    revision = _upload(private=args.private)
    _write_pin(revision, recorded)
    _verify_round_trip()
    return 0


if __name__ == "__main__":
    sys.exit(main())
