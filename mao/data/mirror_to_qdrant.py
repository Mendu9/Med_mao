"""
mao/data/mirror_to_qdrant.py
-----------------------------
One-time (and re-runnable) script that copies all vectors + metadata from
local ChromaDB into Qdrant Cloud.

No re-embedding — reads the raw 768-dim float vectors already stored in
ChromaDB and uploads them directly to Qdrant.

Usage:
    python -m mao.data.mirror_to_qdrant
    python -m mao.data.mirror_to_qdrant --batch-size 200   # slower but safer
    python -m mao.data.mirror_to_qdrant --dry-run          # count only, no upload

Requirements:
    pip install qdrant-client
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import time
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_BATCH = 500


def _get_chroma_collection():
    import chromadb
    from chromadb.config import Settings
    from mao.core.config import cfg

    client = chromadb.HttpClient(
        host=cfg.chroma_host,
        port=cfg.chroma_port,
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_collection(cfg.chroma_collection)
    logger.info("ChromaDB '%s': %d documents", cfg.chroma_collection, col.count())
    return col


def _get_qdrant_client():
    from qdrant_client import QdrantClient
    from mao.core.config import cfg

    if not cfg.qdrant_url:
        raise ValueError("QDRANT_CLUSTER_ENDPOINT not set in .env")
    if not cfg.qdrant_api_key:
        raise ValueError("QDRANT_API_KEY not set in .env")

    client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
    logger.info("Qdrant connected: %s", cfg.qdrant_url)
    return client


def _ensure_qdrant_collection(client: Any, name: str, dim: int) -> None:
    from qdrant_client.models import Distance, VectorParams

    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        info = client.get_collection(name)
        existing_dim = info.config.params.vectors.size
        if existing_dim != dim:
            raise ValueError(
                f"Qdrant collection '{name}' exists with dim={existing_dim}, "
                f"but ChromaDB uses dim={dim}. Delete the Qdrant collection first."
            )
        count = client.count(name).count
        logger.info("Qdrant '%s' already exists: %d points", name, count)
    else:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s' (dim=%d, cosine)", name, dim)


def _chroma_id_to_qdrant_id(chroma_id: str) -> int:
    """Hash a ChromaDB string ID into a 63-bit positive integer for Qdrant."""
    return int(hashlib.md5(chroma_id.encode()).hexdigest(), 16) % (2**63)


def _safe_list(val: Any) -> list:
    """Convert ChromaDB result to a plain Python list.
    ChromaDB may return numpy arrays (2-D for embeddings, 1-D for docs/metas).
    Using 'not val' or 'val or []' on a numpy array raises ValueError.
    """
    if val is None:
        return []
    if hasattr(val, "tolist"):
        return val.tolist()
    return list(val)


def _build_points(
    ids: list[str],
    embeddings: Any,
    documents: Any,
    metadatas: Any,
) -> list[Any]:
    from qdrant_client.models import PointStruct

    emb_list = _safe_list(embeddings)
    doc_list = _safe_list(documents)
    meta_list = _safe_list(metadatas)

    points = []
    for i, chroma_id in enumerate(ids):
        vec = emb_list[i] if i < len(emb_list) else None
        if vec is None or len(vec) == 0:
            continue
        vec_floats: list[float] = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        meta = meta_list[i] if i < len(meta_list) else None
        meta = meta if isinstance(meta, dict) else {}
        payload: dict[str, Any] = {
            "chunk_id": chroma_id,
            "text": doc_list[i] if i < len(doc_list) else "",
            **{k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))},
        }
        points.append(PointStruct(
            id=_chroma_id_to_qdrant_id(chroma_id),
            vector=vec_floats,
            payload=payload,
        ))
    return points


def mirror(batch_size: int = _DEFAULT_BATCH, dry_run: bool = False) -> int:
    """Mirror all ChromaDB vectors to Qdrant. Returns total points upserted."""
    from mao.core.config import cfg

    col = _get_chroma_collection()
    total_docs = col.count()

    if dry_run:
        logger.info("DRY RUN — %d documents would be mirrored", total_docs)
        return 0

    qdrant = _get_qdrant_client()

    # Determine vector dimension from a sample
    sample = col.get(limit=1, include=["embeddings"])
    embs = _safe_list(sample.get("embeddings") if sample else None)
    if len(embs) == 0 or embs[0] is None or len(embs[0]) == 0:
        raise RuntimeError("ChromaDB returned no embeddings — collection may not have been ingested with embeddings stored.")
    dim = len(embs[0])
    logger.info("Vector dimension: %d", dim)

    _ensure_qdrant_collection(qdrant, cfg.qdrant_collection, dim)

    total_upserted = 0
    offset = 0

    while True:
        batch = col.get(
            limit=batch_size,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        ids: list[str] = batch.get("ids") or []
        if not ids:
            break

        points = _build_points(
            ids,
            batch.get("embeddings"),
            batch.get("documents"),
            batch.get("metadatas"),
        )

        if points:
            qdrant.upsert(collection_name=cfg.qdrant_collection, points=points)
            total_upserted += len(points)

        logger.info(
            "Progress: %d / %d  (%.1f%%)",
            offset + len(ids),
            total_docs,
            100.0 * (offset + len(ids)) / max(total_docs, 1),
        )

        offset += len(ids)
        if len(ids) < batch_size:
            break

        time.sleep(0.05)

    logger.info(
        "Mirror complete: %d points in Qdrant '%s'",
        total_upserted,
        cfg.qdrant_collection,
    )
    return total_upserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror ChromaDB vectors to Qdrant Cloud")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH,
                        help=f"Vectors per batch (default: {_DEFAULT_BATCH})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count only — no upload")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt")
    args = parser.parse_args()

    from mao.core.config import cfg
    print(f"\nSource : ChromaDB {cfg.chroma_host}:{cfg.chroma_port} / {cfg.chroma_collection}")
    print(f"Target : Qdrant   {cfg.qdrant_url} / {cfg.qdrant_collection}")
    print(f"Batch  : {args.batch_size} | Dry run: {args.dry_run}\n")

    if not args.dry_run and not args.yes:
        confirm = input("Type 'yes' to start: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return

    n = mirror(batch_size=args.batch_size, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"\nDone. {n} points uploaded to Qdrant.")


if __name__ == "__main__":
    main()
