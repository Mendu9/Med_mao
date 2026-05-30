"""Pre-built biomedical knowledge graph loader.

Loads PrimeKG (Harvard, 129K nodes, 4M edges) and AlzKB v2.0 (234K nodes)
into the unified NetworkX MultiDiGraph as the Layer 2/3 knowledge backbone.

Node schema (all sources)
--------------------------
    name       : str  — canonical display name (used as graph node key)
    node_type  : str  — ontology type (e.g. "disease", "gene/protein")
    source     : str  — "primekg" | "alzkb"
    primekg_id : str  — original PrimeKG x_id / y_id  (PrimeKG only)

Edge schema (all sources)
--------------------------
    relation   : str  — human-readable relation label (display_relation)
    source     : str  — "primekg" | "alzkb"
    edge_type  : str  — "kg"
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)

# PrimeKG Harvard Dataverse file ID
_PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"

# Node types in PrimeKG that are relevant to clinical AI
_PRIMEKG_RELEVANT_TYPES = frozenset([
    "disease", "drug", "gene/protein", "effect/phenotype",
    "biological_process", "pathway",
])

# Disease name fragments to filter PrimeKG to relevant rows
_DOMAIN_KEYWORDS = [
    "alzheimer", "dementia", "cognitive", "amyloid", "tau", "apoe",
    "stroke", "ischemi", "hemorrhag", "cerebrovascular", "cerebral",
    "neurodegenerat", "parkinson", "epilep", "multiple sclerosis",
]


def download_primekg(output_path: str | Path) -> bool:
    """Download PrimeKG kg.csv (~370 MB) from Harvard Dataverse.

    Uses requests.get with stream=True and 8 KB chunks to keep memory flat.
    Logs progress every ~10 MB. Cleans up partial file on failure.

    Returns True if download succeeded or file already exists.
    """
    import requests

    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > 1_000_000:
        logger.info("PrimeKG already downloaded at %s", output_path)
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading PrimeKG (~370 MB) to %s ...", output_path)
    try:
        with requests.get(_PRIMEKG_URL, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            written = 0
            _log_interval = 10 * 1024 * 1024  # 10 MB
            _next_log = _log_interval
            with open(output_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
                        written += len(chunk)
                        if written >= _next_log:
                            pct = (written / total * 100) if total else 0.0
                            logger.info(
                                "PrimeKG download: %.1f MB (%.0f%%)",
                                written / 1024 / 1024,
                                pct,
                            )
                            _next_log += _log_interval
        logger.info(
            "PrimeKG download complete: %.1f MB", output_path.stat().st_size / 1e6
        )
        return True
    except Exception as exc:
        logger.error("PrimeKG download failed: %s", exc)
        if output_path.exists():
            output_path.unlink()
        return False


def load_primekg(
    G: "nx.MultiDiGraph",
    kg_csv_path: str | Path,
    domains: list[str] | None = None,
    max_rows: int = 500_000,
) -> int:
    """Load PrimeKG edges into G, filtered to neurological/AD/stroke domains.

    Node keys are human-readable x_name / y_name strings so they compose
    naturally with the NER-derived entity graph (not numeric PrimeKG IDs).

    kg.csv columns:
        x_index, x_id, x_type, x_name, x_source,
        y_index, y_id, y_type, y_name, y_source,
        relation, display_relation

    Filter: rows where (x_type or y_type) in _PRIMEKG_RELEVANT_TYPES AND
            (x_name or y_name) contains a domain keyword (case-insensitive).

    Node schema: {name, node_type: x_type/y_type, source: 'primekg', primekg_id: x_id/y_id}
    Edge schema: {relation: display_relation, source: 'primekg', edge_type: 'kg'}

    Returns count of edges added.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed; cannot load PrimeKG")
        return 0

    kg_csv_path = Path(kg_csv_path)
    if not kg_csv_path.exists():
        logger.warning("PrimeKG file not found: %s", kg_csv_path)
        return 0

    kw_pattern = "|".join(domains if domains else _DOMAIN_KEYWORDS)
    logger.info("Loading PrimeKG from %s ...", kg_csv_path)
    try:
        chunks = []
        for chunk in pd.read_csv(kg_csv_path, chunksize=50_000, dtype=str, low_memory=False):
            # Filter to relevant node types
            type_mask = (
                chunk["x_type"].isin(_PRIMEKG_RELEVANT_TYPES) |
                chunk["y_type"].isin(_PRIMEKG_RELEVANT_TYPES)
            )
            # Filter to domain-relevant names
            name_mask = (
                chunk["x_name"].str.lower().str.contains(kw_pattern, na=False) |
                chunk["y_name"].str.lower().str.contains(kw_pattern, na=False)
            )
            filtered = chunk[type_mask & name_mask]
            chunks.append(filtered)
            if sum(len(c) for c in chunks) >= max_rows:
                break

        if not chunks:
            logger.warning("PrimeKG: no rows matched domain filter")
            return 0

        df = pd.concat(chunks, ignore_index=True)
        logger.info("PrimeKG: %d rows after domain filter", len(df))

        edge_count = 0
        for _, row in df.iterrows():
            # Use human-readable names as node keys (not numeric IDs)
            x_name = str(row["x_name"])
            y_name = str(row["y_name"])

            if not G.has_node(x_name):
                G.add_node(
                    x_name,
                    name=x_name,
                    node_type=str(row["x_type"]),
                    source="primekg",
                    primekg_id=str(row["x_id"]),
                )
            if not G.has_node(y_name):
                G.add_node(
                    y_name,
                    name=y_name,
                    node_type=str(row["y_type"]),
                    source="primekg",
                    primekg_id=str(row["y_id"]),
                )

            G.add_edge(
                x_name, y_name,
                relation=str(row.get("display_relation", row.get("relation", ""))),
                source="primekg",
                edge_type="kg",
            )
            edge_count += 1

        logger.info("PrimeKG: added %d edges to graph", edge_count)
        return edge_count

    except Exception as exc:
        logger.error("PrimeKG load failed: %s", exc)
        return 0


def load_alzkb(G: "nx.MultiDiGraph", csv_dir: str | Path) -> int:
    """Load AlzKB v2.0 CSV exports into G.

    Source: https://zenodo.org/records/13647592

    Expects CSVs in *csv_dir* named:
      nodes_*.csv  (one per node type: Gene, Drug, BiologicalProcess, ...)
      edges_*.csv  (one per relationship type)

    Also accepts single nodes.csv / edges.csv layout.

    If the directory or files are absent, logs helpful instructions and
    returns 0 — never raises.

    Node schema: {name, node_type, source: 'alzkb', alzkb_id}
    Edge schema: {relation, source: 'alzkb', edge_type: 'kg'}

    Returns count of nodes added.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed; cannot load AlzKB")
        return 0

    csv_dir = Path(csv_dir)
    if not csv_dir.exists():
        logger.warning(
            "AlzKB directory not found: %s\n"
            "Download AlzKB v2.0 CSV exports from https://zenodo.org/records/13647592\n"
            "and place them in that directory.",
            csv_dir,
        )
        return 0

    count = 0

    # Locate node files — prefer nodes_*.csv, fall back to nodes.csv
    node_files = list(csv_dir.glob("nodes_*.csv"))
    if not node_files:
        single = csv_dir / "nodes.csv"
        if single.exists():
            node_files = [single]

    for node_file in node_files:
        node_type = node_file.stem.replace("nodes_", "").lower() or "entity"
        try:
            df = pd.read_csv(node_file, dtype=str, low_memory=False)
            for _, row in df.iterrows():
                node_id = str(row.get("id", row.get("nodeId", row.get("name", ""))))
                name = str(row.get("name", row.get("label", node_id)))
                if name and not G.has_node(name):
                    G.add_node(
                        name,
                        name=name,
                        node_type=node_type,
                        source="alzkb",
                        alzkb_id=node_id,
                    )
                    count += 1
        except Exception as exc:
            logger.warning("AlzKB node file %s failed: %s", node_file.name, exc)

    # Locate edge files — prefer edges_*.csv, fall back to edges.csv
    edge_files = list(csv_dir.glob("edges_*.csv"))
    if not edge_files:
        single_edges = csv_dir / "edges.csv"
        if single_edges.exists():
            edge_files = [single_edges]

    edge_count = 0
    for edge_file in edge_files:
        rel_type = edge_file.stem.replace("edges_", "") or "related_to"
        try:
            df = pd.read_csv(edge_file, dtype=str, low_memory=False)
            src_col = next(
                (c for c in df.columns if "source" in c.lower() or "from" in c.lower() or "start" in c.lower()),
                None,
            )
            tgt_col = next(
                (c for c in df.columns if "target" in c.lower() or "to" in c.lower() or "end" in c.lower()),
                None,
            )
            if src_col and tgt_col:
                for _, row in df.iterrows():
                    src, tgt = str(row[src_col]), str(row[tgt_col])
                    G.add_edge(
                        src, tgt,
                        relation=rel_type,
                        source="alzkb",
                        edge_type="kg",
                    )
                    edge_count += 1
        except Exception as exc:
            logger.warning("AlzKB edge file %s failed: %s", edge_file.name, exc)

    logger.info("AlzKB: added %d nodes, %d edges", count, edge_count)
    return count
