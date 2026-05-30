"""One-time script to download and ingest pre-built biomedical knowledge graphs.

Downloads PrimeKG (~370MB) and optionally AlzKB, merges them into the
unified entity graph at mao/data/entity_graph.json.

Usage:
    python -m mao.data.ingest_knowledge_bases
    python -m mao.data.ingest_knowledge_bases --skip-download  # if already downloaded
    python -m mao.data.ingest_knowledge_bases --ontology-only  # skip PrimeKG
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths relative to project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "mao" / "data"
_PRIMEKG_CSV = _DATA_DIR / "primekg" / "kg.csv"
_ALZKB_DIR = _DATA_DIR / "alzkb"
_GRAPH_PATH = _DATA_DIR / "entity_graph.json"


def run(skip_download: bool = False, ontology_only: bool = False, alzkb_dir: str | None = None) -> dict:
    """Run the full KG ingestion pipeline. Returns stats dict."""
    import networkx as nx
    from mao.rag.graph_builder import load_graph, save_graph

    # Load existing graph (or start fresh)
    G = load_graph()
    if G.number_of_nodes() == 0:
        G = nx.MultiDiGraph()
    logger.info("Starting graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    stats: dict = {"initial_nodes": G.number_of_nodes(), "initial_edges": G.number_of_edges()}

    # Step 1: Load ontology backbone (HPO + MONDO)
    from mao.rag.ontology_loader import load_hpo_subgraph, load_mondo_diseases
    hpo_count = load_hpo_subgraph(G, domains=["alzheimer", "stroke"])
    mondo_count = load_mondo_diseases(G)
    stats["hpo_nodes"] = hpo_count
    stats["mondo_nodes"] = mondo_count
    logger.info("After ontology: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    if not ontology_only:
        # Step 2: Load PrimeKG
        from mao.rag.kg_loader import download_primekg, load_primekg

        if not skip_download:
            _PRIMEKG_CSV.parent.mkdir(parents=True, exist_ok=True)
            ok = download_primekg(_PRIMEKG_CSV)
            if not ok:
                logger.warning("PrimeKG download failed; skipping PrimeKG load")

        if _PRIMEKG_CSV.exists():
            primekg_edges = load_primekg(G, _PRIMEKG_CSV)
            stats["primekg_edges"] = primekg_edges
        else:
            logger.info("PrimeKG CSV not found; run without --skip-download to fetch it")
            stats["primekg_edges"] = 0

        # Step 3: Load AlzKB (optional — requires manual download from Zenodo)
        alzkb_path = Path(alzkb_dir) if alzkb_dir else _ALZKB_DIR
        if alzkb_path.exists():
            from mao.rag.kg_loader import load_alzkb
            alzkb_count = load_alzkb(G, alzkb_path)
            stats["alzkb_nodes"] = alzkb_count
        else:
            logger.info(
                "AlzKB directory not found at %s.\n"
                "To add AlzKB: download CSVs from https://zenodo.org/records/13647592\n"
                "and place them in %s", alzkb_path, _ALZKB_DIR
            )
            stats["alzkb_nodes"] = 0

    # Save merged graph
    save_graph(G, _GRAPH_PATH)
    stats["final_nodes"] = G.number_of_nodes()
    stats["final_edges"] = G.number_of_edges()
    logger.info(
        "Graph saved: %d nodes, %d edges → %s",
        G.number_of_nodes(), G.number_of_edges(), _GRAPH_PATH
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description="Ingest pre-built biomedical knowledge graphs")
    parser.add_argument("--skip-download", action="store_true", help="Skip PrimeKG download (use existing file)")
    parser.add_argument("--ontology-only", action="store_true", help="Load only HPO+MONDO, skip PrimeKG+AlzKB")
    parser.add_argument("--alzkb-dir", default=None, help="Path to AlzKB CSV directory")
    args = parser.parse_args()

    stats = run(
        skip_download=args.skip_download,
        ontology_only=args.ontology_only,
        alzkb_dir=args.alzkb_dir,
    )
    print("\n=== Knowledge Base Ingestion Complete ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
