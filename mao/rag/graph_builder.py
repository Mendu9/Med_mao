"""
mao/rag/graph_builder.py
------------------------
Builds a NetworkX entity graph from ingested documents.

Why NetworkX over Neo4j:
  - Zero infrastructure overhead — pure Python in-process
  - Sufficient for corpus sizes up to ~500K entities
  - Serializable to JSON (no running graph DB to manage)
  - Trivially mockable in tests

Entity extraction strategy:
  - scispaCy NER (en_ner_bc5cdr_md or en_core_sci_lg) for biomedical entities
    (DISEASE, CHEMICAL, etc.) with fallback to en_core_web_sm
  - Co-occurrence within a *sentence* = edge (weighted by frequency)
  - Graph persisted to disk as JSON; reloaded on startup

Graph type: nx.MultiDiGraph
  - Directed so future relation types (e.g. "treats", "causes") can be
    represented as distinct edge keys in the same direction
  - Multi so the same node pair can have multiple relation types

Integration points:
  - data/ingest_wikipedia.py calls build_graph_from_documents()
  - data/ingest_alzheimers.py calls build_graph_from_documents()
  - rag/retriever.py calls expand_via_graph() for graph traversal step
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from mao.core.config import cfg

logger = logging.getLogger(__name__)

# Path where the serialised graph is stored
GRAPH_PATH = cfg.data_dir / "entity_graph.json"

# spaCy model — loaded lazily, cached at module level
_nlp = None  # type: ignore[assignment]

# Graph — loaded lazily, cached at module level (700 MB JSON, must not reload per query)
_graph_cache: "nx.MultiDiGraph | None" = None


def _load_nlp():
    """
    Load the best available spaCy NLP pipeline.

    Priority:
      1. en_ner_bc5cdr_md  — scispaCy BC5CDR model (DISEASE + CHEMICAL NER)
      2. en_core_sci_lg    — scispaCy large general biomedical model
      3. en_core_web_sm    — standard English model (fallback)

    Returns the loaded spaCy Language object.
    """
    import spacy

    for model_name in ("en_ner_bc5cdr_md", "en_core_sci_lg", "en_core_web_sm"):
        try:
            nlp = spacy.load(model_name)
            logger.info("Loaded spaCy model: %s", model_name)
            return nlp
        except OSError:
            logger.debug("spaCy model not found, trying next: %s", model_name)

    raise RuntimeError(
        "No spaCy model available. Install at least en_core_web_sm:\n"
        "  python -m spacy download en_core_web_sm"
    )


def _get_nlp():
    """Return the cached NLP pipeline, loading it on first call."""
    global _nlp
    if _nlp is None:
        _nlp = _load_nlp()
    return _nlp


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> list[tuple[str, str]]:
    """
    Run spaCy NER on *text*.

    Returns list of (entity_text, entity_label) tuples.
    For scispaCy BC5CDR the label will be "DISEASE" or "CHEMICAL".
    For en_core_sci_lg unlabelled spans get "ENTITY".
    For en_core_web_sm the label will be "PERSON", "ORG", "GPE", etc.

    Called by rag/retriever.py — signature must remain stable.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    results: list[tuple[str, str]] = []
    for ent in doc.ents:
        label = ent.label_ if ent.label_ else "ENTITY"
        entity_text = ent.text.strip()
        if len(entity_text) > 1:
            results.append((entity_text, label))
    return results


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph_from_documents(
    documents: list[dict[str, Any]],
) -> nx.MultiDiGraph:
    """
    Build a sentence-level co-occurrence entity graph from a list of document dicts.

    Args:
        documents: Each dict must have a "text" key. Optional keys:
                   "source" (str), "chunk_id" (str) — used as doc_id on nodes/edges.

    Returns:
        nx.MultiDiGraph where:
          - Nodes: entity strings with attributes
              name, node_type, source="document", doc_id
          - Edges: co-occurrence within the same sentence with attributes
              relation="co-occurs_with", weight=1, source=doc_id,
              edge_type="co-occurrence"

    Co-occurrence is sentence-scoped (not document-scoped) to reduce noise:
    two entities linked only if they appear in the same sentence.
    """
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    nlp = _get_nlp()

    for doc_dict in documents:
        text: str = doc_dict.get("text", "")
        doc_id: str = doc_dict.get("chunk_id", "") or doc_dict.get("source", "unknown")

        if not text.strip():
            continue

        spacy_doc = nlp(text)

        # Process each sentence independently — sentence-level co-occurrence
        for sent in spacy_doc.sents:
            # Collect unique entities in this sentence (dedup by text, keep label)
            seen: dict[str, str] = {}
            for ent in sent.ents:
                label = ent.label_ if ent.label_ else "ENTITY"
                entity_text = ent.text.strip()
                if len(entity_text) > 1:
                    seen[entity_text] = label

            sent_entities = list(seen.items())  # list of (text, label)

            # Add / update nodes
            for entity_text, label in sent_entities:
                if not G.has_node(entity_text):
                    G.add_node(
                        entity_text,
                        name=entity_text,
                        node_type=label,
                        source="document",
                        doc_id=doc_id,
                    )

            # Add directed co-occurrence edges between all pairs in this sentence
            for i, (e1_text, _) in enumerate(sent_entities):
                for e2_text, _ in sent_entities[i + 1:]:
                    G.add_edge(
                        e1_text,
                        e2_text,
                        relation="co-occurs_with",
                        weight=1,
                        source=doc_id,
                        edge_type="co-occurrence",
                    )

    logger.info(
        "Graph built: %d nodes, %d edges from %d documents",
        G.number_of_nodes(),
        G.number_of_edges(),
        len(documents),
    )
    return G


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_graph(G: nx.MultiDiGraph, path: Path | None = None) -> None:
    """Serialise graph to JSON (node-link format)."""
    global _graph_cache
    target = path or GRAPH_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(G, edges="edges")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f)
    _graph_cache = G  # update cache so next load_graph() call skips disk read
    logger.info("Graph saved to %s", target)


def load_graph(path: Path | None = None) -> nx.MultiDiGraph:
    """
    Load graph from disk, caching in memory after the first load.

    The graph is ~700 MB JSON — loading it on every retrieval query is
    prohibitively expensive.  The module-level cache means the file is
    parsed once per process lifetime; subsequent calls return instantly.

    Pass a non-default ``path`` to bypass the cache (e.g. in tests).
    Returns an empty MultiDiGraph if the file does not exist.
    """
    global _graph_cache
    target = path or GRAPH_PATH

    # Return cached graph when using the default path
    if path is None and _graph_cache is not None:
        return _graph_cache

    import os as _os
    if _os.getenv("MAO_DISABLE_GRAPH", "0") == "1":
        logger.info("MAO_DISABLE_GRAPH=1 — skipping graph load to save RAM")
        return nx.MultiDiGraph()

    if not target.exists():
        logger.warning("Graph file not found at %s — returning empty graph", target)
        return nx.MultiDiGraph()

    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        G = nx.node_link_graph(data, edges="edges", directed=True, multigraph=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load graph as MultiDiGraph (%s) — converting from legacy format", exc
        )
        G_old = nx.node_link_graph(data, edges="edges")
        G = nx.MultiDiGraph(G_old)
    if not isinstance(G, nx.MultiDiGraph):
        logger.warning(
            "Loaded graph type %s is not MultiDiGraph — converting", type(G).__name__
        )
        G = nx.MultiDiGraph(G)
    logger.info("Graph loaded: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    if path is None:
        _graph_cache = G  # store in cache for subsequent calls
    return G


# ---------------------------------------------------------------------------
# Graph traversal helper (called by rag/retriever.py)
# ---------------------------------------------------------------------------

def expand_via_graph(
    G: nx.MultiDiGraph,
    seed_entities: list[str],
    hops: int = 2,
    max_nodes: int = 50,
    max_depth: int | None = None,
) -> list[str]:
    """
    BFS from *seed_entities* up to *hops* edges deep.

    Returns the list of entity names reachable (excluding seeds themselves).
    Used by the retriever to find related chunks after the initial vector search.

    Uses G.successors() for directed graphs; falls back to G.neighbors() for
    undirected graphs so the function works with both MultiDiGraph and Graph.

    Args:
        G:              The entity graph.
        seed_entities:  Entity names found in the top-20 vector search results.
        hops:           Number of graph hops to expand. Clamped to max 3.
        max_nodes:      Hard cap on returned nodes (avoids explosion).
        max_depth:      Alias for hops — if provided, takes precedence.
    """
    # max_depth is an alias for hops; clamp to a maximum of 3 to prevent explosion
    effective_hops = min(max_depth if max_depth is not None else hops, 3)

    def _neighbours(node: str):
        if hasattr(G, "successors"):
            return G.successors(node)
        return G.neighbors(node)  # type: ignore[attr-defined]

    visited: set[str] = set(seed_entities)
    frontier: set[str] = set(seed_entities) & set(G.nodes())
    expanded: list[str] = []
    actual_hops = 0
    # Cap neighbors per hub node: dense graphs can have nodes with 10K+ successors;
    # iterating all of them makes each BFS hop O(seconds) on 5M+ edge graphs.
    _MAX_NEIGHBORS_PER_NODE = 50

    for _ in range(effective_hops):
        next_frontier: set[str] = set()
        for node in frontier:
            neighbor_count = 0
            for neighbour in _neighbours(node):
                if neighbor_count >= _MAX_NEIGHBORS_PER_NODE:
                    break
                neighbor_count += 1
                if neighbour not in visited:
                    visited.add(neighbour)
                    next_frontier.add(neighbour)
                    expanded.append(neighbour)
                    if len(expanded) >= max_nodes:
                        logger.debug(
                            "Graph expand: %d seeds → %d nodes in %d hops",
                            len(seed_entities), len(expanded), actual_hops + 1,
                        )
                        return expanded
        actual_hops += 1
        frontier = next_frontier
        if not frontier:
            break

    logger.debug(
        "Graph expand: %d seeds → %d nodes in %d hops",
        len(seed_entities), len(expanded), actual_hops,
    )
    return expanded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = [
        {
            "text": (
                "Tau protein accumulates in Alzheimer disease. "
                "Amyloid beta plaques are also observed in Alzheimer disease patients."
            ),
            "source": "wiki_alzheimer",
            "chunk_id": "c1",
        },
        {
            "text": "Donepezil is a cholinesterase inhibitor used to treat Alzheimer disease.",
            "source": "wiki_treatment",
            "chunk_id": "c2",
        },
    ]
    G = build_graph_from_documents(docs)
    print("Nodes:", list(G.nodes(data=True))[:5])
    print("Edges:", list(G.edges(data=True))[:5])
    expanded = expand_via_graph(G, ["Alzheimer disease"], hops=1)
    print("Expanded from 'Alzheimer disease':", expanded)
