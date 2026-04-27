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
  - spaCy NER (en_core_web_sm) for PERSON, ORG, GPE, EVENT, PRODUCT
  - Co-occurrence within a sentence = edge (weighted by frequency)
  - Graph persisted to disk as JSON; reloaded on startup

Integration points:
  - data/ingest_wikipedia.py calls build_graph_from_documents()
  - rag/retriever.py calls expand_via_graph() for graph traversal step
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import spacy

from mao.core.config import cfg

logger = logging.getLogger(__name__)

# Path where the serialised graph is stored
GRAPH_PATH = cfg.data_dir / "entity_graph.json"

# spaCy model — load once
_nlp: spacy.language.Language | None = None

# Entity types we care about
ENTITY_TYPES = {"PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART", "LAW"}


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy model: en_core_web_sm")
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> list[tuple[str, str]]:
    """
    Run spaCy NER on *text*.

    Returns list of (entity_text, entity_label) tuples for the target types.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    return [
        (ent.text.strip(), ent.label_)
        for ent in doc.ents
        if ent.label_ in ENTITY_TYPES and len(ent.text.strip()) > 1
    ]


def build_graph_from_documents(
    documents: list[dict[str, Any]],
) -> nx.Graph:
    """
    Build a co-occurrence entity graph from a list of document dicts.

    Args:
        documents: Each dict must have a "text" key; other keys become
                   node/edge attributes (e.g., "source", "chunk_id").

    Returns:
        NetworkX undirected Graph where:
          - Nodes: entity strings, with "label" and "type" attributes
          - Edges: co-occurrence count (weight) + list of source doc IDs
    """
    G = nx.Graph()
    edge_sources: dict[tuple[str, str], set[str]] = defaultdict(set)

    for doc in documents:
        text: str = doc.get("text", "")
        source: str = doc.get("source", "unknown")
        chunk_id: str = doc.get("chunk_id", "")

        entities = extract_entities(text)
        unique_in_doc = list({e[0] for e in entities})

        # Add / update nodes
        for name, etype in entities:
            if not G.has_node(name):
                G.add_node(name, type=etype, count=1)
            else:
                G.nodes[name]["count"] += 1

        # Add co-occurrence edges (all pairs in the same document)
        for i, a in enumerate(unique_in_doc):
            for b in unique_in_doc[i + 1:]:
                key = tuple(sorted([a, b]))
                if G.has_edge(*key):
                    G.edges[key]["weight"] += 1
                else:
                    G.add_edge(*key, weight=1)
                edge_sources[key].add(chunk_id or source)

    # Attach source lists to edges
    for (a, b), sources in edge_sources.items():
        if G.has_edge(a, b):
            G.edges[a, b]["sources"] = list(sources)

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

def save_graph(G: nx.Graph, path: Path | None = None) -> None:
    """Serialise graph to JSON (node-link format)."""
    target = path or GRAPH_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(G, edges="edges")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f)
    logger.info("Graph saved to %s", target)


def load_graph(path: Path | None = None) -> nx.Graph:
    """
    Load graph from disk.  Returns empty graph if file does not exist.
    """
    target = path or GRAPH_PATH
    if not target.exists():
        logger.warning("Graph file not found at %s — returning empty graph", target)
        return nx.Graph()
    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    G = nx.node_link_graph(data, edges="edges")
    logger.info("Graph loaded: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


# ---------------------------------------------------------------------------
# Graph traversal helper (called by rag/retriever.py)
# ---------------------------------------------------------------------------

def expand_via_graph(
    G: nx.Graph,
    seed_entities: list[str],
    hops: int = 2,
    max_nodes: int = 30,
) -> list[str]:
    """
    BFS from *seed_entities* up to *hops* edges deep.

    Returns the list of entity names reachable (excluding seeds themselves).
    Used by the retriever to find related chunks after the initial vector search.

    Args:
        G:              The entity graph.
        seed_entities:  Entity names found in the top-20 vector search results.
        hops:           Number of graph hops to expand.
        max_nodes:      Hard cap on returned nodes (avoids explosion).
    """
    visited: set[str] = set(seed_entities)
    frontier: set[str] = set(seed_entities) & set(G.nodes())
    expanded: list[str] = []

    for _ in range(hops):
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbour in G.neighbors(node):
                if neighbour not in visited:
                    visited.add(neighbour)
                    next_frontier.add(neighbour)
                    expanded.append(neighbour)
                    if len(expanded) >= max_nodes:
                        return expanded
        frontier = next_frontier
        if not frontier:
            break

    return expanded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = [
        {"text": "Barack Obama was born in Hawaii. He served as President of the United States.", "source": "wiki_obama", "chunk_id": "c1"},
        {"text": "Michelle Obama is married to Barack Obama. She was born in Chicago.", "source": "wiki_michelle", "chunk_id": "c2"},
    ]
    G = build_graph_from_documents(docs)
    print("Nodes:", list(G.nodes(data=True)))
    print("Edges:", list(G.edges(data=True)))
    expanded = expand_via_graph(G, ["Barack Obama"], hops=1)
    print("Expanded from Barack Obama:", expanded)
