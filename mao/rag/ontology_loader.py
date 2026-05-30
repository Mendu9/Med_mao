"""Ontology backbone loader for HPO (Human Phenotype Ontology) and MONDO disease ontology.

Adds structured is_a hierarchy and disease-phenotype links to the entity graph.
Only loads subtrees relevant to Alzheimer's and Stroke to keep memory footprint small.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)

# HPO root terms for relevant subtrees
_HPO_ROOTS = {
    "alzheimer": "HP:0002511",   # Alzheimer disease
    "stroke": "HP:0001297",      # Stroke
    "dementia": "HP:0000726",    # Dementia
    "cognitive": "HP:0100543",   # Cognitive impairment
}

# MONDO disease IDs
_MONDO_DISEASES = {
    "alzheimer": "MONDO:0004975",
    "stroke": "MONDO:0005098",
    "vascular_dementia": "MONDO:0002254",
    "parkinson": "MONDO:0005180",  # related neurodegenerative
}


def load_hpo_subgraph(G: "nx.MultiDiGraph", domains: list[str] | None = None) -> int:
    """Load HPO phenotype terms for specified domains into G.

    Downloads hp.obo from OBO Library (cached after first download).
    Only loads subtrees under the domain root terms to limit size.

    Returns count of nodes added.
    """
    try:
        from pronto import Ontology
    except ImportError:
        logger.warning("pronto not installed. Run: pip install pronto. Skipping HPO load.")
        return 0

    if domains is None:
        domains = ["alzheimer", "stroke"]

    try:
        logger.info("Loading HPO ontology (may take 30s on first run)...")
        hp = Ontology("http://purl.obolibrary.org/obo/hp.obo")
    except Exception as exc:
        logger.warning("Failed to load HPO: %s", exc)
        return 0

    count = 0
    roots_to_load = {k: v for k, v in _HPO_ROOTS.items() if k in domains or k in ("dementia", "cognitive")}

    for domain_name, root_id in roots_to_load.items():
        try:
            root_term = hp[root_id]
        except KeyError:
            logger.warning("HPO term %s not found", root_id)
            continue

        # Load root + all descendants
        for term in root_term.subclasses():
            node_id = term.id
            if node_id not in G.nodes:
                G.add_node(
                    node_id,
                    name=term.name or node_id,
                    node_type="phenotype",
                    ontology_id=node_id,
                    source="hpo",
                    domain=domain_name,
                )
                count += 1

            # Add is_a edges to parents
            for parent in term.superclasses(distance=1, with_self=False):
                if parent.id in G.nodes:
                    G.add_edge(
                        node_id, parent.id,
                        relation="is_a",
                        source="hpo",
                        edge_type="ontology",
                    )

    logger.info("HPO: added %d phenotype nodes", count)
    return count


def load_mondo_diseases(G: "nx.MultiDiGraph") -> int:
    """Load MONDO disease hierarchy for Alzheimer + Stroke subtrees into G.

    Returns count of nodes added.
    """
    try:
        from pronto import Ontology
    except ImportError:
        logger.warning("pronto not installed. Skipping MONDO load.")
        return 0

    try:
        logger.info("Loading MONDO ontology (may take 60s on first run)...")
        mondo = Ontology("http://purl.obolibrary.org/obo/mondo.obo")
    except Exception as exc:
        logger.warning("Failed to load MONDO: %s", exc)
        return 0

    count = 0
    for disease_name, mondo_id in _MONDO_DISEASES.items():
        try:
            root_term = mondo[mondo_id]
        except KeyError:
            logger.warning("MONDO term %s not found", mondo_id)
            continue

        for term in root_term.subclasses():
            node_id = term.id
            if node_id not in G.nodes:
                G.add_node(
                    node_id,
                    name=term.name or node_id,
                    node_type="disease",
                    ontology_id=node_id,
                    source="mondo",
                    domain=disease_name,
                    # Store synonyms for query expansion
                    synonyms=[s.description for s in term.synonyms],
                )
                count += 1

            for parent in term.superclasses(distance=1, with_self=False):
                if parent.id in G.nodes:
                    G.add_edge(
                        node_id, parent.id,
                        relation="is_a",
                        source="mondo",
                        edge_type="ontology",
                    )

    logger.info("MONDO: added %d disease nodes", count)
    return count


def build_synonym_map(G: "nx.MultiDiGraph") -> dict[str, str]:
    """Build a synonym -> canonical_name map from loaded ontology nodes.

    Used by retriever for query expansion.
    Returns {synonym_lower: canonical_name}.
    """
    synonym_map: dict[str, str] = {}
    for node_id, attrs in G.nodes(data=True):
        name = attrs.get("name", "")
        if not name:
            continue
        synonym_map[name.lower()] = name
        for syn in attrs.get("synonyms", []):
            if syn:
                synonym_map[syn.lower()] = name
    return synonym_map
