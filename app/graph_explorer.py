"""Streamlit 3D Plotly knowledge graph explorer for MAO."""
from __future__ import annotations

import logging
import os
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette — kept identical to the original file so streamlit_app.py
# _NODE_COLORS dict stays in sync with the graph display.
# ---------------------------------------------------------------------------
_NODE_TYPE_PALETTE: dict[str, str] = {
    "disease":            "#e07b54",  # orange-red
    "chemical":           "#5b9bd5",  # blue
    "DISEASE":            "#e07b54",
    "CHEMICAL":           "#5b9bd5",
    "drug":               "#9b59b6",  # purple
    "gene/protein":       "#27ae60",  # green
    "phenotype":          "#f39c12",  # amber
    "effect/phenotype":   "#f39c12",
    "biological_process": "#16a085",  # teal
    "pathway":            "#2980b9",  # dark blue
    "anatomy":            "#8e44ad",  # violet
    "entity":             "#7ec8a4",  # mint (extracted, untyped)
    "ENTITY":             "#7ec8a4",
    # Legacy domain fallback
    "alzheimer":          "#e07b54",
    "stroke":             "#5b9bd5",
    "general":            "#7ec8a4",
}

_DEFAULT_COLOR = "#95a5a6"  # grey for unknown types

# Node types produced by generic spaCy en_core_web_sm that are NOT biomedical.
# Nodes carrying any of these types are stripped from the display graph.
_NON_BIOMEDICAL_TYPES: frozenset[str] = frozenset({
    "PERSON", "ORG", "GPE", "LOC", "NORP", "FAC",
    "CARDINAL", "ORDINAL", "PERCENT", "MONEY",
    "DATE", "TIME", "EVENT", "LANGUAGE", "LAW",
    "WORK_OF_ART", "PRODUCT",
})

# Domain keyword sets for domain-filter fallback (when node_type not set)
_DOMAIN_NODE_KEYWORDS: dict[str, list[str]] = {
    "Alzheimer": [
        "apoe", "app", "psen", "trem2", "abca7", "mapt",
        "amyloid", "tau", "lewy", "necroptosis", "microglia",
        "cholinergic", "nmda", "presenilin", "secretase",
        "alzheim", "alzhe", "dementia", "mci",
        "memantine", "donepezil", "donanemab", "lecanemab",
        "csf", "pet", "fdg",
        "alois", "braak",
        "HP:0002511", "MONDO:0004975",
        # Additional biomedical keywords
        "bdnf", "vegf", "mtor", "autophagy", "lysosome",
        "synuclein", "tdp-43", "fus", "granulin", "progranulin",
        "galantamine", "rivastigmine", "cognitive",
        "hippocampus", "entorhinal", "prefrontal",
        "neuritic", "tangle", "plaque", "synapt",
    ],
    "Stroke": [
        "stroke", "ischemi", "hemorrhag", "subarachnoid", "lacunar",
        "emboli", "tia", "infarct",
        "cerebrovascular", "cerebral", "wernicke", "broca",
        "aphasia", "apraxia", "dysphagia", "nihss",
        "alteplase", "clopidogrel", "aspirin", "tpa",
        "doppler", "holter", "ecg", "pfo",
        "HP:0001297", "MONDO:0005098",
        # Additional biomedical keywords
        "thrombect", "penumbra", "collateral",
        "rankin", "anticoagul", "warfarin",
        "dabigatran", "rivaroxaban", "apixaban",
        "carotid", "atrial",
    ],
}

_EDGE_COLORS: dict[str, str] = {
    "ontology":      "#f39c12",
    "semantic":      "#27ae60",
    "curated":       "#9b59b6",
    "co-occurrence": "rgba(85,85,119,0.6)",
}
_DEFAULT_EDGE_COLOR = "rgba(150,150,150,0.3)"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_node_color(node_id: str, attrs: dict[str, Any]) -> str:
    """Return a hex color for a node based on its node_type attribute."""
    node_type = attrs.get("node_type", "")
    if node_type and node_type in _NODE_TYPE_PALETTE:
        return _NODE_TYPE_PALETTE[node_type]
    label = str(node_id).lower()
    for domain, keywords in _DOMAIN_NODE_KEYWORDS.items():
        if any(kw in label for kw in keywords):
            return _NODE_TYPE_PALETTE.get(domain.lower(), _DEFAULT_COLOR)
    return _DEFAULT_COLOR


def _classify_node_domain(node_id: str, attrs: dict[str, Any]) -> str:
    """Return 'Alzheimer' | 'Stroke' | 'general' for a given node."""
    stored = attrs.get("domain", "")
    if stored in ("Alzheimer", "alzheimer"):
        return "Alzheimer"
    if stored in ("Stroke", "stroke"):
        return "Stroke"
    label = str(node_id).lower()
    for domain, keywords in _DOMAIN_NODE_KEYWORDS.items():
        if any(kw in label for kw in keywords):
            return domain
    return "general"


def _node_size(degree: int, min_size: float = 4.0, max_size: float = 15.0) -> float:
    """Map node degree to a Plotly marker size in [min_size, max_size].

    Uses log1p scaling so hub nodes don't overwhelm the visualisation.
    """
    import math
    scaled = math.log1p(degree) * 2.5
    return max(min_size, min(max_size, scaled))


def _load_graph() -> Any:
    """Load the MAO knowledge graph. Returns an empty DiGraph on failure."""
    import networkx as nx
    try:
        from mao.rag.graph_builder import load_graph as _load, clean_graph
        g = _load()
        g = clean_graph(g)  # Remove any non-biomedical nodes
        return g
    except Exception as exc:
        logger.warning("load_graph() failed: %s", exc)
        return nx.DiGraph()


def _compute_layout(
    g: Any,
    layout_choice: str,
) -> dict[Any, tuple[float, float, float]]:
    """Compute 3-D node positions for the requested layout strategy.

    Returns:
        Mapping from node id to (x, y, z) float tuple.
    """
    import math
    import networkx as nx

    if layout_choice == "3D Force":
        pos2d: dict[Any, Any] = nx.spring_layout(g, seed=42, k=0.3)
        degrees = dict(g.degree())
        max_deg = max(degrees.values(), default=1)
        return {
            n: (float(xy[0]), float(xy[1]), degrees.get(n, 0) / max_deg)
            for n, xy in pos2d.items()
        }

    if layout_choice == "3D Sphere":
        nodes = list(g.nodes())
        total = len(nodes)
        pos_sphere: dict[Any, tuple[float, float, float]] = {}
        for i, node in enumerate(nodes):
            # Fibonacci sphere — uniformly distributes points on a unit sphere
            phi = math.acos(1 - 2 * (i + 0.5) / max(total, 1))
            theta = math.pi * (1 + 5 ** 0.5) * i
            pos_sphere[node] = (
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi),
            )
        return pos_sphere

    # "2D Spring" — spring layout projected at z=0
    pos2d_spring: dict[Any, Any] = nx.spring_layout(g, seed=42, k=0.3)
    return {n: (float(xy[0]), float(xy[1]), 0.0) for n, xy in pos2d_spring.items()}


def _build_plotly_figure(
    g: Any,
    pos: dict[Any, tuple[float, float, float]],
    show_edge_labels: bool,
) -> Any:
    """Build and return a Plotly Figure with Scatter3d edge and node traces.

    Args:
        g:                NetworkX graph (already sliced to display subset).
        pos:              Node-id -> (x, y, z) position mapping.
        show_edge_labels: If True, relation label appears in edge hover text.

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    nodes = list(g.nodes())
    degrees = dict(g.degree())

    # --- Edge trace ---------------------------------------------------------
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []

    for u, v, _edata in g.edges(data=True):
        x0, y0, z0 = pos.get(u, (0.0, 0.0, 0.0))
        x1, y1, z1 = pos.get(v, (0.0, 0.0, 0.0))
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line={"width": 1, "color": _DEFAULT_EDGE_COLOR},
        hoverinfo="none",
        showlegend=False,
        name="edges",
    )

    # --- Node trace ---------------------------------------------------------
    node_x = [pos.get(n, (0.0, 0.0, 0.0))[0] for n in nodes]
    node_y = [pos.get(n, (0.0, 0.0, 0.0))[1] for n in nodes]
    node_z = [pos.get(n, (0.0, 0.0, 0.0))[2] for n in nodes]
    node_colors = [_get_node_color(str(n), dict(g.nodes[n])) for n in nodes]
    node_sizes  = [_node_size(degrees.get(n, 0)) for n in nodes]

    hover_texts: list[str] = []
    for n in nodes:
        attrs = dict(g.nodes[n])
        node_type = attrs.get("node_type", "entity")
        source = attrs.get("source", "")
        ontology_id = attrs.get("ontology_id", "")
        deg = degrees.get(n, 0)
        try:
            # Show first 5 connected nodes (mix of in/out for directed graphs)
            nbrs_out = list(g.successors(n))[:3]
            nbrs_in  = list(g.predecessors(n))[:2]
            nbrs = list(dict.fromkeys(nbrs_out + nbrs_in))[:5]
        except Exception:
            nbrs = []
        nbr_str = ", ".join(str(nb) for nb in nbrs) if nbrs else "none"
        lines = [
            f"<b>{n}</b>",
            f"type: {node_type}",
            f"degree: {deg}",
            f"connected: {nbr_str}",
        ]
        if ontology_id:
            lines.append(f"id: {ontology_id}")
        if source:
            lines.append(f"source: {source}")
        hover_texts.append("<br>".join(lines))

    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers",
        marker={
            "size":    node_sizes,
            "color":   node_colors,
            "opacity": 0.85,
            "line":    {"width": 0.5, "color": "rgba(255,255,255,0.2)"},
        },
        text=[str(n) for n in nodes],
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
        name="nodes",
    )

    # --- Figure layout ------------------------------------------------------
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        scene={
            "bgcolor": "#0e1117",
            "xaxis": {
                "showgrid": False, "zeroline": False,
                "showticklabels": False, "title": "",
            },
            "yaxis": {
                "showgrid": False, "zeroline": False,
                "showticklabels": False, "title": "",
            },
            "zaxis": {
                "showgrid": False, "zeroline": False,
                "showticklabels": False, "title": "",
            },
            "camera": {
                "eye": {"x": 1.4, "y": 1.4, "z": 0.8},
                "up":  {"x": 0.0, "y": 0.0, "z": 1.0},
            },
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=620,
        hoverlabel={"bgcolor": "#1e1e2e", "font_color": "#e0e0e0"},
    )
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_graph_explorer() -> None:
    """Render the full graph explorer UI into the current Streamlit context.

    Called by streamlit_app.py inside the 'Graph Explorer' tab.
    Handles MAO_DISABLE_GRAPH=1 and empty-graph edge cases gracefully.

    Does NOT call st.set_page_config() and does NOT import gradio.
    """
    st.markdown("### Knowledge Graph Explorer")
    st.caption(
        "Node colors: Disease · Chemical · Drug · Gene/Protein · "
        "Phenotype · Pathway · Anatomy  |  "
        "Node size scales with degree (connectivity)."
    )

    # Guard: MAO_DISABLE_GRAPH=1 disables graph loading (HF Spaces CPU constraint)
    if os.getenv("MAO_DISABLE_GRAPH", "0") == "1":
        st.warning(
            "Graph explorer is disabled in this environment "
            "(MAO_DISABLE_GRAPH=1). "
            "Run locally with the full stack to enable it."
        )
        return

    # --- Controls -----------------------------------------------------------
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 2, 2, 3])

    with ctrl_col1:
        domain_filter: str = st.selectbox(
            "Domain",
            options=["All", "Alzheimer", "Stroke"],
            index=0,
            key="ge_domain",
        )  # type: ignore[assignment]

    with ctrl_col2:
        layout_choice: str = st.selectbox(
            "Layout",
            options=["3D Force", "3D Sphere", "2D Spring"],
            index=0,
            key="ge_layout",
        )  # type: ignore[assignment]

    with ctrl_col3:
        max_nodes: int = st.slider(
            "Max nodes",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            key="ge_max_nodes",
            help="Cap to top-N by degree for performance. Full graph may have 100K+ nodes.",
        )

    with ctrl_col4:
        search_query: str = st.text_input(
            "Search / highlight node",
            placeholder="e.g. APOE, amyloid, stroke",
            key="ge_search",
        )

    show_edge_labels: bool = st.checkbox(
        "Show edge labels in hover",
        value=False,
        key="ge_edge_labels",
    )

    # --- Load graph ---------------------------------------------------------
    with st.spinner("Loading knowledge graph..."):
        g = _load_graph()

    if g.number_of_nodes() == 0:
        st.warning(
            "Knowledge graph is empty. Run data ingestion first:\n\n"
            "```bash\npython -m mao.data.ingest_wikipedia\n```"
        )
        return

    # --- Filter non-biomedical nodes (PERSON, GPE, LOC, etc.) ---------------
    display_nodes = [
        n for n in g.nodes()
        if g.nodes[n].get("node_type", "ENTITY") not in _NON_BIOMEDICAL_TYPES
    ]
    if len(display_nodes) < g.number_of_nodes():
        g = g.subgraph(display_nodes).copy()

    # Capture full-graph stats before any slicing
    full_node_count = g.number_of_nodes()
    full_edge_count = g.number_of_edges()

    # --- Domain filter ------------------------------------------------------
    if domain_filter != "All":
        matched: set = {
            n for n in g.nodes()
            if _classify_node_domain(str(n), dict(g.nodes[n])) == domain_filter
        }
        neighbours: set = set()
        for n in matched:
            try:
                neighbours.update(g.successors(n))
                neighbours.update(g.predecessors(n))
            except Exception:
                pass
        keep = matched | neighbours
        if keep:
            g = g.subgraph(keep).copy()
        else:
            st.warning(
                f"No nodes found for domain '{domain_filter}'. "
                "Try 'All' or a different domain."
            )
            return

    # --- Search filter — collect highlight set before capping ---------------
    search_highlight: set = set()
    if search_query.strip():
        q_lower = search_query.strip().lower()
        search_highlight = {n for n in g.nodes() if q_lower in str(n).lower()}
        if not search_highlight:
            st.info(f"No nodes matched '{search_query}'. Showing full subgraph.")

    # --- Cap to max_nodes by degree -----------------------------------------
    top_nodes = sorted(g.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
    top_node_set = {n for n, _ in top_nodes}
    # Always include search-matched nodes (up to 20 extras) even if outside top-N
    if search_highlight:
        extras = list(search_highlight - top_node_set)[:20]
        top_node_set.update(extras)
    g = g.subgraph(top_node_set).copy()

    # --- Stats row ----------------------------------------------------------
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Displayed nodes", g.number_of_nodes())
    s2.metric("Displayed edges", g.number_of_edges())
    s3.metric("Full graph nodes", f"{full_node_count:,}")
    s4.metric("Full graph edges", f"{full_edge_count:,}")

    type_counts: dict[str, int] = {}
    for _, attrs in g.nodes(data=True):
        nt = attrs.get("node_type", "entity")
        type_counts[nt] = type_counts.get(nt, 0) + 1

    if type_counts:
        top_types = sorted(type_counts.items(), key=lambda x: -x[1])[:8]
        st.caption(
            "Node types (displayed): "
            + "  |  ".join(f"{nt}: {cnt:,}" for nt, cnt in top_types)
        )

    # --- Layout + render ----------------------------------------------------
    with st.spinner(f"Computing {layout_choice} layout for {g.number_of_nodes()} nodes..."):
        pos = _compute_layout(g, layout_choice)

    try:
        import plotly.graph_objects as go

        fig = _build_plotly_figure(g, pos, show_edge_labels=show_edge_labels)

        # Overlay a highlight trace for search matches
        if search_highlight:
            nodes_in_view = list(g.nodes())
            hl_x: list[float] = []
            hl_y: list[float] = []
            hl_z: list[float] = []
            hl_text: list[str] = []
            for n in nodes_in_view:
                if n in search_highlight:
                    x, y, z = pos.get(n, (0.0, 0.0, 0.0))
                    hl_x.append(x)
                    hl_y.append(y)
                    hl_z.append(z)
                    hl_text.append(str(n))
            if hl_x:
                fig.add_trace(go.Scatter3d(
                    x=hl_x, y=hl_y, z=hl_z,
                    mode="markers+text",
                    marker={
                        "size": 14,
                        "color": "#ffffff",
                        "opacity": 1.0,
                        "line": {"width": 2, "color": "#f39c12"},
                    },
                    text=hl_text,
                    textfont={"color": "#f39c12", "size": 11},
                    textposition="top center",
                    hoverinfo="text",
                    showlegend=False,
                    name="search_match",
                ))

        st.plotly_chart(fig, use_container_width=True)

        # --- Graph Statistics expander --------------------------------------
        with st.expander("Graph Statistics", expanded=False):
            total_nodes = g.number_of_nodes()
            total_edges = g.number_of_edges()

            # Count by node type
            stat_type_counts: dict[str, int] = {}
            for _n, _attrs in g.nodes(data=True):
                nt = _attrs.get("node_type", "unknown")
                stat_type_counts[nt] = stat_type_counts.get(nt, 0) + 1

            stat_col1, stat_col2 = st.columns(2)
            with stat_col1:
                st.metric("Total Nodes", total_nodes)
                st.metric("Total Edges", total_edges)
            with stat_col2:
                st.write("**Node Types:**")
                for nt, count in sorted(
                    stat_type_counts.items(), key=lambda x: -x[1]
                )[:10]:
                    st.write(f"- {nt}: {count}")

    except ImportError:
        st.error(
            "plotly is not installed. Add `plotly>=5.18.0` to requirements.txt "
            "and reinstall: `pip install plotly>=5.18.0`"
        )
        return

    # --- Node detail expander -----------------------------------------------
    if search_query.strip() and search_highlight:
        with st.expander(
            f"Node details — {len(search_highlight)} match(es) for '{search_query}'",
            expanded=True,
        ):
            for n in list(search_highlight)[:10]:
                if n not in g.nodes:
                    continue
                attrs = dict(g.nodes[n])
                node_type = attrs.get("node_type", "entity")
                deg = g.degree(n)
                try:
                    successors   = list(g.successors(n))[:5]
                    predecessors = list(g.predecessors(n))[:5]
                except Exception:
                    successors, predecessors = [], []

                st.markdown(f"**{n}**")
                dcol1, dcol2, dcol3 = st.columns(3)
                dcol1.markdown(f"Type: `{node_type}`")
                dcol2.markdown(f"Degree: `{deg}`")
                dcol3.markdown(f"Domain: `{_classify_node_domain(str(n), attrs)}`")
                if successors:
                    st.caption("Outgoing: " + ", ".join(str(s) for s in successors))
                if predecessors:
                    st.caption("Incoming: " + ", ".join(str(p) for p in predecessors))
                st.divider()

    # --- Node-type breakdown table ------------------------------------------
    with st.expander("Node type breakdown", expanded=False):
        if type_counts:
            import pandas as pd
            df = pd.DataFrame(
                sorted(type_counts.items(), key=lambda x: -x[1]),
                columns=["node_type", "count"],
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No type data available.")


def build_graph_explorer_tab(*args, **kwargs):
    """Compatibility stub for legacy frontend.py import — does nothing."""
    pass
