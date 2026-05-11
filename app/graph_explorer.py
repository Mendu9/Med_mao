from __future__ import annotations

import logging
from typing import Any

import gradio as gr

logger = logging.getLogger(__name__)

# Domain keyword sets used to colour-classify graph nodes
_DOMAIN_NODE_KEYWORDS: dict[str, list[str]] = {
    "alzheimer": [
        "alzheimer", "amyloid", "tau", "apoe", "dementia", "donepezil",
        "memantine", "cholinesterase", "plaques", "neurodegeneration",
    ],
    "stroke": [
        "stroke", "ischemic", "hemorrhagic", "tpa", "thrombus", "infarct",
        "aneurysm", "clot", "cerebrovascular", "alteplase",
    ],
    "general": [],
}

_DOMAIN_PALETTE: dict[str, str] = {
    "alzheimer": "#e07b54",
    "stroke": "#5b9bd5",
    "general": "#7ec8a4",
}


def _classify_node(label: str) -> str:
    lower = label.lower()
    for domain, keywords in _DOMAIN_NODE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return domain
    return "general"


def load_graph():
    from mao.rag.graph_builder import load_graph as _load
    return _load()


def _vector_search(query: str, n: int) -> list[dict[str, Any]]:
    from mao.rag.retriever import _vector_search as _vs
    return _vs(query=query, n=n)


def render_entity_graph(domain_filter: str = "all") -> str:
    """Load NetworkX graph and return self-contained pyvis HTML string.

    Args:
        domain_filter: "all" | "alzheimer" | "stroke" | "general"
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p>pyvis not installed. Run: pip install pyvis</p>"

    try:
        g = load_graph()
    except Exception as exc:
        logger.warning("load_graph() failed: %s", exc)
        return f"<p>Error loading graph: {exc}</p>"

    if g.number_of_nodes() == 0:
        return "<p>No graph data available. Run data ingestion first.</p>"

    top_nodes = sorted(g.degree(), key=lambda x: x[1], reverse=True)[:200]
    top_node_set = {n for n, _ in top_nodes}
    subgraph = g.subgraph(top_node_set)

    # Apply domain filter — keep matching nodes plus one-hop neighbours
    if domain_filter != "all":
        filtered = {n for n in subgraph.nodes() if _classify_node(str(n)) == domain_filter}
        neighbors: set = set()
        for n in filtered:
            neighbors.update(subgraph.neighbors(n))
        subgraph = subgraph.subgraph(filtered | neighbors)

    if subgraph.number_of_nodes() == 0:
        return f"<p>No nodes found for domain '{domain_filter}'. Try 'all' or another domain.</p>"

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        directed=False,
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)

    for node, degree in subgraph.degree():
        label = str(node)
        domain = _classify_node(label)
        color = _DOMAIN_PALETTE.get(domain, _DOMAIN_PALETTE["general"])
        size = max(10, min(50, degree * 3))
        net.add_node(
            label,
            label=label,
            size=size,
            color=color,
            title=f"{label} (domain: {domain}, degree: {degree})",
        )

    for u, v, data in subgraph.edges(data=True):
        weight = data.get("weight", 1)
        net.add_edge(str(u), str(v), value=weight)

    try:
        return net.generate_html()
    except Exception as exc:
        logger.warning("pyvis HTML generation failed: %s", exc)
        return f"<p>Graph render error: {exc}</p>"


def search_chunks(domain: str, query: str, top_k: int = 20) -> list[list]:
    """Search ChromaDB and return rows for gr.Dataframe."""
    try:
        results = _vector_search(query=query, n=top_k)
        rows = []
        for chunk in results:
            meta = chunk.get("metadata", {})
            source = chunk.get("source", meta.get("source", ""))
            chunk_id = chunk.get("chunk_id", meta.get("chunk_id", ""))
            text = chunk.get("text", "")
            rows.append([
                source,
                chunk_id,
                round(float(chunk.get("score", 0.0)), 4),
                text[:200],
            ])
        return rows
    except Exception as exc:
        logger.warning("search_chunks failed: %s", exc)
        return []


def switch_view(view_name: str) -> tuple[dict, dict]:
    """Return gr.update dicts to show/hide view sections."""
    is_graph = view_name == "Entity Graph"
    return (
        {"visible": is_graph, "__type__": "update"},
        {"visible": not is_graph, "__type__": "update"},
    )


def build_graph_explorer_tab() -> gr.Tab:
    """Build and return the Graph Explorer tab block."""
    with gr.Tab("Graph Explorer") as tab:
        gr.Markdown("## Graph Explorer")

        view_radio = gr.Radio(
            choices=["Entity Graph", "Chunk Browser"],
            value="Entity Graph",
            label="View",
            interactive=True,
        )

        with gr.Column(visible=True) as graph_section:
            gr.Markdown("### Entity Graph")
            with gr.Row():
                domain_filter = gr.Dropdown(
                    choices=["all", "alzheimer", "stroke", "general"],
                    value="all",
                    label="Domain Filter",
                    scale=2,
                )
                refresh_btn = gr.Button("Refresh Graph", variant="primary", scale=1)
                graph_stats = gr.Markdown("_Select a domain filter and click Refresh Graph._", scale=3)
            graph_html = gr.HTML(value="<p>Click 'Refresh Graph' to load.</p>")

        with gr.Column(visible=False) as chunk_section:
            gr.Markdown("### Chunk Browser")
            domain_radio = gr.Radio(
                choices=["alzheimer", "stroke"],
                value="alzheimer",
                label="Domain",
            )
            with gr.Row():
                search_box = gr.Textbox(
                    placeholder="Search chunks...",
                    label="Query",
                    scale=4,
                )
                search_btn = gr.Button("Search", variant="primary", scale=1)
                browse_btn = gr.Button("Browse All", scale=1)

            chunk_table = gr.Dataframe(
                headers=["source", "chunk_id", "score", "snippet"],
                datatype=["str", "str", "number", "str"],
                label="Results",
                interactive=False,
                wrap=True,
            )

        def _refresh_graph(df: str) -> tuple[str, str]:
            html = render_entity_graph(domain_filter=df)
            if "No graph data" in html or "No nodes found" in html:
                stats = f"_No nodes for domain '{df}'._"
            else:
                stats = f"_Graph loaded (domain: {df})._"
            return html, stats

        def _search(domain: str, query: str) -> list[list]:
            return search_chunks(domain=domain, query=query) if query.strip() else []

        def _browse_all(domain: str) -> list[list]:
            return search_chunks(domain=domain, query=domain, top_k=50)

        view_radio.change(
            fn=switch_view,
            inputs=[view_radio],
            outputs=[graph_section, chunk_section],
        )
        refresh_btn.click(
            fn=_refresh_graph,
            inputs=[domain_filter],
            outputs=[graph_html, graph_stats],
        )
        search_btn.click(
            fn=_search,
            inputs=[domain_radio, search_box],
            outputs=[chunk_table],
        )
        browse_btn.click(
            fn=_browse_all,
            inputs=[domain_radio],
            outputs=[chunk_table],
        )

    return tab
