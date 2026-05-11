import pytest
import networkx as nx
from unittest.mock import patch


def test_render_empty_graph() -> None:
    """Empty NetworkX graph returns HTML containing 'No graph data'."""
    empty_graph = nx.Graph()
    with patch("app.graph_explorer.load_graph", return_value=empty_graph):
        from app.graph_explorer import render_entity_graph
        html = render_entity_graph()
    assert "No graph data" in html


def test_render_with_nodes() -> None:
    """Graph with 5 nodes returns non-empty HTML string."""
    g = nx.Graph()
    nodes = ["Alzheimer", "Amyloid", "Tau", "Donepezil", "Memory"]
    g.add_nodes_from(nodes)
    g.add_edges_from([("Alzheimer", "Amyloid"), ("Alzheimer", "Tau"), ("Donepezil", "Memory")])
    with patch("app.graph_explorer.load_graph", return_value=g):
        from app.graph_explorer import render_entity_graph
        html = render_entity_graph()
    assert isinstance(html, str)
    assert len(html) > 100
    assert any(n in html for n in nodes)


def _make_chunks(n: int) -> list[dict]:
    return [
        {
            "text": f"Chunk {i} about Alzheimer treatment and donepezil therapy.",
            "score": round(0.9 - i * 0.05, 2),
            "source": f"doc_{i}.pdf",
            "chunk_id": f"c{i:03d}",
        }
        for i in range(n)
    ]


def test_search_chunks_returns_rows() -> None:
    """mock _vector_search returning 3 chunks produces 3 rows each with 4 columns."""
    with patch("app.graph_explorer._vector_search", return_value=_make_chunks(3)):
        from app.graph_explorer import search_chunks
        rows = search_chunks(domain="alzheimer", query="donepezil treatment")
    assert len(rows) == 3
    for row in rows:
        assert len(row) == 4


def test_search_chunks_on_error() -> None:
    """_vector_search raising Exception returns empty list without propagating."""
    with patch("app.graph_explorer._vector_search", side_effect=Exception("ChromaDB down")):
        from app.graph_explorer import search_chunks
        rows = search_chunks(domain="alzheimer", query="anything")
    assert rows == []


def test_switch_view_graph() -> None:
    """Radio='Entity Graph' -> graph section visible=True, chunk section visible=False."""
    from app.graph_explorer import switch_view
    graph_update, chunk_update = switch_view("Entity Graph")
    assert graph_update["visible"] is True
    assert chunk_update["visible"] is False


def test_switch_view_chunks() -> None:
    """Radio='Chunk Browser' -> graph section visible=False, chunk section visible=True."""
    from app.graph_explorer import switch_view
    graph_update, chunk_update = switch_view("Chunk Browser")
    assert graph_update["visible"] is False
    assert chunk_update["visible"] is True
