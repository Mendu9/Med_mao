"""Tests for app/graph_explorer.py — Streamlit + Plotly 3D graph explorer."""
from __future__ import annotations

import networkx as nx
from unittest.mock import MagicMock, patch


def test_get_node_color_known_type():
    """Nodes with a known node_type get a non-default palette color."""
    from app.graph_explorer import _get_node_color, _DEFAULT_COLOR
    color = _get_node_color("amyloid-beta", {"node_type": "disease"})
    assert color != _DEFAULT_COLOR
    assert color.startswith("#")


def test_get_node_color_unknown_type():
    """Nodes with an unknown type fall back to the default grey color."""
    from app.graph_explorer import _get_node_color, _DEFAULT_COLOR
    color = _get_node_color("unknown-node", {"node_type": "xyzzy"})
    assert color == _DEFAULT_COLOR


def test_node_size_scales_with_degree():
    """Higher degree nodes get larger marker sizes, clamped to [min, max]."""
    from app.graph_explorer import _node_size
    small = _node_size(0)
    medium = _node_size(10)
    large = _node_size(1000)
    assert small <= medium <= large
    assert small >= 4.0
    assert large <= 15.0


def test_classify_node_domain_alzheimer():
    """Node ID containing 'amyloid' is classified as Alzheimer domain."""
    from app.graph_explorer import _classify_node_domain
    domain = _classify_node_domain("amyloid-beta", {})
    assert domain == "Alzheimer"


def test_classify_node_domain_stroke():
    """Node ID containing 'ischemi' is classified as Stroke domain."""
    from app.graph_explorer import _classify_node_domain
    domain = _classify_node_domain("ischemic_stroke", {})
    assert domain == "Stroke"


def test_classify_node_domain_unknown():
    """Node ID with no known keywords is classified as 'general'."""
    from app.graph_explorer import _classify_node_domain
    domain = _classify_node_domain("completely_unknown_xyz", {})
    assert domain == "general"


def test_render_graph_explorer_disabled(monkeypatch):
    """render_graph_explorer shows warning when MAO_DISABLE_GRAPH=1."""
    monkeypatch.setenv("MAO_DISABLE_GRAPH", "1")
    mock_st = MagicMock()
    with patch.dict("sys.modules", {"streamlit": mock_st}):
        import importlib, app.graph_explorer as ge
        importlib.reload(ge)
        ge.render_graph_explorer()
    mock_st.warning.assert_called_once()


def test_render_graph_explorer_empty_graph(monkeypatch):
    """render_graph_explorer shows info message when graph has no nodes."""
    monkeypatch.setenv("MAO_DISABLE_GRAPH", "0")
    mock_st = MagicMock()
    # st.columns must return a list of context-manager mocks, not a single MagicMock
    col = MagicMock()
    col.__enter__ = MagicMock(return_value=col)
    col.__exit__ = MagicMock(return_value=False)
    mock_st.columns.return_value = [col, col, col, col]
    mock_st.selectbox.return_value = "All"
    mock_st.slider.return_value = 200
    mock_st.text_input.return_value = ""
    mock_st.checkbox.return_value = False
    empty_graph = nx.MultiDiGraph()
    with patch.dict("sys.modules", {"streamlit": mock_st}):
        import importlib, app.graph_explorer as ge
        importlib.reload(ge)
        with patch.object(ge, "_load_graph", return_value=empty_graph):
            ge.render_graph_explorer()
    mock_st.warning.assert_called()
