"""Tests for visualization utilities."""

import networkx as nx
import pytest

from networkx_backbone import compare_graphs, graph_difference


def test_graph_difference_nodes_and_edges():
    G = nx.path_graph([0, 1, 2, 3])
    H = nx.Graph()
    H.add_nodes_from([0, 1, 2])
    H.add_edge(0, 1)

    diff = graph_difference(G, H)

    assert diff["removed_nodes"] == {3}
    assert diff["kept_nodes"] == {0, 1, 2}
    assert diff["kept_edges"] == {(0, 1)}
    assert diff["removed_edges"] == {(1, 2), (2, 3)}


def test_graph_difference_directed_orientation():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 0), (1, 2)])
    H = nx.DiGraph()
    H.add_edges_from([(0, 1)])

    diff = graph_difference(G, H)
    assert diff["kept_edges"] == {(0, 1)}
    assert diff["removed_edges"] == {(1, 0), (1, 2)}


def test_compare_graphs_returns_diff():
    pytest.importorskip("matplotlib")

    G = nx.cycle_graph(4)
    H = nx.path_graph([0, 1, 2])

    fig, ax, diff = compare_graphs(G, H, return_diff=True)
    assert fig is not None
    assert ax is not None
    assert 3 in diff["removed_nodes"]
