"""Shared test fixtures for networkx_backbone tests."""

import networkx as nx
import pytest


@pytest.fixture
def weighted_triangle():
    """Simple weighted triangle graph: 0-1 (3.0), 1-2 (1.0), 0-2 (2.0)."""
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 1.0), (0, 2, 2.0)])
    return G


@pytest.fixture
def weighted_star():
    """Weighted star graph centered on node 0."""
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 5.0), (0, 2, 3.0), (0, 3, 1.0)])
    return G


@pytest.fixture
def karate():
    """Zachary's karate club graph (unweighted)."""
    return nx.karate_club_graph()


@pytest.fixture
def complete4():
    """Complete graph on 4 nodes (unweighted)."""
    return nx.complete_graph(4)


@pytest.fixture
def bipartite_graph():
    """Simple bipartite graph with agents [1,2,3] and artifacts ['a','b','c']."""
    B = nx.Graph()
    B.add_edges_from([
        (1, "a"), (1, "b"),
        (2, "a"), (2, "c"),
        (3, "b"), (3, "c"),
    ])
    return B
