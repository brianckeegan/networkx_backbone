"""Shared test fixtures for networkx_backbone tests.

Setting ``NETWORKX_TEST_BACKEND`` runs the suite under NetworkX's dispatch
test harness, as NetworkX's own ``conftest.py`` does for its tests: every
call to a dispatchable function (ours and NetworkX's) converts its input
graphs to the named backend, calls that backend, and, for functions that
return graphs, checks that the result equals the plain NetworkX result::

    NETWORKX_TEST_BACKEND=nx_loopback pytest --doctest-modules networkx_backbone tests

``nx_loopback`` is the test backend that ships with NetworkX.  Set
``NETWORKX_FALLBACK_TO_NX=1`` to run functions a backend does not implement
with NetworkX instead of marking their tests as expected failures.
"""

import os

import networkx as nx
import pytest


def pytest_configure(config):
    backend = os.environ.get("NETWORKX_TEST_BACKEND")
    config.nx_test_backend = backend
    if not backend:
        return
    if backend == "nx_loopback":
        # NetworkX hides its loopback backend outside of its own test runs.
        from importlib.metadata import entry_points

        (ep,) = entry_points(name="nx_loopback", group="networkx.backends")
        nx.utils.backends.backends["nx_loopback"] = ep
        nx.utils.backends.backend_info["nx_loopback"] = {}
        nx.config.backends = nx.utils.Config(
            nx_loopback=nx.utils.Config(), **nx.config.backends
        )
    nx.config.fallback_to_nx = bool(os.environ.get("NETWORKX_FALLBACK_TO_NX"))
    dispatchable = nx.utils.backends._dispatchable
    if hasattr(dispatchable, "_call_if_any_backends_installed"):  # NetworkX >= 3.5
        dispatchable.__call__ = dispatchable._call_if_any_backends_installed


def pytest_collection_modifyitems(config, items):
    # As in NetworkX's conftest: switch to the backend only after collection,
    # so module-level test inputs are built with plain NetworkX.
    backend = config.nx_test_backend
    if backend and backend != "networkx":
        nx.utils.backends._dispatchable._is_testing = True
        nx.config.backend_priority.algos = [backend]
        nx.config.backend_priority.generators = [backend]


@pytest.fixture
def weighted_triangle():
    """Simple weighted triangle graph: 0-1 (3.0), 1-2 (1.0), 0-2 (2.0)."""
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 1.0), (0, 2, 2.0)])
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
    B.add_edges_from(
        [
            (1, "a"),
            (1, "b"),
            (2, "a"),
            (2, "c"),
            (3, "b"),
            (3, "c"),
        ]
    )
    return B


@pytest.fixture
def davis_southern_women_graph():
    """Davis Southern Women bipartite attendance network."""
    return nx.davis_southern_women_graph()


@pytest.fixture
def davis_women_nodes(davis_southern_women_graph):
    """Women partition from the Davis Southern Women graph."""
    return [
        n
        for n, data in davis_southern_women_graph.nodes(data=True)
        if data.get("bipartite") == 0
    ]


@pytest.fixture
def les_miserables_weighted():
    """Weighted co-appearance graph from Victor Hugo's *Les Miserables*."""
    return nx.les_miserables_graph()


@pytest.fixture
def les_miserables_unweighted(les_miserables_weighted):
    """Unweighted version of Les Miserables preserving nodes but dropping edge attrs."""
    G = nx.Graph()
    G.add_nodes_from(les_miserables_weighted.nodes(data=True))
    G.add_edges_from(les_miserables_weighted.edges())
    return G
