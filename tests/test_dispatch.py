"""NetworkX backend dispatching (``nx._dispatchable``).

The graph-in, graph-out functions are registered with NetworkX's dispatcher
under ``"backbone_" + name``, so a backend can implement them and callers can
pick one with ``backend="..."``.  These tests use ``nx_loopback``, the test
backend that ships with NetworkX: it converts the input graph to its own graph
class, using what the decorator declares, and runs the NetworkX code.  A
result equal to the plain call shows that the declared conversion keeps every
attribute the function needs and returns.

Running the whole suite with ``NETWORKX_TEST_BACKEND=nx_loopback`` (see
``tests/conftest.py``) does the same for every call in every test.
"""

import warnings

import networkx as nx
import pytest

import networkx_backbone as nb
from networkx_backbone._dispatch import DISPATCH_PREFIX

pytest.importorskip("numpy")
pytest.importorskip("scipy")

# Every dispatchable function, with the extra arguments it needs on the
# karate club graph (edge "weight" and "flag", node "club", graph "name").
DISPATCHABLE = {
    # statistical
    nb.disparity_filter: {},
    nb.noise_corrected_filter: {},
    nb.marginal_likelihood_filter: {},
    nb.ecm_filter: {},
    nb.lans_filter: {},
    nb.multiple_linkage_analysis: {},
    nb.polya_filter: {},
    nb.gloss_filter: {},
    # structural
    nb.global_threshold_filter: {"threshold": 2},
    nb.strongest_n_ties: {},
    nb.global_sparsification: {},
    nb.primary_linkage_analysis: {},
    nb.edge_betweenness_filter: {},
    nb.node_degree_filter: {},
    nb.high_salience_skeleton: {},
    nb.metric_backbone: {},
    nb.ultrametric_backbone: {},
    nb.doubly_stochastic_filter: {},
    nb.h_backbone: {},
    nb.modularity_backbone: {"seed": 1},
    nb.planar_maximally_filtered_graph: {},
    nb.maximum_spanning_tree_backbone: {},
    # proximity
    nb.neighborhood_overlap: {},
    nb.jaccard_backbone: {},
    nb.dice_backbone: {},
    nb.cosine_backbone: {},
    nb.hub_promoted_index: {},
    nb.hub_depressed_index: {},
    nb.lhn_local_index: {},
    nb.preferential_attachment_score: {},
    nb.adamic_adar_index: {},
    nb.resource_allocation_index: {},
    nb.graph_distance_proximity: {},
    nb.local_path_index: {"weight": "weight"},
    # hybrid
    nb.glab_filter: {},
    # unweighted
    nb.sparsify: {"escore": "random", "seed": 1},
    nb.lspar: {},
    nb.local_degree: {},
    # filters
    nb.multigraph_to_weighted: {"aggregate": "sum"},
    nb.threshold_filter: {"score": "weight", "threshold": 3, "mode": "above"},
    nb.fraction_filter: {"score": "weight", "fraction": 0.5},
    nb.boolean_filter: {"score": "flag"},
}
IDS = [f.__name__ for f in DISPATCHABLE]


@pytest.fixture
def loopback(monkeypatch):
    """Make NetworkX's ``nx_loopback`` test backend available to ``backend=``."""
    from importlib.metadata import entry_points

    backends = nx.utils.backends
    (ep,) = entry_points(name="nx_loopback", group="networkx.backends")
    monkeypatch.setitem(backends.backends, "nx_loopback", ep)
    monkeypatch.setitem(backends.backend_info, "nx_loopback", {})
    monkeypatch.setattr(
        nx.config,
        "backends",
        nx.utils.Config(**{"nx_loopback": nx.utils.Config(), **nx.config.backends}),
    )
    dispatchable = backends._dispatchable
    if hasattr(dispatchable, "_call_if_any_backends_installed"):  # NetworkX >= 3.5
        monkeypatch.setattr(
            dispatchable, "__call__", dispatchable._call_if_any_backends_installed
        )
    return "nx_loopback"


@pytest.fixture
def karate():
    G = nx.karate_club_graph()
    for _, _, d in G.edges(data=True):
        d["flag"] = d["weight"] > 2
    return G


def _snapshot(G):
    return (
        G.is_directed(),
        G.is_multigraph(),
        dict(G.graph),
        dict(G.nodes(data=True)),
        {frozenset((u, v)): d for u, v, d in G.edges(data=True)},
    )


def test_dispatch_names_are_prefixed_and_registered():
    registry = nx.utils.backends._registered_algorithms
    for func in DISPATCHABLE:
        name = DISPATCH_PREFIX + func.__name__
        registered = registry[name]
        assert (registered.__module__, registered.__name__) == (
            func.__module__,
            func.__name__,
        )
    ours = {n for n in registry if n.startswith(DISPATCH_PREFIX)}
    assert ours == {DISPATCH_PREFIX + f.__name__ for f in DISPATCHABLE}


def test_names_that_would_collide_with_networkx():
    # Without the prefix these two would clash with NetworkX's own
    # link-prediction functions and fail at import time.
    registry = nx.utils.backends._registered_algorithms
    for name in ("adamic_adar_index", "resource_allocation_index"):
        assert registry[name].__module__.startswith("networkx.")


@pytest.mark.parametrize("func", DISPATCHABLE, ids=IDS)
def test_signature_gains_backend_keyword(func):
    import inspect

    assert "backend" in inspect.signature(func).parameters
    assert func.__wrapped__ is not None


@pytest.mark.parametrize("func", DISPATCHABLE, ids=IDS)
def test_loopback_round_trip_keeps_all_attributes(func, karate, loopback):
    kwargs = DISPATCHABLE[func]
    with warnings.catch_warnings():
        # doubly_stochastic_filter: Sinkhorn-Knopp does not converge on karate.
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = func(karate, **kwargs)
        got = func(karate, backend=loopback, **kwargs)
    assert getattr(got, "__networkx_backend__", None) == "nx_loopback"
    assert _snapshot(got) == _snapshot(expected)
    # The input graph's own attributes come through as well as the new ones.
    assert got.graph["name"] == karate.graph["name"]
    assert all(got.nodes[n]["club"] == karate.nodes[n]["club"] for n in got)


def test_backend_networkx_is_the_default(karate):
    H = nb.disparity_filter(karate, backend="networkx")
    assert _snapshot(H) == _snapshot(nb.disparity_filter(karate))


def test_unknown_backend_raises(karate):
    with pytest.raises(ImportError, match="not installed"):
        nb.disparity_filter(karate, backend="no_such_backend")


def test_graph_type_checks_run_before_dispatch(loopback):
    with pytest.raises(nx.NetworkXNotImplemented):
        nb.metric_backbone(nx.MultiGraph([(0, 1)]), backend=loopback)
    with pytest.raises(nx.NetworkXNotImplemented):
        nb.lspar(nx.DiGraph([(0, 1)]), backend=loopback)
