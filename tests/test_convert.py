"""Tests for to_graph and matrix / edge-list input to the dispatch wrappers."""

import networkx as nx
import numpy as np
import pytest

import networkx_backbone as nb

sp = pytest.importorskip("scipy.sparse")


def _nx_type(G):
    """``type(G)``, mapping a backend's graph subclass to its NetworkX class.

    Under ``NETWORKX_TEST_BACKEND=nx_loopback`` with NetworkX 3.4, the
    NetworkX constructors that ``to_graph`` calls return the loopback
    backend's subclass of ``nx.Graph``/``nx.DiGraph``.
    """
    cls = type(G)
    if getattr(cls, "__networkx_backend__", "networkx") != "networkx":
        cls = cls.__base__
    return cls


# Scorers whose output does not depend on node or edge insertion order (see
# ``order_invariant`` in tests/test_api_contract.py), so a graph rebuilt from
# its adjacency matrix must give bit-identical scores.  ``h_backbone`` is left
# out: its betweenness sums differ in the last bit with adjacency order.
ORDER_INVARIANT_SCORERS = [
    nb.disparity_filter,
    nb.noise_corrected_filter,
    nb.marginal_likelihood_filter,
    nb.ecm_filter,
    nb.lans_filter,
    nb.multiple_linkage_analysis,
    nb.node_degree_filter,
    nb.high_salience_skeleton,
    nb.metric_backbone,
    nb.jaccard_backbone,
    nb.dice_backbone,
    nb.cosine_backbone,
    nb.neighborhood_overlap,
    nb.hub_promoted_index,
    nb.hub_depressed_index,
    nb.lhn_local_index,
    nb.preferential_attachment_score,
    nb.adamic_adar_index,
    nb.resource_allocation_index,
    nb.graph_distance_proximity,
    nb.local_path_index,
    nb.glab_filter,
    nb.lspar,
    nb.local_degree,
]

WEIGHTED_METHODS = [
    ("disparity", {}),
    ("mlf", {}),
    ("lans", {}),
    ("global", {"threshold": 3}),
]
UNWEIGHTED_METHODS = [("sparsify", {"seed": 1}), ("lspar", {}), ("local_degree", {})]


def _edges(G):
    """Edge data keyed by node pair (unordered for undirected graphs)."""
    key = tuple if G.is_directed() else frozenset
    return {key((u, v)): d for u, v, d in G.edges(data=True)}


def _pvalues(G):
    return {frozenset((u, v)): p for u, v, p in G.edges(data="disparity_pvalue")}


def _same(H1, H2):
    """Same nodes, edges and attributes (bit-identical scores)."""
    return (
        H1.is_directed() == H2.is_directed()
        and dict(H1.nodes) == dict(H2.nodes)
        and _edges(H1) == _edges(H2)
    )


@pytest.fixture
def les_mis():
    return nx.les_miserables_graph()


def _matrices(G):
    """The adjacency matrix of G in every supported container."""
    A = nx.to_scipy_sparse_array(G, nodelist=list(G))
    return {
        "csr_array": A,
        "coo_array": A.tocoo(),
        "csr_matrix": sp.csr_matrix(A),
        "ndarray": A.toarray(),
    }


# ---------------------------------------------------------------------------
# to_graph: adjacency matrices
# ---------------------------------------------------------------------------


def test_to_graph_round_trips_adjacency(les_mis):
    for kind, A in _matrices(les_mis).items():
        G = nb.to_graph(A, nodelist=list(les_mis))
        assert _nx_type(G) is nx.Graph, kind
        assert list(G) == list(les_mis), kind
        assert dict(G.edges) == dict(les_mis.edges), kind


def test_to_graph_default_labels_and_list_input():
    G = nb.to_graph([[0, 2.5], [2.5, 0]])
    assert list(G) == [0, 1]
    assert G[0][1] == {"weight": 2.5}


def test_to_graph_keeps_isolated_nodes():
    A = np.zeros((3, 3))
    A[0, 1] = A[1, 0] = 1.0
    G = nb.to_graph(sp.csr_array(A), nodelist="abc")
    assert list(G) == ["a", "b", "c"]
    assert list(G.edges) == [("a", "b")]


def test_to_graph_directed():
    A = np.array([[0, 1, 0], [3, 0, 2], [0, 0, 0]])
    for M in (A, sp.csr_array(A)):
        D = nb.to_graph(M, directed=True, nodelist="xyz")
        assert _nx_type(D) is nx.DiGraph
        assert sorted(D.edges(data="weight")) == [
            ("x", "y", 1),
            ("y", "x", 3),
            ("y", "z", 2),
        ]


def test_to_graph_rejects_asymmetric_undirected():
    A = np.array([[0, 1], [2, 0]])
    for M in (A, sp.csr_array(A)):
        with pytest.raises(ValueError, match="not symmetric"):
            nb.to_graph(M)


def test_to_graph_drops_explicit_sparse_zeros():
    # An explicit zero must not be an edge (and must not overwrite (1, 0)).
    A = sp.csr_array(([0.0, 0.0, 4.0, 4.0], ([0, 1, 1, 2], [1, 0, 2, 1])), (3, 3))
    G = nb.to_graph(A)
    assert list(G.edges(data="weight")) == [(1, 2, 4.0)]


def test_to_graph_weight_name_and_none():
    A = np.array([[0, 2], [2, 0]])
    assert nb.to_graph(A, weight="w")[0][1] == {"w": 2}
    assert nb.to_graph(A, weight=None)[0][1] == {}
    assert nb.to_graph(sp.csr_array(A), weight=None)[0][1] == {}


def test_to_graph_boolean_matrix():
    A = np.array([[False, True], [True, False]])
    assert nb.to_graph(A)[0][1] == {"weight": 1}


def test_to_graph_rejects_bad_matrices_and_nodelists():
    for data, match in [
        (np.zeros((2, 3)), "square"),
        (np.zeros(3), "square"),
        (sp.csr_array(np.zeros((2, 3))), "square"),
        (np.array([["a", "b"], ["b", "a"]]), "real-valued"),
    ]:
        with pytest.raises(ValueError, match=match):
            nb.to_graph(data)
    A = np.zeros((2, 2))
    with pytest.raises(ValueError, match="nodelist has 3 entries"):
        nb.to_graph(A, nodelist="abc")
    with pytest.raises(ValueError, match="duplicates"):
        nb.to_graph(A, nodelist="aa")
    with pytest.raises(ValueError, match="hashable"):
        nb.to_graph(A, nodelist=[[1], [2]])


def test_to_graph_returns_graphs_unchanged(les_mis):
    assert nb.to_graph(les_mis) is les_mis
    M = nx.MultiGraph([(0, 1), (0, 1)])
    assert nb.to_graph(M) is M


def test_to_graph_rejects_other_types():
    with pytest.raises(TypeError, match="Expected a NetworkX graph"):
        nb.to_graph({0: {1: {}}})


# ---------------------------------------------------------------------------
# to_graph: pandas edge lists
# ---------------------------------------------------------------------------


@pytest.fixture
def pd():
    return pytest.importorskip("pandas")


def test_to_graph_round_trips_edgelist(pd, les_mis):
    df = nx.to_pandas_edgelist(les_mis)
    G = nb.to_graph(df)
    assert _same(G, les_mis)


def test_to_graph_edgelist_column_names(pd):
    df = pd.DataFrame({"from": ["a", "a"], "to": ["b", "c"], "w": [3.0, 1.0]})
    G = nb.to_graph(df, source="from", target="to", weight="w")
    assert sorted(G.edges(data=True)) == [
        ("a", "b", {"w": 3.0}),
        ("a", "c", {"w": 1.0}),
    ]
    # Only the weight column is copied, and only when present.
    df["other"] = 1
    G = nb.to_graph(df, source="from", target="to", weight="missing")
    assert all(d == {} for _, _, d in G.edges(data=True))


def test_to_graph_directed_edgelist(pd):
    df = pd.DataFrame({"source": [1, 2], "target": [2, 1], "weight": [1.0, 5.0]})
    D = nb.to_graph(df, directed=True)
    assert _nx_type(D) is nx.DiGraph
    assert D[2][1]["weight"] == 5.0
    with pytest.raises(ValueError, match=r"1 duplicate row.*node pair"):
        nb.to_graph(df)


def test_to_graph_edgelist_errors(pd):
    df = pd.DataFrame({"u": [1], "v": [2]})
    with pytest.raises(ValueError, match="no column"):
        nb.to_graph(df)
    dup = pd.DataFrame({"source": [1, 1], "target": [2, 2]})
    with pytest.raises(ValueError, match=r"\(source, target\) pair"):
        nb.to_graph(dup, directed=True)


# ---------------------------------------------------------------------------
# Results are identical to the graph path
# ---------------------------------------------------------------------------


def test_scorers_identical_on_converted_matrix(les_mis):
    nodes = list(les_mis)
    G = nb.to_graph(nx.to_scipy_sparse_array(les_mis, nodelist=nodes), nodelist=nodes)
    differ = [
        scorer.__name__
        for scorer in ORDER_INVARIANT_SCORERS
        if not _same(scorer(les_mis), scorer(G))
    ]
    assert not differ, f"scores differ on the converted matrix: {differ}"


def test_backbone_from_weighted_matrix(les_mis):
    nodes = list(les_mis)
    failures = []
    for method, kwargs in WEIGHTED_METHODS:
        expected = nb.backbone_from_weighted(les_mis, method=method, **kwargs)
        for kind in ("csr_array", "csr_matrix", "ndarray"):
            A = _matrices(les_mis)[kind]
            got = nb.backbone_from_weighted(A, method=method, nodelist=nodes, **kwargs)
            if not _same(got, expected):
                failures.append(f"backbone_from_weighted({kind}, method={method!r})")
            # backbone() reads a matrix as an adjacency matrix for weighted methods.
            got = nb.backbone(A, method=method, nodelist=nodes, **kwargs)
            if not _same(got, expected):
                failures.append(f"backbone({kind}, method={method!r})")
    assert not failures, failures


def test_backbone_from_unweighted_matrix(karate):
    A = nx.to_scipy_sparse_array(karate, nodelist=list(karate), weight=None)
    for method, kwargs in UNWEIGHTED_METHODS:
        expected = nb.backbone_from_unweighted(karate, method=method, **kwargs)
        got = nb.backbone_from_unweighted(
            A, method=method, nodelist=list(karate), **kwargs
        )
        assert sorted(got.edges) == sorted(expected.edges), method
        # The matrix values are not copied as weights.
        assert all("weight" not in d for _, _, d in got.edges(data=True)), method
        got = nb.backbone(A, method=method, nodelist=list(karate), **kwargs)
        assert sorted(got.edges) == sorted(expected.edges), method


def test_backbone_from_weighted_directed_matrix():
    D = nx.DiGraph()
    D.add_weighted_edges_from(
        [(0, 1, 5.0), (1, 0, 1.0), (0, 2, 2.0), (2, 1, 4.0), (1, 2, 3.0)]
    )
    A = nx.to_scipy_sparse_array(D, nodelist=[0, 1, 2])
    got = nb.backbone_from_weighted(A, alpha=1.0, directed=True)
    assert got.is_directed()
    assert _same(got, nb.backbone_from_weighted(D, alpha=1.0))
    with pytest.raises(ValueError, match="not symmetric"):
        nb.backbone_from_weighted(A)


def test_backbone_from_weighted_edgelist(pd, les_mis):
    df = nx.to_pandas_edgelist(les_mis, source="u", target="v")
    for method, kwargs in WEIGHTED_METHODS:
        expected = nb.backbone_from_weighted(les_mis, method=method, **kwargs)
        got = nb.backbone_from_weighted(
            df, method=method, source="u", target="v", **kwargs
        )
        assert _same(got, expected), method
    # A custom weight column.
    df = nx.to_pandas_edgelist(les_mis).rename(columns={"weight": "count"})
    expected = nb.backbone_from_weighted(les_mis)
    got = nb.backbone_from_weighted(df, weight="count")
    assert _pvalues(got) == _pvalues(expected)


def test_backbone_auto_dispatch_edgelist(pd, les_mis, karate):
    # A weighted edge list gets the disparity filter, as a weighted graph does.
    df = nx.to_pandas_edgelist(les_mis)
    assert _same(nb.backbone(df), nb.backbone(les_mis))
    # Custom columns are recognized as an edge list too.
    renamed = df.rename(columns={"source": "a", "target": "b", "weight": "w"})
    got = nb.backbone(renamed, source="a", target="b", weight="w")
    assert _pvalues(got) == _pvalues(nb.backbone(les_mis))
    # Without a weight column the edge list is sparsified.
    el = nx.to_pandas_edgelist(karate)[["source", "target"]]
    unweighted = nx.from_pandas_edgelist(el)
    expected = nb.backbone_from_unweighted(unweighted, method="sparsify", seed=3)
    got = nb.backbone(el, seed=3)
    assert 0 < got.number_of_edges() < karate.number_of_edges()
    assert _same(got, expected)


def test_backbone_auto_dispatch_matrix_is_incidence(davis_southern_women_graph):
    # Without a method a matrix is still an agents x artifacts incidence matrix.
    B = davis_southern_women_graph
    women = sorted((n for n, d in B.nodes(data=True) if d["bipartite"] == 0), key=str)
    events = sorted((n for n, d in B.nodes(data=True) if d["bipartite"] == 1), key=str)
    M = nx.bipartite.biadjacency_matrix(B, row_order=women, column_order=events)
    got = nb.backbone(M, agent_labels=women, artifact_labels=events)
    assert _same(got, nb.backbone(B))


def test_wrappers_ignore_conversion_options_for_graphs(les_mis):
    expected = nb.backbone_from_weighted(les_mis)
    got = nb.backbone_from_weighted(les_mis, source="x", target="y")
    assert _same(got, expected)
