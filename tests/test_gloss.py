"""Known-answer tests for gloss_filter (Radicchi, Ramasco & Fortunato, 2011).

The expected p-values come from the paper's Eqs. (1)-(4), evaluated by hand,
in closed form, or by enumerating every weight assignment of the null model
(topology fixed, i.i.d. weights from the observed weight distribution).
Shared behavior (attributes, p-value range, weight validation, multigraphs,
empty graphs, input order) is covered by ``test_api_contract.py``.
"""

import itertools
import math
from collections import Counter
from fractions import Fraction

import networkx as nx
import pytest

import networkx_backbone as nb

np = pytest.importorskip("numpy")


def _pvalues(H):
    return {(u, v): d["gloss_pvalue"] for u, v, d in H.edges(data=True)}


def _enumerated_pvalues(G, weight="weight"):
    """Exact P(w_e >= observed | endpoint strengths) under the GloSS null.

    Enumerates every assignment of i.i.d. weights from the empirical weight
    distribution to the fixed edges, so it implements the null model of
    Sec. II A directly rather than through Eqs. (2)-(4).
    """
    edges = list(G.edges())
    observed = [G[u][v][weight] for u, v in edges]
    p_obs = {w: Fraction(c, len(edges)) for w, c in Counter(observed).items()}

    def ends(u, v):
        return (("out", u), ("in", v)) if G.is_directed() else (u, v)

    def strengths(assignment):
        s = Counter()
        for (u, v), w in zip(edges, assignment):
            a, b = ends(u, v)
            s[a] += w
            s[b] += w
        return s

    s_obs = strengths(observed)
    num = Counter()
    den = Counter()
    for assignment in itertools.product(p_obs, repeat=len(edges)):
        s = strengths(assignment)
        prob = math.prod(p_obs[w] for w in assignment)
        for e, (u, v) in enumerate(edges):
            a, b = ends(u, v)
            if s[a] == s_obs[a] and s[b] == s_obs[b]:
                den[e] += prob
                if assignment[e] >= observed[e]:
                    num[e] += prob
    return {edge: num[e] / den[e] for e, edge in enumerate(edges)}


@pytest.fixture
def small_graph():
    G = nx.Graph()
    G.add_weighted_edges_from(
        [(0, 1, 1), (0, 2, 3), (1, 2, 2), (2, 3, 1), (1, 3, 3), (3, 4, 2), (0, 4, 1)]
    )
    return G


@pytest.fixture
def small_digraph():
    D = nx.DiGraph()
    D.add_weighted_edges_from(
        [
            (0, 1, 1),
            (0, 2, 4),
            (0, 3, 2),
            (1, 0, 4),
            (1, 2, 1),
            (2, 0, 2),
            (2, 3, 4),
            (3, 1, 1),
            (3, 2, 2),
        ]
    )
    return D


def test_matches_enumerated_null_undirected(small_graph):
    expected = _enumerated_pvalues(small_graph)
    # Eq. (4) by hand: P_obs: P(1) = 3/7, P(2) = 2/7, P(3) = 2/7.  Edge (0, 2)
    # has w = 3; node 0 has k = 3, s = 5 and node 2 has k = 3, s = 6.  In units
    # of 1/49, F(s, 2) = 9, 12, 16, 8, 4 for s = 2..6 (Eq. 3), so Eq. (4) is
    #   P(3) F(2, 2) F(3, 2) / sum_w P(w) F(5 - w, 2) F(6 - w, 2)
    #   = 2*9*12 / (3*16*8 + 2*12*16 + 2*9*12) = 216 / 984 = 9/41.
    assert expected[(0, 2)] == Fraction(9, 41)
    got = _pvalues(nb.gloss_filter(small_graph))
    for edge, p in expected.items():
        assert got[edge] == pytest.approx(float(p), rel=1e-12, abs=1e-15)


def test_matches_enumerated_null_directed(small_digraph):
    # Directed edges i -> j use (k_out, s_out) of i and (k_in, s_in) of j.
    expected = _enumerated_pvalues(small_digraph)
    assert expected[(0, 2)] == Fraction(1, 3)
    assert expected[(2, 3)] == Fraction(1, 2)
    got = _pvalues(nb.gloss_filter(small_digraph))
    for edge, p in expected.items():
        assert got[edge] == pytest.approx(float(p), rel=1e-12, abs=1e-15)


def _two_valued_pvalue(p, k_i, n_i, k_j, n_j):
    """Eq. (4) for a heavy edge when weights take two values.

    With heavy weights of probability p, F is binomial, and Eq. (4) reduces to
    1 / (1 + p/(1-p) * (k_i - n_i)(k_j - n_j) / (n_i n_j)), where n_i counts
    the heavy edges of i (the tested edge included).
    """
    return 1.0 / (1.0 + p / (1 - p) * (k_i - n_i) * (k_j - n_j) / (n_i * n_j))


def test_two_valued_closed_form_at_large_degree():
    # Hub "B" has 4001 edges, only 3 of them heavy: F(s_B - w, 4000) is about
    # 1e-379, below the smallest double, and an FFT would return round-off.
    G = nx.Graph()
    G.add_edge("A", "B", weight=2)
    G.add_weighted_edges_from(("A", ("a", i), 2 if i < 990 else 1) for i in range(1000))
    G.add_weighted_edges_from(("B", ("b", i), 2 if i < 2 else 1) for i in range(4000))
    p_heavy = 993 / G.number_of_edges()
    expected = _two_valued_pvalue(p_heavy, 1001, 991, 4001, 3)
    H = nb.gloss_filter(G)
    assert H["A"]["B"]["gloss_pvalue"] == pytest.approx(expected, rel=1e-9)
    # Leaves have degree 1: their edge weight is fixed by their strength.
    assert H["A"][("a", 0)]["gloss_pvalue"] == 1.0


def test_two_valued_closed_form_random_graph():
    G = nx.gnm_random_graph(60, 400, seed=3)
    rng = np.random.default_rng(3)
    for u, v in G.edges():
        G[u][v]["weight"] = 5 if rng.random() < 0.3 else 2
    p_heavy = sum(w == 5 for *_, w in G.edges(data="weight")) / G.number_of_edges()
    heavy = Counter()
    for u, v, w in G.edges(data="weight"):
        if w == 5:
            heavy[u] += 1
            heavy[v] += 1
    H = nb.gloss_filter(G)
    for u, v, d in H.edges(data=True):
        if G[u][v]["weight"] == 2:
            assert d["gloss_pvalue"] == 1.0  # the lightest weight
        else:
            k_u, k_v = G.degree(u), G.degree(v)
            expected = _two_valued_pvalue(p_heavy, k_u, heavy[u], k_v, heavy[v])
            assert d["gloss_pvalue"] == pytest.approx(expected, rel=1e-9)


def test_degree_one_endpoint_gives_one():
    G = nx.star_graph(4)
    nx.set_edge_attributes(G, {(0, 1): 9, (0, 2): 1, (0, 3): 1, (0, 4): 2}, "weight")
    G.add_edge(5, 6, weight=9)  # both endpoints have degree 1
    assert set(_pvalues(nb.gloss_filter(G)).values()) == {1.0}


def test_equal_weights_give_one(small_graph):
    nx.set_edge_attributes(small_graph, 2.5, "weight")
    assert set(_pvalues(nb.gloss_filter(small_graph)).values()) == {1.0}


def test_invariances(small_graph):
    """Same p-values after an affine change of the weights, on a finer lattice,
    as a symmetric DiGraph, and with a self-loop added."""
    expected = _pvalues(nb.gloss_filter(small_graph))

    def assert_same(got, what):
        for edge, p in expected.items():
            assert got[edge] == pytest.approx(p, rel=1e-12), (what, edge)

    shifted = small_graph.copy()
    for *_, d in shifted.edges(data=True):
        d["weight"] = 3 * d["weight"] + 2
    assert_same(_pvalues(nb.gloss_filter(shifted)), "shifted and scaled")
    halves = small_graph.copy()
    for *_, d in halves.edges(data=True):
        d["weight"] = d["weight"] / 2  # non-integer weights on a 0.5 lattice
    assert_same(_pvalues(nb.gloss_filter(halves)), "halved")
    # A finer resolution does not change an exact lattice.
    got = _pvalues(nb.gloss_filter(small_graph, resolution=0.25))
    assert_same(got, "resolution=0.25")
    # A symmetric DiGraph gives each direction the undirected p-value.
    got = _pvalues(nb.gloss_filter(nx.DiGraph(small_graph)))
    assert_same(got, "directed (u, v)")
    assert_same({(u, v): p for (v, u), p in got.items()}, "directed (v, u)")
    # Self-loops are left out of the null.
    looped = small_graph.copy()
    looped.add_edge(2, 2, weight=100)
    H = nb.gloss_filter(looped)
    assert H[2][2]["gloss_pvalue"] == 1.0
    assert_same(_pvalues(H), "self-loop added")


def test_coarse_resolution_rounds_weights_and_bad_resolution_raises(small_graph):
    # A spacing of 2 rounds weights 1, 2, 3 to lattice points 0, 1, 1 (weights
    # 1, 3, 3): the two heavier values merge and no edge is significant.
    H = nb.gloss_filter(small_graph, resolution=2)
    merged = small_graph.copy()
    for *_, d in merged.edges(data=True):
        d["weight"] = 1 if d["weight"] == 1 else 3
    assert _pvalues(H) == pytest.approx(_pvalues(nb.gloss_filter(merged)))
    for bad in (0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="resolution"):
            nb.gloss_filter(small_graph, resolution=bad)


def test_iid_weights_give_uniform_pvalues():
    # Sec. II B / Fig. 1: with i.i.d. weights no edge is special, so
    # P(alpha <= a) is about a (at most a, as discrete weights are conservative).
    G = nx.random_regular_graph(8, 500, seed=1)
    rng = np.random.default_rng(0)
    for u, v in G.edges():
        G[u][v]["weight"] = int(rng.geometric(0.3))
    p = np.array(list(_pvalues(nb.gloss_filter(G)).values()))
    for a in (0.05, 0.1, 0.2, 0.5):
        assert a / 2 <= np.mean(p <= a) <= a + 0.01
