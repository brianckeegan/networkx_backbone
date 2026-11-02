"""Known-answer tests for ``polya_filter`` (Marcaccioli & Livan, 2019).

Equation numbers refer to the paper (Nature Communications 10, 745) and,
for Eqs. (15)-(16), to its Supplementary Information.  The exact p-values
are checked against an independent reference that plays the Pólya urn
forward draw by draw in rational arithmetic, rather than against the
beta-binomial formula the implementation uses.  Shared behavior
(attributes, p-value range, weight validation, multigraphs, empty graphs,
input order) is covered by ``test_api_contract.py``.
"""

import math
from fractions import Fraction

import networkx as nx
import pytest

import networkx_backbone as nb

pytest.importorskip("scipy")
from scipy import special, stats


def urn_tail(w, k, s, a):
    """P(at least w red draws in s) for the urn of Fig. 1, exactly.

    The urn starts with one red and k - 1 black balls; after each draw the
    ball goes back with *a* more balls of its color (a = 0: simple
    replacement).
    """
    a = Fraction(a)
    dist = {0: Fraction(1)}  # red balls drawn so far -> probability
    for n in range(s):
        nxt = {}
        for x, p in dist.items():
            red = 1 + a * x
            black = (k - 1) + a * (n - x)
            nxt[x + 1] = nxt.get(x + 1, 0) + p * red / (red + black)
            nxt[x] = nxt.get(x, 0) + p * black / (red + black)
        dist = nxt
    return sum(p for x, p in dist.items() if x >= w)


def star(weights):
    """Hub "c" linked to one leaf per weight; leaf 0 carries weights[0]."""
    G = nx.Graph()
    G.add_weighted_edges_from(("c", i, w) for i, w in enumerate(weights))
    return G


def pvalues(H):
    return {frozenset((u, v)): d["polya_pvalue"] for u, v, d in H.edges(data=True)}


# ---------------------------------------------------------------------------
# Exact p-values, Eqs. (1)-(2)
# ---------------------------------------------------------------------------


def test_figure_1_example():
    """Fig. 1: k = 3, s = 4, w = 2, a = 1 gives 1/15 + 2/15 + 1/5 = 6/15."""
    G = star([2, 1, 1])
    G = nx.relabel_nodes(G, {"c": "A", 0: "B", 1: "C", 2: "D"})
    H = nb.polya_filter(G, a=1)
    # B has degree 1, so the edge is tested from A only.
    assert H["A"]["B"]["polya_pvalue"] == pytest.approx(6 / 15, rel=1e-12)
    assert urn_tail(2, 3, 4, 1) == Fraction(6, 15)
    # The three favorable outcomes of the figure, via Eq. (1).
    pmf = stats.betabinom(4, 1, 2).pmf
    assert [pmf(4), pmf(3), pmf(2)] == pytest.approx([1 / 15, 2 / 15, 1 / 5])


def test_exact_pvalue_matches_urn():
    mismatches = []
    for k in (2, 3, 4):
        for a in (0, Fraction(1, 2), 1, 2, 3):
            for s in range(k, 8):
                for w in range(1, s - (k - 1) + 1):
                    rest = s - w
                    others = [
                        rest // (k - 1) + (i < rest % (k - 1)) for i in range(k - 1)
                    ]
                    H = nb.polya_filter(star([w, *others]), a=float(a))
                    got = H["c"][0]["polya_pvalue"]
                    expected = float(urn_tail(w, k, s, a))
                    if got != pytest.approx(expected, rel=1e-10):
                        mismatches.append(
                            f"w={w} k={k} s={s} a={a}: {got} != {expected}"
                        )
    assert not mismatches, "\n".join(mismatches)


def test_a_zero_is_binomial():
    """At a = 0 the null is Binomial(s, 1/k) (Eq. (1) at a = 0)."""
    G = nx.les_miserables_graph()
    H = nb.polya_filter(G, a=0)

    def tail(n, w):
        k, s = G.degree(n), int(G.degree(n, weight="weight"))
        return 1.0 if k == 1 else stats.binom.sf(w - 1, s, 1 / k)

    for u, v, d in H.edges(data=True):
        w = int(G[u][v]["weight"])
        expected = min(tail(u, w), tail(v, w))
        assert d["polya_pvalue"] == pytest.approx(expected, rel=1e-9)


def test_large_a_tends_to_one_over_k():
    """As a -> infinity the p-value tends to 1/k (paper, 'The backbone family')."""
    H = nb.polya_filter(star([5, 2, 1, 1]), a=1e8)
    for leaf in range(4):
        assert H["c"][leaf]["polya_pvalue"] == pytest.approx(1 / 4, rel=1e-6)


def test_degree_one_endpoints_give_pvalue_one():
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 3), (2, 3, 7)])
    H = nb.polya_filter(G)
    assert all(p == 1.0 for p in pvalues(H).values())
    assert math.isnan(H.graph["polya_a"])


def test_directed_uses_out_and_in_views():
    """Supplementary Note 2: test (u, v) from u's out- and v's in-links."""
    D = nx.DiGraph()
    D.add_weighted_edges_from(
        [(0, 1, 2), (0, 2, 1), (0, 3, 1), (3, 1, 1), (2, 1, 3), (1, 0, 4)]
    )
    H = nb.polya_filter(D, a=1)
    for u, v, w in D.edges(data="weight"):
        tests = []
        for k, s in (
            (D.out_degree(u), D.out_degree(u, weight="weight")),
            (D.in_degree(v), D.in_degree(v, weight="weight")),
        ):
            tests.append(1.0 if k == 1 else float(urn_tail(w, k, s, 1)))
        assert H[u][v]["polya_pvalue"] == pytest.approx(min(tests), rel=1e-10)
    # Edge 0 -> 1: out-test of node 0 is the Fig. 1 urn (k=3, s=4, w=2).
    assert H[0][1]["polya_pvalue"] <= 6 / 15 + 1e-12


def test_backbones_shrink_as_a_grows():
    """Eq. (3): at fixed significance, P_a2 is a subset of P_a1 for a1 <= a2."""
    G = nx.les_miserables_graph()
    grid = [0, 0.1, 0.3, 0.5, 1, 2, 5, 20]
    scored = [pvalues(nb.polya_filter(G, a=a)) for a in grid]
    for alpha in (0.05, 0.01, 0.05 / (2 * G.number_of_edges())):
        kept = [{e for e, p in s.items() if p < alpha} for s in scored]
        assert all(kept[i + 1] <= kept[i] for i in range(len(kept) - 1)), alpha
        assert len(kept[0]) > len(kept[-1])


# ---------------------------------------------------------------------------
# Large-strength approximation, Eqs. (6) and (8)
# ---------------------------------------------------------------------------


def test_approximation_at_a_one_is_disparity_filter():
    """Eq. (8): Eq. (6) at a = 1 is the disparity filter p-value."""
    G = nx.les_miserables_graph()
    H = nb.polya_filter(G, a=1, approximate=True)
    D = nb.disparity_filter(G)
    for u, v, d in H.edges(data=True):
        assert d["polya_pvalue"] == pytest.approx(
            D[u][v]["disparity_pvalue"], rel=1e-12, abs=1e-15
        )


def test_approximation_known_value():
    """Eq. (6) by hand for k = 3, s = 4, w = 2, a = 2."""
    expected = (1 / math.gamma(0.5)) * (1 - 2 / 4) ** 1 * (2 * 3 / (4 * 2)) ** -0.5
    H = nb.polya_filter(star([2, 1, 1]), a=2, approximate=True)
    assert H["c"][0]["polya_pvalue"] == pytest.approx(expected, rel=1e-12)


def test_approximation_is_clipped_to_one():
    # (w k / (s a))**(1/a - 1) is large for small w when a > 1.
    H = nb.polya_filter(star([1, 50, 50]), a=4, approximate=True)
    assert H["c"][0]["polya_pvalue"] == 1.0


def test_non_integer_weights_use_approximation():
    G = star([2.5, 1.25, 1.0])
    H = nb.polya_filter(G, a=1.5)
    expected = nb.polya_filter(G, a=1.5, approximate=True)
    assert pvalues(H) == pvalues(expected)
    with pytest.raises(nx.NetworkXError, match="integer weights"):
        nb.polya_filter(G, a=1.5, approximate=False)
    with pytest.raises(nx.NetworkXError, match="a=None"):
        nb.polya_filter(G)


def test_integer_valued_floats_are_counts():
    assert pvalues(nb.polya_filter(star([2.0, 1.0, 1.0]), a=1)) == pvalues(
        nb.polya_filter(star([2, 1, 1]), a=1)
    )


# ---------------------------------------------------------------------------
# Maximum-likelihood a, Supplementary Note 5
# ---------------------------------------------------------------------------


def _rows(G):
    """(w, k, s) terms of Eq. (15): every link seen from each endpoint, k > 1."""
    for i in G:
        k, s = G.degree(i), int(G.degree(i, weight="weight"))
        if k > 1:
            for j in G[i]:
                yield int(G[i][j]["weight"]), k, s


def _loglik(G, a):
    """Eq. (15), written with scipy's beta-binomial (independent of the code)."""
    return sum(stats.betabinom.logpmf(w, s, 1 / a, (k - 1) / a) for w, k, s in _rows(G))


def _eq16(G, a):
    """Left-hand side of Eq. (16)."""
    psi = special.digamma
    return sum(
        -(k - 1) * psi((k + a * s - a * w - 1) / a)
        + k * psi(k / a + s)
        + (k - 1) * psi((k - 1) / a)
        - k * psi(k / a)
        - psi(w + 1 / a)
        + psi(1 / a)
        for w, k, s in _rows(G)
    )


def test_ml_estimate_les_miserables():
    G = nx.les_miserables_graph()
    H = nb.polya_filter(G)
    a_ml = H.graph["polya_a"]
    assert a_ml == pytest.approx(0.3397028, rel=1e-6)
    # It solves Eq. (16) ...
    assert abs(_eq16(G, a_ml)) < 1e-6
    # ... and maximizes Eq. (15) over a grid of alternatives.
    best = _loglik(G, a_ml)
    for a in (0.01, 0.1, 0.3, 0.33, 0.35, 0.4, 1, 3, 10):
        assert _loglik(G, a) < best
    # a=None is the same as passing the estimate.
    assert pvalues(H) == pytest.approx(pvalues(nb.polya_filter(G, a=a_ml)))

    # The scores compose with threshold_filter's multiple-testing correction.
    raw = nb.threshold_filter(H, "polya_pvalue", 0.05)
    bonf = nb.threshold_filter(H, "polya_pvalue", 0.05, mtc="bonferroni")
    assert 0 < bonf.number_of_edges() <= raw.number_of_edges()
    # The paper's own rule: Bonferroni over L = 2m tests.
    paper = nb.threshold_filter(H, "polya_pvalue", 0.05 / (2 * G.number_of_edges()))
    assert paper.number_of_edges() <= bonf.number_of_edges()


def test_ml_estimate_at_binomial_boundary():
    """Equal weights are under-dispersed, so the likelihood peaks at a = 0."""
    G = nx.complete_graph(5)
    nx.set_edge_attributes(G, 3, "weight")
    H = nb.polya_filter(G)
    assert H.graph["polya_a"] == 0.0
    expected = stats.binom.sf(2, 12, 1 / 4)
    assert all(p == pytest.approx(expected) for p in pvalues(H).values())
    # Eq. (6) is undefined at a = 0.
    with pytest.raises(ValueError, match="estimate of a is 0"):
        nb.polya_filter(G, approximate=True)


def test_explicit_a_is_recorded():
    H = nb.polya_filter(star([2, 1, 1]), a=2.5)
    assert H.graph["polya_a"] == 2.5


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def test_invalid_arguments():
    for a in (-0.5, math.inf, math.nan):
        with pytest.raises(ValueError, match="a must be"):
            nb.polya_filter(star([2, 1, 1]), a=a)
    with pytest.raises(ValueError, match="approximate"):
        nb.polya_filter(star([2, 1, 1]), a=1, approximate="yes")
    with pytest.raises(ValueError, match="a > 0"):
        nb.polya_filter(star([2, 1, 1]), a=0, approximate=True)
