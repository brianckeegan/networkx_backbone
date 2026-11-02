"""Calibration of null models: under its own null, a p-value is uniform.

A statistical scorer is calibrated when, for weights drawn from the null model
it assumes, its p-values are uniform on [0, 1], so that a threshold *alpha*
rejects a fraction *alpha* of tests.  Each test here simulates weights from a
scorer's null model, collects p-values of **independent endpoint tests**, and
checks them with a Kolmogorov-Smirnov test and the rejection rates at
``alpha = 0.01`` and ``0.05``.

The simulated graphs are disjoint stars: every leaf has degree 1, so it cannot
reject the null and each edge's p-value is the test from its star's center
(see the degree-1 rule in ``disparity_filter``).  Taking one edge per star
makes the samples independent, which the Kolmogorov-Smirnov test assumes.
Seeds are fixed, so the tests are deterministic; the sizes keep each test well
under a second.

Negative controls check that the harness has the power to detect a wrong null
model, and :func:`test_or_rule_is_anti_conservative` documents that combining
two calibrated endpoint tests by their minimum is not calibrated.
"""

import math

import networkx as nx
import pytest

import networkx_backbone as nb
from networkx_backbone.statistical import _disparity_beta_tails

pytest.importorskip("numpy")
pytest.importorskip("scipy")
import numpy as np
from scipy import stats

ALPHAS = (0.01, 0.05)


def _stars(rng, draw, n_stars=4000, k_range=(2, 12), directed=False):
    """Disjoint stars with weights ``draw(k)``; returns the graph and test edges.

    Star *i* has center ``("c", i)`` and ``k`` leaves ``("l", i, j)``, with
    ``k`` uniform on *k_range*.  The test edge of each star is the one to its
    first leaf.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    tests = []
    for i in range(n_stars):
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        for j, w in enumerate(draw(k)):
            G.add_edge(("c", i), ("l", i, j), weight=float(w))
        tests.append((("c", i), ("l", i, 0)))
    return G, tests


def _pvalues(H, tests, attr="disparity_pvalue"):
    return np.array([H.edges[e][attr] for e in tests])


def _assert_uniform(p, what):
    """KS test and rejection rates of i.i.d. p-values against Uniform(0, 1)."""
    p = np.asarray(p)
    n = len(p)
    ks = stats.kstest(p, "uniform")
    assert ks.pvalue > 0.01, f"{what}: KS D = {ks.statistic:.4f} (p = {ks.pvalue:.2g})"
    for alpha in ALPHAS:
        rate = float(np.mean(p <= alpha))
        se = math.sqrt(alpha * (1 - alpha) / n)
        assert abs(rate - alpha) < 4 * se, (
            f"{what}: P(p <= {alpha}) = {rate:.4f}, expected {alpha} +/- {4 * se:.4f}"
        )


def _gamma(rng, shape, heterogeneous_scale=True):
    """Gamma(shape) weights, with a random scale per star when requested."""

    def draw(k):
        scale = rng.lognormal(0.0, 1.0) if heterogeneous_scale else 1.0
        return rng.gamma(shape, scale, size=k)

    return draw


# ---------------------------------------------------------------------------
# disparity_filter, a = 1: the uniform share null of Serrano et al. (2009)
# ---------------------------------------------------------------------------


def test_disparity_uniform_null_is_calibrated():
    # Exponential weights conditioned on the node strength have uniform
    # spacings as shares: exactly the null of the disparity filter.
    rng = np.random.default_rng(1)
    G, tests = _stars(rng, _gamma(rng, 1.0))
    _assert_uniform(_pvalues(nb.disparity_filter(G), tests), "disparity a=1")


def test_disparity_uniform_null_is_miscalibrated_otherwise():
    # Negative control: Gamma(0.3) weights are more heterogeneous than the
    # uniform null allows, so it rejects too often; Gamma(3) weights are more
    # even, so it rejects too rarely.  The harness must see both.
    for shape, direction in [(0.3, "liberal"), (3.0, "strict")]:
        rng = np.random.default_rng(2)  # each case draws the same as on its own
        G, tests = _stars(rng, _gamma(rng, shape))
        p = _pvalues(nb.disparity_filter(G), tests)
        assert stats.kstest(p, "uniform").pvalue < 1e-6, shape
        rate = np.mean(p <= 0.05)
        assert rate > 0.08 if direction == "liberal" else rate < 0.02, shape


# ---------------------------------------------------------------------------
# disparity_filter, a != 1: symmetric Dirichlet shares, Beta(1/a, (k-1)/a)
# ---------------------------------------------------------------------------


def test_disparity_concentration_null_is_calibrated():
    # i.i.d. Gamma(c) weights (any scale) give symmetric Dirichlet(c) shares,
    # the null of disparity_filter(a=1/c).
    for shape in (0.3, 3.0):
        rng = np.random.default_rng(3)  # each case draws the same as on its own
        G, tests = _stars(rng, _gamma(rng, shape))
        H = nb.disparity_filter(G, a=1 / shape)
        _assert_uniform(_pvalues(H, tests), f"disparity a={1 / shape:g}")


def test_disparity_concentration_null_is_calibrated_directed():
    # Out-stars: each edge is tested from the center's out-links (the leaf
    # has in-degree 1).
    rng = np.random.default_rng(4)
    G, tests = _stars(rng, _gamma(rng, 0.5), directed=True)
    H = nb.disparity_filter(G, a=2.0)
    _assert_uniform(_pvalues(H, tests), "directed disparity a=2")


def test_disparity_fitted_a_is_calibrated():
    # a="ml" fits one parameter to all stars and then tests the same weights;
    # with thousands of profiles the reuse of the data is negligible.
    for shape in (0.3, 1.0, 3.0):
        rng = np.random.default_rng(5)  # each case draws the same as on its own
        G, tests = _stars(rng, _gamma(rng, shape))
        H = nb.disparity_filter(G, a="ml")
        assert H.graph["disparity_a"] == pytest.approx(1 / shape, rel=0.05), shape
        what = f"disparity a='ml' (true a={1 / shape:g})"
        _assert_uniform(_pvalues(H, tests), what)


def test_disparity_fitted_a_is_calibrated_on_random_graphs():
    # Beyond stars: on a G(n, m) graph with i.i.d. Gamma(0.5) weights every
    # node's share profile is symmetric Dirichlet(0.5).  The test from one
    # endpoint of each edge is uniform; the p-values within a node are
    # dependent, hence the looser KS threshold.
    rng = np.random.default_rng(6)
    G = nx.gnm_random_graph(1500, 9000, seed=6)
    G.remove_nodes_from([u for u in list(G) if G.degree(u) < 2])
    for u, v in G.edges():
        G[u][v]["weight"] = float(rng.gamma(0.5))
    a_hat = nb.disparity_filter(G, a="ml").graph["disparity_a"]
    assert a_hat == pytest.approx(2.0, rel=0.05)
    s, k = dict(G.degree(weight="weight")), dict(G.degree())
    tests = [(d["weight"], s[u], k[u]) for u, v, d in G.edges(data=True)]
    table = _disparity_beta_tails(set(tests), 1 / a_hat)
    p = np.array([table[t][0] for t in tests])
    assert stats.kstest(p, "uniform").statistic < 0.02
    for alpha in ALPHAS:
        assert np.mean(p <= alpha) == pytest.approx(alpha, rel=0.2)


# ---------------------------------------------------------------------------
# The edge p-value is the minimum over the two endpoints
# ---------------------------------------------------------------------------


def test_or_rule_is_anti_conservative():
    # Each endpoint test is calibrated, but disparity_pvalue is the smaller
    # of the two, so more than alpha of the edges fall below alpha even when
    # the null model holds (documented in the Notes of disparity_filter).
    rng = np.random.default_rng(7)
    rates = []
    for seed in range(5):
        G = nx.random_regular_graph(8, 400, seed=seed)
        for u, v in G.edges():
            G[u][v]["weight"] = float(rng.exponential())
        p = np.array(
            [d["disparity_pvalue"] for *_, d in nb.disparity_filter(G).edges(data=True)]
        )
        rates.append(np.mean(p <= 0.05))
    assert 0.07 < np.mean(rates) < 0.09


# ---------------------------------------------------------------------------
# SDSM: discrete p-values are valid (P(p <= alpha) <= alpha) under the null
# ---------------------------------------------------------------------------


def _sdsm_null_pvalues(method, reps=3, shape=(300, 250), density=0.01):
    """SDSM p-values of every agent pair, for incidences drawn from the null.

    Every cell is an independent Bernoulli(*density*) draw, which is the
    BiCM with equal degrees; SDSM refits the BiCM on each draw, as in use.
    """
    rng = np.random.default_rng(108)
    out = []
    for _ in range(reps):
        R = (rng.random(shape) < density).astype(int)
        H = nb.sdsm(R, pvalue_method=method)
        out.append([p for *_, p in H.edges(data="sdsm_pvalue")])
    return np.concatenate(out)


@pytest.mark.parametrize("method", ["auto", "exact"])
def test_sdsm_sparse_null_is_valid(method):
    # Issue #108: the counts are discrete, so P(p <= alpha) is at most alpha
    # (well below it at alpha = 0.05, where most pairs never co-occur).
    p = _sdsm_null_pvalues(method)
    for alpha in (0.05, 0.01, 1e-3, 1e-4):
        se = math.sqrt(alpha / len(p))
        assert np.mean(p <= alpha) <= alpha + 4 * se, alpha


def test_sdsm_rna_sparse_null_is_anti_conservative():
    # The refined normal approximation's far upper tail is too light when a
    # pair's expected co-occurrence is small (here about 0.025).
    p = _sdsm_null_pvalues("rna")
    assert np.mean(p <= 1e-4) > 3e-4
