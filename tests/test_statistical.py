"""Known-answer tests for the statistical scorers (nx_backbone.statistical).

Each check is tied to an equation in the cited paper, so a regression here
means the implementation drifted from the method it claims to implement.
Shared behavior (attributes, p-value ranges, validation, multigraphs, empty
graphs, input order) is covered by ``test_api_contract.py``.
"""

import math
import random
import warnings

import networkx as nx
import pytest

import nx_backbone as nb

pytest.importorskip("scipy")
from scipy import special, stats


@pytest.fixture
def count_graph():
    G = nx.Graph()
    G.add_weighted_edges_from(
        [("a", "b", 5), ("a", "c", 1), ("b", "c", 2), ("c", "d", 3), ("b", "d", 1)]
    )
    return G


@pytest.fixture
def hub_star():
    """Hub "c" with one very strong edge and several weak edges."""
    G = nx.Graph()
    G.add_weighted_edges_from(
        [("c", 1, 100.0), ("c", 2, 1.0), ("c", 3, 1.0), ("c", 4, 1.0)]
    )
    return G


# ---------------------------------------------------------------------------
# Disparity filter -- Serrano et al. (2009)
# ---------------------------------------------------------------------------


class TestDisparity:
    def test_formula(self, count_graph):
        H = nb.disparity_filter(count_graph)
        # Edge a-b: from a, p = 5/6, k = 2 -> (1 - 5/6)^1; from b, p = 5/8, k=3.
        expected = min((1 - 5 / 6) ** 1, (1 - 5 / 8) ** 2)
        assert H["a"]["b"]["disparity_pvalue"] == pytest.approx(expected)

    def test_unsigned_strong_edge(self, hub_star):
        H = nb.disparity_filter(hub_star)
        assert all("sign" not in d for _, _, d in H.edges(data=True))
        # one-tailed p-value: (1 - w/s)^(k-1) for the strong edge
        assert H["c"][1]["disparity_pvalue"] == pytest.approx((1 - 100 / 103) ** 3)

    def test_directed_uses_source_and_target(self):
        # Five sources each send one link into v; one of them is heavy.  From
        # each source (out-degree 1) nothing is testable, but from v's
        # incoming links the heavy one stands out.
        G = nx.DiGraph()
        G.add_weighted_edges_from(
            [(f"s{i}", "v", w) for i, w in enumerate([100, 1, 1, 1, 1])]
        )
        H = nb.disparity_filter(G)
        expected = (1 - 100 / 104) ** 4
        assert H["s0"]["v"]["disparity_pvalue"] == pytest.approx(expected)
        assert H["s1"]["v"]["disparity_pvalue"] > 0.5

    def test_signed_marks_strong_and_weak(self, hub_star):
        H = nb.disparity_filter(hub_star, signed=True)
        assert H["c"][1]["sign"] == 1  # strong edge
        assert H["c"][2]["sign"] == -1  # weak edge

    def test_signed_pvalue_is_two_sided(self, hub_star):
        # Two-sided p-value is twice the smaller one-tailed tail (clipped to 1).
        unsigned = nb.disparity_filter(hub_star)
        signed = nb.disparity_filter(hub_star, signed=True)
        p_hi = unsigned["c"][1]["disparity_pvalue"]
        assert signed["c"][1]["disparity_pvalue"] == pytest.approx(min(1.0, 2 * p_hi))


def test_signed_adds_sign_and_two_sided_pvalue(les_miserables_weighted):
    for scorer, attr in [
        (nb.disparity_filter, "disparity_pvalue"),
        (nb.mlf, "ml_pvalue"),
        (nb.lans, "lans_pvalue"),
        (nb.ecm_filter, "ecm_pvalue"),
        (nb.noise_corrected_filter, "nc_pvalue"),
    ]:
        H = scorer(les_miserables_weighted, signed=True)
        edges = list(H.edges(data=True))
        assert all(d.get("sign") in (-1, 1) for *_, d in edges), scorer.__name__
        assert all(0.0 <= d[attr] <= 1.0 for *_, d in edges), scorer.__name__


def test_aliases_match_full_names(les_miserables_weighted):
    G = les_miserables_weighted
    for alias, full, attr in [
        (nb.disparity, nb.disparity_filter, "disparity_pvalue"),
        (nb.mlf, nb.marginal_likelihood_filter, "ml_pvalue"),
        (nb.lans, nb.lans_filter, "lans_pvalue"),
    ]:
        for signed in (False, True):
            got = nx.get_edge_attributes(alias(G, signed=signed), attr)
            assert got == nx.get_edge_attributes(full(G, signed=signed), attr), (
                alias.__name__
            )


# ---------------------------------------------------------------------------
# Noise-corrected filter -- Coscia & Neffke (2017)
# ---------------------------------------------------------------------------


# Scores and standard deviations from the authors' reference implementation
# (``backboning.noise_corrected``, Python 3 / NetworkX 2 release from
# michelecoscia.com), run on the fixtures below with ``return_self_loops=True``;
# undirected input was given with both directions of every edge.  Kept inline
# so the test suite does not depend on backboning.py or pandas.
NC_UNDIRECTED_EDGES = [(0, 1, 5), (0, 2, 1), (1, 2, 2), (2, 3, 3), (1, 3, 1)]
NC_DIRECTED_EDGES = [
    (0, 1, 5), (1, 0, 2), (0, 2, 1), (2, 3, 4),
    (3, 0, 3), (1, 3, 1), (3, 1, 6), (2, 1, 2),
]  # fmt: skip
NC_REFERENCE = {
    "undirected": {
        (0, 1): (0.428571428571, 0.032108591699),
        (0, 2): (-0.2, 0.364425896468),
        (1, 2): (0.0, 0.161512001277),
        (1, 3): (-0.142857142857, 0.338230798806),
        (2, 3): (0.5, 0.019829734799),
    },
    "directed": {
        (0, 1): (0.212121212121, 0.001520612374),
        (0, 2): (0.6, 0.029898792212),
        (1, 0): (0.52380952381, 0.003209286937),
        (1, 3): (0.230769230769, 0.207002530301),
        (2, 1): (-0.238095238095, 0.207066784412),
        (2, 3): (0.52380952381, 0.038571114064),
        (3, 0): (0.230769230769, 0.04163111544),
        (3, 1): (0.103448275862, 0.018318205422),
    },
}


class TestNoiseCorrected:
    def test_matches_reference_implementation(self):
        for kind in ("undirected", "directed"):
            G = nx.DiGraph() if kind == "directed" else nx.Graph()
            edges = NC_DIRECTED_EDGES if kind == "directed" else NC_UNDIRECTED_EDGES
            G.add_weighted_edges_from(edges)
            H = nb.noise_corrected_filter(G)
            for (u, v), (score, sdev) in NC_REFERENCE[kind].items():
                assert H[u][v]["nc_score"] == pytest.approx(score, abs=1e-11), kind
                assert H[u][v]["nc_sdev"] == pytest.approx(sdev, rel=1e-9), kind

    def test_hand_computed_lift_score(self):
        # Undirected edges count in both directions: n.. = 2 * 12 = 24,
        # s_0 = 6, s_1 = 8, so kappa = 24 / 48 and the lift is 5 / 2.
        G = nx.Graph()
        G.add_weighted_edges_from(NC_UNDIRECTED_EDGES)
        H = nb.noise_corrected_filter(G)
        assert H[0][1]["nc_score"] == pytest.approx((2.5 - 1) / (2.5 + 1))

    def test_hand_computed_complete_graph(self):
        # Unit-weight K5: every node has strength 4 and n.. = 2 * 10 = 20, so
        # each edge has lift 20 / 16 = 5/4 and score 1/9.  The legacy z-score,
        # which counted W once but strengths twice, is negative here.
        G = nx.complete_graph(5)
        nx.set_edge_attributes(G, 1, "weight")
        H = nb.noise_corrected_filter(G)
        n, ni, nij = 20, 4, 1
        mean = ni * ni / n**2  # 0.04
        var = ni * ni * (n - ni) ** 2 / (n**4 * (n - 1))
        alpha = mean**2 / var * (1 - mean) - mean
        beta = mean / var * (1 - mean**2) - (1 - mean)
        assert (alpha, beta) == pytest.approx((1.1, 28.68))
        p_hat = (alpha + nij) / (alpha + beta + n)
        kappa = n / ni**2
        d = 1 / ni**2 - n * 2 * ni / ni**4  # -0.5625
        deriv = 2 * (kappa + nij * d) / (kappa * nij + 1) ** 2
        sdev = math.sqrt(n * p_hat * (1 - p_hat)) * abs(deriv)
        z = (1 / 9) / sdev
        for *_, data in H.edges(data=True):
            assert data["nc_score"] == pytest.approx(1 / 9)
            assert data["nc_sdev"] == pytest.approx(sdev)
            # Default: exact tail P(X >= 1) of X ~ B(n.., mean).
            assert data["nc_pvalue"] == pytest.approx(1 - (1 - mean) ** n)
        D = nb.noise_corrected_filter(G, pvalue_method="delta")
        for *_, data in D.edges(data=True):
            assert data["nc_sdev"] == pytest.approx(sdev)
            assert data["nc_pvalue"] == pytest.approx(stats.norm.sf(z))

        legacy = nb.noise_corrected_filter(G, method="zscore")
        assert legacy[0][1]["nc_score"] == pytest.approx(-0.518, abs=1e-3)

    def test_pvalue_matches_delta_rule(self, les_miserables_weighted):
        # score > delta * sdev  <=>  nc_pvalue < 1 - Phi(delta)
        H = nb.noise_corrected_filter(les_miserables_weighted, pvalue_method="delta")
        cut = stats.norm.sf(1.64)
        for *_, d in H.edges(data=True):
            assert (d["nc_score"] > 1.64 * d["nc_sdev"]) == (d["nc_pvalue"] < cut)

    def test_signed_is_two_sided(self):
        G = nx.Graph()
        G.add_weighted_edges_from(NC_UNDIRECTED_EDGES)
        H = nb.noise_corrected_filter(G, signed=True, pvalue_method="delta")
        for (u, v), (score, sdev) in NC_REFERENCE["undirected"].items():
            z = score / sdev
            expected = min(1.0, 2 * stats.norm.sf(abs(z)))
            assert H[u][v]["nc_pvalue"] == pytest.approx(expected, abs=1e-9)
            assert H[u][v]["sign"] == (1 if z > 0 else -1)
        assert H[2][3]["sign"] == 1 and H[2][3]["nc_pvalue"] < 1e-10
        assert H[0][2]["sign"] == -1
        # Score 0 (lift exactly 1): the tails tie, so the sign is -1 as in R.
        assert H[1][2]["nc_score"] == 0.0
        assert (H[1][2]["nc_pvalue"], H[1][2]["sign"]) == (1.0, -1)

    def test_legacy_zscore_formula(self, count_graph):
        H = nb.noise_corrected_filter(count_graph, method="zscore")
        s = dict(count_graph.degree(weight="weight"))
        W = sum(d["weight"] for *_, d in count_graph.edges(data=True))
        p = s["a"] * s["b"] / W**2
        z = (5 - W * p) / math.sqrt(W * p * (1 - p))
        assert H["a"]["b"]["nc_score"] == pytest.approx(z)
        assert H["a"]["b"]["nc_pvalue"] == pytest.approx(stats.norm.sf(z))
        assert "nc_sdev" not in H["a"]["b"]

    def test_degenerate_prior_is_untestable(self):
        # A single edge carries all the weight: the prior has no variance.
        G = nx.DiGraph([(0, 1, {"weight": 3})])
        H = nb.noise_corrected_filter(G)
        assert H[0][1]["nc_score"] == 0.0
        assert math.isinf(H[0][1]["nc_sdev"])
        assert H[0][1]["nc_pvalue"] == 1.0

    def test_invalid_method_raises(self, count_graph):
        with pytest.raises(ValueError, match="method"):
            nb.noise_corrected_filter(count_graph, method="bogus")
        with pytest.raises(ValueError, match="pvalue_method"):
            nb.noise_corrected_filter(count_graph, pvalue_method="exact")

    @pytest.mark.parametrize("directed", [False, True])
    def test_pvalue_is_exact_binomial_tail(self, directed):
        # #105: nc_pvalue is P(X >= n_ij) for X ~ B(n.., n_i. n_.j / n..^2),
        # with undirected edges counted in both directions.
        G = nx.gnm_random_graph(30, 60, seed=2, directed=directed)
        rng = random.Random(0)
        for *_, d in G.edges(data=True):
            d["weight"] = rng.randint(1, 4)
        if directed:
            s_out = dict(G.out_degree(weight="weight"))
            s_in = dict(G.in_degree(weight="weight"))
            n = G.size(weight="weight")
        else:
            s_out = s_in = dict(G.degree(weight="weight"))
            n = 2 * G.size(weight="weight")
        H = nb.noise_corrected_filter(G, signed=False)
        S = nb.noise_corrected_filter(G, signed=True)
        for u, v, d in H.edges(data=True):
            p, k = s_out[u] * s_in[v] / n**2, d["weight"]
            upper, lower = stats.binom.sf(k - 1, n, p), stats.binom.cdf(k, n, p)
            assert d["nc_pvalue"] == pytest.approx(upper, rel=1e-9)
            assert S[u][v]["nc_pvalue"] == pytest.approx(
                min(1.0, 2 * min(upper, lower)), rel=1e-9
            )
            assert S[u][v]["sign"] == (1 if upper < lower else -1)
        if directed:
            # Same null as the marginal likelihood filter on a directed graph.
            M = nb.marginal_likelihood_filter(G)
            for u, v, d in H.edges(data=True):
                assert d["nc_pvalue"] == pytest.approx(M[u][v]["ml_pvalue"], rel=1e-9)

    def test_unit_weight_noise_is_not_significant(self):
        # The reproducer of #105: every weight 1, no structure.  The delta
        # rule kept all 40 edges with a median p-value near 1e-65.
        G = nx.gnm_random_graph(30, 40, seed=1)
        nx.set_edge_attributes(G, 1, "weight")
        p = sorted(
            d["nc_pvalue"] for *_, d in nb.noise_corrected_filter(G).edges(data=True)
        )
        assert sum(x < 0.05 for x in p) == 4
        assert p[len(p) // 2] == pytest.approx(0.1176, abs=1e-4)
        D = nb.noise_corrected_filter(G, pvalue_method="delta")
        assert all(d["nc_pvalue"] < 0.05 for *_, d in D.edges(data=True))

    def test_non_integer_counts_interpolate(self):
        # Real counts use the incomplete beta function: between the integer
        # tails and equal to them at integers.
        def pvalue(w):
            G = nx.DiGraph([(0, 1, {"weight": w}), (1, 2, {"weight": 3})])
            G.add_edge(2, 0, weight=2)
            G.add_edge(0, 2, weight=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return nb.noise_corrected_filter(G)[0][1]["nc_pvalue"]

        p1, p15, p2 = pvalue(1), pvalue(1.5), pvalue(2)
        assert p1 > p15 > p2
        n, n_i, n_j = 7.5, 2.5, 1.5
        assert p15 == pytest.approx(
            special.betainc(1.5, n - 0.5, n_i * n_j / n**2), rel=1e-12
        )

    def test_pure_python_betainc_matches_scipy(self):
        from nx_backbone.statistical import _betainc

        rng = random.Random(0)
        for _ in range(500):
            a, b = 10 ** rng.uniform(-1, 5), 10 ** rng.uniform(-1, 5)
            x = rng.random() ** rng.choice([1, 3, 10])
            ref = float(special.betainc(a, b, x))
            if ref > 1e-250:
                assert _betainc(a, b, x) == pytest.approx(ref, rel=1e-7, abs=1e-14)
        assert (_betainc(2.0, 3.0, 0.0), _betainc(2.0, 3.0, 1.0)) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Marginal likelihood filter -- Dianati (2016), Eqs. 1-2
# ---------------------------------------------------------------------------


class TestMarginalLikelihood:
    def test_undirected_formula(self, count_graph):
        H = nb.marginal_likelihood_filter(count_graph)
        strength = dict(count_graph.degree(weight="weight"))
        T = sum(strength.values()) / 2
        for u, v, d in H.edges(data=True):
            p = strength[u] * strength[v] / (2 * T * T)
            expected = stats.binom.sf(count_graph[u][v]["weight"] - 1, T, p)
            assert d["ml_pvalue"] == pytest.approx(expected)
        p = nx.get_edge_attributes(H, "ml_pvalue")
        assert min(p, key=p.get) == ("a", "b")  # strongest edge most significant

    def test_directed_formula(self):
        G = nx.DiGraph()
        G.add_weighted_edges_from([(0, 1, 4), (1, 2, 1), (2, 0, 2), (0, 2, 3)])
        H = nb.marginal_likelihood_filter(G)
        s_out = dict(G.out_degree(weight="weight"))
        s_in = dict(G.in_degree(weight="weight"))
        T = sum(s_out.values())
        for u, v, d in H.edges(data=True):
            p = s_out[u] * s_in[v] / T**2
            expected = stats.binom.sf(G[u][v]["weight"] - 1, T, p)
            assert d["ml_pvalue"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# ECM filter -- Gemmetto, Cardillo & Garlaschelli (2017)
# ---------------------------------------------------------------------------


def _ecm_expectations(fit, directed):
    import numpy as np

    a_out, b_out, a_in, b_in, act_out, act_in, _ = fit
    n = len(a_out)
    xxyy = np.exp(-(a_out[:, None] + a_in[None, :]))
    yy = np.exp(-(b_out[:, None] + b_in[None, :]))
    p = xxyy / (1 - yy + xxyy)
    p[~(act_out[:, None] & act_in[None, :])] = 0.0
    p[np.eye(n, dtype=bool)] = 0.0
    w = p / (1 - yy)
    return p, w


class TestECM:
    def test_fit_reproduces_degrees_and_strengths(self, les_miserables_weighted):
        import numpy as np

        from nx_backbone.statistical import _fit_ecm

        for directed in (False, True):
            G = les_miserables_weighted
            if directed:
                G = nx.DiGraph(G)
            nodes = list(G)
            out_deg = G.out_degree if directed else G.degree
            k_out = np.array([out_deg(v) for v in nodes], float)
            s_out = np.array([out_deg(v, weight="weight") for v in nodes], float)
            k_in, s_in = k_out, s_out
            if directed:
                k_in = np.array([G.in_degree(v) for v in nodes], float)
                s_in = np.array([G.in_degree(v, weight="weight") for v in nodes], float)
            fit = _fit_ecm(k_out, s_out, k_in, s_in, directed, 1000, 1e-8)
            assert fit[-1] < 1e-6, directed
            p, w = _ecm_expectations(fit, directed)
            msg = f"directed={directed}"
            np.testing.assert_allclose(p.sum(1), k_out, rtol=1e-6, err_msg=msg)
            np.testing.assert_allclose(w.sum(1), s_out, rtol=1e-6, err_msg=msg)
            np.testing.assert_allclose(p.sum(0), k_in, rtol=1e-6, err_msg=msg)
            np.testing.assert_allclose(w.sum(0), s_in, rtol=1e-6, err_msg=msg)

    def test_pvalue_is_geometric_tail(self, count_graph):
        # p-value = p_ij * (y_i y_j)^(w - 1): heavier links are rarer.
        H = nb.ecm_filter(count_graph)
        G2 = count_graph.copy()
        G2["a"]["b"]["weight"] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            H2 = nb.ecm_filter(G2)
        assert H["a"]["b"]["ecm_pvalue"] < H2["a"]["b"]["ecm_pvalue"]

    def test_pvalues_strictly_positive(self, les_miserables_weighted):
        # The pinned Les Misérables count is in test_les_miserables.py.
        H = nb.ecm_filter(les_miserables_weighted)
        assert all(0 < p <= 1 for *_, p in H.edges(data="ecm_pvalue"))

    def test_warns_when_solver_does_not_converge(self, les_miserables_weighted):
        with pytest.warns(RuntimeWarning, match="did not converge"):
            nb.ecm_filter(les_miserables_weighted, max_iter=1)

    def test_self_loop_gets_pvalue_one(self, count_graph):
        count_graph.add_edge("a", "a", weight=4)
        H = nb.ecm_filter(count_graph)
        assert H["a"]["a"]["ecm_pvalue"] == 1.0
        loop = nb.ecm_filter(count_graph, signed=True)["a"]["a"]
        # Untested: both tails are 1, a tie, signed -1 (#59).
        assert (loop["ecm_pvalue"], loop["sign"]) == (1.0, -1)

    def test_signed_tails_from_fitted_multipliers(self, count_graph):
        # Upper tail p_ij (y_i y_j)^(w-1), lower tail 1 - p_ij (y_i y_j)^w,
        # two-sided p-value 2 * min(tails) (#25).
        import numpy as np

        from nx_backbone.statistical import _fit_ecm

        nodes = list(count_graph)
        k = np.array([count_graph.degree(v) for v in nodes], float)
        s = np.array([count_graph.degree(v, weight="weight") for v in nodes], float)
        fit = _fit_ecm(k, s, k, s, False, 1000, 1e-6)
        p, _ = _ecm_expectations(fit, False)
        yy = np.exp(-(fit[1][:, None] + fit[3][None, :]))
        unsigned = nb.ecm_filter(count_graph)
        signed = nb.ecm_filter(count_graph, signed=True)
        for u, v, w in count_graph.edges(data="weight"):
            i, j = nodes.index(u), nodes.index(v)
            p_hi = p[i, j] * yy[i, j] ** (w - 1)
            p_lo = 1 - p[i, j] * yy[i, j] ** w
            assert unsigned[u][v]["ecm_pvalue"] == pytest.approx(p_hi, rel=1e-6)
            assert signed[u][v]["ecm_pvalue"] == pytest.approx(
                min(1.0, 2 * min(p_hi, p_lo)), rel=1e-6
            )
            assert signed[u][v]["sign"] == (1 if p_hi < p_lo else -1)

    def test_directed_classes_are_exact(self):
        # Nodes are grouped by (k_out, s_out, k_in, s_in) (#21).  Grouping by
        # out- and in-pairs separately is not exact, because each node's own
        # pair is excluded; this graph has many nodes that share an out-pair
        # but not an in-pair, so the fitted margins catch the difference.
        import numpy as np

        from nx_backbone.statistical import _fit_ecm

        G = nx.gnp_random_graph(120, 0.05, seed=1, directed=True)
        rng = np.random.default_rng(1)
        for u, v in G.edges():
            G[u][v]["weight"] = int(rng.integers(1, 4))
        G.add_node("isolated")
        G.add_edge("source", 0, weight=2)  # out-degree only
        nodes = list(G)
        k_out = np.array([G.out_degree(v) for v in nodes], float)
        s_out = np.array([G.out_degree(v, weight="weight") for v in nodes], float)
        k_in = np.array([G.in_degree(v) for v in nodes], float)
        s_in = np.array([G.in_degree(v, weight="weight") for v in nodes], float)
        fit = _fit_ecm(k_out, s_out, k_in, s_in, True, 1000, 1e-9)
        assert fit[-1] < 1e-9
        p, w = _ecm_expectations(fit, True)
        np.testing.assert_allclose(p.sum(1), k_out, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(w.sum(1), s_out, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(p.sum(0), k_in, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(w.sum(0), s_in, rtol=1e-8, atol=1e-12)

    def test_newton_cg_matches_dense_newton(self, les_miserables_weighted, monkeypatch):
        # Large problems use Newton-CG with chunked Hessian-vector products;
        # force it (and tiny chunks) on Les Miserables and compare.
        import nx_backbone.statistical as st

        graphs = [les_miserables_weighted, nx.DiGraph(les_miserables_weighted)]
        dense = [st.ecm_filter(G, tol=1e-10, signed=True) for G in graphs]
        monkeypatch.setattr(st, "_ECM_NEWTON_MAX_PARAMS", 0)
        monkeypatch.setattr(st, "_ECM_CHUNK", 64)
        for G, expected in zip(graphs, dense):
            for cache in (0, 1 << 22):
                monkeypatch.setattr(st, "_ECM_CACHE_PAIRS", cache)
                sparse = st.ecm_filter(G, tol=1e-10, signed=True)
                case = (G.is_directed(), cache)
                for u, v, d in expected.edges(data=True):
                    assert sparse[u][v]["ecm_pvalue"] == pytest.approx(
                        d["ecm_pvalue"], rel=1e-8, abs=1e-14
                    ), case
                    assert sparse[u][v]["sign"] == d["sign"], case

    def test_scales_to_thousands_of_nodes(self):
        # 5,000 nodes: the class-reduced fit converges without warnings.
        # ecm_filter warns whenever the fit's residual exceeds tol (1e-6 by
        # default), so turning that warning into an error checks convergence
        # without fitting a second time.
        import random

        G = nx.barabasi_albert_graph(5000, 3, seed=0)
        rng = random.Random(0)
        for u, v in G.edges():
            G[u][v]["weight"] = rng.randint(1, 20)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            H = nb.ecm_filter(G)
        assert all(0 < p <= 1 for *_, p in H.edges(data="ecm_pvalue"))


# ---------------------------------------------------------------------------
# LANS -- Foti, Hughes & Rockmore (2011); multiple linkage analysis
# ---------------------------------------------------------------------------


class TestLANS:
    def test_directed_heavy_edge_significant_from_target_side(self):
        # Five sources each send one link to ``v``; the heavy edge is the
        # strongest incoming link of ``v`` and must get p = 0 from v's side
        # (#11, contributed in #34).
        G = nx.DiGraph()
        for i, w in enumerate([100, 1, 1, 1, 1]):
            G.add_edge(f"s{i}", "v", weight=w)
        H = nb.lans_filter(G)
        assert H["s0"]["v"]["lans_pvalue"] == 0.0

    def test_directed_uses_target_in_edges_not_only_source_out_edges(self):
        # Edge (u, v1) is mid-ranked among u's out-edges (source-side eCDF =
        # 0.5) but is the strongest incoming link of v1, which drives p to 0.
        G = nx.DiGraph()
        G.add_weighted_edges_from([("u", "v1", 100), ("u", "w1", 200), ("x", "v1", 1)])
        H = nb.lans_filter(G)
        assert H["u"]["v1"]["lans_pvalue"] == 0.0

    def test_empirical_cdf(self, count_graph):
        H = nb.lans_filter(count_graph)
        # Edge c-d: at c weights are [1, 2, 3] -> F(3) = 1; at d [1, 3] -> 1.
        assert H["c"]["d"]["lans_pvalue"] == 0.0
        # Edge a-c: at a [1, 5] -> F(1) = 1/2; at c F(1) = 1/3 -> 1 - 1/2.
        assert H["a"]["c"]["lans_pvalue"] == pytest.approx(0.5)

    def test_signed_lower_tail_excludes_the_link(self, count_graph):
        # Upper tail #{weights > w} / k, lower tail #{weights < w} / k, each
        # the minimum over the two endpoints (#58).  Weights: a [1, 5],
        # b [1, 2, 5], c [1, 2, 3], d [1, 3].
        H = nb.lans_filter(count_graph, signed=True)
        got = {
            tuple(sorted((u, v))): (d["lans_pvalue"], d["sign"])
            for u, v, d in H.edges(data=True)
        }
        assert got == {
            # The weakest link of both endpoints: lower tail 0.
            ("a", "c"): (0.0, -1),
            ("b", "d"): (0.0, -1),
            # 1/3 above and 1/3 below at both b and c: an exact tie, signed
            # -1 as in R (#59).
            ("b", "c"): (pytest.approx(2 / 3), -1),
            ("a", "b"): (0.0, 1),
            ("c", "d"): (0.0, 1),
        }
        # Absent pairs (weight 0) do not change these and get lower tail 0.
        count_graph.add_node("z")  # no links: not testable, not added
        full = nb.lans_filter(count_graph, signed=True, missing_as_zero=True)
        assert {e: full.edges[e] for e in H.edges} == dict(H.edges.items())
        assert full["a"]["d"] == {"weight": 0, "lans_pvalue": 0.0, "sign": -1}
        assert set(full["z"]) == set()

    def test_signed_tails_are_mirror_images(self, les_miserables_weighted):
        # Reversing the order of the weights swaps the two tails exactly, so
        # every p-value is unchanged and every sign flips, except at ties.
        G = les_miserables_weighted
        R = G.copy()
        for _, _, d in R.edges(data=True):
            d["weight"] = 1 / d["weight"]
        H = nb.lans_filter(G, signed=True)
        H_rev = nb.lans_filter(R, signed=True)
        signs = set()
        for e, d in H.edges.items():
            assert H_rev.edges[e]["lans_pvalue"] == d["lans_pvalue"]
            signs.add((d["sign"], H_rev.edges[e]["sign"]))
        # (-1, -1) is a tie: upper == lower, signed -1 both ways.
        assert signs == {(1, -1), (-1, 1), (-1, -1)}

    def test_signed_degree_one_tie_is_negative(self):
        # The only link of a degree-1 node is both its strongest and its
        # weakest: at that node both tails are 0, so the link's upper and
        # lower tails are both 0.  This exact tie is signed -1, as in R
        # backbone 3.0.4 (#59), and the signed backbone keeps it as weak.
        G = nx.Graph()
        G.add_weighted_edges_from([("leaf", "hub", 2), ("hub", "a", 1), ("a", "b", 1)])
        G.add_weighted_edges_from([("hub", "c", 3), ("c", "b", 5)])
        H = nb.lans_filter(G, signed=True)
        assert H["leaf"]["hub"] == {"weight": 2, "lans_pvalue": 0.0, "sign": -1}
        B = nb.backbone_from_weighted(G, method="lans", alpha=0.05, signed=True)
        assert B["leaf"]["hub"]["sign"] == -1
        # Not a tie: the strongest link of both of its endpoints.
        assert (H["c"]["b"]["lans_pvalue"], H["c"]["b"]["sign"]) == (0.0, 1)

    def test_directed_signed_weak_from_either_side(self):
        # u -> t is mid-ranked among u's out-links [1, 5, 9] but the weakest
        # in-link of t [5, 7, 8]; s -> r is the weakest out-link of s
        # [2, 4, 6] but mid-ranked among r's in-links [1, 2, 3] (#11).
        D = nx.DiGraph()
        D.add_weighted_edges_from([("u", "t", 5), ("u", "p", 1), ("u", "q", 9)])
        D.add_weighted_edges_from([("a", "t", 7), ("b", "t", 8)])
        D.add_weighted_edges_from([("s", "r", 2), ("s", "m", 4), ("s", "n", 6)])
        D.add_weighted_edges_from([("c", "r", 1), ("d", "r", 3)])
        H = nb.lans_filter(D, signed=True)
        for edge in (("u", "t"), ("s", "r")):
            d = H.edges[edge]
            assert (d["lans_pvalue"], d["sign"]) == (0.0, -1), edge
        # Reversing every edge swaps out- and in-links: same result.
        Hr = nb.lans_filter(D.reverse(), signed=True)
        for (u, v), d in H.edges.items():
            assert Hr[v][u] == d

    def test_strongest_link_of_each_node_is_kept(self, les_miserables_weighted):
        H = nb.lans_filter(les_miserables_weighted)
        for node in H:
            if H.degree(node) == 0:
                continue
            best = max(H[node], key=lambda v: H[node][v]["weight"])
            assert H[node][best]["lans_pvalue"] == 0.0

    def test_mla_invalid_alpha_raises(self, weighted_triangle):
        with pytest.raises(ValueError):
            nb.multiple_linkage_analysis(weighted_triangle, alpha=1.5)


# ---------------------------------------------------------------------------
# Multiple linkage analysis -- Van Nuffel (2007), dominance fitting
# ---------------------------------------------------------------------------


def _shares(weights, n=None):
    """Normalized profile, padded with zero flows to length n."""
    total = sum(weights)
    p = [w / total for w in weights]
    return p + [0.0] * ((n or len(p)) - len(p))


def _ideal(k, n):
    return [1 / k] * k + [0.0] * (n - k)


def _dominance_deviation(weights, k):
    """Squared deviation of a sorted profile from k equal dominant links."""
    p = _shares(weights)
    return sum((a - b) ** 2 for a, b in zip(p, _ideal(k, len(p))))


def _weaver_variance(weights, k):
    """Weaver's variance: mean squared deviation over the k candidate links."""
    p = _shares(weights)
    return sum((p[r] - 1 / k) ** 2 for r in range(k)) / k


def _cod(weights, k, n):
    """Squared Pearson correlation of the zero-padded profile and the ideal."""
    p, t = _shares(weights, n), _ideal(k, n)
    mp, mt = sum(p) / n, sum(t) / n
    cov = sum((a - mp) * (b - mt) for a, b in zip(p, t))
    vp = sum((a - mp) ** 2 for a in p)
    vt = sum((b - mt) ** 2 for b in t)
    return cov**2 / (vp * vt)


def _star(weights, n=None, source="hub"):
    """DiGraph with out-links of the given weights from *source* to 0, 1, ...

    Isolated extra nodes pad the graph to n other nodes (zero flows).
    """
    G = nx.DiGraph()
    G.add_weighted_edges_from((source, i, w) for i, w in enumerate(weights))
    G.add_nodes_from(range(len(weights), n or len(weights)))
    return G


# Table 1 of Van Nuffel, Derudder & Witlox (2010): the flows of spatial units
# X and Y to eight other units a-h (zeros included), ranked largest first, and
# the published coefficient of determination for 1-5 equal links.
VAN_NUFFEL_TABLE_1 = {
    "X": ([50, 43, 42, 21, 12, 7], [0.31, 0.55, 0.88, 0.80, 0.63], 3),
    "Y": ([85, 32, 20, 19, 12, 5, 2], [0.85, 0.67, 0.51, 0.44, 0.34], 1),
}


class TestMultipleLinkageAnalysis:
    def test_reproduces_van_nuffel_2010_table_1(self):
        for unit in ("X", "Y"):
            weights, published, significant = VAN_NUFFEL_TABLE_1[unit]
            cods = [_cod(weights, k, 8) for k in range(1, 6)]
            assert cods == pytest.approx(published, abs=0.005), unit

            H = nb.multiple_linkage_analysis(_star(weights, 8, source=unit))
            assert H.nodes[unit]["mla_rank"] == significant, unit
            kept = sorted(v for _, v, keep in H.edges(data="mla_keep") if keep)
            assert kept == list(range(significant)), unit

    def test_deviation_measures_disagree(self):
        for unit, profile_k, weaver_k in [
            # X: (50, 43, 42, 21, 12, 7) / 175.  D_3 = 3616/91875 > D_4 =
            # 3013/122500 < D_5 = 1112/30625; V_3 = 1714/275625 > V_4 =
            # 2241/490000 < V_5 = 1063/153125.  COD picks 3.
            ("X", 4, 4),
            # Y: (85, 32, 20, 19, 12, 5, 2) / 175.  D_k is smallest at k = 4
            # (12757/122500); V_k decreases all the way to V_7 = 4808/214375.
            # COD picks 1.
            ("Y", 4, 7),
        ]:
            weights = VAN_NUFFEL_TABLE_1[unit][0]
            ks = range(1, len(weights) + 1)
            devs = [_dominance_deviation(weights, k) for k in ks]
            variances = [_weaver_variance(weights, k) for k in ks]
            assert 1 + devs.index(min(devs)) == profile_k, unit
            assert 1 + variances.index(min(variances)) == weaver_k, unit
            if unit == "X":
                assert devs[2:5] == pytest.approx(
                    [3616 / 91875, 3013 / 122500, 1112 / 30625]
                )
                assert variances[2:5] == pytest.approx(
                    [1714 / 275625, 2241 / 490000, 1063 / 153125]
                )
            else:
                assert devs[3] == pytest.approx(12757 / 122500)
                assert variances[6] == pytest.approx(4808 / 214375)

            G = _star(weights, 8, source=unit)
            ranks = {
                d: nb.multiple_linkage_analysis(G, deviation=d).nodes[unit]["mla_rank"]
                for d in ("cod", "profile", "weaver")
            }
            assert ranks == {
                "cod": VAN_NUFFEL_TABLE_1[unit][2],
                "profile": profile_k,
                "weaver": weaver_k,
            }, unit

    @pytest.mark.parametrize(
        ("weights", "deviations", "cods", "dominant"),
        [
            # One dominant link: profile (10, 1, 1, 1) / 13, n = 4.
            # D_1 = (3/13)^2 + 3 (1/13)^2 = 12/169; COD_1 = 1.
            (
                [10, 1, 1, 1],
                [12 / 169, 0.263314, 0.327416, 0.359467],
                [1, 1 / 3, 1 / 9],
                1,
            ),
            # Two: profile (10, 10, 1, 1) / 22.
            # D_2 = 2 (10/22 - 1/2)^2 + 2 (1/22)^2 = 4/484; COD_2 = 1.
            (
                [10, 10, 1, 1],
                [246 / 484, 4 / 484, 0.114325, 0.167355],
                [1 / 3, 1, 1 / 3],
                2,
            ),
            # Three: profile (10, 10, 10, 1, 1) / 32.
            # D_3 = 3 (10/32 - 1/3)^2 + 2 (1/32)^2 = 3/2304 + 2/1024.
            (
                [10, 10, 10, 1, 1],
                [0.669922, 0.169922, 3 / 2304 + 2 / 1024, 0.060547, 0.094922],
                [1 / 6, 4 / 9, 1, 3 / 8],
                3,
            ),
        ],
    )
    def test_hand_computed_dominant_links(self, weights, deviations, cods, dominant):
        n = len(weights)
        devs = [_dominance_deviation(weights, k) for k in range(1, n + 1)]
        assert devs == pytest.approx(deviations, abs=1e-6)
        # With n = number of links, the all-equal ideal has no variance.
        assert [_cod(weights, k, n) for k in range(1, n)] == pytest.approx(cods)

        for deviation in ("cod", "profile", "weaver"):
            H = nb.multiple_linkage_analysis(_star(weights), deviation=deviation)
            assert H.nodes["hub"]["mla_rank"] == dominant, deviation
            kept = sorted(v for _, v, keep in H.edges(data="mla_keep") if keep)
            assert kept == list(range(dominant)), deviation
            # Targets have no out-links in a directed graph.
            assert all(H.nodes[i]["mla_rank"] == 0 for i in range(n)), deviation

    def test_evenly_spread_keeps_all(self):
        for deviation in ("cod", "profile", "weaver"):
            for n in (4, 6):
                H = nb.multiple_linkage_analysis(
                    _star([4, 4, 4, 4], n), deviation=deviation
                )
                assert H.nodes["hub"]["mla_rank"] == 4, (deviation, n)

    def test_cod_counts_zero_flows(self):
        # Profile (6, 3, 1) / 10.  Linked to all n = 3 other nodes:
        # COD_1 = 16/19 > COD_2 = 49/76.  With two more nodes (n = 5):
        # COD_1 = 10/13 < COD_2 = 125/156.  D_k and V_k ignore the padding.
        assert _cod([6, 3, 1], 1, 3) == pytest.approx(16 / 19)
        assert _cod([6, 3, 1], 2, 3) == pytest.approx(49 / 76)
        assert _cod([6, 3, 1], 1, 5) == pytest.approx(10 / 13)
        assert _cod([6, 3, 1], 2, 5) == pytest.approx(125 / 156)
        for n, cod_k in [(3, 1), (5, 2)]:
            G = _star([6, 3, 1], n)
            assert nb.multiple_linkage_analysis(G).nodes["hub"]["mla_rank"] == cod_k
            for d in ("profile", "weaver"):
                H = nb.multiple_linkage_analysis(G, deviation=d)
                assert H.nodes["hub"]["mla_rank"] == 2

    def test_tied_links_kept_together(self):
        # (6, 3, 3) / 12 = (1/2, 1/4, 1/4): D_1 = 3/8, D_2 = 1/8 (splits the
        # tie, not considered), D_3 = 1/24 -> keep all three.  Weaver:
        # V_1 = 1/4, V_2 = 1/32, V_3 = 1/72.
        devs = [_dominance_deviation([6, 3, 3], k) for k in (1, 2, 3)]
        assert devs == pytest.approx([3 / 8, 1 / 8, 1 / 24])
        variances = [_weaver_variance([6, 3, 3], k) for k in (1, 2, 3)]
        assert variances == pytest.approx([1 / 4, 1 / 32, 1 / 72])
        for deviation in ("profile", "weaver"):
            H = nb.multiple_linkage_analysis(_star([6, 3, 3]), deviation=deviation)
            assert H.nodes["hub"]["mla_rank"] == 3, deviation

    def test_weaver_tied_split_not_considered(self):
        # (8, 8, 5, 5, 3, 3) / 32 with V_5 < V_6: k = 5 would split the tied
        # 3s, so Weaver keeps all six (k = 4 fits worse than k = 6).
        weights = [8, 8, 5, 5, 3, 3]
        v = [_weaver_variance(weights, k) for k in range(1, 7)]
        assert v[4] < v[5] < v[3]
        H = nb.multiple_linkage_analysis(_star(weights), deviation="weaver")
        assert H.nodes["hub"]["mla_rank"] == 6

    def test_equal_fits_go_to_smaller_rank(self):
        # (3, 3, 1, 1) / 8: D_2 = D_3 = D_4 = 1/16 and V_2 = V_3 = V_4 = 1/64;
        # k = 3 would also split a tie.  The most parsimonious fit, k = 2,
        # wins.  (COD_2 = 1 is the unique best.)
        devs = [_dominance_deviation([3, 3, 1, 1], k) for k in (2, 3, 4)]
        assert devs == pytest.approx([1 / 16] * 3)
        variances = [_weaver_variance([3, 3, 1, 1], k) for k in (2, 3, 4)]
        assert variances == pytest.approx([1 / 64] * 3)
        for deviation in ("cod", "profile", "weaver"):
            H = nb.multiple_linkage_analysis(_star([3, 3, 1, 1]), deviation=deviation)
            assert H.nodes["hub"]["mla_rank"] == 2, deviation
            kept = sorted(v for _, v, keep in H.edges(data="mla_keep") if keep)
            assert kept == [0, 1], deviation

    def test_undirected_or_rule(self):
        # Hub h: (10, 1, 1, 1) -> keeps only h-a.  For b, h-b is the stronger
        # of its two links, so the OR rule keeps it; for c, h-c is weak.
        G = nx.Graph()
        G.add_weighted_edges_from(
            [("h", "a", 10), ("h", "b", 1), ("h", "c", 1), ("h", "d", 1)]
        )
        G.add_weighted_edges_from([("b", "x", 0.1), ("c", "y", 5), ("d", "z", 5)])
        for deviation in ("cod", "profile", "weaver"):
            H = nb.multiple_linkage_analysis(G, deviation=deviation)
            assert H.nodes["h"]["mla_rank"] == 1
            assert H.nodes["b"]["mla_rank"] == 1
            assert H["h"]["a"]["mla_keep"]
            assert H["h"]["b"]["mla_keep"]  # dominant for b only
            assert not H["h"]["c"]["mla_keep"]  # dominant for neither endpoint

    def test_lans_method_matches_netbone_convention(self, les_miserables_weighted):
        H = nb.multiple_linkage_analysis(
            les_miserables_weighted, alpha=0.05, method="lans"
        )
        L = nb.lans_filter(les_miserables_weighted)
        for u, v, d in H.edges(data=True):
            assert d["mla_pvalue"] == L[u][v]["lans_pvalue"]
            assert d["mla_keep"] == (d["mla_pvalue"] <= 0.05)

    def test_every_node_keeps_its_strongest_link(self, les_miserables_weighted):
        G = les_miserables_weighted
        for deviation in ("cod", "profile", "weaver"):
            H = nb.multiple_linkage_analysis(G, deviation=deviation)
            for node in H:
                best = max(H[node], key=lambda v: H[node][v]["weight"])
                assert H[node][best]["mla_keep"], (deviation, node)
                assert H.nodes[node]["mla_rank"] >= 1, (deviation, node)

    def test_invalid_method_or_deviation_raises(self, weighted_triangle):
        with pytest.raises(ValueError, match="method"):
            nb.multiple_linkage_analysis(weighted_triangle, method="bogus")
        with pytest.raises(ValueError, match="deviation"):
            nb.multiple_linkage_analysis(weighted_triangle, deviation="bogus")


# ---------------------------------------------------------------------------
# missing_as_zero: absent dyads tested as weight-0 edges (Backbone 3.0, #26)
# ---------------------------------------------------------------------------

_MISSING_SCORERS = [
    (nb.disparity_filter, "disparity_pvalue"),
    (nb.lans_filter, "lans_pvalue"),
    (nb.marginal_likelihood_filter, "ml_pvalue"),
    (nb.ecm_filter, "ecm_pvalue"),
    (nb.noise_corrected_filter, "nc_pvalue"),
]


def _zero_edges(H, attr):
    """``{sorted pair: (pvalue, sign)}`` for the weight-0 edges of *H*."""
    return {
        tuple(sorted((u, v))): (d[attr], d.get("sign"))
        for u, v, d in H.edges(data=True)
        if d["weight"] == 0
    }


@pytest.fixture
def small_graph():
    """Strengths a=5, b=5, c=4, d=2, e=0; degrees 2, 2, 3, 1, 0."""
    G = nx.Graph()
    G.add_weighted_edges_from(
        [("a", "b", 4), ("a", "c", 1), ("b", "c", 1), ("c", "d", 2)]
    )
    G.add_node("e")
    return G


@pytest.fixture
def two_pairs():
    """Two strong pairs a-b and c-d joined by one weak edge, plus isolate e.

    T = 41; strengths a=21, b=20, c=21, d=20.
    """
    G = nx.Graph()
    G.add_weighted_edges_from([("a", "b", 20), ("c", "d", 20), ("a", "c", 1)])
    G.add_node("e")
    return G


class TestMissingAsZero:
    def test_default_unchanged_and_existing_edges_unaffected(self, count_graph):
        count_graph.add_node("z")
        edges = set(map(frozenset, count_graph.edges()))
        failures = []
        for scorer, _ in _MISSING_SCORERS:
            for signed in (False, True):
                case = f"{scorer.__name__}(signed={signed})"
                default = scorer(count_graph, signed=signed)
                if set(map(frozenset, default.edges())) != edges:
                    failures.append(f"{case}: default changed the edge set")
                off = scorer(count_graph, signed=signed, missing_as_zero=False)
                if not nx.utils.graphs_equal(default, off):
                    failures.append(f"{case}: missing_as_zero=False is not the default")
                full = scorer(count_graph, signed=signed, missing_as_zero=True)
                # Weight-0 pairs do not change the null model of existing edges.
                if any(full[u][v] != d for u, v, d in default.edges(data=True)):
                    failures.append(f"{case}: existing edges changed")
                added = set(map(frozenset, full.edges())) - edges
                if not added or any(full.edges[tuple(e)]["weight"] != 0 for e in added):
                    failures.append(f"{case}: no weight-0 pairs added")
        assert not failures, "\n".join(failures)

    def test_disparity_and_lans_hand_values(self, small_graph):
        # A weight of 0 is below every link: upper tail 1, lower tail 0 from
        # any endpoint with links (degree >= 2 for the disparity filter).
        # Pairs with the isolate "e" have a 0/0 normalized weight and are not
        # added (R: NA).  The degree of d stays 1; (a, d) is tested from a.
        for scorer, attr in [
            (nb.disparity_filter, "disparity_pvalue"),
            (nb.lans_filter, "lans_pvalue"),
        ]:
            signed = scorer(small_graph, signed=True, missing_as_zero=True)
            assert _zero_edges(signed, attr) == {
                ("a", "d"): (0.0, -1),
                ("b", "d"): (0.0, -1),
            }, scorer.__name__
            unsigned = scorer(small_graph, missing_as_zero=True)
            assert _zero_edges(unsigned, attr) == {
                ("a", "d"): (1.0, None),
                ("b", "d"): (1.0, None),
            }, scorer.__name__

    def test_directed_disparity_and_lans(self):
        # Out-degrees a=2, b=1, c=1; in-degrees a=1, b=1, c=2; e isolated.
        D = nx.DiGraph()
        D.add_weighted_edges_from(
            [("a", "b", 3), ("b", "c", 1), ("c", "a", 2), ("a", "c", 1)]
        )
        D.add_node("e")
        disp = nb.disparity_filter(D, signed=True, missing_as_zero=True)
        lans = nb.lans_filter(D, signed=True, missing_as_zero=True)
        # Only b->a and c->b are testable (e has no out- or in-links).  Both
        # sides of each have one link, which cannot reject the disparity null.
        zeros = {(u, v) for u, v, w in disp.edges(data="weight") if w == 0}
        assert zeros == {("b", "a"), ("c", "b")}
        assert all(disp[u][v]["disparity_pvalue"] == 1.0 for u, v in zeros)
        assert {(u, v) for u, v, w in lans.edges(data="weight") if w == 0} == zeros
        assert all(lans[u][v]["lans_pvalue"] == 0.0 for u, v in zeros)
        assert all(lans[u][v]["sign"] == -1 for u, v in zeros)

    def test_mlf_hand_values(self, two_pairs):
        # Absent pair (u, v) is X = 0 under B(T, s_u s_v / (2 T^2)): upper
        # tail 1, lower tail (1 - p)^T.  Pairs with the isolate are tested
        # (as in R) and get p-value 1.
        T = 41
        H = nb.marginal_likelihood_filter(two_pairs, signed=True, missing_as_zero=True)
        got = _zero_edges(H, "ml_pvalue")
        lo_ad = (1 - 21 * 20 / (2 * T * T)) ** T
        lo_bd = (1 - 20 * 20 / (2 * T * T)) ** T
        assert got[("a", "d")] == (pytest.approx(2 * lo_ad), -1)
        assert got[("b", "c")] == (pytest.approx(2 * lo_ad), -1)
        assert got[("b", "d")] == (pytest.approx(2 * lo_bd), -1)
        assert got[("b", "d")][0] == pytest.approx(0.0111048103)
        for x in "abcd":  # both tails are 1: a tie, signed -1
            assert got[(x, "e")] == (1.0, -1)
        assert len(got) == 7
        unsigned = nb.marginal_likelihood_filter(two_pairs, missing_as_zero=True)
        assert {p for p, _ in _zero_edges(unsigned, "ml_pvalue").values()} == {1.0}

    def test_noise_corrected_hand_values(self, two_pairs):
        # Pair (b, d): n_i = n_j = 20, n = 82 (undirected weights counted
        # twice).  n_ij = 0 gives lift 0, score -1 and derivative 2 * kappa.
        n, ni, nj = 82.0, 20.0, 20.0
        E = ni * nj / n**2
        V = ni * nj * (n - ni) * (n - nj) / (n**4 * (n - 1))
        a = E**2 * (1 - E) / V - E
        b = E * (1 - E**2) / V - (1 - E)
        p_post = a / (a + b + n)
        sdev = 2 * (n / (ni * nj)) * math.sqrt(n * p_post * (1 - p_post))
        H = nb.noise_corrected_filter(
            two_pairs, signed=True, missing_as_zero=True, pvalue_method="delta"
        )
        d = H["b"]["d"]
        assert d["weight"] == 0
        assert d["nc_score"] == -1.0
        assert d["nc_sdev"] == pytest.approx(sdev)
        assert d["nc_sdev"] == pytest.approx(0.6746729794)
        assert d["nc_pvalue"] == pytest.approx(math.erfc(1 / (sdev * math.sqrt(2))))
        assert d["sign"] == -1
        # Binomial default: upper tail 1, lower tail P(X = 0) = (1 - E)^n.
        B = nb.noise_corrected_filter(two_pairs, signed=True, missing_as_zero=True)
        assert B["b"]["d"]["nc_sdev"] == pytest.approx(sdev)
        assert B["b"]["d"]["nc_pvalue"] == pytest.approx(min(1, 2 * (1 - E) ** n))
        assert B["b"]["d"]["sign"] == -1
        # The isolate has a zero margin: undefined lift, not added.
        assert not any(H.has_edge(x, "e") for x in "abcd")
        assert len(_zero_edges(H, "nc_pvalue")) == 3

    def test_noise_corrected_zscore_hand_values(self, two_pairs):
        # Legacy z = (0 - W p) / sqrt(W p (1 - p)), p = s_b s_d / W^2, W = 41.
        W, p = 41, 20 * 20 / 41**2
        z = -math.sqrt(W * p / (1 - p))
        H = nb.noise_corrected_filter(
            two_pairs, method="zscore", signed=True, missing_as_zero=True
        )
        assert H["b"]["d"]["nc_score"] == pytest.approx(z)
        assert H["b"]["d"]["nc_pvalue"] == pytest.approx(math.erfc(-z / math.sqrt(2)))
        assert not H.has_edge("b", "e")

    def test_ecm_lower_tail_is_one_minus_link_probability(self):
        import numpy as np

        from nx_backbone.statistical import _fit_ecm

        G = nx.Graph()
        G.add_weighted_edges_from(
            [
                ("a", "b", 20),
                ("c", "d", 20),
                ("a", "c", 1),
                ("b", "d", 1),
                ("a", "e", 1),
                ("b", "f", 1),
            ]
        )
        G.add_node("g")
        nodes = list(G)
        k = np.array([G.degree(v) for v in nodes], float)
        s = np.array([G.degree(v, weight="weight") for v in nodes], float)
        fit = _fit_ecm(k, s, k, s, False, 1000, 1e-6)
        p, _ = _ecm_expectations(fit, False)
        H = nb.ecm_filter(G, signed=True, missing_as_zero=True)
        i, j = nodes.index("a"), nodes.index("d")
        # P(w_ad <= 0) = 1 - p_ad and P(w_ad >= 0) = 1.
        assert H["a"]["d"]["ecm_pvalue"] == pytest.approx(2 * (1 - p[i, j]), rel=1e-6)
        assert H["a"]["d"]["ecm_pvalue"] < 1.0
        assert H["a"]["d"]["sign"] == -1
        # The isolate g has x = 0, so p_gj = 0 and both tails are 1.
        assert H["a"]["g"]["ecm_pvalue"] == 1.0
        unsigned = nb.ecm_filter(G, missing_as_zero=True)
        assert {pv for pv, _ in _zero_edges(unsigned, "ecm_pvalue").values()} == {1.0}

    def test_edgeless_graph_adds_nothing(self):
        for scorer, _ in _MISSING_SCORERS:
            H = scorer(nx.empty_graph(4), signed=True, missing_as_zero=True)
            assert H.number_of_edges() == 0, scorer.__name__

    def test_warns_when_output_is_dense(self, two_pairs, monkeypatch):
        import nx_backbone.statistical as st

        for scorer, _ in _MISSING_SCORERS:
            monkeypatch.setattr(st, "_MISSING_AS_ZERO_WARN_DYADS", 6)
            # 5 nodes: 10 pairs - 3 edges = 7 absent pairs > 6.
            with pytest.warns(
                RuntimeWarning, match="missing_as_zero=True adds 7"
            ) as rec:
                scorer(two_pairs, signed=True, missing_as_zero=True)
            # The warning points at the caller, not at library internals.
            assert any(r.filename == __file__ for r in rec), scorer.__name__
            monkeypatch.setattr(st, "_MISSING_AS_ZERO_WARN_DYADS", 7)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                scorer(two_pairs, signed=True, missing_as_zero=True)

    def test_aliases_forward_missing_as_zero(self, two_pairs):
        for alias, full in [
            (nb.disparity, nb.disparity_filter),
            (nb.mlf, nb.marginal_likelihood_filter),
            (nb.lans, nb.lans_filter),
        ]:
            got = alias(two_pairs, signed=True, missing_as_zero=True)
            want = full(two_pairs, signed=True, missing_as_zero=True)
            assert nx.utils.graphs_equal(got, want)


# ---------------------------------------------------------------------------
# Generalized disparity null: disparity_filter(a=...)
# ---------------------------------------------------------------------------


def _center_star(weights):
    """Star with center "c"; leaves have degree 1, so p-values come from "c"."""
    G = nx.Graph()
    G.add_weighted_edges_from(("c", i, w) for i, w in enumerate(weights))
    return G


def _disparity_closed_form(G, weight="weight"):
    """The a=1 p-values exactly as networkx-backbone 0.2 computes them."""
    s, k = dict(G.degree(weight=weight)), dict(G.degree())

    def tail(w, x):
        if k[x] <= 1:
            return 1.0
        return max((1.0 - min(w / s[x], 1.0)) ** (k[x] - 1), 0.0)

    return {
        (u, v): min(tail(d[weight], u), tail(d[weight], v))
        for u, v, d in G.edges(data=True)
    }


class TestDisparityConcentration:
    def test_a_one_is_bit_identical(self, les_miserables_weighted):
        G = les_miserables_weighted
        closed_form = _disparity_closed_form(G)
        for a in (1, 1.0):
            H = nb.disparity_filter(G, a=a)
            got = nx.get_edge_attributes(H, "disparity_pvalue")
            assert got == closed_form, a  # exact, not approx
            for kw in ({"signed": True}, {"signed": True, "missing_as_zero": True}):
                assert nx.utils.graphs_equal(
                    nb.disparity_filter(G, a=a, **kw), nb.disparity_filter(G, **kw)
                ), (a, kw)
            assert "disparity_a" not in H.graph, a

    def test_hand_computed_values(self):
        for weights, a, expected in [
            # k = 3, a = 1/2 (c = 2): Beta(2, 4) upper tail at x = 1/2 is
            # 5 (1 - x)^4 - 4 (1 - x)^5 = 3/16.
            ((2, 1, 1), 0.5, 3 / 16),
            # k = 3, a = 2 (c = 1/2): Beta(1/2, 1) has cdf sqrt(x), so the
            # upper tail at x = 1/4 is 1/2.
            ((1, 1.5, 1.5), 2.0, 0.5),
        ]:
            H = nb.disparity_filter(_center_star(weights), a=a)
            assert H["c"][0]["disparity_pvalue"] == pytest.approx(
                expected, rel=1e-12
            ), a

    def test_beta_tail(self, hub_star):
        k, x = 4, 100 / 103
        for a in (0.3, 0.5, 2.0, 7.0):
            H = nb.disparity_filter(hub_star, a=a, signed=True)
            hi = stats.beta.sf(x, 1 / a, (k - 1) / a)
            lo = stats.beta.cdf(x, 1 / a, (k - 1) / a)
            assert H["c"][1]["disparity_pvalue"] == pytest.approx(
                min(1.0, 2 * min(hi, lo)), rel=1e-9
            ), a
            assert H["c"][1]["sign"] == (1 if hi <= lo else -1), a

    def test_close_to_one_is_close_to_closed_form(self, les_miserables_weighted):
        G = les_miserables_weighted
        exact = _disparity_closed_form(G)
        near = nx.get_edge_attributes(
            nb.disparity_filter(G, a=1 + 1e-12), "disparity_pvalue"
        )
        assert near == pytest.approx(exact, rel=1e-6, abs=1e-12)

    def test_larger_a_gives_sparser_backbones(self, les_miserables_weighted):
        counts = [
            nb.threshold_filter(
                nb.disparity_filter(les_miserables_weighted, a=a),
                "disparity_pvalue",
                0.05,
            ).number_of_edges()
            for a in (0.25, 0.5, 1.0, 2.0, 4.0)
        ]
        assert counts == [37, 20, 9, 5, 2]

    def test_scale_invariant(self, les_miserables_weighted):
        G = les_miserables_weighted
        G4 = nb.transform_weights(G, lambda w: 4 * w)  # exact in floating point
        for a in (2.0, "ml"):
            H, H4 = nb.disparity_filter(G, a=a), nb.disparity_filter(G4, a=a)
            assert nx.get_edge_attributes(H4, "disparity_pvalue") == pytest.approx(
                nx.get_edge_attributes(H, "disparity_pvalue"), rel=1e-9
            )

    def test_directed_uses_out_and_in_profiles(self):
        G = nx.DiGraph()
        G.add_weighted_edges_from(
            [(f"s{i}", "v", w) for i, w in enumerate([100, 1, 1, 1, 1])]
        )
        H = nb.disparity_filter(G, a=2.0)
        expected = stats.beta.sf(100 / 104, 0.5, 4 / 2)
        assert H["s0"]["v"]["disparity_pvalue"] == pytest.approx(expected, rel=1e-9)

    def test_missing_as_zero_pairs_match_a_one(self, two_pairs):
        # A weight-0 pair has upper tail 1 and lower tail 0 whatever a is.
        base = nb.disparity_filter(two_pairs, signed=True, missing_as_zero=True)
        for a in (0.5, 3.0, "ml"):
            H = nb.disparity_filter(two_pairs, a=a, signed=True, missing_as_zero=True)
            assert set(H.edges()) == set(base.edges())
            for u, v, d in H.edges(data=True):
                if d["weight"] == 0:
                    assert d == base[u][v]

    def test_signed_marks_strong_and_weak(self, hub_star):
        H = nb.disparity_filter(hub_star, a=2.0, signed=True)
        assert H["c"][1]["sign"] == 1
        assert H["c"][2]["sign"] == -1

    def test_ml_estimate_les_miserables(self, les_miserables_weighted):
        H = nb.disparity_filter(les_miserables_weighted, a="ml")
        a_hat = H.graph["disparity_a"]
        assert a_hat == pytest.approx(0.525125, rel=1e-5)
        assert nb.threshold_filter(H, "disparity_pvalue", 0.05).number_of_edges() == 19
        # The p-values are those of the fixed a = a_hat.
        fixed = nb.disparity_filter(les_miserables_weighted, a=a_hat)
        assert nx.get_edge_attributes(H, "disparity_pvalue") == nx.get_edge_attributes(
            fixed, "disparity_pvalue"
        )

    def test_ml_estimate_maximizes_the_likelihood(self, les_miserables_weighted):
        from scipy.special import gammaln

        G = les_miserables_weighted
        a_hat = nb.disparity_filter(G, a="ml").graph["disparity_a"]

        def loglik(c):
            total = 0.0
            for u in G:
                ws = [d["weight"] for d in G[u].values()]
                if len(ws) > 1:
                    s = sum(ws)
                    k = len(ws)
                    total += gammaln(k * c) - k * gammaln(c)
                    total += (c - 1) * sum(math.log(w / s) for w in ws)
            return total

        c_hat = 1 / a_hat
        assert loglik(c_hat) > loglik(c_hat * 1.01)
        assert loglik(c_hat) > loglik(c_hat / 1.01)

    def test_ml_directed_symmetric_matches_undirected(self, les_miserables_weighted):
        G = les_miserables_weighted
        a_und = nb.disparity_filter(G, a="ml").graph["disparity_a"]
        a_dir = nb.disparity_filter(nx.DiGraph(G), a="ml").graph["disparity_a"]
        assert a_dir == pytest.approx(a_und, rel=1e-9)

    def test_ml_without_informative_nodes(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=3.0)
        H = nb.disparity_filter(G, a="ml")
        assert math.isnan(H.graph["disparity_a"])
        assert H[0][1]["disparity_pvalue"] == 1.0

    def test_ml_equal_shares_warns_and_uses_bound(self):
        # Every share is exactly 1/k: the likelihood grows without bound as
        # the concentration c = 1/a grows.
        G = nx.complete_graph(5)
        nx.set_edge_attributes(G, 1.0, "weight")
        with pytest.warns(RuntimeWarning, match="edge of the search interval"):
            H = nb.disparity_filter(G, a="ml")
        assert H.graph["disparity_a"] == pytest.approx(1e-4)

    def test_invalid_a(self, hub_star):
        for a in (0, -1.0, math.nan, math.inf, "mle", None, True):
            with pytest.raises(ValueError, match="a must be"):
                nb.disparity_filter(hub_star, a=a)

    def test_alias_forwards_a(self, les_miserables_weighted):
        G = les_miserables_weighted
        assert nx.utils.graphs_equal(
            nb.disparity(G, a="ml"), nb.disparity_filter(G, a="ml")
        )

    def test_large_strength_limit_of_polya_filter(self):
        # polya_filter's exact beta-binomial p-value (Eq. (1)-(2) of
        # Marcaccioli & Livan, 2019) converges to disparity_filter(a=a) as the
        # strength grows with the shares fixed.
        a, shares = 2.0, (4, 3, 2, 1)
        gaps = []
        for scale in (1, 100, 10_000):
            G = _center_star([scale * x for x in shares])
            p_polya = nb.polya_filter(G, a=a)["c"][0]["polya_pvalue"]
            p_beta = nb.disparity_filter(G, a=a)["c"][0]["disparity_pvalue"]
            gaps.append(abs(p_polya - p_beta) / p_beta)
        assert gaps[0] > gaps[1] > gaps[2]
        assert gaps[2] < 1e-3


# ---------------------------------------------------------------------------
# Count unit of the count-based nulls: unit=
# ---------------------------------------------------------------------------

_COUNT_SCORERS = [
    (nb.marginal_likelihood_filter, "ml_pvalue", {}),
    (nb.ecm_filter, "ecm_pvalue", {}),
    (nb.noise_corrected_filter, "nc_pvalue", {}),
    (nb.noise_corrected_filter, "nc_pvalue", {"method": "zscore"}),
]
_COUNT_IDS = ["mlf", "ecm", "nc", "nc-zscore"]


class TestCountUnit:
    def test_default_unit_is_unchanged(self, les_miserables_weighted):
        G = les_miserables_weighted
        for (scorer, _, kw), name in zip(_COUNT_SCORERS, _COUNT_IDS):
            for extra in ({}, {"signed": True, "missing_as_zero": True}):
                assert nx.utils.graphs_equal(
                    scorer(G, unit=1, **kw, **extra), scorer(G, **kw, **extra)
                ), (name, extra)

    def test_unit_recovers_integer_counts(self, les_miserables_weighted):
        # Weights recorded as unit * count (with floating-point error) give
        # the p-values of the integer counts, without a warning.
        G = les_miserables_weighted
        failures = []
        for (scorer, attr, kw), name in zip(_COUNT_SCORERS, _COUNT_IDS):
            want = scorer(G, signed=True, **kw)
            for unit in (0.1, 2.5, 7):
                scaled = nb.transform_weights(G, lambda w, unit=unit: unit * w)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        got = scorer(scaled, unit=unit, signed=True, **kw)
                except Warning as w:
                    failures.append(f"{name} unit={unit}: warned {w}")
                    continue
                if any(
                    got[u][v][attr] != pytest.approx(d[attr], rel=1e-9, abs=1e-15)
                    or got[u][v]["sign"] != d["sign"]
                    for u, v, d in want.edges(data=True)
                ):
                    failures.append(f"{name} unit={unit}: p-values or signs differ")
        assert not failures, "\n".join(failures)

    def test_mlf_and_ecm_unit_is_exact_after_rounding(self, les_miserables_weighted):
        G = les_miserables_weighted
        scaled = nb.transform_weights(G, lambda w: 0.1 * w)
        for scorer, attr in [
            (nb.marginal_likelihood_filter, "ml_pvalue"),
            (nb.ecm_filter, "ecm_pvalue"),
        ]:
            got = nx.get_edge_attributes(scorer(scaled, unit=0.1), attr)
            assert got == nx.get_edge_attributes(scorer(G), attr)

    def test_results_depend_on_unit(self, les_miserables_weighted):
        # The Les Misérables counts quoted in the docstrings: unit=0.1 counts
        # every weight as ten events (126, 72 and 126 edges at unit=1, pinned
        # in test_les_miserables.py).
        got = {}
        for scorer, attr in [
            (nb.marginal_likelihood_filter, "ml_pvalue"),
            (nb.ecm_filter, "ecm_pvalue"),
            (nb.noise_corrected_filter, "nc_pvalue"),
        ]:
            H = scorer(les_miserables_weighted, unit=0.1)
            got[attr] = nb.threshold_filter(H, attr, 0.05).number_of_edges()
        assert got == {"ml_pvalue": 215, "ecm_pvalue": 89, "nc_pvalue": 215}

    def test_nc_score_does_not_depend_on_unit(self, les_miserables_weighted):
        G = les_miserables_weighted
        base = nx.get_edge_attributes(nb.noise_corrected_filter(G), "nc_score")
        got = nx.get_edge_attributes(nb.noise_corrected_filter(G, unit=0.1), "nc_score")
        assert got == pytest.approx(base, rel=1e-12)

    def test_warns_on_non_integer_counts(self, count_graph):
        count_graph["a"]["c"]["weight"] = 0.4
        for scorer, match in [
            (nb.marginal_likelihood_filter, "below 0.5 becomes 0"),
            (nb.ecm_filter, "below 1.5 becomes 1"),
            (nb.noise_corrected_filter, "depend on unit"),
        ]:
            with pytest.warns(UserWarning, match=match) as rec:
                H = scorer(count_graph)
            assert "1 of 5 weights are not integer counts" in str(rec[0].message)
            assert any(r.filename == __file__ for r in rec), scorer.__name__
            if scorer is nb.marginal_likelihood_filter:
                assert H["a"]["c"]["ml_pvalue"] == 1.0  # rounded to 0 unit edges
            # unit=0.2 makes every weight an integer count again.
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                scorer(count_graph, unit=0.2)

    def test_invalid_unit(self, count_graph):
        for scorer, _, kw in _COUNT_SCORERS:
            for unit in (0, -1.0, math.nan, math.inf, True, "1"):
                with pytest.raises(ValueError, match="unit must be"):
                    scorer(count_graph, unit=unit, **kw)

    def test_alias_forwards_unit(self, les_miserables_weighted):
        G = les_miserables_weighted
        assert nx.utils.graphs_equal(
            nb.mlf(G, unit=0.5), nb.marginal_likelihood_filter(G, unit=0.5)
        )
