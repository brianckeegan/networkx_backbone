"""Known-answer tests for the statistical scorers (networkx_backbone.statistical).

Each check is tied to an equation in the cited paper, so a regression here
means the implementation drifted from the method it claims to implement.
Shared behaviour (attributes, p-value ranges, validation, multigraphs, empty
graphs, input order) is covered by ``test_api_contract.py``.
"""

import math
import warnings

import networkx as nx
import pytest

import networkx_backbone as nb

pytest.importorskip("scipy")
from scipy import stats


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


@pytest.mark.parametrize(
    ("scorer", "attr"),
    [
        (nb.disparity_filter, "disparity_pvalue"),
        (nb.mlf, "ml_pvalue"),
        (nb.lans, "lans_pvalue"),
        (nb.ecm_filter, "ecm_pvalue"),
        (nb.noise_corrected_filter, "nc_pvalue"),
    ],
)
def test_signed_adds_sign_and_two_sided_pvalue(scorer, attr, les_miserables_weighted):
    H = scorer(les_miserables_weighted, signed=True)
    assert all("sign" in d and d["sign"] in (-1, 1) for _, _, d in H.edges(data=True))
    assert all(0.0 <= d[attr] <= 1.0 for _, _, d in H.edges(data=True))


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
    @pytest.mark.parametrize("kind", ["undirected", "directed"])
    def test_matches_reference_implementation(self, kind):
        G = nx.DiGraph() if kind == "directed" else nx.Graph()
        edges = NC_DIRECTED_EDGES if kind == "directed" else NC_UNDIRECTED_EDGES
        G.add_weighted_edges_from(edges)
        H = nb.noise_corrected_filter(G)
        for (u, v), (score, sdev) in NC_REFERENCE[kind].items():
            assert H[u][v]["nc_score"] == pytest.approx(score, abs=1e-11)
            assert H[u][v]["nc_sdev"] == pytest.approx(sdev, rel=1e-9)

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
            assert data["nc_pvalue"] == pytest.approx(stats.norm.sf(z))

        legacy = nb.noise_corrected_filter(G, method="zscore")
        assert legacy[0][1]["nc_score"] == pytest.approx(-0.518, abs=1e-3)

    def test_pvalue_matches_delta_rule(self, les_miserables_weighted):
        # score > delta * sdev  <=>  nc_pvalue < 1 - Phi(delta)
        H = nb.noise_corrected_filter(les_miserables_weighted)
        cut = stats.norm.sf(1.64)
        for *_, d in H.edges(data=True):
            assert (d["nc_score"] > 1.64 * d["nc_sdev"]) == (d["nc_pvalue"] < cut)

    def test_signed_is_two_sided(self):
        G = nx.Graph()
        G.add_weighted_edges_from(NC_UNDIRECTED_EDGES)
        H = nb.noise_corrected_filter(G, signed=True)
        for (u, v), (score, sdev) in NC_REFERENCE["undirected"].items():
            z = score / sdev
            expected = min(1.0, 2 * stats.norm.sf(abs(z)))
            assert H[u][v]["nc_pvalue"] == pytest.approx(expected, abs=1e-9)
            assert H[u][v]["sign"] == (1 if z >= 0 else -1)
        assert H[2][3]["sign"] == 1 and H[2][3]["nc_pvalue"] < 1e-10
        assert H[0][2]["sign"] == -1

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
    @pytest.mark.parametrize("directed", [False, True])
    def test_fit_reproduces_degrees_and_strengths(
        self, les_miserables_weighted, directed
    ):
        import numpy as np

        from networkx_backbone.statistical import _fit_ecm

        G = nx.DiGraph(les_miserables_weighted) if directed else les_miserables_weighted
        nodes = list(G)
        k_out = np.array(
            [G.out_degree(v) if directed else G.degree(v) for v in nodes], float
        )
        s_out = np.array(
            [
                G.out_degree(v, weight="weight")
                if directed
                else G.degree(v, weight="weight")
                for v in nodes
            ],
            float,
        )
        k_in = np.array([G.in_degree(v) for v in nodes], float) if directed else k_out
        s_in = (
            np.array([G.in_degree(v, weight="weight") for v in nodes], float)
            if directed
            else s_out
        )
        fit = _fit_ecm(k_out, s_out, k_in, s_in, directed, 1000, 1e-8)
        assert fit[-1] < 1e-6
        p, w = _ecm_expectations(fit, directed)
        np.testing.assert_allclose(p.sum(1), k_out, rtol=1e-6)
        np.testing.assert_allclose(w.sum(1), s_out, rtol=1e-6)
        np.testing.assert_allclose(p.sum(0), k_in, rtol=1e-6)
        np.testing.assert_allclose(w.sum(0), s_in, rtol=1e-6)

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
        assert nb.ecm_filter(count_graph, signed=True)["a"]["a"]["ecm_pvalue"] == 1.0

    def test_signed_tails_from_fitted_multipliers(self, count_graph):
        # Upper tail p_ij (y_i y_j)^(w-1), lower tail 1 - p_ij (y_i y_j)^w,
        # two-sided p-value 2 * min(tails) (#25).
        import numpy as np

        from networkx_backbone.statistical import _fit_ecm

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
            assert signed[u][v]["sign"] == (1 if p_hi <= p_lo else -1)


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
# Multiple linkage analysis -- Van Nuffel et al. (2010), dominance fitting
# ---------------------------------------------------------------------------


def _dominance_deviation(weights, k):
    """Squared deviation of a sorted profile from k equal dominant links."""
    total = sum(weights)
    return sum(
        (w / total - (1 / k if r < k else 0.0)) ** 2 for r, w in enumerate(weights)
    )


class TestMultipleLinkageAnalysis:
    @pytest.mark.parametrize(
        ("weights", "deviations", "dominant"),
        [
            # One dominant link: profile (10, 1, 1, 1) / 13.
            # D_1 = (3/13)^2 + 3 (1/13)^2 = 12/169.
            ([10, 1, 1, 1], [12 / 169, 0.263314, 0.327416, 0.359467], 1),
            # Two: profile (10, 10, 1, 1) / 22.
            # D_2 = 2 (10/22 - 1/2)^2 + 2 (1/22)^2 = 4/484.
            ([10, 10, 1, 1], [246 / 484, 4 / 484, 0.114325, 0.167355], 2),
            # Three: profile (10, 10, 10, 1, 1) / 32.
            # D_3 = 3 (10/32 - 1/3)^2 + 2 (1/32)^2 = 3/2304 + 2/1024.
            (
                [10, 10, 10, 1, 1],
                [0.669922, 0.169922, 3 / 2304 + 2 / 1024, 0.060547, 0.094922],
                3,
            ),
            # Evenly spread: keep all of them.
            ([4, 4, 4, 4], [0.75, 0.25, 0.083333, 0.0], 4),
        ],
    )
    def test_hand_computed_dominant_links(self, weights, deviations, dominant):
        devs = [_dominance_deviation(weights, k) for k in range(1, len(weights) + 1)]
        assert devs == pytest.approx(deviations, abs=1e-6)

        G = nx.DiGraph()
        G.add_weighted_edges_from(("hub", i, w) for i, w in enumerate(weights))
        H = nb.multiple_linkage_analysis(G)
        assert H.nodes["hub"]["mla_rank"] == dominant
        kept = sorted(v for _, v, keep in H.edges(data="mla_keep") if keep)
        assert kept == list(range(dominant))
        # Targets have no out-links in a directed graph.
        assert all(H.nodes[i]["mla_rank"] == 0 for i in range(len(weights)))

    def test_tied_links_kept_together(self):
        # (6, 3, 3) / 12 = (1/2, 1/4, 1/4): D_1 = 3/8, D_2 = 1/8 (splits the
        # tie, not considered), D_3 = 1/24 -> keep all three.
        G = nx.DiGraph()
        G.add_weighted_edges_from([("u", "a", 6), ("u", "b", 3), ("u", "c", 3)])
        devs = [_dominance_deviation([6, 3, 3], k) for k in (1, 2, 3)]
        assert devs == pytest.approx([3 / 8, 1 / 8, 1 / 24])
        H = nb.multiple_linkage_analysis(G)
        assert H.nodes["u"]["mla_rank"] == 3

    def test_equal_fits_go_to_smaller_rank(self):
        # (3, 3, 1, 1) / 8: D_2 = D_3 = D_4 = 1/16; k = 3 would also split a
        # tie.  The most parsimonious fit, k = 2, wins.
        devs = [_dominance_deviation([3, 3, 1, 1], k) for k in (2, 3, 4)]
        assert devs == pytest.approx([1 / 16] * 3)
        G = nx.DiGraph()
        G.add_weighted_edges_from([("u", i, w) for i, w in enumerate([3, 3, 1, 1])])
        H = nb.multiple_linkage_analysis(G)
        assert H.nodes["u"]["mla_rank"] == 2
        assert sorted(v for _, v, keep in H.edges(data="mla_keep") if keep) == [0, 1]

    def test_undirected_or_rule(self):
        # Hub h: (10, 1, 1, 1) -> keeps only h-a.  For b, h-b is the stronger
        # of its two links, so the OR rule keeps it; for c, h-c is weak.
        G = nx.Graph()
        G.add_weighted_edges_from(
            [("h", "a", 10), ("h", "b", 1), ("h", "c", 1), ("h", "d", 1)]
        )
        G.add_weighted_edges_from([("b", "x", 0.1), ("c", "y", 5), ("d", "z", 5)])
        H = nb.multiple_linkage_analysis(G)
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
        H = nb.multiple_linkage_analysis(les_miserables_weighted)
        for node in H:
            best = max(H[node], key=lambda v: H[node][v]["weight"])
            assert H[node][best]["mla_keep"]
            assert H.nodes[node]["mla_rank"] >= 1

    def test_invalid_method_raises(self, weighted_triangle):
        with pytest.raises(ValueError, match="method"):
            nb.multiple_linkage_analysis(weighted_triangle, method="bogus")
