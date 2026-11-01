"""Tests for networkx_backbone.bipartite: projections, null models, wrappers.

Shared behavior of every projection scorer (attributes, p-value ranges,
agent-set validation, directed/multigraph rejection, generator input, empty
graphs, mixed labels, no input mutation) is covered by ``test_api_contract.py``.
"""

import itertools
import warnings

import networkx as nx
import numpy as np
import pytest
from scipy import stats

import networkx_backbone as nb
from networkx_backbone import (
    backbone,
    backbone_from_projection,
    backbone_from_unweighted,
    backbone_from_weighted,
    bipartite_projection,
    fastball,
    fdsm,
    hyper_projection,
    probs_projection,
    sdsm,
    simple_projection,
    ycn_projection,
)
from networkx_backbone import bipartite as bp

AGENTS = [1, 2, 3]


def _davis_matrix(B, women):
    """Dense ``(agents, artifacts, R, co-occurrence)`` of a bipartite graph."""
    agents, artifacts, R = bp._incidence(B, women)
    R = R.toarray()
    observed = R @ R.T
    np.fill_diagonal(observed, 0)
    return agents, artifacts, R, observed


def _brute_pb_pmf(p):
    """Poisson-binomial pmf by enumerating all outcomes."""
    pmf = np.zeros(len(p) + 1)
    for bits in itertools.product([0, 1], repeat=len(p)):
        pmf[sum(bits)] += np.prod([q if b else 1 - q for q, b in zip(p, bits)])
    return pmf


def _weights(G):
    return {frozenset((u, v)): w for u, v, w in G.edges(data="weight")}


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------


def test_davis_southern_women_fixture(davis_southern_women_graph, davis_women_nodes):
    assert nx.is_bipartite(davis_southern_women_graph)
    assert davis_southern_women_graph.number_of_nodes() == 32
    assert davis_southern_women_graph.number_of_edges() == 89
    assert len(davis_women_nodes) == 18


@pytest.mark.parametrize(
    ("func", "method", "weight"),
    [
        (simple_projection, "simple", 1.0),
        (hyper_projection, "hyper", 0.5),
        (probs_projection, "probs", 0.25),
        (ycn_projection, "ycn", 1.0 / 12.0),
    ],
)
def test_projection_weights(bipartite_graph, func, method, weight):
    projection = func(bipartite_graph, agent_nodes=AGENTS)
    assert projection.number_of_nodes() == 3
    assert projection.number_of_edges() == 3
    for _, _, data in projection.edges(data=True):
        assert data["weight"] == pytest.approx(weight)
    dispatched = bipartite_projection(bipartite_graph, AGENTS, method=method)
    assert _weights(dispatched) == _weights(projection)


def test_probs_projection_directed():
    B = nx.Graph()
    B.add_edges_from([("a", "x"), ("a", "y"), ("b", "x"), ("c", "y")])
    projection = probs_projection(B, agent_nodes=["a", "b", "c"], directed=True)
    assert isinstance(projection, nx.DiGraph)
    assert projection["a"]["b"]["weight"] == pytest.approx(0.25)
    assert projection["b"]["a"]["weight"] == pytest.approx(0.5)


def test_bipartite_projection_dispatch_unknown_method_raises(bipartite_graph):
    with pytest.raises(ValueError):
        bipartite_projection(bipartite_graph, agent_nodes=AGENTS, method="unknown")


def test_scorers_annotate_projection_weights(
    davis_southern_women_graph, davis_women_nodes
):
    B, women = davis_southern_women_graph, davis_women_nodes
    for scored, method in [
        (sdsm(B, women, projection="hyper"), "hyper"),
        (fdsm(B, women, trials=20, seed=42, projection="ycn"), "ycn"),
        (backbone_from_projection(B, women, "fixedrow", 0.5, "probs"), "probs"),
    ]:
        expected = _weights(bipartite_projection(B, women, method=method))
        got = _weights(scored)
        assert got, method
        assert got == {e: pytest.approx(expected.get(e, 0.0)) for e in got}, method


def test_invalid_projection_raises(bipartite_graph):
    with pytest.raises(ValueError):
        sdsm(bipartite_graph, agent_nodes=AGENTS, projection="invalid")


def test_validation_errors(bipartite_graph):
    with pytest.raises(nx.NetworkXError, match="not bipartite"):
        sdsm(nx.complete_graph(3), agent_nodes=[0, 1])
    with pytest.raises(nx.NetworkXError, match="not in B"):
        sdsm(bipartite_graph, agent_nodes=[99])


# ---------------------------------------------------------------------------
# BiCM
# ---------------------------------------------------------------------------


class TestBiCM:
    def test_margins_match_observed_degrees(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        B = davis_southern_women_graph
        P, agents, artifacts = nb.bicm(B, davis_women_nodes, return_labels=True)
        row = np.array([B.degree(a) for a in agents])
        col = np.array([B.degree(f) for f in artifacts])
        np.testing.assert_allclose(P.sum(axis=1), row, atol=1e-6)
        np.testing.assert_allclose(P.sum(axis=0), col, atol=1e-6)
        assert np.all((P > 0) & (P < 1))

    def test_is_maximum_entropy_form(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        # p_ik = x_i y_k / (1 + x_i y_k)  <=>  logit(P) has rank one in the
        # additive sense: logit(p_ik) = a_i + b_k.
        P = nb.bicm(davis_southern_women_graph, davis_women_nodes)
        L = np.log(P / (1 - P))
        resid = L - L.mean(1, keepdims=True) - L.mean(0, keepdims=True) + L.mean()
        assert np.abs(resid).max() < 1e-8

    def test_degenerate_rows_and_columns_fixed(self):
        # agent "full" links every artifact, "iso" links none; artifact "z"
        # is linked only by "full" (column degree 1).
        B = nx.Graph()
        B.add_edges_from([("full", f) for f in "xyz"])
        B.add_edges_from([("a", "x"), ("b", "y"), ("a", "y")])
        B.add_node("iso")
        P, agents, artifacts = nb.bicm(B, ["full", "iso", "a", "b"], return_labels=True)
        ai = {v: i for i, v in enumerate(agents)}
        fi = {v: i for i, v in enumerate(artifacts)}
        np.testing.assert_array_equal(P[ai["full"]], 1.0)
        np.testing.assert_array_equal(P[ai["iso"]], 0.0)
        assert P[ai["a"], fi["z"]] == 0.0 and P[ai["b"], fi["z"]] == 0.0
        R = np.array([[int(B.has_edge(a, f)) for f in artifacts] for a in agents])
        np.testing.assert_allclose(P.sum(axis=1), R.sum(axis=1), atol=1e-6)
        np.testing.assert_allclose(P.sum(axis=0), R.sum(axis=0), atol=1e-6)

    def test_non_convergence_warns(self, davis_southern_women_graph, davis_women_nodes):
        with pytest.warns(RuntimeWarning, match="did not converge"):
            nb.bicm(davis_southern_women_graph, davis_women_nodes, max_iter=1)

    def test_converged_fit_does_not_warn(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            nb.bicm(davis_southern_women_graph, davis_women_nodes)


def test_fastball_preserves_marginals():
    m = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=int,
    )
    randomized = fastball(m, n_swaps=50, seed=42)
    assert randomized.shape == m.shape
    assert np.array_equal(randomized.sum(axis=1), m.sum(axis=1))
    assert np.array_equal(randomized.sum(axis=0), m.sum(axis=0))


# ---------------------------------------------------------------------------
# Poisson-binomial helpers and SDSM p-values
# ---------------------------------------------------------------------------


class TestPoissonBinomial:
    def test_exact_matches_brute_force(self):
        for seed in range(4):
            p = np.random.default_rng(seed).random(9)
            pmf = _brute_pb_pmf(p)
            np.testing.assert_allclose(bp._poisson_binomial_pmf(p), pmf, atol=1e-13)
            for k in range(-1, 11):
                expected_sf = pmf[max(k, 0) :].sum()
                expected_cdf = pmf[: max(k + 1, 0)].sum()
                assert bp._poisson_binomial_sf(p, k) == pytest.approx(
                    expected_sf, abs=1e-12
                ), (seed, k)
                assert bp._poisson_binomial_cdf(p, k) == pytest.approx(
                    expected_cdf, abs=1e-12
                ), (seed, k)

    def test_rna_formula(self):
        p = np.random.default_rng(1).random(30) * 0.6
        mu, sigma = p.sum(), np.sqrt((p * (1 - p)).sum())
        gamma = (p * (1 - p) * (1 - 2 * p)).sum() / sigma**3

        def G(k):
            x = (k + 0.5 - mu) / sigma
            val = stats.norm.cdf(x) + gamma * (1 - x**2) * stats.norm.pdf(x) / 6
            return min(max(val, 0.0), 1.0)

        for k in range(20):
            upper, lower = bp._rna_tails(mu, (p**2).sum(), (p**3).sum(), np.array([k]))
            assert upper[0] == pytest.approx(1 - G(k - 1) if k > 0 else 1.0, abs=1e-12)
            assert lower[0] == pytest.approx(G(k), abs=1e-12)

    def test_rna_close_to_exact(self, davis_southern_women_graph, davis_women_nodes):
        _, _, R, obs = _davis_matrix(davis_southern_women_graph, davis_women_nodes)
        P = bp._bicm_fit(R)
        k = obs[np.triu_indices(len(R), 1)]
        exact, _ = bp._pairwise_pb_tails(P, k, method="exact")
        rna, _ = bp._pairwise_pb_tails(P, k, method="rna")
        assert np.abs(exact - rna).max() < 0.03

    def test_sdsm_exact_pvalues_match_brute_force(self):
        B = nx.Graph()
        B.add_edges_from(
            [
                ("a", 1),
                ("a", 2),
                ("a", 3),
                ("b", 1),
                ("b", 2),
                ("c", 3),
                ("c", 4),
                ("d", 1),
                ("d", 4),
                ("d", 5),
                ("e", 2),
                ("e", 5),
            ]
        )
        agents = ["a", "b", "c", "d", "e"]
        P, ag, _ = nb.bicm(B, agents, return_labels=True)
        R = np.array([[int(B.has_edge(u, f)) for f in [1, 2, 3, 4, 5]] for u in ag])
        obs = R @ R.T
        for method in ("exact", "auto"):
            for signed in (False, True):
                H = nb.sdsm(B, agents, signed=signed, pvalue_method=method)
                for i, j in itertools.combinations(range(len(ag)), 2):
                    pmf = _brute_pb_pmf(P[i] * P[j])
                    hi, lo = pmf[obs[i, j] :].sum(), pmf[: obs[i, j] + 1].sum()
                    expected = min(1.0, 2 * min(hi, lo)) if signed else hi
                    got = H[ag[i]][ag[j]]["sdsm_pvalue"]
                    assert got == pytest.approx(expected, abs=1e-10)

    def test_invalid_pvalue_method(self, bipartite_graph):
        with pytest.raises(ValueError):
            nb.sdsm(bipartite_graph, [1, 2, 3], pvalue_method="normal")


# ---------------------------------------------------------------------------
# SDSM-EC
# ---------------------------------------------------------------------------


class TestSDSMEC:
    def _cells(self, B, women):
        agents, artifacts, R, _ = _davis_matrix(B, women)
        required = [(agents[0], artifacts[k]) for k in np.flatnonzero(R[0])[:2]]
        prohibited = [
            (agents[i], artifacts[k])
            for i in range(1, 4)
            for k in range(5)
            if R[i, k] == 0
        ]
        return agents, artifacts, R, required, prohibited

    def test_refit_preserves_margins(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        agents, artifacts, R, required, prohibited = self._cells(
            davis_southern_women_graph, davis_women_nodes
        )
        a_idx = {v: i for i, v in enumerate(agents)}
        f_idx = {v: i for i, v in enumerate(artifacts)}
        req = bp._constraint_mask(required, "required", a_idx, f_idx, R.shape)
        pro = bp._constraint_mask(prohibited, "prohibited", a_idx, f_idx, R.shape)
        P = bp._bicm_fit(R, required=req, prohibited=pro)
        np.testing.assert_allclose(P.sum(axis=1), R.sum(axis=1), atol=1e-6)
        np.testing.assert_allclose(P.sum(axis=0), R.sum(axis=0), atol=1e-6)
        assert np.all(P[req] == 1.0) and np.all(P[pro] == 0.0)

    def test_constraints_accept_numpy_array(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        _, _, _, _, prohibited = self._cells(
            davis_southern_women_graph, davis_women_nodes
        )
        as_list = nb.sdsm(
            davis_southern_women_graph, davis_women_nodes, prohibited=prohibited
        )
        as_array = nb.sdsm(
            davis_southern_women_graph,
            davis_women_nodes,
            prohibited=np.array(prohibited),
        )
        for u, v, p in as_list.edges(data="sdsm_pvalue"):
            assert as_array[u][v]["sdsm_pvalue"] == pytest.approx(p)

    def test_reversed_pair_raises(self, davis_southern_women_graph, davis_women_nodes):
        _, _, _, _, prohibited = self._cells(
            davis_southern_women_graph, davis_women_nodes
        )
        a, f = prohibited[0]
        with pytest.raises(nx.NetworkXError, match="ordered"):
            nb.sdsm(davis_southern_women_graph, davis_women_nodes, prohibited=[(f, a)])

    def test_malformed_pairs_raise(self, davis_southern_women_graph, davis_women_nodes):
        for bad in [["abc"], [("x",)], [("a", "b", "c")], [5]]:
            with pytest.raises(nx.NetworkXError, match="pairs"):
                nb.sdsm(davis_southern_women_graph, davis_women_nodes, required=bad)

    def test_unknown_constraint_nodes_raise(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        B = davis_southern_women_graph
        with pytest.raises(nx.NetworkXError):
            sdsm(B, agent_nodes=davis_women_nodes, prohibited=[("missing", "absent")])

    def test_agent_agent_pair_raises(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        pair = (davis_women_nodes[0], davis_women_nodes[1])
        with pytest.raises(nx.NetworkXError):
            nb.sdsm(davis_southern_women_graph, davis_women_nodes, prohibited=[pair])

    def test_contradictory_constraints_raise(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        _, _, _, required, prohibited = self._cells(
            davis_southern_women_graph, davis_women_nodes
        )
        with pytest.raises(nx.NetworkXError):
            nb.sdsm(
                davis_southern_women_graph, davis_women_nodes, prohibited=required[:1]
            )
        with pytest.raises(nx.NetworkXError):
            nb.sdsm(
                davis_southern_women_graph, davis_women_nodes, required=prohibited[:1]
            )

    def test_empty_constraints_equal_unconstrained(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        base = nb.sdsm(davis_southern_women_graph, davis_women_nodes)
        empty = nb.sdsm(
            davis_southern_women_graph,
            davis_women_nodes,
            prohibited=[],
            required=np.empty((0, 2)),
        )
        for u, v, p in base.edges(data="sdsm_pvalue"):
            assert empty[u][v]["sdsm_pvalue"] == pytest.approx(p)


# ---------------------------------------------------------------------------
# FDSM
# ---------------------------------------------------------------------------


class TestFDSMNull:
    def test_samples_preserve_degree_sequence(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        _, _, R, _ = _davis_matrix(davis_southern_women_graph, davis_women_nodes)
        samples = list(bp._fdsm_samples(R, 200, np.random.default_rng(0)))
        assert len(samples) == 200
        for M in samples:
            np.testing.assert_array_equal(M.sum(axis=1), R.sum(axis=1))
            np.testing.assert_array_equal(M.sum(axis=0), R.sum(axis=0))
            assert set(np.unique(M)) <= {0, 1}
        # The chain actually moves.
        assert any(not np.array_equal(M, R) for M in samples)

    @pytest.mark.parametrize("signed", [False, True])
    def test_pvalues_strictly_positive(
        self, davis_southern_women_graph, davis_women_nodes, signed
    ):
        trials = 50
        H = nb.fdsm(
            davis_southern_women_graph,
            davis_women_nodes,
            trials=trials,
            seed=0,
            signed=signed,
        )
        pvals = [p for *_, p in H.edges(data="fdsm_pvalue")]
        assert min(pvals) >= 1 / (trials + 1) - 1e-12
        assert max(pvals) <= 1.0

    def test_pvalue_matches_exact_null_on_tiny_graph(self):
        # 2x2 identity-like matrix: the FDSM space is {I, anti-I}, both with
        # co-occurrence 0, so the upper-tail p-value is exactly 1.
        B = nx.Graph([("a", "x"), ("b", "y"), ("c", "x"), ("c", "y")])
        H = nb.fdsm(B, ["a", "b", "c"], trials=20, seed=1)
        assert H["a"]["b"]["fdsm_pvalue"] == pytest.approx(1.0)

    def test_seed_types_reproducible(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        def pvalues(seed):
            H = nb.fdsm(
                davis_southern_women_graph, davis_women_nodes, trials=30, seed=seed
            )
            return dict(nx.get_edge_attributes(H, "fdsm_pvalue"))

        for make_seed in (np.random.default_rng, np.random.RandomState, int):
            assert pvalues(make_seed(3)) == pvalues(make_seed(3)), make_seed

    def test_invalid_trials_and_burn_in_raise(self, bipartite_graph):
        with pytest.raises(ValueError):
            fdsm(bipartite_graph, agent_nodes=AGENTS, trials=0)
        with pytest.raises(ValueError):
            nb.fdsm(bipartite_graph, AGENTS, trials=5, burn_in=-1)


# ---------------------------------------------------------------------------
# Fixed fill / row / column models
# ---------------------------------------------------------------------------


def _random_bipartite(n_agents, n_artifacts, density, seed):
    rng = np.random.default_rng(seed)
    R = (rng.random((n_agents, n_artifacts)) < density).astype(int)
    B = nx.Graph()
    agents = [f"a{i}" for i in range(n_agents)]
    artifacts = [f"f{k}" for k in range(n_artifacts)]
    B.add_nodes_from(agents)
    B.add_nodes_from(artifacts)
    B.add_edges_from((agents[i], artifacts[k]) for i, k in zip(*np.nonzero(R)))
    return B, agents, artifacts, R


class TestFixedModels:
    @pytest.mark.parametrize(
        "dims", [(3, 3, 4), (4, 3, 7), (3, 4, 5), (2, 4, 6), (4, 4, 3)]
    )
    def test_fixedfill_pmf_matches_enumeration(self, dims):
        na, nf, T = dims
        counts = np.zeros(nf + 1)
        for cells in itertools.combinations(range(na * nf), T):
            M = np.zeros(na * nf, dtype=int)
            M[list(cells)] = 1
            M = M.reshape(na, nf)
            counts[M[0] @ M[1]] += 1
        np.testing.assert_allclose(
            bp._fixedfill_pmf(na, nf, T), counts / counts.sum(), atol=1e-12
        )

    def test_fixedfill_pvalues(self):
        B, agents, _, R = _random_bipartite(3, 4, 0.5, seed=2)
        pmf = bp._fixedfill_pmf(3, 4, int(R.sum()))
        H = nb.fixedfill(B, agents, alpha=np.inf, signed=False)
        obs = R @ R.T
        for i, j in itertools.combinations(range(3), 2):
            assert H[agents[i]][agents[j]]["fixedfill_pvalue"] == pytest.approx(
                pmf[obs[i, j] :].sum()
            )

    def test_fixedcol_matches_enumeration(self):
        # Columns: x={a,b}, y={a,b}, z={a,c,d}, w={b,c}
        B = nx.Graph()
        cols = {"x": "ab", "y": "ab", "z": "acd", "w": "bc"}
        for f, members in cols.items():
            B.add_edges_from((a, f) for a in members)
        agents = ["a", "b", "c", "d"]
        counts = np.zeros(len(cols) + 1)
        combos = [list(itertools.combinations(range(4), len(m))) for m in cols.values()]
        for choice in itertools.product(*combos):
            counts[sum(1 for ch in choice if 0 in ch and 1 in ch)] += 1
        pmf = counts / counts.sum()
        H = nb.fixedcol(B, agents, alpha=np.inf, signed=False)
        assert H["a"]["b"]["fixedcol_pvalue"] == pytest.approx(pmf[2:].sum())
        Hs = nb.fixedcol(B, agents, alpha=np.inf, signed=True)
        assert Hs["a"]["b"]["fixedcol_pvalue"] == pytest.approx(
            min(1.0, 2 * min(pmf[2:].sum(), pmf[:3].sum()))
        )

    def test_fixedrow_matches_hypergeometric(self):
        B, agents, _, R = _random_bipartite(6, 8, 0.4, seed=5)
        H = nb.fixedrow(B, agents, alpha=np.inf)
        obs = R @ R.T
        d = R.sum(axis=1)
        for i, j in itertools.combinations(range(6), 2):
            expected = stats.hypergeom.sf(obs[i, j] - 1, 8, d[i], d[j])
            assert H[agents[i]][agents[j]]["fixedrow_pvalue"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Agent-set inference
# ---------------------------------------------------------------------------


class TestAgentInference:
    def test_disconnected_inference_ambiguous(self):
        B = nx.Graph([("a", "x"), ("b", "x"), ("c", "y"), ("d", "y")])
        with pytest.raises(nx.AmbiguousSolution):
            nb.backbone_from_projection(B, method="fixedrow")

    def test_inference_uses_bipartite_attribute(self):
        B = nx.Graph([("a", "x"), ("b", "x"), ("c", "y"), ("d", "y")])
        nx.set_node_attributes(B, {n: 0 if n in "abcd" else 1 for n in B}, "bipartite")
        H = nb.backbone_from_projection(B, method="fixedrow", alpha=1.0)
        assert set(H.nodes()) == {"a", "b", "c", "d"}

    def test_inference_connected_graph(self, bipartite_graph):
        H = nb.backbone_from_projection(bipartite_graph, method="fixedrow", alpha=1.0)
        assert set(H.nodes()) in ({1, 2, 3}, {"a", "b", "c"})


# ---------------------------------------------------------------------------
# Wrapper thresholding
# ---------------------------------------------------------------------------


class TestWrapperThresholding:
    @pytest.mark.parametrize(
        "method", ["sdsm", "fdsm", "fixedfill", "fixedrow", "fixedcol"]
    )
    @pytest.mark.parametrize("signed", [False, True])
    def test_threshold_consistent_across_methods(
        self, davis_southern_women_graph, davis_women_nodes, method, signed
    ):
        kw = {"trials": 100, "seed": 4} if method == "fdsm" else {}
        alpha = 0.2
        H = nb.backbone_from_projection(
            davis_southern_women_graph,
            davis_women_nodes,
            method=method,
            alpha=alpha,
            signed=signed,
            **kw,
        )
        attr = f"{method}_pvalue"
        assert all(d[attr] < alpha for *_, d in H.edges(data=True))
        if signed:
            assert all(d["sign"] in (-1, 1) for *_, d in H.edges(data=True))
        # Same set as scoring every pair and thresholding by hand.
        scored = getattr(nb, method)(
            davis_southern_women_graph,
            davis_women_nodes,
            alpha=np.inf,
            signed=signed,
            **kw,
        )
        expected = {
            frozenset((u, v)) for u, v, p in scored.edges(data=attr) if p < alpha
        }
        assert {frozenset(e) for e in H.edges()} == expected
        assert (
            H.number_of_edges()
            < len(davis_women_nodes) * (len(davis_women_nodes) - 1) // 2
        )

    @pytest.mark.parametrize("method", ["sdsm", "fixedrow"])
    def test_mtc_forwarded(self, davis_southern_women_graph, davis_women_nodes, method):
        alpha = 0.3
        scored = getattr(nb, method)(
            davis_southern_women_graph, davis_women_nodes, alpha=np.inf
        )
        attr = f"{method}_pvalue"
        edges = list(scored.edges(data=attr))
        adjusted = nb.adjust_pvalues([p for *_, p in edges], "bonferroni")
        expected = {
            frozenset((u, v)) for (u, v, _), q in zip(edges, adjusted) if q < alpha
        }
        H = nb.backbone_from_projection(
            davis_southern_women_graph,
            davis_women_nodes,
            method=method,
            alpha=alpha,
            mtc="bonferroni",
        )
        assert {frozenset(e) for e in H.edges()} == expected

    def test_backbone_auto_dispatch_thresholds(self, davis_southern_women_graph):
        H = nb.backbone(davis_southern_women_graph, alpha=0.1)
        assert all(d["sdsm_pvalue"] < 0.1 for *_, d in H.edges(data=True))
        assert H.number_of_edges() < 153


# ---------------------------------------------------------------------------
# High-level wrappers
# ---------------------------------------------------------------------------


def test_backbone_from_projection_sdsm_with_custom_projection(
    davis_southern_women_graph, davis_women_nodes
):
    backbone_graph = backbone_from_projection(
        davis_southern_women_graph,
        davis_women_nodes,
        method="sdsm",
        alpha=0.5,
        projection="ycn",
    )
    scored = sdsm(davis_southern_women_graph, davis_women_nodes, projection="ycn")
    expected = {
        frozenset((u, v)) for u, v, p in scored.edges(data="sdsm_pvalue") if p < 0.5
    }
    assert backbone_graph.number_of_nodes() == len(davis_women_nodes)
    assert {frozenset(e) for e in backbone_graph.edges()} == expected
    assert expected
    for _, _, data in backbone_graph.edges(data=True):
        assert data["sdsm_pvalue"] < 0.5
        assert "weight" in data


def test_backbone_from_weighted_dispatch(weighted_triangle):
    backbone_graph = backbone_from_weighted(
        weighted_triangle, method="disparity", alpha=1.0
    )
    assert backbone_graph.number_of_nodes() == weighted_triangle.number_of_nodes()
    with pytest.raises(ValueError):
        backbone_from_weighted(weighted_triangle, method="global")


class TestWeightedMissingAsZero:
    """``backbone_from_weighted(missing_as_zero=...)`` (Backbone 3.0, #26)."""

    @pytest.fixture
    def two_pairs(self):
        # T = 41; strengths a=21, b=20, c=21, d=20.  Absent: a-d, b-c, b-d.
        G = nx.Graph()
        G.add_weighted_edges_from([("a", "b", 20), ("c", "d", 20), ("a", "c", 1)])
        return G

    @staticmethod
    def _zeros(H):
        return {
            tuple(sorted((u, v))): d["sign"]
            for u, v, d in H.edges(data=True)
            if d["weight"] == 0
        }

    def test_default_returns_no_absent_pairs(self, two_pairs):
        for method in ("disparity", "mlf", "lans"):
            H = backbone_from_weighted(two_pairs, method=method, signed=True)
            assert self._zeros(H) == {}
            assert set(map(frozenset, H.edges())) <= set(
                map(frozenset, two_pairs.edges())
            )

    def test_mlf_keeps_significantly_weak_absent_pairs(self, two_pairs):
        # Two-sided p-values 2 (1 - p)^41: a-d and b-c 0.00841, b-d 0.0111.
        H = backbone_from_weighted(
            two_pairs, method="mlf", signed=True, missing_as_zero=True
        )
        assert self._zeros(H) == {("a", "d"): -1, ("b", "c"): -1, ("b", "d"): -1}
        # The absent pairs join the Bonferroni family (6 tests, not 3), and
        # 6 * 0.00841 > 0.05.
        H = backbone_from_weighted(
            two_pairs, method="mlf", signed=True, missing_as_zero=True, mtc="bonferroni"
        )
        assert self._zeros(H) == {}

    def test_disparity_and_lans_keep_every_testable_absent_pair(self, two_pairs):
        # A weight of 0 has lower tail 0 under both nulls (as in R).
        H = backbone_from_weighted(
            two_pairs, method="lans", signed=True, missing_as_zero=True
        )
        assert self._zeros(H) == {("a", "d"): -1, ("b", "c"): -1, ("b", "d"): -1}
        # For the disparity filter b and d both have degree 1 and cannot
        # reject the null; a and c have degree 2.
        H = backbone_from_weighted(
            two_pairs, method="disparity", signed=True, missing_as_zero=True
        )
        assert self._zeros(H) == {("a", "d"): -1, ("b", "c"): -1}

    def test_backbone_dispatch_forwards_flag(self, two_pairs):
        from networkx_backbone import backbone

        H = backbone(two_pairs, method="mlf", signed=True, missing_as_zero=True)
        assert len(self._zeros(H)) == 3

    def test_global_rejects_missing_as_zero(self, two_pairs):
        with pytest.raises(ValueError, match="missing_as_zero"):
            backbone_from_weighted(
                two_pairs, method="global", threshold=1, missing_as_zero=True
            )


def test_backbone_from_weighted_multigraph_auto_collapse():
    G = nx.MultiGraph()
    G.add_edge(0, 1, edge_type="a")
    G.add_edge(0, 1, edge_type="b")
    G.add_edge(0, 1, edge_type="b")
    G.add_edge(1, 2, edge_type="a")
    backbone_graph = backbone_from_weighted(
        G, method="global", threshold=2, edge_type_attr="edge_type"
    )
    assert backbone_graph.number_of_edges() == 1
    assert backbone_graph.has_edge(0, 1)


def test_backbone_from_weighted_multidigraph_auto_collapse():
    G = nx.MultiDiGraph()
    G.add_edge(0, 1)
    G.add_edge(0, 1)
    G.add_edge(1, 0)
    backbone_graph = backbone_from_weighted(G, method="global", threshold=2)
    assert isinstance(backbone_graph, nx.DiGraph)
    assert list(backbone_graph.edges()) == [(0, 1)]


def test_backbone_from_unweighted_dispatch(karate):
    backbone_graph = backbone_from_unweighted(karate, method="lspar", s=0.5)
    assert backbone_graph.number_of_nodes() == karate.number_of_nodes()


def test_backbone_dispatch(
    davis_southern_women_graph, davis_women_nodes, weighted_triangle
):
    proj_backbone = backbone(
        davis_southern_women_graph,
        method="sdsm",
        agent_nodes=davis_women_nodes,
        alpha=0.5,
    )
    weighted_backbone = backbone(weighted_triangle, method="disparity", alpha=1.0)
    assert set(proj_backbone.nodes()).issubset(set(davis_women_nodes))
    assert weighted_backbone.number_of_nodes() == weighted_triangle.number_of_nodes()

    with pytest.raises(ValueError):
        backbone(weighted_triangle, method="not_a_method")


class TestSignedProjections:
    # Signed fixed models and the wrapper are covered by TestWrapperThresholding.

    def test_sdsm_signed_adds_sign(self, davis_southern_women_graph, davis_women_nodes):
        H = sdsm(davis_southern_women_graph, agent_nodes=davis_women_nodes, signed=True)
        assert {d["sign"] for _, _, d in H.edges(data=True)} == {-1, 1}
        assert all(0.0 <= d["sdsm_pvalue"] <= 1.0 for _, _, d in H.edges(data=True))
        H = sdsm(davis_southern_women_graph, agent_nodes=davis_women_nodes)
        assert all("sign" not in d for _, _, d in H.edges(data=True))

    def test_fdsm_signed_deterministic(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        kw = {
            "agent_nodes": davis_women_nodes,
            "trials": 200,
            "seed": 7,
            "signed": True,
        }
        a = fdsm(davis_southern_women_graph, **kw)
        b = fdsm(davis_southern_women_graph, **kw)
        assert all("sign" in d for _, _, d in a.edges(data=True))
        assert {(u, v): d["sign"] for u, v, d in a.edges(data=True)} == {
            (u, v): d["sign"] for u, v, d in b.edges(data=True)
        }

    @pytest.mark.parametrize("model", ["fixedrow", "fdsm"])
    def test_tied_tails_are_signed_negative(self, model):
        # Agent a uses every artifact, so its co-occurrence with any agent is
        # that agent's degree under the null as well: both tails are exactly
        # 1 (fdsm: every sample is >= and <= the observed count).  A tie
        # upper == lower is signed -1, as in R backbone (#59).
        B = nx.Graph()
        B.add_nodes_from("abc", bipartite=0)
        B.add_nodes_from([1, 2, 3, 4], bipartite=1)
        B.add_edges_from([("a", 1), ("a", 2), ("a", 3), ("a", 4)])
        B.add_edges_from([("b", 1), ("b", 2), ("c", 3)])
        kw = {"trials": 50, "seed": 1} if model == "fdsm" else {}
        H = getattr(nb, model)(B, "abc", alpha=np.inf, signed=True, **kw)
        for other in "bc":
            assert H["a"][other][f"{model}_pvalue"] == 1.0
            assert H["a"][other]["sign"] == -1


class TestSDSMEdgeConstraints:
    def test_constraints_change_null(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        B = davis_southern_women_graph
        women = davis_women_nodes
        artifacts = sorted(
            (n for n, d in B.nodes(data=True) if d["bipartite"] == 1), key=str
        )
        present = next((women[0], f) for f in artifacts if B.has_edge(women[0], f))
        absent = next((women[0], f) for f in artifacts if not B.has_edge(women[0], f))

        base = sdsm(B, agent_nodes=women)
        prohibited = sdsm(B, agent_nodes=women, prohibited=[absent])
        required = sdsm(B, agent_nodes=women, required=[present])

        diff_pro = sum(
            1
            for u, v, d in prohibited.edges(data=True)
            if abs(d["sdsm_pvalue"] - base[u][v]["sdsm_pvalue"]) > 1e-9
        )
        diff_req = sum(
            1
            for u, v, d in required.edges(data=True)
            if abs(d["sdsm_pvalue"] - base[u][v]["sdsm_pvalue"]) > 1e-9
        )
        assert diff_pro > 0 and diff_req > 0

    def test_constraints_compose_with_signed(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        artifact = next(n for n in davis_southern_women_graph[davis_women_nodes[0]])
        H = sdsm(
            davis_southern_women_graph,
            agent_nodes=davis_women_nodes,
            signed=True,
            required=[(davis_women_nodes[0], artifact)],
        )
        assert all("sign" in d for _, _, d in H.edges(data=True))


# ---------------------------------------------------------------------------
# Projection formulas (issue #13): hand-computed values and sources
# ---------------------------------------------------------------------------

# Agents A, B, C; artifact x = {A, B, C} (degree 3), y = {A, B} (degree 2),
# z = {A} (degree 1).  Agent degrees: A = 3, B = 2, C = 1; |E| = 6.
TINY_EDGES = [("A", "x"), ("B", "x"), ("C", "x"), ("A", "y"), ("B", "y"), ("A", "z")]
TINY_AGENTS = ["A", "B", "C"]
TINY_EXPECTED = {
    # |Γ(u) ∩ Γ(v)|
    "simple": {"AB": 2.0, "AC": 1.0, "BC": 1.0},
    # Σ 1/d(z): AB = 1/3 + 1/2, AC = BC = 1/3
    "hyper": {"AB": 5 / 6, "AC": 1 / 3, "BC": 1 / 3},
    # Newman: Σ 1/(d(z) - 1): AB = 1/2 + 1, AC = BC = 1/2
    "newman": {"AB": 3 / 2, "AC": 1 / 2, "BC": 1 / 2},
    # mean of (1/d(u)) Σ 1/d(z) and (1/d(v)) Σ 1/d(z):
    # AB = (5/18 + 5/12) / 2, AC = (1/9 + 1/3) / 2, BC = (1/6 + 1/3) / 2
    "probs": {"AB": 25 / 72, "AC": 2 / 9, "BC": 1 / 4},
    # π = d / |E| = (1/2, 1/3, 1/6); π_u T_uv = hyper / 6
    "ycn": {"AB": 5 / 36, "AC": 1 / 18, "BC": 1 / 18},
}
TINY_PROBS_DIRECTED = {
    ("A", "B"): 5 / 18,
    ("B", "A"): 5 / 12,
    ("A", "C"): 1 / 9,
    ("C", "A"): 1 / 3,
    ("B", "C"): 1 / 6,
    ("C", "B"): 1 / 3,
}


def _tiny():
    return nx.Graph(TINY_EDGES)


@pytest.mark.parametrize("method", list(TINY_EXPECTED))
def test_projection_hand_computed(method):
    P = bipartite_projection(_tiny(), TINY_AGENTS, method=method)
    got = {"".join(sorted((u, v))): w for u, v, w in P.edges(data="weight")}
    assert got == pytest.approx(TINY_EXPECTED[method], rel=1e-12)


def test_projection_functions_match_dispatch():
    B = _tiny()
    for func, kw, method in [
        (simple_projection, {}, "simple"),
        (hyper_projection, {}, "hyper"),
        (hyper_projection, {"newman": True}, "newman"),
        (probs_projection, {}, "probs"),
        (ycn_projection, {}, "ycn"),
    ]:
        got = _weights(func(B, TINY_AGENTS, **kw))
        assert got == _weights(bipartite_projection(B, TINY_AGENTS, method=method))


def test_probs_directed_hand_computed():
    P = probs_projection(_tiny(), TINY_AGENTS, directed=True)
    got = {(u, v): w for u, v, w in P.edges(data="weight")}
    assert got == pytest.approx(TINY_PROBS_DIRECTED, rel=1e-12)
    # Each row is a two-step random walk: with the return to u it sums to 1.
    artifact_degrees = {"A": (3, 2, 1), "B": (3, 2), "C": (3,)}
    for u, d in zip("ABC", (3, 2, 1)):
        back = sum(1 / (d * k) for k in artifact_degrees[u])
        out = sum(w for (s, _), w in got.items() if s == u)
        assert out + back == pytest.approx(1.0)


def test_probs_matches_yildirim_coscia_example():
    # Yildirim & Coscia (2014): a1 = {b1, b4, b5}, a2 = {b1, b3}, no other
    # links.  ProbS s(a1, a2) = 1/4 (the share of a2's resource reaching a1)
    # and HeatS s(a1, a2) = 1/6 (the share of a1's resource reaching a2).
    B = nx.Graph([("a1", "b1"), ("a1", "b4"), ("a1", "b5"), ("a2", "b1"), ("a2", "b3")])
    P = probs_projection(B, ["a1", "a2"], directed=True)
    assert P["a2"]["a1"]["weight"] == pytest.approx(1 / 4)
    assert P["a1"]["a2"]["weight"] == pytest.approx(1 / 6)


def test_ycn_directed_is_stationary_flow():
    # π_u T_uv with T the ProbS transition matrix including returns.
    P = ycn_projection(_tiny(), TINY_AGENTS, directed=True)
    pi = {"A": 1 / 2, "B": 1 / 3, "C": 1 / 6}
    for (u, v), t in TINY_PROBS_DIRECTED.items():
        assert P[u][v]["weight"] == pytest.approx(pi[u] * t, rel=1e-12)


def test_ycn_equals_hyper_over_edges_on_connected_graph(
    davis_southern_women_graph, davis_women_nodes
):
    B = davis_southern_women_graph
    ycn = _weights(ycn_projection(B, davis_women_nodes))
    hyper = _weights(hyper_projection(B, davis_women_nodes))
    E = B.number_of_edges()
    assert ycn == pytest.approx({e: w / E for e, w in hyper.items()}, rel=1e-9)


def test_ycn_disconnected_components_scaled_by_agent_share():
    # Two components; power iteration from the uniform vector keeps each
    # component's share of agents: π_u = (n_c / n) d(u) / |E_c|.
    B = nx.Graph([("a", "x"), ("b", "x"), ("c", "y"), ("d", "y"), ("e", "y")])
    P = ycn_projection(B, ["a", "b", "c", "d", "e"])
    assert P["a"]["b"]["weight"] == pytest.approx((2 / 5) * (1 / 2) / 2)
    assert P["c"]["d"]["weight"] == pytest.approx((3 / 5) * (1 / 3) / 3)


def _bare(B):
    """*B* without node or graph attributes.

    NetworkX's projections copy them into the result but do not declare that
    to the dispatcher, which fails the ``NETWORKX_TEST_BACKEND`` harness's
    result comparison.  Only the edge weights are compared here.
    """
    return nx.Graph(B.edges())


def test_simple_equals_networkx_weighted_projection(
    davis_southern_women_graph, davis_women_nodes
):
    B = davis_southern_women_graph
    ours = _weights(simple_projection(B, davis_women_nodes))
    theirs = _weights(
        nx.bipartite.weighted_projected_graph(_bare(B), davis_women_nodes)
    )
    assert ours == theirs


def test_newman_equals_networkx_collaboration_projection(
    davis_southern_women_graph, davis_women_nodes
):
    B = davis_southern_women_graph
    ours = _weights(hyper_projection(B, davis_women_nodes, newman=True))
    theirs = _weights(
        nx.bipartite.collaboration_weighted_projected_graph(_bare(B), davis_women_nodes)
    )
    assert ours == pytest.approx(theirs, rel=1e-12)
    for alias in ("newman", "collaboration"):
        got = _weights(bipartite_projection(B, davis_women_nodes, method=alias))
        assert got == pytest.approx(theirs, rel=1e-12)


def test_hyper_differs_from_newman(davis_southern_women_graph, davis_women_nodes):
    # Coscia & Rossi's 1/d(z) and Newman's 1/(d(z) - 1) differ on every pair.
    B = davis_southern_women_graph
    hyper = _weights(hyper_projection(B, davis_women_nodes))
    newman = _weights(hyper_projection(B, davis_women_nodes, newman=True))
    assert hyper.keys() == newman.keys()
    assert all(hyper[e] < newman[e] for e in hyper)
    pair = frozenset(("Olivia Carleton", "Sylvia Avondale"))
    assert hyper[pair] == pytest.approx(1 / 12)
    assert newman[pair] == pytest.approx(1 / 11)


def test_scorers_accept_newman_projection(
    davis_southern_women_graph, davis_women_nodes
):
    B, women = davis_southern_women_graph, davis_women_nodes
    expected = _weights(hyper_projection(B, women, newman=True))
    got = _weights(sdsm(B, women, projection="newman"))
    assert got == {e: pytest.approx(expected.get(e, 0.0)) for e in got}


def test_directed_symmetric_projection_raises():
    for method in ("simple", "hyper", "newman"):
        with pytest.raises(ValueError, match="directed"):
            bipartite_projection(_tiny(), TINY_AGENTS, method=method, directed=True)


# ---------------------------------------------------------------------------
# Sparse end-to-end implementation (issue #30): dense reference
# ---------------------------------------------------------------------------


def _dense_tails(method, R, **kw):
    """Pre-sparse dense implementation: tails for np.triu_indices pairs."""
    na, nf = R.shape
    iu, ju = np.triu_indices(na, 1)
    k = (R @ R.T)[iu, ju]
    if method == "sdsm":
        P = bp._bicm_fit(R)
        return bp._pairwise_pb_tails(P, k, method=kw["pvalue_method"], need_lower=True)
    if method == "fixedfill":
        return bp._pmf_tails(bp._fixedfill_pmf(na, nf, int(R.sum())), k)
    if method == "fixedcol":
        c = R.sum(axis=0).astype(float)
        pmf = bp._poisson_binomial_pmf(np.clip(c * (c - 1) / (na * (na - 1)), 0, 1))
        return bp._pmf_tails(pmf, k)
    if method == "fixedrow":
        d = R.sum(axis=1)
        return (
            stats.hypergeom.sf(k - 1, nf, d[iu], d[ju]),
            stats.hypergeom.cdf(k, nf, d[iu], d[ju]),
        )
    if method == "fdsm":
        rng = np.random.default_rng(kw["seed"])
        exceed = np.zeros(len(iu), dtype=np.int64)
        below = np.zeros(len(iu), dtype=np.int64)
        for S in bp._fdsm_samples(R, kw["trials"], rng):
            co = (S @ S.T)[iu, ju]
            exceed += co >= k
            below += co <= k
        return (exceed + 1) / (kw["trials"] + 1), (below + 1) / (kw["trials"] + 1)
    raise AssertionError(method)


def _dense_weights(R, method):
    """Pre-sparse dense projection weight matrix (undirected)."""
    if method == "simple":
        W = (R @ R.T).astype(float)
    else:
        d = R.sum(axis=0).astype(float)
        H = (R * np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)) @ R.T
        deg = R.sum(axis=1).astype(float)
        T = np.divide(H, deg[:, None], out=np.zeros_like(H), where=deg[:, None] > 0)
        if method == "hyper":
            W = H
        elif method == "probs":
            W = 0.5 * (T + T.T)
        else:  # ycn: power iteration from the uniform vector
            pi = np.full(len(R), 1 / len(R))
            for _ in range(1000):
                nxt = pi @ T
                done = np.abs(nxt - pi).sum() <= 1e-12
                pi = nxt
                if done:
                    break
            U = (pi / pi.sum())[:, None] * T
            W = 0.5 * (U + U.T)
    np.fill_diagonal(W, 0.0)
    return W


DENSE_CASES = [
    ("sdsm", {"pvalue_method": "exact"}),
    ("sdsm", {"pvalue_method": "rna"}),
    ("fixedfill", {}),
    ("fixedrow", {}),
    ("fixedcol", {}),
    ("fdsm", {"trials": 60, "seed": 7}),
]


@pytest.mark.parametrize(("method", "kw"), DENSE_CASES)
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("projection", ["simple", "hyper", "probs", "ycn"])
def test_sparse_matches_dense_reference_on_davis(
    davis_southern_women_graph, davis_women_nodes, method, kw, signed, projection
):
    B, women = davis_southern_women_graph, davis_women_nodes
    agents, _, R, _ = _davis_matrix(B, women)
    p_hi, p_lo = _dense_tails(method, R, **kw)
    if signed:
        expected, sign = bp._signed_pvalues(p_hi, p_lo)
    else:
        expected, sign = np.clip(p_hi, 0, 1), None
    W = _dense_weights(R, projection)

    H = getattr(nb, method)(
        B, women, alpha=np.inf, signed=signed, projection=projection, **kw
    )
    iu, ju = np.triu_indices(len(agents), 1)
    edges = [H[agents[i]][agents[j]] for i, j in zip(iu, ju)]
    got = np.array([d[f"{method}_pvalue"] for d in edges])
    if kw.get("pvalue_method") == "rna":
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)
    else:  # exact Poisson-binomial, closed forms and Monte Carlo: bit-identical
        np.testing.assert_array_equal(got, expected)
    if signed:
        np.testing.assert_array_equal([d["sign"] for d in edges], sign)
    got_w = [d["weight"] for d in edges]
    np.testing.assert_allclose(got_w, W[iu, ju], rtol=1e-12, atol=0)


@pytest.mark.parametrize("method", ["sdsm", "fixedrow", "fixedcol", "fdsm"])
def test_row_blocks_do_not_change_results(
    davis_southern_women_graph, davis_women_nodes, monkeypatch, method
):
    B, women = davis_southern_women_graph, davis_women_nodes
    kw = {"trials": 30, "seed": 1} if method == "fdsm" else {}
    common = {"method": method, "alpha": 0.5, "signed": True, "projection": "ycn"}

    def run():
        return (
            backbone_from_projection(B, women, mtc="bh", **common, **kw),
            backbone_from_projection(B, women, **common, **kw),
            bipartite_projection(B, women, method="probs", directed=True),
        )

    one_block = run()
    monkeypatch.setattr(bp, "_MAX_BLOCK_CELLS", 40)  # blocks of 2 agent rows
    many_blocks = run()
    for G, H in zip(one_block, many_blocks):
        assert list(H.edges(data=True)) == list(G.edges(data=True))


@pytest.mark.parametrize("seed", range(6))
def test_bicm_classes_bit_identical_to_dense_fit(seed):
    rng = np.random.default_rng(seed)
    R = (rng.random((9, 7)) < 0.4).astype(int)
    R[0] = 1  # full row
    R[1] = 0  # empty row
    R[:, 2] = 0  # empty column
    if seed % 2:
        R[:, 3] = 1  # full column
    Q, row_class, col_class = bp._bicm_classes(R.sum(axis=1), R.sum(axis=0))
    np.testing.assert_array_equal(Q[row_class][:, col_class], bp._bicm_fit(R))


def test_bicm_classes_on_davis(davis_southern_women_graph, davis_women_nodes):
    _, _, R, _ = _davis_matrix(davis_southern_women_graph, davis_women_nodes)
    Q, rc, cc = bp._bicm_classes(R.sum(axis=1), R.sum(axis=0))
    np.testing.assert_array_equal(Q[rc][:, cc], bp._bicm_fit(R))
    assert Q.shape[0] < R.shape[0]  # degree classes compress the fit


@pytest.mark.parametrize(
    "method", ["none", "bonferroni", "holm", "hochberg", "bh", "fdr", "by"]
)
def test_adjust_array_matches_adjust_pvalues(method):
    rng = np.random.default_rng(3)
    p = np.round(rng.random(200) ** 3, 2)  # many ties
    np.testing.assert_array_equal(
        bp._adjust_array(p, method), nb.adjust_pvalues(p.tolist(), method)
    )


def test_invalid_mtc_raises(davis_southern_women_graph, davis_women_nodes):
    with pytest.raises(ValueError, match="method must be one of"):
        backbone_from_projection(
            davis_southern_women_graph, davis_women_nodes, mtc="bogus"
        )


def test_scale_sparse_incidence_stays_fast():
    # 3000 agents x 800 artifacts at 1% density: 4.5 million agent pairs.
    # The dense implementation needed 40-80 s and about 2 GB per model here;
    # the blocked sparse path takes well under a second per model.
    sp = pytest.importorskip("scipy.sparse")
    M = sp.random_array((3000, 800), density=0.01, rng=np.random.default_rng(0))
    M.data[:] = 1
    for method in ("sdsm", "fixedcol"):
        H = backbone_from_projection(M, method=method, alpha=1e-3)
        assert H.number_of_nodes() == 3000
        assert 0 < H.number_of_edges() < 3000 * 2999 // 2
        assert all(d[f"{method}_pvalue"] < 1e-3 for *_, d in H.edges(data=True))


# ---------------------------------------------------------------------------
# Incidence-matrix input (issue #20)
# ---------------------------------------------------------------------------


def _davis_incidence(B, women, kind):
    sp = pytest.importorskip("scipy.sparse")
    events = sorted(set(B) - set(women), key=str)
    dense = nx.bipartite.biadjacency_matrix(
        B, row_order=women, column_order=events
    ).toarray()
    if kind == "pandas":
        pd = pytest.importorskip("pandas")
        return pd.DataFrame(dense, index=women, columns=events), {}
    M = {
        "numpy": lambda: dense,
        "bool": lambda: dense.astype(bool),
        "csr_array": lambda: sp.csr_array(dense),
        "csr_matrix": lambda: sp.csr_matrix(dense),
        "coo_array": lambda: sp.coo_array(dense),
    }[kind]()
    return M, {"agent_labels": women, "artifact_labels": events}


MATRIX_KINDS = ["numpy", "bool", "csr_array", "csr_matrix", "coo_array", "pandas"]


def _same_graph(G, H):
    assert list(G.nodes) == list(H.nodes)
    assert list(G.edges(data=True)) == list(H.edges(data=True))


@pytest.mark.parametrize("kind", MATRIX_KINDS)
@pytest.mark.parametrize(
    "method", ["sdsm", "fdsm", "fixedfill", "fixedrow", "fixedcol"]
)
def test_matrix_input_identical_to_graph(
    davis_southern_women_graph, davis_women_nodes, kind, method
):
    B, women = davis_southern_women_graph, davis_women_nodes
    M, labels = _davis_incidence(B, women, kind)
    kw = {"trials": 30, "seed": 2} if method == "fdsm" else {}
    common = {"method": method, "alpha": 0.3, "projection": "ycn", "signed": True}
    _same_graph(
        backbone_from_projection(M, **labels, **common, **kw),
        backbone_from_projection(B, women, **common, **kw),
    )
    _same_graph(
        getattr(nb, method)(M, **labels, alpha=np.inf, **kw),
        getattr(nb, method)(B, women, alpha=np.inf, **kw),
    )


@pytest.mark.parametrize("kind", MATRIX_KINDS)
def test_matrix_input_projections_and_bicm(
    davis_southern_women_graph, davis_women_nodes, kind
):
    B, women = davis_southern_women_graph, davis_women_nodes
    M, labels = _davis_incidence(B, women, kind)
    for method in ("simple", "hyper", "newman", "probs", "ycn"):
        _same_graph(
            bipartite_projection(M, method=method, **labels),
            bipartite_projection(B, women, method=method),
        )
    _same_graph(
        hyper_projection(M, newman=True, **labels),
        hyper_projection(B, women, newman=True),
    )
    P, agents, artifacts = nb.bicm(M, return_labels=True, **labels)
    P0, agents0, artifacts0 = nb.bicm(B, women, return_labels=True)
    np.testing.assert_array_equal(P, P0)
    assert (agents, artifacts) == (agents0, artifacts0)


def test_matrix_default_labels_and_constraints():
    M = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0]])
    B = nx.Graph()
    B.add_nodes_from(range(4), bipartite=0)
    B.add_nodes_from((f"f{k}" for k in range(4)), bipartite=1)
    B.add_edges_from((int(i), f"f{k}") for i, k in zip(*np.nonzero(M)))
    H = sdsm(M, prohibited=[(0, 3)], required=[(2, 3)], signed=True)
    assert set(H) == {0, 1, 2, 3}
    H0 = sdsm(B, prohibited=[(0, "f3")], required=[(2, "f3")], signed=True)
    _same_graph(H, H0)
    # backbone() routes a matrix to the projection backbone.
    _same_graph(nb.backbone(M, alpha=0.5), backbone_from_projection(M, alpha=0.5))


def test_agent_nodes_inferred_by_scorers(davis_southern_women_graph):
    B = davis_southern_women_graph
    women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    _same_graph(sdsm(B), sdsm(B, women))
    _same_graph(simple_projection(B), simple_projection(B, women))


@pytest.mark.parametrize(
    ("M", "kw", "match"),
    [
        (np.array([[1, 2], [0, 1]]), {}, "binary"),
        (np.array([[1.0, np.nan], [0, 1]]), {}, "binary"),
        (np.array([1, 0, 1]), {}, "two-dimensional"),
        (np.array([["a", "b"]]), {}, "numeric"),
        (np.eye(2), {"agent_labels": ["x"]}, "agent_labels has 1"),
        (np.eye(2), {"artifact_labels": ["x", "x"]}, "duplicates"),
        (np.eye(2), {"agent_labels": [[1], [2]]}, "hashable"),
        (np.eye(2), {"agent_nodes": [0, 1]}, "agent_labels"),
    ],
)
def test_invalid_matrix_input_raises(M, kw, match):
    with pytest.raises(ValueError, match=match):
        backbone_from_projection(M, **kw)


def test_sparse_duplicates_and_explicit_zeros():
    sp = pytest.importorskip("scipy.sparse")
    M = sp.coo_array(([1, 1], ([0, 0], [0, 0])), shape=(2, 2))  # sums to 2
    with pytest.raises(ValueError, match="binary"):
        simple_projection(M)
    explicit_zero = sp.csr_array(
        (np.array([1, 0]), np.array([0, 1]), np.array([0, 2, 2])), shape=(2, 2)
    )
    assert simple_projection(explicit_zero).number_of_edges() == 0


def test_labels_rejected_for_graph_input(bipartite_graph):
    with pytest.raises(ValueError, match="incidence matrix"):
        simple_projection(bipartite_graph, AGENTS, agent_labels=AGENTS)
