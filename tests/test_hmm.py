"""Tests for the two-state edge-activity HMM (#73).

The engine (``networkx_backbone._hmm``): the exact algorithms are checked
against brute-force enumeration of all 2^T state paths; EM by parameter
recovery on simulated data with fixed seeds.  The public
:func:`~networkx_backbone.edge_persistence_hmm` is checked against the engine
on hand-built and perturbed snapshots.
"""

import itertools
import math
import warnings

import networkx as nx
import pytest

np = pytest.importorskip("numpy")

import networkx_backbone as nb  # noqa: E402
from networkx_backbone import _hmm  # noqa: E402

TRUE_BINARY = {"a01": 0.1, "a10": 0.2, "alpha": 0.05, "beta": 0.15}
TRUE_PVALUE = {"a01": 0.1, "a10": 0.2, "a": 0.2}


def _emission(o, known, s, p, observation):
    if not known:
        return 1.0
    if observation == "binary":
        on = p["alpha"] if s == 0 else 1.0 - p["beta"]
        return on if o == 1 else 1.0 - on
    return 1.0 if s == 0 else p["a"] * o ** (p["a"] - 1.0)


def _brute_force(o, m, p, observation="binary"):
    """Posterior, likelihood and MAP path of one edge by enumerating paths."""
    T = len(o)
    pi1 = p["pi1"] if "pi1" in p else p["a01"] / (p["a01"] + p["a10"])
    A = [[1 - p["a01"], p["a01"]], [p["a10"], 1 - p["a10"]]]
    total = 0.0
    post = np.zeros(T)
    best, best_path = -1.0, None
    for path in itertools.product((0, 1), repeat=T):
        prob = pi1 if path[0] else 1 - pi1
        for t in range(T):
            if t:
                prob *= A[path[t - 1]][path[t]]
            prob *= _emission(o[t], m[t], path[t], p, observation)
        total += prob
        post += prob * np.array(path)
        if prob > best:
            best, best_path = prob, path
    return post / total, math.log(total), np.array(best_path)


def _random_params(rng, observation):
    p = {"a01": rng.uniform(0.05, 0.6), "a10": rng.uniform(0.05, 0.6)}
    if observation == "binary":
        p["alpha"] = rng.uniform(0.0, 0.4)
        p["beta"] = rng.uniform(0.0, 0.4)
    else:
        p["a"] = rng.uniform(0.1, 0.9)
    return p


def _simulate(params, E, T, seed, observation="binary"):
    rng = np.random.default_rng(seed)
    pi1 = params["a01"] / (params["a01"] + params["a10"])
    s = np.zeros((E, T), dtype=int)
    s[:, 0] = rng.random(E) < pi1
    for t in range(1, T):
        u = rng.random(E)
        s[:, t] = np.where(s[:, t - 1] == 1, u >= params["a10"], u < params["a01"])
    if observation == "binary":
        on = np.where(s == 1, 1 - params["beta"], params["alpha"])
        return (rng.random((E, T)) < on).astype(float), s
    pv = rng.random((E, T))
    return np.where(s == 1, pv ** (1.0 / params["a"]), pv), s


@pytest.mark.parametrize("observation", ["binary", "pvalue"])
def test_matches_brute_force(observation):
    rng = np.random.default_rng(1)
    for case in range(12):
        T = [1, 2, 3, 5, 8, 10][case % 6]
        E = 4
        p = _random_params(rng, observation)
        if case % 3 == 0:
            p["pi1"] = rng.uniform(0.05, 0.95)
        if observation == "binary":
            o = (rng.random((E, T)) < 0.5).astype(float)
        else:
            o = rng.random((E, T))
        m = rng.random((E, T)) < 0.8
        post, ll = _hmm.forward_backward(o, m, p, observation=observation)
        path = _hmm.viterbi(o, m, p, observation=observation)
        for e in range(E):
            bpost, bll, bpath = _brute_force(o[e], m[e], p, observation)
            np.testing.assert_allclose(post[e], bpost, rtol=1e-10, atol=1e-12)
            assert ll[e] == pytest.approx(bll, rel=1e-10, abs=1e-12)
            np.testing.assert_array_equal(path[e], bpath)


def test_viterbi_ties_go_to_inactive():
    p = {"a01": 0.5, "a10": 0.5, "alpha": 0.1, "beta": 0.1}
    # All steps unknown: every path is equally likely.
    path = _hmm.viterbi(np.zeros((2, 6)), np.zeros((2, 6), dtype=bool), p)
    np.testing.assert_array_equal(path, 0)
    # Symmetric evidence (one 1 and one 0 per step under a symmetric model)
    # with the last step unknown: the last state is a tie and goes to 0.
    o = np.array([[1.0, 0.0, 0.0]])
    m = np.array([[True, True, False]])
    np.testing.assert_array_equal(_hmm.viterbi(o, m, p), [[1, 0, 0]])


def test_noise_free_emissions_reproduce_observations():
    rng = np.random.default_rng(2)
    o = (rng.random((20, 15)) < 0.4).astype(float)
    m = rng.random((20, 15)) < 0.7
    p = {"a01": 0.2, "a10": 0.3, "alpha": 0.0, "beta": 0.0}
    post, ll = _hmm.forward_backward(o, m, p)
    path = _hmm.viterbi(o, m, p)
    np.testing.assert_allclose(post[m], o[m], atol=1e-12)
    np.testing.assert_array_equal(path[m], o[m])
    assert np.all(np.isfinite(ll))


def test_memoryless_chain_gives_per_step_posteriors():
    rng = np.random.default_rng(3)
    o = (rng.random((10, 12)) < 0.5).astype(float)
    m = rng.random((10, 12)) < 0.8
    # Identical transition rows (a01 = 1 - a10): steps are independent with
    # prior a01, so each posterior uses only its own observation.
    for a01, alpha, beta in [(0.5, 0.2, 0.3), (0.3, 0.1, 0.25)]:
        p = {"a01": a01, "a10": 1 - a01, "alpha": alpha, "beta": beta}
        post, _ = _hmm.forward_backward(o, m, p)
        lik1 = np.where(o == 1, 1 - beta, beta)
        lik0 = np.where(o == 1, alpha, 1 - alpha)
        expected = a01 * lik1 / (a01 * lik1 + (1 - a01) * lik0)
        expected[~m] = a01
        np.testing.assert_allclose(post, expected, rtol=1e-12)


def test_unknown_steps_carry_no_evidence():
    rng = np.random.default_rng(4)
    o = (rng.random((6, 9)) < 0.5).astype(float)
    m = rng.random((6, 9)) < 0.6
    m[0] = False
    p = dict(TRUE_BINARY)
    post, ll = _hmm.forward_backward(o, m, p)
    path = _hmm.viterbi(o, m, p)
    # Values at unknown steps are ignored, including NaN.
    o2 = np.where(m, o, np.nan)
    post2, ll2 = _hmm.forward_backward(o2, None, p)
    np.testing.assert_array_equal(post, post2)
    np.testing.assert_array_equal(ll, ll2)
    np.testing.assert_array_equal(path, _hmm.viterbi(o2, None, p))
    # An edge with no known step keeps the stationary prior and likelihood 1.
    pi1 = p["a01"] / (p["a01"] + p["a10"])
    np.testing.assert_allclose(post[0], pi1, rtol=1e-12)
    assert ll[0] == 0.0
    # Impossible observations: -inf likelihood and NaN posteriors.
    q = {"a01": 0.0, "a10": 0.0, "alpha": 0.0, "beta": 0.0}
    post, ll = _hmm.forward_backward([[1.0, 0.0], [1.0, 1.0]], None, q)
    assert ll[0] == -math.inf and np.isnan(post[0]).all()
    np.testing.assert_allclose(post[1], 1.0)


def test_parameter_recovery_binary():
    o, s = _simulate(TRUE_BINARY, E=800, T=40, seed=10)
    m = np.random.default_rng(11).random(o.shape) < 0.9
    res = _hmm.fit_hmm(o, m)
    assert res["converged"]
    for name, value in TRUE_BINARY.items():
        assert res["params"][name] == pytest.approx(value, abs=0.015), name
    # Posterior accuracy with fitted parameters is close to the Bayes rate
    # (the accuracy with the true parameters).
    true_post, _ = _hmm.forward_backward(o, m, TRUE_BINARY)
    bayes = np.mean((true_post > 0.5) == s)
    fitted = np.mean((res["posterior"] > 0.5) == s)
    assert bayes > 0.9
    assert fitted == pytest.approx(bayes, abs=0.003)
    assert res["path"].shape == o.shape
    assert np.mean(res["path"] == s) > 0.9


def test_parameter_recovery_pvalue():
    o, s = _simulate(TRUE_PVALUE, E=800, T=40, seed=12, observation="pvalue")
    res = _hmm.fit_hmm(o, observation="pvalue")
    assert res["converged"]
    for name, value in TRUE_PVALUE.items():
        assert res["params"][name] == pytest.approx(value, abs=0.015), name
    true_post, _ = _hmm.forward_backward(o, None, TRUE_PVALUE, observation="pvalue")
    bayes = np.mean((true_post > 0.5) == s)
    assert np.mean((res["posterior"] > 0.5) == s) == pytest.approx(bayes, abs=0.003)


def test_fixed_parameters_and_fitted_initial():
    o, _ = _simulate(TRUE_BINARY, E=800, T=30, seed=13)
    res = _hmm.fit_hmm(o, fixed={"alpha": 0.05, "a01": 0.1})
    assert res["params"]["alpha"] == 0.05 and res["params"]["a01"] == 0.1
    assert res["params"]["a10"] == pytest.approx(0.2, abs=0.015)
    assert res["params"]["beta"] == pytest.approx(0.15, abs=0.015)
    # A free pi1 (exact M-step): EM never decreases the likelihood.
    res = _hmm.fit_hmm(o, initial="fit")
    hist = np.array(res["loglik_history"])
    assert np.all(np.diff(hist) >= -1e-8 * abs(hist[-1]))
    assert res["params"]["pi1"] == pytest.approx(1 / 3, abs=0.04)
    res = _hmm.fit_hmm(o[:100], fixed={"pi1": 0.25})
    assert res["params"]["pi1"] == 0.25


def test_label_switching_is_resolved():
    o, _ = _simulate(TRUE_BINARY, E=500, T=30, seed=14)
    ref = _hmm.fit_hmm(o)
    # Start EM in the mirrored labeling (alpha > 1 - beta).
    mirrored = {"a01": 0.2, "a10": 0.1, "alpha": 0.85, "beta": 0.95}
    res = _hmm.fit_hmm(o, init=mirrored)
    assert res["label_swapped"]
    assert res["params"]["alpha"] < 1 - res["params"]["beta"]
    for name in TRUE_BINARY:
        assert res["params"][name] == pytest.approx(ref["params"][name], abs=1e-4)
    np.testing.assert_allclose(res["posterior"], ref["posterior"], atol=1e-3)
    # With some parameters fixed, no relabeling: the free emission parameter
    # is clipped to the constraint, with a warning.
    with pytest.warns(RuntimeWarning, match="identifiability"):
        res = _hmm.fit_hmm(np.zeros((5, 6)), fixed={"alpha": 0.5})
    assert res["constrained"] and not res["label_swapped"]
    assert res["params"]["alpha"] + res["params"]["beta"] < 1
    # p-values denser near 1 than uniform (Beta(2, 1)): a is held below 1.
    pv = np.sqrt(np.random.default_rng(0).random((50, 10)))
    with pytest.warns(RuntimeWarning, match="near 0"):
        res = _hmm.fit_hmm(pv, observation="pvalue")
    assert res["constrained"] and res["params"]["a"] < 1


def test_restarts_nonconvergence_and_empty_input():
    o, _ = _simulate(TRUE_BINARY, E=100, T=20, seed=15)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        res = _hmm.fit_hmm(o, max_iter=1)
    assert not res["converged"] and res["n_iter"] == 1
    one = _hmm.fit_hmm(o)
    a = _hmm.fit_hmm(o, n_init=3, seed=7)
    b = _hmm.fit_hmm(o, n_init=3, seed=7)
    assert a["params"] == b["params"]
    assert a["loglik"] >= one["loglik"] - 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        empty = _hmm.fit_hmm(np.zeros((0, 5)))
    assert empty["converged"] and empty["posterior"].shape == (0, 5)


def test_invalid_input_raises():
    p = dict(TRUE_BINARY)
    bad_calls = [
        lambda: _hmm.forward_backward([1, 0], None, p),  # 1-D
        lambda: _hmm.forward_backward([[1, 0]], [[True]], p),  # mask shape
        lambda: _hmm.forward_backward([[2.0, 0.0]], None, p),  # not 0/1
        lambda: _hmm.forward_backward(
            [[1.5]], None, {**TRUE_PVALUE}, observation="pvalue"
        ),
        lambda: _hmm.forward_backward([[1, 0]], None, p, observation="gauss"),
        lambda: _hmm.forward_backward([[1, 0]], None, {"a01": 0.1}),  # missing
        lambda: _hmm.forward_backward([[1, 0]], None, {**p, "alpha": 1.2}),
        lambda: _hmm.forward_backward([[1, 0]], None, {**p, "gamma": 0.1}),
        lambda: _hmm.viterbi([[1, 0]], None, {**p, "alpha": 0.9, "beta": 0.2}),
        lambda: _hmm.fit_hmm([[1, 0]], fixed={"alpha": 0.6, "beta": 0.5}),
        lambda: _hmm.fit_hmm([[1, 0]], initial="uniform"),
        lambda: _hmm.fit_hmm([[1, 0]], max_iter=0),
        lambda: _hmm.fit_hmm([[1, 0]], n_init=0),
        lambda: _hmm.fit_hmm([[0.5]], observation="pvalue", init={"a": 1.5}),
        lambda: _hmm.fit_hmm(
            [[1.0, 0.0]], fixed={"a01": 0, "a10": 0, "alpha": 0, "beta": 0}
        ),  # impossible data
    ]
    for call in bad_calls:
        with pytest.raises(nx.NetworkXError):
            call()


# ---------------------------------------------------------------------------
# Public function: edge_persistence_hmm
# ---------------------------------------------------------------------------

FIXED = {"a01": 0.2, "a10": 0.3, "alpha": 0.1, "beta": 0.2}


def _late_joiner():
    """Six steps; "c" joins at step 2 and "d" is present only at step 1."""
    S = {t: nx.Graph() for t in range(6)}
    for t, G in S.items():
        G.add_nodes_from("ab" if t < 2 else "abc")
    for t in (0, 1, 2, 4, 5):
        S[t].add_edge("a", "b")
    for t in (2, 3, 5):
        S[t].add_edge("b", "c")
    S[1].add_edge("a", "d")
    return S


def _expected(obs, mask, params, observation="binary"):
    """Posterior and Viterbi path of one edge from the engine."""
    mask = np.array([mask], dtype=bool)
    post, _ = _hmm.forward_backward([obs], mask, params, observation=observation)
    path = _hmm.viterbi([obs], mask, params, observation=observation)
    return post[0], path[0]


# Fitting three edges reaches the identifiability boundary (with a warning).
@pytest.mark.filterwarnings("ignore:HMM identifiability:RuntimeWarning")
def test_edge_persistence_hmm_presence():
    S = _late_joiner()
    H = nb.edge_persistence_hmm(S, params=FIXED, min_steps=3)
    cases = {  # edge: (observations, eligible steps)
        ("a", "b"): ([1, 1, 1, 0, 1, 1], [1] * 6),
        ("b", "c"): ([0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 1]),
        ("a", "d"): ([0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]),
    }
    for (u, v), (obs, elig) in cases.items():
        d = H[u][v]
        post, path = _expected(obs, elig, FIXED)
        steps = [t for t in range(6) if elig[t]]
        assert d["hmm_posterior"] == pytest.approx({t: post[t] for t in steps})
        assert d["hmm_score"] == pytest.approx(max(post[steps]))
        assert d["hmm_active_fraction"] == pytest.approx(np.mean(post[steps]))
        assert d["hmm_n_active"] == int(path[steps].sum())
        assert d["hmm_keep"] is (d["hmm_n_active"] >= 3)
    assert H["a"]["b"]["hmm_keep"] and not H["a"]["d"]["hmm_keep"]
    meta = H.graph["temporal"]
    assert meta["method"] == "edge_persistence_hmm" and meta["n_steps"] == 6
    assert meta["parameters"]["observe"] == "presence"
    assert meta["hmm"]["fixed"] == sorted(FIXED)
    assert meta["hmm"]["pi1"] == pytest.approx(0.4)
    # The three input forms agree, with fixed and with fitted parameters.
    for params in (FIXED, None):
        ref = nb.edge_persistence_hmm(S, params=params)
        model = dict(ref.graph["temporal"]["hmm"])
        fixed = model.pop("fixed")
        for M in (nb.events_from_snapshots(S), nb.spans_from_snapshots(S)):
            got = nb.edge_persistence_hmm(M, params=params)
            fit = dict(got.graph["temporal"]["hmm"])
            assert fit.pop("fixed") == fixed and fit == pytest.approx(model)
            for u, v, d in ref.edges(data=True):
                assert got[u][v]["hmm_posterior"] == pytest.approx(d["hmm_posterior"])
    # With weight and min_weight, a light edge counts as unobserved (a missing
    # weight counts as 1).
    W = {t: G.copy() for t, G in S.items()}
    W[1]["a"]["b"]["weight"] = 0.5
    HW = nb.edge_persistence_hmm(W, params=FIXED, weight="weight", min_weight=0.75)
    post, _ = _expected([1, 0, 1, 0, 1, 1], [1] * 6, FIXED)
    assert HW["a"]["b"]["hmm_posterior"][1] == pytest.approx(post[1])


def _perturbed_les_mis():
    """Four perturbed Les Misérables snapshots; Marius and Gavroche join late."""
    G = nx.les_miserables_graph()
    S = {}
    for t in range(4):
        H = G.copy()
        for i, (u, v, d) in enumerate(list(H.edges(data=True))):
            d["weight"] += (i * (t + 1)) % 4
            if (i + t) % 5 == 0:
                H.remove_edge(u, v)
        if t < 2:
            H.remove_nodes_from(["Marius", "Gavroche"])
        S[t] = H
    return S


def test_edge_persistence_hmm_keep_and_pvalue_observations():
    S = _perturbed_les_mis()
    scored = {t: nb.disparity_filter(G) for t, G in S.items()}
    kept = {
        t: nb.threshold_filter(H, "disparity_pvalue", 0.05) for t, H in scored.items()
    }
    for observe, observation in (("keep", "binary"), ("pvalue", "pvalue")):
        H = nb.edge_persistence_hmm(S, observe=observe, method="disparity_filter")
        model = H.graph["temporal"]["hmm"]
        assert model["observation"] == observation and model["converged"]
        names = ("a01", "a10", "alpha", "beta", "a", "pi1")
        params = {k: model[k] for k in names if k in model}
        for u, v, d in H.edges(data=True):
            elig = [u in G and v in G for G in S.values()]
            if observe == "keep":
                obs = [float(kept[t].has_edge(u, v)) for t in S]
            else:
                obs = [
                    scored[t][u][v]["disparity_pvalue"] if S[t].has_edge(u, v) else 1
                    for t in S
                ]
            post, _ = _expected(obs, elig, params, observation)
            steps = [t for t in S if elig[t]]
            assert d["hmm_posterior"] == pytest.approx({t: post[t] for t in steps})
    B = nb.boolean_filter(H, "hmm_keep")
    text = nb.describe_backbone("edge_persistence_hmm", original=H, backbone=B)
    assert "hidden Markov model" in text
    assert "scored with the disparity filter" in text
    assert "p-values were the observations" in text
    assert "estimated by expectation-maximization pooled over all edges" in text
    H = nb.edge_persistence_hmm(_late_joiner(), params=FIXED)
    text = nb.describe_backbone("edge_persistence_hmm", backbone=H)
    assert "were fixed at a01 = 0.2, a10 = 0.3, alpha = 0.1, beta = 0.2" in text
    assert "whether the edge was present" in text


def _block_and_flicker(seed=0, n_nodes=20, T=20, run=10, p_noise=0.08):
    """Snapshots with a persistent block, random flicker and light noise.

    Four block edges are present in one run of *run* consecutive steps, four
    flicker edges in *run* steps drawn at random; every other pair appears
    with probability *p_noise* in each step (#112).
    """
    rng = np.random.default_rng(seed)
    block = [(0, 1), (2, 3), (4, 5), (6, 7)]
    flicker = [(8, 9), (10, 11), (12, 13), (14, 15)]
    S = [nx.Graph() for _ in range(T)]
    for G in S:
        G.add_nodes_from(range(n_nodes))
    for e in block:
        start = int(rng.integers(0, T - run + 1))
        for t in range(start, start + run):
            S[t].add_edge(*e)
    for e in flicker:
        for t in rng.choice(T, run, replace=False):
            S[int(t)].add_edge(*e)
    planted = set(block) | set(flicker)
    for G in S:
        for u, v in itertools.combinations(range(n_nodes), 2):
            if (u, v) not in planted and rng.random() < p_noise:
                G.add_edge(u, v)
    return S, block, flicker


def test_edge_persistence_hmm_unobserved_active_diagnostic():
    # The diagnostic counts the decoded active steps and the share of them
    # in which the edge was not observed.
    S = _late_joiner()
    meta = nb.edge_persistence_hmm(S, params=FIXED).graph["temporal"]["hmm"]
    n_active = unobserved = 0
    for obs, elig in (
        ([1, 1, 1, 0, 1, 1], [1] * 6),
        ([0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 1]),
        ([0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]),
    ):
        _, path = _expected(obs, elig, FIXED)
        active = (path == 1) & np.array(elig, dtype=bool)
        n_active += int(active.sum())
        unobserved += int((active & (np.array(obs) == 0)).sum())
    assert n_active > unobserved > 0
    assert meta["n_active_steps"] == n_active
    assert meta["unobserved_active_share"] == pytest.approx(unobserved / n_active)
    # No active step: the share is undefined.
    quiet = nb.edge_persistence_hmm(S, params={**FIXED, "a01": 0.0, "pi1": 0.0})
    assert quiet.graph["temporal"]["hmm"]["n_active_steps"] == 0
    assert quiet.graph["temporal"]["hmm"]["unobserved_active_share"] is None
    # p-values have no "unobserved" flag.
    H = nb.edge_persistence_hmm(
        _perturbed_les_mis(), observe="pvalue", method="disparity_filter"
    )
    assert H.graph["temporal"]["hmm"]["unobserved_active_share"] is None
    assert H.graph["temporal"]["hmm"]["n_active_steps"] == sum(
        d["hmm_n_active"] for _, _, d in H.edges(data=True)
    )


def test_edge_persistence_hmm_pooled_miss_rate_tradeoff():
    # #112: on a mixture of persistent and randomly flickering edges, the
    # pooled fit trades switching for a large miss rate, so the flicker gets
    # more decoded active steps than the block.  The record exposes this.
    S, block, flicker = _block_and_flicker(seed=0)

    def n_active(H, edges):
        return np.mean([H.edges[e]["hmm_n_active"] for e in edges])

    fitted = nb.edge_persistence_hmm(S)
    meta = fitted.graph["temporal"]["hmm"]
    assert meta["beta"] > 0.25
    assert meta["unobserved_active_share"] > 0.25
    assert meta["n_active_steps"] == sum(
        d["hmm_n_active"] for _, _, d in fitted.edges(data=True)
    )
    assert n_active(fitted, flicker) > n_active(fitted, block)
    # Fixing the miss rate reads the gaps as inactivity instead.
    fixed = nb.edge_persistence_hmm(S, params={"beta": 0.05})
    assert fixed.graph["temporal"]["hmm"]["unobserved_active_share"] < 0.1
    assert n_active(fixed, flicker) < n_active(fixed, block)


def test_edge_persistence_hmm_errors():
    S = _late_joiner()
    call = nb.edge_persistence_hmm
    bad = [
        (ValueError, lambda: call(S, observe="flags")),
        (ValueError, lambda: call(S, observe="keep")),
        (ValueError, lambda: call(S, method="disparity_filter")),
        (ValueError, lambda: call(S, threshold=0.1)),
        (ValueError, lambda: call(S, alpha=0.1)),
        (ValueError, lambda: call(S, min_steps=0)),
        (ValueError, lambda: call(S, sign=0)),
        (ValueError, lambda: call(S, observe="keep", method="mst", min_weight=1)),
        (ValueError, lambda: call(S, observe="pvalue", method="mst")),
        (ValueError, lambda: call(S, observe="pvalue", method="disparity", mtc="bh")),
        (ValueError, lambda: call(S, freq=1)),
        (nx.NetworkXError, lambda: call(S, params={"alpha": 2})),
        (nx.NetworkXError, lambda: call(S, params={"gamma": 0.1})),
        (nx.NetworkXNotImplemented, lambda: call([nx.MultiGraph([(0, 1)])])),
    ]
    for err, f in bad:
        with pytest.raises(err):
            f()
