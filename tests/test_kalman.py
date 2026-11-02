"""Tests for ``kalman_backbone`` (issue #72) and its numerical core.

The array-level tests check ``networkx_backbone._kalman``; the last three
check the graph-level function in ``networkx_backbone.temporal``.  The API
contract is checked in tests/test_api_contract.py.
"""

import warnings

import networkx as nx
import pytest

np = pytest.importorskip("numpy")
stats = pytest.importorskip("scipy.stats")

import networkx_backbone as nb  # noqa: E402
from networkx_backbone import _kalman as kal  # noqa: E402
from networkx_backbone.statistical import _combine_tails  # noqa: E402

MODES = ("smoother", "filter")


def _random_masks(rng, shape, frac_missing=0.3):
    return np.where(rng.random(shape) < frac_missing, np.nan, 1.0)


def test_null_sd_recursion_matches_brute_force():
    # The smoother is linear in y: running it on unit vectors gives the weight
    # matrix A, and the null variance is the column sum of A².
    rng = np.random.default_rng(0)
    T = 12
    masks = [
        np.ones(T, bool),
        np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0], bool),  # leading/trailing gaps
        np.eye(T, dtype=bool)[4],  # a single observed step
    ]
    for mode in MODES:
        for q in (0.0, 0.01, 0.05, 1.0, 100.0):
            for times in (None, np.cumsum(rng.uniform(0.2, 3.0, T))):
                for mask in masks:
                    unit = np.where(mask[None, :], np.eye(T), np.nan)
                    level = kal.kalman_scores(unit, q, times=times, mode=mode).level
                    brute = (np.where(mask[:, None], level, 0.0) ** 2).sum(axis=0)
                    Y = np.where(mask, 0.0, np.nan)[None, :]
                    sd = kal.kalman_scores(Y, q, times=times, mode=mode).null_sd[0]
                    np.testing.assert_allclose(sd**2, brute, rtol=0, atol=1e-12)


def test_per_step_null_calibration():
    rng = np.random.default_rng(1)
    E, T = 4000, 25
    Y = rng.standard_normal((E, T))
    for mode in MODES:
        for missing in (False, True):
            Yin = Y * _random_masks(rng, Y.shape) if missing else Y
            p = kal.kalman_scores(Yin, 0.05, mode=mode).pvalue
            for t in (0, 12, T - 1):
                col = p[:, t][~np.isnan(p[:, t])]
                assert len(col) > 0.5 * E
                assert stats.kstest(col, "uniform").pvalue > 1e-3, (mode, missing, t)
                assert 0.04 < np.mean(col < 0.05) < 0.06


def test_posterior_sd_is_not_a_null_sd():
    # The design note's warning: √P gives miscalibrated (conservative) tests.
    rng = np.random.default_rng(2)
    res = kal.kalman_scores(rng.standard_normal((20000, 40)), 0.05)
    naive = stats.norm.sf(res.level / res.level_sd)
    assert np.mean(naive < 0.05) < 0.02
    assert abs(np.mean(res.pvalue < 0.05) - 0.05) < 0.003


def test_summary_null_calibration():
    rng = np.random.default_rng(3)
    E, T = 2000, 15
    Y = rng.standard_normal((E, T))
    patterns = np.where(rng.random((4, T)) < 0.3, np.nan, 1.0)
    for mode in MODES:
        for Yin in (Y, Y * patterns[rng.integers(0, 4, E)]):
            stat, p = kal.max_score_test(Yin, 0.1, mode=mode, n_sim=20000, seed=4)
            assert np.all(
                stat == np.nanmax(kal.kalman_scores(Yin, 0.1, mode=mode).score, axis=1)
            )
            assert stats.kstest(p, "uniform").pvalue > 1e-3, mode
            assert 0.035 < np.mean(p < 0.05) < 0.065
            assert p.min() >= 1 / 20001
    # Beyond the Monte Carlo floor the Bonferroni bound refines the p-value.
    Y = np.array([[0.0, 9.0, 0.0], [0.5, 0.2, 0.1]])
    stat, p = kal.max_score_test(Y, 1e6, n_sim=99, seed=0)
    assert p[0] == pytest.approx(3 * stats.norm.sf(stat[0])) and p[0] < 1 / 100
    assert p[1] > 0.1


def test_limits_stouffer_and_identity():
    rng = np.random.default_rng(5)
    E, T = 50, 10
    p_in = rng.random((E, T))
    Y = kal.probit_scores(p_in) * _random_masks(rng, (E, T))
    obs = ~np.isnan(Y)
    Y[0] = np.nan  # an edge with no observed step
    obs[0] = False
    y0 = np.where(obs, Y, 0.0)
    # q → 0: Stouffer over the observed steps (all of them, or the past only).
    stouffer = y0[1:].sum(axis=1) / np.sqrt(obs[1:].sum(axis=1))
    for q in (0.0, 1e-12):
        z = kal.kalman_scores(Y, q).score
        np.testing.assert_allclose(z[1:], np.repeat(stouffer[:, None], T, 1), atol=1e-9)
        assert np.all(np.isnan(z[0]))
    with np.errstate(invalid="ignore", divide="ignore"):
        past = np.cumsum(y0, axis=1) / np.sqrt(np.cumsum(obs, axis=1))
    np.testing.assert_allclose(kal.kalman_scores(Y, 0.0, mode="filter").score, past)
    # q → ∞: the input p-values at the observed steps.
    for mode in MODES:
        p = kal.kalman_scores(Y, 1e12, mode=mode).pvalue
        np.testing.assert_allclose(p[obs], p_in[obs], atol=1e-6)


def test_power_in_a_signal_window():
    rng = np.random.default_rng(6)
    E, T = 3000, 40
    mu = np.zeros(T)
    mu[10:20] = 0.8
    Y = rng.standard_normal((E, T)) + mu
    p = kal.kalman_scores(Y, 0.05).pvalue
    per_step = stats.norm.sf(Y)
    assert np.mean(p[:, 15] < 0.05) > 0.65  # prototype: 0.73
    assert np.mean(per_step[:, 15] < 0.05) < 0.25  # prototype: 0.20
    _, p_sum = kal.max_score_test(Y, 0.05, n_sim=4999, seed=0)
    bonferroni = np.mean(T * per_step.min(axis=1) < 0.05)
    assert np.mean(p_sum < 0.05) > bonferroni + 0.25


def test_time_reversal_and_edge_order_invariance():
    rng = np.random.default_rng(7)
    E, T = 30, 16
    Y = rng.standard_normal((E, T)) * _random_masks(rng, (E, T))
    times = np.cumsum(rng.uniform(0.5, 2.0, T))
    fwd = kal.kalman_scores(Y, 0.3, times=times)
    rev = kal.kalman_scores(Y[:, ::-1], 0.3, times=-times[::-1])
    for name in ("level", "level_sd", "null_sd", "score"):
        np.testing.assert_allclose(
            getattr(rev, name)[:, ::-1], getattr(fwd, name), rtol=1e-9, atol=1e-12
        )
    # Rows are independent, and the summary's simulation depends only on the
    # missingness pattern, so permuting edges permutes every output.
    perm = rng.permutation(E)
    s1, p1 = kal.max_score_test(Y, 0.3, times=times, n_sim=999, seed=8)
    s2, p2 = kal.max_score_test(Y[perm], 0.3, times=times, n_sim=999, seed=8)
    np.testing.assert_array_equal(s2, s1[perm])
    np.testing.assert_array_equal(p2, p1[perm])


def test_half_life_mapping():
    T, s = 300, 150
    unit = np.eye(T)[[s]]
    for h in (1.0, 2.5, 10.0):
        q = kal.half_life_to_q(h)
        assert kal.q_to_half_life(q) == pytest.approx(h, rel=1e-12)
        ratio = 2 ** (-1 / h)
        # Filter: weight of y_s in m_{s+k} decays by 1 − K = 2^(−1/h) per step.
        w = kal.kalman_scores(unit, q, mode="filter").level[0]
        np.testing.assert_allclose(w[s + 1 : s + 20] / w[s : s + 19], ratio, rtol=1e-8)
        # Smoother: the kernel decays at the same rate on both sides.
        a = kal.kalman_scores(unit, q).level[0]
        np.testing.assert_allclose(a[s + 1 : s + 20] / a[s : s + 19], ratio, rtol=1e-8)
        np.testing.assert_allclose(a[s - 20 : s - 1] / a[s - 19 : s], ratio, rtol=1e-8)
    assert kal.half_life_to_q(1.0) == pytest.approx(0.5)
    assert kal.q_to_half_life(0.0) == np.inf


def test_estimate_q():
    rng = np.random.default_rng(9)
    E, T, q = 1500, 30, 0.2
    times = np.cumsum(rng.uniform(0.5, 2.0, T))
    steps = np.concatenate(([1e3], np.diff(times)))  # wide first step: diffuse start
    X = np.cumsum(rng.standard_normal((E, T)) * np.sqrt(q * steps), axis=1)
    Y = (X + rng.standard_normal((E, T))) * _random_masks(rng, (E, T))
    q_hat, ll = kal.estimate_q(Y, times=times)
    assert q_hat == pytest.approx(q, rel=0.15)
    assert ll == pytest.approx(kal.pooled_loglik(Y, q_hat, times=times))
    assert ll >= max(kal.pooled_loglik(Y, f * q_hat, times=times) for f in (0.9, 1.1))
    # Standardized innovations are N(0, 1) under the fitted model.
    nu = kal.kalman_scores(Y, q_hat, times=times).innovation
    assert np.nanstd(nu) == pytest.approx(1.0, abs=0.03)
    # Pooled likelihood over null edges pushes q to the lower bound.
    q0, _ = kal.estimate_q(rng.standard_normal((E, T)))
    assert q0 < 1e-3


def test_probit_signed_reconstruction():
    rng = np.random.default_rng(10)
    z = rng.standard_normal(2000) * 3
    pairs = [_combine_tails(stats.norm.sf(v), stats.norm.cdf(v), v) for v in z]
    p2, sign = np.array(pairs).T
    z = np.clip(z, -stats.norm.isf(1e-16), stats.norm.isf(1e-16))  # see y below
    np.testing.assert_allclose(kal.probit_scores(p2, sign=sign), z, atol=1e-8)
    # Signed input keeps full precision in the lower tail; a one-sided p near 1
    # cannot (1 − p rounds), so check the unsigned path where it is exact.
    mid = np.abs(z) < 5
    np.testing.assert_allclose(
        kal.probit_scores(stats.norm.sf(z[mid])), z[mid], atol=1e-8
    )
    # Extreme p-values stay finite and symmetric; NaN stays unknown.
    y = kal.probit_scores([0.0, 1.0, np.nan])
    assert y[0] == -y[1] == pytest.approx(stats.norm.isf(1e-16))
    assert np.isnan(y[2])


def test_pit_on_discrete_pvalues():
    rng = np.random.default_rng(11)
    null = stats.binom(6, 0.3)
    x = null.rvs(20000, random_state=rng)
    p, p_strict = null.sf(x - 1), null.sf(x)  # P(X >= x), P(X > x)
    raw = kal.probit_scores(p)
    assert stats.kstest(raw, "norm").pvalue < 1e-6  # not N(0, 1)
    rand = kal.probit_scores(p, p_strict=p_strict, pit="random", seed=12)
    assert stats.kstest(rand, "norm").pvalue > 1e-3
    again = kal.probit_scores(p, p_strict=p_strict, pit="random", seed=12)
    np.testing.assert_array_equal(rand, again)
    mid = kal.probit_scores(p, p_strict=p_strict)  # pit="mid"
    assert abs(mid.mean()) < 0.03 < abs(raw.mean())
    assert mid.var() < 1  # conservative after smoothing
    # Raw discrete p-values keep the smoothed tests valid but conservative:
    # the smoothing weights are nonnegative.
    P = null.sf(null.rvs((4000, 20), random_state=rng) - 1)
    res = kal.kalman_scores(kal.probit_scores(P), 0.1)
    _, p_sum = kal.max_score_test(kal.probit_scores(P), 0.1, n_sim=999, seed=0)
    for pv in (res.pvalue[:, 10], p_sum):
        assert np.mean(pv < 0.05) < 0.04
    out = kal.discrete_pit([np.nan, 0.4], [np.nan, 0.1], pit="random", seed=0)
    assert np.isnan(out[0]) and 0.1 <= out[1] <= 0.4


def test_unknown_steps_and_innovations():
    Y = np.array([[np.nan, 1.0, np.nan, 2.0], [np.nan] * 4])
    for mode in MODES:
        res = kal.kalman_scores(Y, 0.5, mode=mode)
        assert np.all(np.isnan(res.score[1])) and np.all(np.isnan(res.pvalue[1]))
        stat, p = kal.max_score_test(Y, 0.5, mode=mode, n_sim=99, seed=0)
        assert np.isnan(stat[1]) and np.isnan(p[1]) and 0 < p[0] <= 1
    # No innovation at an unknown step or at the first (diffuse) observation.
    nu = kal.kalman_scores(Y, 0.5).innovation[0]
    assert np.isnan(nu[:3]).all()
    # ν = 2 − 1 with S = P_1 + q·Δt + 1 = 1 + 2·0.5 + 1.
    assert nu[3] == pytest.approx(1 / np.sqrt(3))


def test_input_validation():
    Y = np.zeros((2, 3))
    bad = [
        lambda: kal.kalman_scores(Y, 0.1, mode="backward"),
        lambda: kal.kalman_scores(Y, -1.0),
        lambda: kal.kalman_scores(Y, np.inf),
        lambda: kal.kalman_scores(Y, 0.1, times=[0, 2, 1]),
        lambda: kal.kalman_scores(Y, 0.1, times=[0, 1]),
        lambda: kal.kalman_scores(np.zeros(3), 0.1),
        lambda: kal.kalman_scores(np.full((1, 2), np.inf), 0.1),
        lambda: kal.max_score_test(Y, 0.1, n_sim=0),
        lambda: kal.estimate_q(np.array([[1.0, np.nan], [np.nan, 1.0]])),
        lambda: kal.estimate_q(Y, bounds=(0, 1)),
        lambda: kal.half_life_to_q(0),
        lambda: kal.q_to_half_life(-1),
        lambda: kal.probit_scores([1.5]),
        lambda: kal.probit_scores([0.5], sign=[0]),
        lambda: kal.probit_scores([0.5], sign=[1], p_strict=[0.2]),
        lambda: kal.discrete_pit([0.5], [0.2], pit="upper"),
        lambda: kal.discrete_pit([0.2], [0.5]),
        lambda: kal.discrete_pit([0.2, 0.1], [0.1]),
    ]
    for call in bad:
        with pytest.raises(ValueError):
            call()


# ---------------------------------------------------------------------------
# Graph level: kalman_backbone and kalman_snapshots
# ---------------------------------------------------------------------------


def _karate_snapshots(n_steps=8, seed=0):
    """Weighted karate snapshots; node 33 joins at step 2, edges flicker."""
    rng = np.random.default_rng(seed)
    G = nx.karate_club_graph()
    S = {}
    for t in range(n_steps):
        S[t] = nx.Graph()
        S[t].add_nodes_from(n for n in G if t >= 2 or n != 33)
        for u, v in G.edges:
            if u in S[t] and v in S[t] and rng.random() < 0.8:
                S[t].add_edge(u, v, weight=float(1 + rng.poisson(2.0)))
    return S


def test_kalman_backbone_matches_the_array_core():
    S = _karate_snapshots()
    times = list(S)
    for mode in MODES:
        H = nb.kalman_backbone(S, "disparity", mode=mode, half_life=2, seed=1)
        ends = list(H.edges)
        # By hand: score every eligible pair, an absent one as weight 0.
        P = np.full((len(ends), len(times)), np.nan)
        for k, t in enumerate(times):
            D = nb.disparity_filter(S[t], missing_as_zero=True)
            for j, (u, v) in enumerate(ends):
                if D.has_edge(u, v):
                    P[j, k] = D[u][v]["disparity_pvalue"]
        Y = kal.probit_scores(P * (2 - P))  # the Šidák correction
        q = kal.half_life_to_q(2)
        res = kal.kalman_scores(Y, q, mode=mode)
        stat, p_sum = kal.max_score_test(Y, q, mode=mode, n_sim=9999, seed=1)
        for j, (u, v) in enumerate(ends):
            d = H[u][v]
            eligible = [t for t in times if u in S[t] and v in S[t]]
            if mode == "filter":
                eligible = [t for t in eligible if not np.isnan(res.pvalue[j, t])]
            assert list(d["kalman_pvalues"]) == eligible
            assert d["kalman_n_obs"] == np.sum(~np.isnan(P[j]))
            np.testing.assert_allclose(
                [d["kalman_pvalues"][t] for t in eligible], res.pvalue[j, eligible]
            )
            np.testing.assert_allclose(
                [d["kalman_levels"][t] for t in eligible], res.level[j, eligible]
            )
            assert d["kalman_pvalue"] == pytest.approx(p_sum[j])
            assert d["kalman_score"] == pytest.approx(stat[j])
            assert set(d["kalman_innovations"]) < set(eligible)
        assert len(H[32][33]["kalman_pvalues"]) == len(times) - 2  # joins late
        meta = H.graph["temporal"]
        assert meta["method"] == "kalman_backbone" and meta["n_steps"] == 8
        fit = meta["kalman"]
        assert fit["q"] == pytest.approx(q) and fit["half_life"] == pytest.approx(2)
        assert (fit["q_source"], fit["absent"], fit["mode"]) == (
            "half_life",
            "zero",
            mode,
        )
    # kalman_snapshots: one graph per step holding that step's values.
    steps = nb.kalman_snapshots(H)
    assert list(steps) == times
    assert set(steps[0]) <= set(S[0]) and 33 not in steps[1]
    d = steps[4][32][33]
    assert d["kalman_pvalue"] == H[32][33]["kalman_pvalues"][4]
    assert d["kalman_level"] == H[32][33]["kalman_levels"][4]
    with pytest.raises(nx.NetworkXError, match="kalman_backbone"):
        nb.kalman_snapshots(nb.observation_fraction(S))


def test_kalman_backbone_q_steps_and_describe():
    S = _karate_snapshots(6)
    H = nb.kalman_backbone(S, nb.disparity_filter, n_sim=99, seed=0)
    fit = H.graph["temporal"]["kalman"]
    assert fit["q_source"] == "likelihood" and 1e-6 <= fit["q"] <= 1e3
    assert fit["half_life"] == pytest.approx(kal.q_to_half_life(fit["q"]))
    Hq = nb.kalman_backbone(S, "disparity", q=fit["q"], n_sim=99, seed=0)
    assert Hq.graph["temporal"]["kalman"]["loglik"] == pytest.approx(fit["loglik"])
    assert Hq.edges[0, 1]["kalman_pvalues"] == pytest.approx(
        H.edges[0, 1]["kalman_pvalues"]
    )
    # Unequal snapshot times scale q by the gap (in units of the smallest);
    # binned events are equally spaced steps.
    U = dict(zip([0, 1, 2, 4, 8, 9], S.values()))
    HU = nb.kalman_backbone(U, "disparity", q=0.3, n_sim=99, seed=0)
    assert HU.graph["temporal"]["kalman"]["step_unit"] == 1
    assert HU.edges[0, 1]["kalman_pvalues"] != pytest.approx(
        nb.kalman_backbone(S, "disparity", q=0.3, n_sim=99, seed=0).edges[0, 1][
            "kalman_pvalues"
        ]
    )
    E = nb.events_from_snapshots({10 * t: G for t, G in S.items()})
    HE = nb.kalman_backbone(E, "disparity", freq=10, q=0.3, n_sim=99, seed=0)
    assert HE.graph["temporal"]["kalman"]["step_unit"] is None
    assert HE.graph["temporal"]["input_form"] == "events"
    B = nb.threshold_filter(H, "kalman_pvalue", 0.05)
    text = nb.describe_backbone("kalman_backbone", original=H, backbone=B, alpha=0.05)
    for phrase in (
        "Kalman smoothing of per-step edge significance (Kalman, 1960; Rauch",
        "6 time steps, given as snapshots",
        "scored with the disparity filter (Serrano et al., 2009).",
        "estimated by maximum pooled likelihood",
        "simulated null distribution",
    ):
        assert phrase in text, phrase


def test_kalman_backbone_scorers_and_errors():
    S = _karate_snapshots(5)
    # A normal-model test gives the same one-sided scores signed or not.
    kw = {"q": 0.3, "n_sim": 99, "seed": 0, "pvalue_method": "delta"}
    one = nb.kalman_backbone(S, "noise_corrected_filter", **kw)
    two = nb.kalman_backbone(S, "noise_corrected_filter", signed=True, **kw)
    for u, v in [(0, 1), (32, 33)]:
        assert one[u][v]["kalman_pvalues"] == pytest.approx(
            two[u][v]["kalman_pvalues"], abs=1e-8
        )
    # The default noise-corrected p-values are exact binomial tails (#105),
    # so they are discrete.
    for nc_kw in ({}, {"pvalue_method": "binomial"}):
        with pytest.raises(ValueError, match="discrete p-values"):
            nb.kalman_backbone(S, "noise_corrected_filter", q=0.3, **nc_kw)
    with pytest.raises(ValueError, match="are discrete"):
        nb.kalman_backbone(
            S, "noise_corrected_filter", q=0.3, discrete="raw", signed=True
        )
    H = nb.kalman_backbone(
        S, "noise_corrected_filter", q=0.3, discrete="raw", n_sim=99, seed=0
    )
    assert 0 <= H.edges[0, 1]["kalman_pvalue"] <= 1
    # glab_filter cannot score absent edges: they become unknown, with a warning.
    with pytest.warns(UserWarning, match="treated as unknown"):
        H = nb.kalman_backbone(S, "glab_filter", q=0.3, n_sim=99, seed=0)
    assert H.graph["temporal"]["kalman"]["absent"] == "unknown"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        H = nb.kalman_backbone(
            S, "disparity", q=0.3, n_sim=99, seed=0, missing_as_zero=False
        )
    assert H.graph["temporal"]["kalman"]["absent"] == "unknown"
    # Discrete p-values are refused unless used raw (valid, conservative).
    with pytest.raises(ValueError, match="discrete p-values"):
        nb.kalman_backbone(S, "lans_filter", q=0.3)
    H = nb.kalman_backbone(S, "lans_filter", q=0.3, discrete="raw", n_sim=99)
    assert 0 <= H.edges[0, 1]["kalman_pvalue"] <= 1
    # Without the Šidák correction the disparity p-values are smaller.
    raw = nb.kalman_backbone(S, "disparity", q=0.3, sidak=False, n_sim=99, seed=0)
    sid = nb.kalman_backbone(S, "disparity", q=0.3, n_sim=99, seed=0)
    assert raw.edges[0, 1]["kalman_pvalues"][2] < sid.edges[0, 1]["kalman_pvalues"][2]
    bad = [
        ({"method": "disparity", "signed": True, "q": 0.3}, "combine two endpoints"),
        ({"method": "disparity", "q": 0.3, "half_life": 2}, "not both"),
        ({"method": "disparity", "mode": "both"}, "mode"),
        ({"method": "disparity", "discrete": "mid"}, "discrete"),
        ({"method": "disparity", "n_sim": 0}, "n_sim"),
        ({"method": "disparity", "q": -1}, "q must"),
        ({"method": "disparity", "half_life": 0}, "half_life"),
        ({"method": "high_salience_skeleton"}, "not one"),
        ({"method": "sdsm"}, "bipartite"),
    ]
    for kwargs, match in bad:
        with pytest.raises(ValueError, match=match):
            nb.kalman_backbone(S, **kwargs)
    # q cannot be estimated when no edge is scored twice.
    with pytest.raises(ValueError, match="pass q= or half_life="):
        nb.kalman_backbone([nx.Graph(S[0].edges(data=True))], "disparity")
