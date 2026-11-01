"""Tests for the numerical core of ``kalman_backbone`` (issue #72)."""

import pytest

np = pytest.importorskip("numpy")
stats = pytest.importorskip("scipy.stats")

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
