"""Numerical core of the planned ``kalman_backbone`` (issue #72), on arrays only.

This private module holds the numerics of proposal P1 in
``docs/design/temporal-backbones.md``; the public, graph-level function will
live in :mod:`networkx_backbone.temporal` and call these helpers.  Everything
here works on numpy arrays with one row per edge and one column per time step;
numpy and scipy are imported lazily inside the functions.

Model
-----
Each edge's per-step p-value ``p_t`` from a static scorer becomes a probit score
``y_t = Φ⁻¹(1 − p_t)``, which is N(0, 1) under that scorer's null
(:func:`probit_scores`).  The latent standardized signal follows a local-level
model with a known observation variance of 1::

    x_t = x_{t-1} + η_t,   η_t ~ N(0, q·Δt_t)
    y_t = x_t + ε_t,       ε_t ~ N(0, 1)

A NaN in ``y`` marks an unknown step: the filter predicts but does not update
(gain 0).  The initial state is diffuse (infinite prior variance), handled
exactly: the first observed step sets the level to its ``y`` with variance 1.
So an edge's results depend only on its observed steps and their times.

Exact null
----------
For fixed ``q`` the filtered and smoothed means are linear in ``y`` with
weights that depend only on ``q``, the times and the edge's missingness
pattern.  Under the edge-global null (``y_t`` iid N(0, 1) at observed steps)
the smoothed mean is therefore N(0, sd₀²), and ``sd₀`` follows from an O(T)
recursion: the Joseph form for the filter (it holds for any gain, so it gives
the error variance when the true ``q`` is 0) plus a backward recursion for the
smoother (:func:`_null_variance`).  The per-step p-value is
``1 − Φ(m_t / sd₀,t)``.  The posterior standard deviation ``√P`` answers a
different (Bayesian) question and gives miscalibrated p-values.

For fixed ``q`` the method is a time-localized, weighted Stouffer combination:
``q → 0`` gives Stouffer's Z over all observed steps and ``q → ∞`` returns the
input scores at observed steps.

The model and recursions follow Section 3 (P1) of
``docs/design/temporal-backbones.md``, which cites the standard Kalman filter,
Joseph form and Rauch–Tung–Striebel smoother equations from its source papers.
Using the recursions to test backbone edges is a novel adaptation.

Examples
--------
>>> import numpy as np
>>> from networkx_backbone import _kalman
>>> Y = np.array([[0.5, np.nan, 2.0, 1.5]])
>>> res = _kalman.kalman_scores(Y, q=0.0)
>>> bool(np.allclose(res.score, (0.5 + 2.0 + 1.5) / np.sqrt(3)))  # Stouffer
True
"""

from __future__ import annotations

from typing import NamedTuple

__all__: list[str] = []

#: Probit scores are clipped to ``±Φ⁻¹(1 − _P_FLOOR)`` (about ±8.22), so a
#: p-value of exactly 0 or 1 gives a finite score and one step cannot dominate.
_P_FLOOR = 1e-16

_MODES = ("smoother", "filter")
_PITS = ("mid", "random")


# --------------------------------------------------------------------------
# Input transforms
# --------------------------------------------------------------------------


def _as_generator(seed):
    from networkx_backbone.bipartite import _as_generator as as_generator

    return as_generator(seed)


def discrete_pit(p, p_strict, pit="mid", seed=None):
    """Probability integral transform of discrete upper-tail p-values.

    A discrete null makes ``p = P(X ≥ x)`` stochastically larger than uniform,
    so its probit score is not N(0, 1).  With ``p_strict = P(X > x)`` (the
    p-value of the next more extreme attainable outcome), the mid-p value is
    ``(p + p_strict) / 2`` and the randomized value is
    ``p_strict + U·(p − p_strict)`` with ``U ~ Uniform(0, 1)``.  The randomized
    value is exactly uniform under the null; the mid-p value has mean 1/2 but a
    smaller variance, so it is slightly conservative after smoothing.

    Parameters
    ----------
    p : array_like
        Upper-tail p-values ``P(X ≥ x)``; NaN marks an unknown step.
    p_strict : array_like
        ``P(X > x)``, the same shape as ``p``, with ``0 ≤ p_strict ≤ p``.
    pit : {"mid", "random"}, optional (default="mid")
        Which transform to apply.
    seed : integer, random_state, or None (default)
        Random seed for ``pit="random"``.

    Returns
    -------
    numpy.ndarray
        Continuous p-values, NaN where ``p`` is NaN.

    Examples
    --------
    >>> from networkx_backbone._kalman import discrete_pit
    >>> discrete_pit([1.0, 0.5], [0.0, 0.25]).tolist()
    [0.5, 0.375]
    """
    import numpy as np

    if pit not in _PITS:
        raise ValueError(f"pit must be one of {_PITS}, got {pit!r}")
    p = np.asarray(p, dtype=float)
    p_strict = np.asarray(p_strict, dtype=float)
    if p.shape != p_strict.shape:
        raise ValueError("p and p_strict must have the same shape")
    known = ~np.isnan(p)
    if np.any(known & ~((p_strict >= 0) & (p_strict <= p + 1e-12) & (p <= 1 + 1e-12))):
        raise ValueError("need 0 <= p_strict <= p <= 1 for every known step")
    if pit == "mid":
        return np.where(known, 0.5 * (p + p_strict), np.nan)
    u = _as_generator(seed).random(p.shape)
    return np.where(known, p_strict + u * (p - p_strict), np.nan)


def probit_scores(p, *, sign=None, p_strict=None, pit="mid", seed=None):
    """Convert per-step p-values to probit scores ``y = Φ⁻¹(1 − p)``.

    Parameters
    ----------
    p : array_like
        Per-step p-values in [0, 1]; NaN marks an unknown step.  One-sided
        (upper tail, small = strong edge) unless *sign* is given.
    sign : array_like or None, optional (default=None)
        For signed scorers: *p* is then two-sided, ``min(1, 2·min(upper,
        lower))``, and *sign* is ``+1`` where the upper tail is the smaller one
        and ``-1`` otherwise (the library's ``sign`` edge attribute).  For a
        continuous null the upper tail is ``p/2`` (sign +1) or ``1 − p/2``
        (sign −1), so ``y = sign · Φ⁻¹(1 − p/2)``, computed without
        cancellation in either tail.
    p_strict : array_like or None, optional (default=None)
        For discrete one-sided p-values: ``P(X > x)``, the p-value of the next
        more extreme outcome; see :func:`discrete_pit`.  Not supported together
        with *sign*.
    pit : {"mid", "random"}, optional (default="mid")
        Transform used when *p_strict* is given.
    seed : integer, random_state, or None (default)
        Random seed for ``pit="random"``.

    Returns
    -------
    numpy.ndarray
        Probit scores, NaN at unknown steps, clipped to about ±8.22
        (``Φ⁻¹(1 − 1e-16)``) so p-values of exactly 0 or 1 stay finite.

    Examples
    --------
    >>> from networkx_backbone._kalman import probit_scores
    >>> probit_scores([0.5, 0.025]).round(3).tolist()
    [0.0, 1.96]
    >>> probit_scores([0.05, 0.05], sign=[1, -1]).round(3).tolist()
    [1.96, -1.96]
    """
    import numpy as np
    from scipy.stats import norm

    p = np.asarray(p, dtype=float)
    known = ~np.isnan(p)
    if np.any(known & ((p < 0) | (p > 1))):
        raise ValueError("p-values must lie in [0, 1]")
    if p_strict is not None:
        if sign is not None:
            raise ValueError("p_strict (discrete p-values) and sign cannot be combined")
        p = discrete_pit(p, p_strict, pit=pit, seed=seed)
    y_max = norm.isf(_P_FLOOR)
    if sign is None:
        y = norm.isf(np.where(known, p, 0.5))
    else:
        sign = np.broadcast_to(np.asarray(sign, dtype=float), p.shape)
        if np.any(known & (np.abs(sign) != 1)):
            raise ValueError("sign must be +1 or -1 at every known step")
        y = np.where(sign > 0, 1.0, -1.0) * norm.isf(np.where(known, p, 1.0) / 2)
    return np.where(known, np.clip(y, -y_max, y_max), np.nan)


# --------------------------------------------------------------------------
# Kalman filter and RTS smoother
# --------------------------------------------------------------------------


def _check_inputs(Y, q, times, mode="smoother"):
    """Validate the panel and return ``(Y, observed, qdt)``."""
    import numpy as np

    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[1] == 0:
        raise ValueError("Y must be a 2-D (edges, steps) array with at least one step")
    if np.any(np.isinf(Y)):
        raise ValueError("Y must not contain infinite values (NaN marks unknown)")
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    return Y, ~np.isnan(Y), _process_variance(q, times, Y.shape[1])


def _process_variance(q, times, T):
    """Per-step process variance ``q·Δt_t`` (entry 0 is unused: diffuse prior)."""
    import numpy as np

    q = float(q)
    if not (np.isfinite(q) and q >= 0):
        raise ValueError(f"q must be a finite number >= 0, got {q!r}")
    if times is None:
        dt = np.ones(T - 1)
    else:
        times = np.asarray(times, dtype=float)
        if times.shape != (T,) or not np.all(np.isfinite(times)):
            raise ValueError(f"times must be {T} finite numbers")
        dt = np.diff(times)
        if np.any(dt <= 0):
            raise ValueError("times must be strictly increasing")
    return np.concatenate(([0.0], q * dt))


class _Gains(NamedTuple):
    """Gains and variances of the filter and smoother (independent of ``y``)."""

    K: object  # filter gain K_t (0 at unknown steps)
    omk: object  # 1 − K_t, computed without cancellation
    Pf: object  # filtered variance P_t (inf before the first observation)
    G: object  # smoother gain G_t = P_t / P_{t+1|t}, shape (n, T − 1)
    omg: object  # 1 − G_t


def _gains(observed, qdt):
    """Run the variance recursions for each row of the boolean *observed*."""
    import numpy as np

    n, T = observed.shape
    K = np.zeros((n, T))
    omk = np.ones((n, T))
    Pf = np.empty((n, T))
    P = np.full(n, np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        for t in range(T):
            pred = P + qdt[t]  # P_{t|t-1}
            obs = observed[:, t]
            diffuse = np.isinf(pred)
            K[:, t] = np.where(obs, np.where(diffuse, 1.0, pred / (pred + 1.0)), 0.0)
            omk[:, t] = np.where(obs, np.where(diffuse, 0.0, 1.0 / (pred + 1.0)), 1.0)
            # P_t = P_{t|t-1} / (P_{t|t-1} + 1), which is 1 for a diffuse prior.
            P = np.where(obs, np.where(diffuse, 1.0, pred / (pred + 1.0)), pred)
            Pf[:, t] = P
        before = Pf[:, :-1]
        pred_next = before + qdt[1:]
        inf = np.isinf(before)
        # Before the first observation P_t is infinite and the limit G_t = 1.
        G = np.where(inf, 1.0, before / pred_next)
        omg = np.where(inf, 0.0, qdt[1:] / pred_next)
    return _Gains(K, omk, Pf, G, omg)


def _filter_means(Y, observed, gains):
    """Filtered means and standardized innovations; gains broadcast over rows."""
    import numpy as np

    y = np.where(observed, Y, 0.0)
    n, T = y.shape
    mf = np.empty((n, T))
    innov = np.empty((n, T))  # raw ν_t = y_t − m_{t−1}; callers mask and scale
    m = np.zeros(n)
    for t in range(T):
        innov[:, t] = y[:, t] - m
        m = gains.omk[:, t] * m + gains.K[:, t] * y[:, t]
        mf[:, t] = m
    return mf, innov


def _smoother_means(mf, gains):
    """RTS smoothed means ``m_{t|T} = (1 − G_t) m_t + G_t m_{t+1|T}``."""
    import numpy as np

    ms = np.empty_like(mf)
    ms[:, -1] = mf[:, -1]
    for t in range(mf.shape[1] - 2, -1, -1):
        ms[:, t] = gains.omg[:, t] * mf[:, t] + gains.G[:, t] * ms[:, t + 1]
    return ms


def _smoother_variances(gains, qdt):
    """RTS posterior variances ``P_{t|T} = (1 − G_t) P_t + G_t² P_{t+1|T}``."""
    import numpy as np

    Pf = gains.Pf
    Ps = np.empty_like(Pf)
    Ps[:, -1] = Pf[:, -1]
    with np.errstate(invalid="ignore"):
        for t in range(Pf.shape[1] - 2, -1, -1):
            Ps[:, t] = np.where(
                np.isinf(Pf[:, t]),
                qdt[t + 1] + Ps[:, t + 1],  # diffuse limit
                gains.omg[:, t] * Pf[:, t] + gains.G[:, t] ** 2 * Ps[:, t + 1],
            )
    return Ps


def _null_variance(gains, mode="smoother"):
    """Exact variance of the filtered or smoothed mean under the null.

    Under H0 the observed ``y_s`` are iid N(0, 1).  The filtered mean
    ``m_t = (1 − K_t) m_{t−1} + K_t y_t`` has variance
    ``v_t = (1 − K_t)² v_{t−1} + K_t²`` (the Joseph form with true process
    variance 0).  The smoothed mean ``m_{t|T} = (1 − G_t) m_t + G_t m_{t+1|T}``
    has variance::

        V_t = (1 − G_t)² v_t + G_t² V_{t+1} + 2 G_t (1 − G_t) v_t (1 − K_{t+1}) χ_{t+1}
        χ_t = (1 − G_t) + G_t (1 − K_{t+1}) χ_{t+1},    V_T = v_T,  χ_T = 1

    where ``χ_t`` is the weight of ``m_t`` in ``m_{t|T}`` (so that
    ``Cov(m_t, m_{t+1|T}) = χ_{t+1} (1 − K_{t+1}) v_t``).  O(T) per edge.
    """
    import numpy as np

    K, omk, G, omg = gains.K, gains.omk, gains.G, gains.omg
    n, T = K.shape
    v = np.empty((n, T))
    prev = np.zeros(n)
    for t in range(T):
        prev = omk[:, t] ** 2 * prev + K[:, t] ** 2
        v[:, t] = prev
    if mode == "filter":
        return v
    V = np.empty((n, T))
    V[:, -1] = v[:, -1]
    chi = np.ones(n)
    for t in range(T - 2, -1, -1):
        cov = v[:, t] * omk[:, t + 1] * chi
        V[:, t] = (
            omg[:, t] ** 2 * v[:, t]
            + G[:, t] ** 2 * V[:, t + 1]
            + 2 * G[:, t] * omg[:, t] * cov
        )
        chi = omg[:, t] + G[:, t] * omk[:, t + 1] * chi
    return V


class KalmanResult(NamedTuple):
    """Per-step outputs of :func:`kalman_scores`; every array is (edges, steps).

    Attributes
    ----------
    level : numpy.ndarray
        Smoothed (or filtered) mean of the latent signal, ``m_t``.
    level_sd : numpy.ndarray
        Posterior standard deviation ``√P_t`` (Bayesian; for plotting only).
    null_sd : numpy.ndarray
        Exact null standard deviation ``sd₀,t`` of ``level``.
    score : numpy.ndarray
        ``level / null_sd``, N(0, 1) under the null (NaN with no observation).
    pvalue : numpy.ndarray
        One-sided p-value ``1 − Φ(score)``.
    innovation : numpy.ndarray
        Standardized one-step prediction error ``ν_t / √S_t`` (NaN at unknown
        steps and at each edge's first observation).  Calibrated only under the
        fitted model, not under the null.
    """

    level: object
    level_sd: object
    null_sd: object
    score: object
    pvalue: object
    innovation: object


def kalman_scores(Y, q, *, times=None, mode="smoother"):
    """Local-level Kalman filter / RTS smoother with exact null p-values.

    Parameters
    ----------
    Y : array_like
        (edges, steps) probit scores; NaN marks an unknown step.
    q : float
        Process variance per unit of time (``q ≥ 0``).
    times : array_like or None, optional (default=None)
        Strictly increasing step times; ``None`` means ``0, 1, …, T − 1``.  The
        process variance between steps ``t − 1`` and ``t`` is
        ``q·(times[t] − times[t − 1])``.
    mode : {"smoother", "filter"}, optional (default="smoother")
        ``"smoother"`` uses all steps (RTS); ``"filter"`` uses only the past.

    Returns
    -------
    KalmanResult
        Per-step level, posterior sd, null sd, z-score, p-value and
        standardized innovation.

    Examples
    --------
    >>> import numpy as np
    >>> from networkx_backbone._kalman import kalman_scores
    >>> res = kalman_scores(np.array([[1.0, 2.0, 0.0]]), q=1e12)
    >>> res.score.round(6).tolist()  # q → ∞ returns the inputs
    [[1.0, 2.0, 0.0]]
    """
    import numpy as np
    from scipy.stats import norm

    Y, observed, qdt = _check_inputs(Y, q, times, mode)
    gains = _gains(observed, qdt)
    mf, innov = _filter_means(Y, observed, gains)
    # Innovation variance S_t = P_{t−1} + q·Δt_t + 1 (infinite while diffuse).
    with np.errstate(invalid="ignore"):
        S = np.full(Y.shape, np.nan)
        S[:, 1:] = gains.Pf[:, :-1] + qdt[1:] + 1.0
        innov = np.where(observed & np.isfinite(S), innov / np.sqrt(S), np.nan)
    if mode == "smoother":
        level = _smoother_means(mf, gains)
        post = _smoother_variances(gains, qdt)
    else:
        level, post = mf, gains.Pf
    null_sd = np.sqrt(_null_variance(gains, mode))
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(null_sd > 0, level / null_sd, np.nan)
    return KalmanResult(
        level=level,
        level_sd=np.sqrt(post),
        null_sd=null_sd,
        score=score,
        pvalue=norm.sf(score),
        innovation=innov,
    )


# --------------------------------------------------------------------------
# Choosing q
# --------------------------------------------------------------------------


def pooled_loglik(Y, q, *, times=None):
    """Pooled Gaussian log-likelihood from the prediction-error decomposition.

    ``log p(y) = Σ_e Σ_t log N(ν_{e,t}; 0, S_{e,t})`` over the observed steps,
    skipping each edge's first observation (its prediction is diffuse).

    Parameters
    ----------
    Y : array_like
        (edges, steps) probit scores; NaN marks an unknown step.
    q : float
        Process variance per unit of time.
    times : array_like or None, optional (default=None)
        Step times, as in :func:`kalman_scores`.

    Returns
    -------
    float
        The pooled log-likelihood.

    Examples
    --------
    >>> import numpy as np
    >>> from networkx_backbone._kalman import pooled_loglik
    >>> ll = pooled_loglik(np.zeros((1, 2)), q=0.0)  # one innovation, S = 2
    >>> bool(np.isclose(ll, -0.5 * np.log(2 * np.pi * 2)))
    True
    """
    import numpy as np

    Y, observed, qdt = _check_inputs(Y, q, times)
    gains = _gains(observed, qdt)
    _, nu = _filter_means(Y, observed, gains)
    with np.errstate(invalid="ignore"):
        S = gains.Pf[:, :-1] + qdt[1:] + 1.0
    nu, obs = nu[:, 1:], observed[:, 1:] & np.isfinite(S)
    S = np.where(obs, S, 1.0)
    nu = np.where(obs, nu, 0.0)
    return float(np.sum(np.where(obs, -0.5 * (np.log(2 * np.pi * S) + nu**2 / S), 0)))


def estimate_q(Y, *, times=None, bounds=(1e-6, 1e3)):
    """Maximum pooled-likelihood estimate of ``q`` by a bounded search on log q.

    Pooling over mostly-null edges pushes the estimate toward the lower bound
    (no time variation), so a user-chosen half-life (:func:`half_life_to_q`) is
    often the better choice for backboning.

    Parameters
    ----------
    Y : array_like
        (edges, steps) probit scores; NaN marks an unknown step.
    times : array_like or None, optional (default=None)
        Step times, as in :func:`kalman_scores`.
    bounds : tuple of float, optional (default=(1e-6, 1e3))
        Search interval for ``q``.

    Returns
    -------
    q : float
        The estimate, within *bounds*.
    loglik : float
        The pooled log-likelihood at *q*.

    Examples
    --------
    >>> import numpy as np
    >>> from networkx_backbone._kalman import estimate_q
    >>> rng = np.random.default_rng(0)
    >>> X = np.cumsum(rng.normal(0, np.sqrt(0.2), (500, 30)), axis=1)
    >>> q, _ = estimate_q(X + rng.standard_normal(X.shape))
    >>> bool(0.15 < q < 0.25)
    True
    """
    import numpy as np
    from scipy.optimize import minimize_scalar

    Y, observed, _ = _check_inputs(Y, 0.0, times)
    if not np.any(observed.sum(axis=1) >= 2):
        raise ValueError("estimating q needs an edge with at least two observed steps")
    lo, hi = (float(b) for b in bounds)
    if not 0 < lo < hi < np.inf:
        raise ValueError("bounds must satisfy 0 < low < high < inf")
    res = minimize_scalar(
        lambda lq: -pooled_loglik(Y, np.exp(lq), times=times),
        bounds=(np.log(lo), np.log(hi)),
        method="bounded",
        options={"xatol": 1e-4},
    )
    q = float(np.exp(res.x))
    best = (q, -float(res.fun))
    for edge in (lo, hi):  # the bounded search never evaluates the endpoints
        ll = pooled_loglik(Y, edge, times=times)
        if ll > best[1]:
            best = (edge, ll)
    return best


def half_life_to_q(half_life):
    """Process variance whose steady-state kernel halves every *half_life* steps.

    With unit steps the local-level filter converges to an exponentially
    weighted moving average ``m_t = (1 − K) m_{t−1} + K y_t``: the steady-state
    predicted variance ``P'`` solves ``P' = P'/(P' + 1) + q``, i.e.
    ``P'² = q (1 + P')``, and ``K = P'/(1 + P')``.  The weight of ``y_{t−k}``
    is ``K (1 − K)^k``, and the RTS smoother's steady-state gain is also
    ``1 − K``, so its two-sided kernel decays as ``(1 − K)^|k|`` too.  A
    half-life ``h`` means ``(1 − K)^h = 1/2``, so ``K = 1 − 2^(−1/h)``, and
    substituting ``P' = K/(1 − K)`` gives ``q = K² / (1 − K)``.

    Parameters
    ----------
    half_life : float
        Half-life in steps (units of time), ``> 0``.

    Returns
    -------
    float
        The process variance ``q``.

    Examples
    --------
    >>> from networkx_backbone._kalman import half_life_to_q
    >>> half_life_to_q(1.0)  # K = 1/2
    0.5
    """
    import math

    h = float(half_life)
    if not (math.isfinite(h) and h > 0):
        raise ValueError(f"half_life must be a finite number > 0, got {half_life!r}")
    log_keep = -math.log(2.0) / h  # log(1 − K)
    K = -math.expm1(log_keep)
    return K * K / math.exp(log_keep)


def q_to_half_life(q):
    """Inverse of :func:`half_life_to_q` (``inf`` for ``q = 0``).

    Parameters
    ----------
    q : float
        Process variance per step, ``≥ 0``.

    Returns
    -------
    float
        Steady-state half-life in steps.

    Examples
    --------
    >>> from networkx_backbone._kalman import q_to_half_life
    >>> q_to_half_life(0.5)
    1.0
    """
    import math

    q = float(q)
    if not (math.isfinite(q) and q >= 0):
        raise ValueError(f"q must be a finite number >= 0, got {q!r}")
    if q == 0:
        return math.inf
    pred = 0.5 * (q + math.sqrt(q * q + 4 * q))  # steady-state P'
    return math.log(2.0) / math.log1p(pred)  # 1 − K = 1/(1 + P')


# --------------------------------------------------------------------------
# Edge-level summary test
# --------------------------------------------------------------------------


def max_score_test(Y, q, *, times=None, mode="smoother", n_sim=9999, seed=None):
    """Edge-level test across steps: the maximum over t of the per-step z.

    The statistic is ``max_t score_t`` (one-sided, as the per-step tests).
    Its null distribution depends only on ``q``, the times, the mode and the
    edge's missingness pattern, so it is simulated once per distinct pattern
    from ``n_sim`` null series (iid N(0, 1) at the observed steps).  The same
    base draws are reused for every pattern (common random numbers).  The
    p-value is the Monte Carlo ``(1 + #{null ≥ stat}) / (1 + n_sim)``, which is
    valid (never anti-conservative) and cannot go below ``1/(n_sim + 1)``.
    Edges with the same pattern share one simulated null, so their p-values
    share its Monte Carlo error: at ``n_sim=9999`` the realized level of a
    0.05 test is within about ±0.004 (two standard errors) of 0.05.  The cost
    is O(n_sim · T) per distinct pattern.

    Parameters
    ----------
    Y : array_like
        (edges, steps) probit scores; NaN marks an unknown step.
    q : float
        Process variance per unit of time.
    times : array_like or None, optional (default=None)
        Step times, as in :func:`kalman_scores`.
    mode : {"smoother", "filter"}, optional (default="smoother")
        As in :func:`kalman_scores`.
    n_sim : int, optional (default=9999)
        Null series simulated per missingness pattern.
    seed : integer, random_state, or None (default)
        Random seed for the simulation.

    Returns
    -------
    stat : numpy.ndarray
        ``max_t score_t`` per edge (NaN for an edge with no observed step).
    pvalue : numpy.ndarray
        Monte Carlo p-value per edge (NaN for an edge with no observed step).

    Examples
    --------
    >>> import numpy as np
    >>> from networkx_backbone._kalman import max_score_test
    >>> Y = np.array([[0.1, 3.5, 3.0, 0.2], [0.0, -1.0, np.nan, 0.5]])
    >>> stat, p = max_score_test(Y, q=0.5, n_sim=999, seed=1)
    >>> bool(p[0] < 0.05 < p[1])
    True
    """
    import numpy as np

    Y, observed, qdt = _check_inputs(Y, q, times, mode)
    n_sim = int(n_sim)
    if n_sim < 1:
        raise ValueError("n_sim must be a positive integer")
    with np.errstate(invalid="ignore"):
        score = kalman_scores(Y, q, times=times, mode=mode).score
    has_obs = observed.any(axis=1)
    stat = np.full(Y.shape[0], np.nan)
    stat[has_obs] = np.nanmax(score[has_obs], axis=1)
    pvalue = np.full(Y.shape[0], np.nan)
    patterns, inverse = np.unique(observed, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    base = _as_generator(seed).standard_normal((n_sim, Y.shape[1]))
    for k, pattern in enumerate(patterns):
        if not pattern.any():
            continue
        null = np.sort(_null_max_scores(base, pattern, qdt, mode))
        rows = np.flatnonzero(inverse == k)
        n_ge = n_sim - np.searchsorted(null, stat[rows], side="left")
        pvalue[rows] = (1.0 + n_ge) / (1.0 + n_sim)
    return stat, pvalue


def _null_max_scores(base, pattern, qdt, mode):
    """``max_t score_t`` for null series *base* observed on *pattern*."""
    import numpy as np

    observed = np.broadcast_to(pattern, base.shape)
    gains = _gains(pattern[None, :], qdt)
    mf, _ = _filter_means(base, observed, gains)
    level = _smoother_means(mf, gains) if mode == "smoother" else mf
    sd = np.sqrt(_null_variance(gains, mode))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(sd > 0, level / sd, -np.inf)
    return z.max(axis=1)
