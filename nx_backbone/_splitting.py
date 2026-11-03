"""Adaptive multilevel splitting for FDSM co-occurrence tails (private).

Used by ``fdsm(refine="splitting")`` (issue #77).

Under the FDSM null the incidence matrix is uniform over all binary matrices
with the observed row and column sums.  For one agent pair ``(i, j)`` the
score is the co-occurrence ``C_ij = |S_i & S_j|``, where ``S_r`` is the set
of columns of row ``r``.  :func:`_splitting_tail` estimates
``P(C_ij >= x_obs)`` by sequential Monte Carlo over the nested level sets
``{C_ij >= l_1} > ... > {C_ij >= l_n = x_obs}`` (Cerou, Del Moral, Furon &
Guyader 2012, Algorithm 2, with integer levels), moving particles inside a
level set with a restricted curveball kernel.

:func:`_exact_cooccurrence_pmf` computes the exact null distribution of
``C_ij`` for small matrices by dynamic programming over columns; it is the
reference the estimator is validated against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__: list[str] = []


# =====================================================================
# Exact reference: dynamic programming over columns
# =====================================================================


def _exact_cooccurrence_pmf(row_sums, col_sums, i, j):
    """Exact pmf of ``C_ij`` over binary matrices with the given margins.

    Columns are filled one at a time.  The state is the remaining degree of
    rows ``i`` and ``j`` and the multiset of the other rows' remaining
    degrees (rows with equal remaining degree are interchangeable), and each
    state carries the number of partial matrices that reach it for every
    co-occurrence count so far, as exact integers.

    Returns
    -------
    pmf : list of fractions.Fraction
        ``pmf[x] = P(C_ij = x)``.
    total : int
        Number of matrices with these margins.
    """
    from fractions import Fraction

    rows = [int(r) for r in row_sums]
    cols = sorted((int(c) for c in col_sums), reverse=True)
    if sum(rows) != sum(cols):
        raise ValueError("row and column sums differ")
    if i == j:
        raise ValueError("i and j must differ")
    K = min(rows[i], rows[j])
    others = tuple(sorted((r for k, r in enumerate(rows) if k not in (i, j) and r)))
    n_cols = len(cols)

    def choose_others(groups, t):
        """Yield (ways, new_groups) choosing ``t`` rows from the others."""
        vals = list(groups)
        out = []

        def rec(idx, left, ways, acc):
            if left == 0:
                new = {}
                for v, m in groups.items():
                    new[v] = new.get(v, 0) + m
                for v, tv in acc:
                    new[v] -= tv
                    if v - 1 > 0:
                        new[v - 1] = new.get(v - 1, 0) + tv
                key = tuple(sorted(v for v, m in new.items() for _ in range(m) if m))
                out.append((ways, key))
                return
            if idx == len(vals):
                return
            v = vals[idx]
            m = groups[v]
            for tv in range(min(m, left) + 1):
                rec(
                    idx + 1,
                    left - tv,
                    ways * math.comb(m, tv),
                    (*acc, (v, tv)) if tv else acc,
                )

        rec(0, t, 1, ())
        return out

    states = {(rows[i], rows[j], others): [1] + [0] * K}
    for k, c in enumerate(cols):
        left_cols = n_cols - k - 1
        nxt = {}
        for (ri, rj, oth), poly in states.items():
            groups = {}
            for v in oth:
                groups[v] = groups.get(v, 0) + 1
            for a in (0, 1):
                if a > ri:
                    continue
                for b in (0, 1):
                    if b > rj:
                        continue
                    t = c - a - b
                    if t < 0 or t > len(oth):
                        continue
                    for ways, key in choose_others(groups, t):
                        ni, nj = ri - a, rj - b
                        if max((ni, nj, *key), default=0) > left_cols:
                            continue
                        dest = nxt.setdefault((ni, nj, key), [0] * (K + 1))
                        shift = a & b
                        for x in range(K + 1 - shift):
                            if poly[x]:
                                dest[x + shift] += ways * poly[x]
        states = nxt
    final = states.get((0, 0, ()), [0] * (K + 1))
    total = sum(final)
    if total == 0:
        raise ValueError("no binary matrix has these margins")
    return [Fraction(v, total) for v in final], total


def _exact_upper_tail(row_sums, col_sums, i, j, x):
    """Exact ``P(C_ij >= x)`` as a float."""
    pmf, _ = _exact_cooccurrence_pmf(row_sums, col_sums, i, j)
    return float(sum(pmf[x:], start=0 * pmf[0]))


# =====================================================================
# Moves that keep the uniform law on a level set
# =====================================================================
#
# Every move below takes ``prng``, a :class:`_Rand`: Python's generator for
# scalar draws (much faster than NumPy's for one number at a time) and
# NumPy's for shuffling lists (faster than Python's), both seeded from the
# caller's NumPy generator so results stay reproducible.


class _Rand:
    """Fast scalar uniforms and list shuffles from one NumPy generator."""

    __slots__ = ("random", "shuffle")

    def __init__(self, rng):
        import random

        self.random = random.Random(int(rng.integers(2**63))).random
        self.shuffle = rng.shuffle


def _truncated_hypergeometric(hot, cold, k, lo, hi, prng):
    """Draw ``h`` in ``[lo, hi]`` with weight ``C(hot, h) C(cold, k - h)``."""
    if lo == hi:
        return lo
    weights = [math.comb(hot, h) * math.comb(cold, k - h) for h in range(lo, hi + 1)]
    u = prng.random() * sum(weights)
    for h, w in zip(range(lo, hi + 1), weights):
        u -= w
        if u < 0:
            return h
    return hi


def _restricted_trade(rows, r1, r2, i, j, level, prng):
    """Heat-bath curveball trade between rows ``r1`` and ``r2`` on a level set.

    Rows ``r1`` and ``r2`` redistribute the columns they do not share,
    keeping their degrees, uniformly at random among the redistributions
    that keep ``C_ij = |rows[i] & rows[j]|`` at least ``level``.  Without the
    constraint this is the curveball trade (Strona et al. 2014), which is
    symmetric with the uniform law as stationary law (Carstens 2015).  The
    set of allowed redistributions is the same from each of its members, so
    the constrained trade is symmetric too and keeps the uniform law on the
    level set.

    Only a trade that involves exactly one of ``i`` and ``j`` can change
    ``C_ij``: a trade between ``i`` and ``j`` keeps their common columns.
    For a trade between ``i`` and ``r``, the number ``h`` of pool columns in
    ``rows[j]`` that row ``i`` receives is hypergeometric; it is drawn from
    that law truncated to ``C_ij >= level``.
    """
    s1 = rows[r1]
    s2 = rows[r2]
    only1 = s1 - s2
    if not only1:
        return
    only2 = s2 - s1
    if not only2:
        return
    common = s1 & s2
    pool = list(only1 | only2)
    k1 = len(only1)
    if (r1 == i or r1 == j) == (r2 == i or r2 == j):
        prng.shuffle(pool)
        rows[r1] = common | set(pool[:k1])
        rows[r2] = common | set(pool[k1:])
        return
    # ``a`` is the constrained row (i or j), ``b`` its partner.
    if r1 == i or r1 == j:
        a, b, ka = r1, r2, k1
    else:
        a, b, ka = r2, r1, len(pool) - k1
    partner = rows[j] if a == i else rows[i]
    hot = [c for c in pool if c in partner]
    cold = [c for c in pool if c not in partner]
    base = len(common & partner)
    lo = max(level - base, 0, ka - len(cold))
    hi = min(len(hot), ka)
    h = _truncated_hypergeometric(len(hot), len(cold), ka, lo, hi, prng)
    prng.shuffle(hot)
    prng.shuffle(cold)
    rows[a] = common | set(hot[:h]) | set(cold[: ka - h])
    rows[b] = common | set(hot[h:]) | set(cold[ka - h :])


def _pair_block_trade(rows, i, j, others, prng):
    """Trade the shared columns of ``i`` and ``j`` with two other rows.

    For two rows ``r``, ``s`` drawn uniformly from *others* (the rows other
    than ``i`` and ``j``), the columns where rows ``(i, j, r, s)`` read
    ``(1, 1, 0, 0)`` or ``(0, 0, 1, 1)`` are redistributed uniformly,
    keeping how many read ``(1, 1, 0, 0)``.  Every row and column sum and
    ``C_ij`` are unchanged, and the set of redistributions is the same from
    each of its members, so this heat-bath move is symmetric.  It is the
    move that changes ``rows[i]`` and ``rows[j]`` together, which single
    trades cannot do without lowering ``C_ij`` when ``rows[i]`` and
    ``rows[j]`` must be identical.
    """
    n = len(others)
    if n < 2:
        return
    a = int(prng.random() * n)
    b = int(prng.random() * (n - 1))
    r, s = others[a], others[b + (b >= a)]
    si, sj, sr, ss = rows[i], rows[j], rows[r], rows[s]
    A = (si & sj) - sr - ss
    if not A:
        return
    D = (sr & ss) - si - sj
    if not D:
        return
    pool = list(A | D)
    prng.shuffle(pool)
    new_a = set(pool[: len(A)])
    new_d = set(pool[len(A) :])
    rows[i] = (si - A) | new_a
    rows[j] = (sj - A) | new_a
    rows[r] = (sr - D) | new_d
    rows[s] = (ss - D) | new_d


def _move(rows, i, j, level, n_moves, prng, focus=0.5, block=0.2, others=None):
    """Apply ``n_moves`` moves that keep ``C_ij >= level``.

    Each move is a :func:`_pair_block_trade` with probability *block*, and
    otherwise a :func:`_restricted_trade` whose first row is ``i`` or ``j``
    with probability *focus* (and uniform otherwise) and whose second row is
    uniform among the rest.  The rows are drawn independently of the state,
    and a state-independent mixture of moves that keep the uniform law on
    the level set (reversibly) keeps it too.
    """
    n = len(rows)
    if others is None:
        others = [k for k in range(n) if k != i and k != j]
    rand = prng.random
    for _ in range(n_moves):
        if rand() < block:
            _pair_block_trade(rows, i, j, others, prng)
            continue
        if rand() < focus:
            r1 = i if rand() < 0.5 else j
        else:
            r1 = int(rand() * n)
        r2 = int(rand() * (n - 1))
        r2 += r2 >= r1
        _restricted_trade(rows, r1, r2, i, j, level, prng)


# =====================================================================
# Splitting estimator
# =====================================================================


@dataclass
class SplittingResult:
    """Output of :func:`_splitting_tail`."""

    pvalue_rb: float  # estimate with the Rao-Blackwellized last stage
    se_rb: float  # its Lee-Whiteley standard error
    pvalue: float  # plain estimate (product of survival fractions)
    se: float  # its Lee-Whiteley standard error
    se_ideal: float  # standard error of the idealized algorithm
    levels: list[int]
    fractions: list[float]
    n_moves: int
    extinct: bool
    eve_survivors: int


def _rb_phi(rows, i, j, level, x):
    """``K 1{C_ij >= x}`` at one particle, for a one-trade kernel ``K``.

    ``K`` picks ``a`` in ``{i, j}`` and a partner ``b`` outside ``{i, j}``
    uniformly and applies :func:`_restricted_trade` for the level set
    ``{C_ij >= level}``.  ``K`` keeps the uniform law on that level set, so
    the mean of ``K 1{C_ij >= x}`` over particles estimates the same
    probability as the mean of the indicator, with less variance (a
    Rao-Blackwellization of the last stage).  For each ``(a, b)`` the new
    count is ``base + h`` with ``h`` hypergeometric truncated to the level
    set, so ``K 1{C_ij >= x}`` is a finite sum.
    """
    c_now = len(rows[i] & rows[j])
    total = 0.0
    count = 0
    for a in (i, j):
        partner = rows[j] if a == i else rows[i]
        sa = rows[a]
        for b, sb in enumerate(rows):
            if b == i or b == j:
                continue
            count += 1
            only_a = sa - sb
            only_b = sb - sa
            if not only_a or not only_b:
                total += c_now >= x
                continue
            pool = only_a | only_b
            ka = len(only_a)
            hot = len(pool & partner)
            cold = len(pool) - hot
            base = len(sa & sb & partner)
            lo = max(level - base, 0, ka - cold)
            hi = min(hot, ka)
            num = den = 0
            for h in range(lo, hi + 1):
                w = math.comb(hot, h) * math.comb(cold, ka - h)
                den += w
                if base + h >= x:
                    num += w
            total += num / den
    return total / count if count else float(c_now >= x)


def _lee_whiteley(phi, eve, G, n_steps):
    """Estimate and standard error of ``G * mean(phi)`` from the genealogy.

    Lee & Whiteley (2018), eq. (4): with ``N`` particles, ``n`` resampling
    steps and ``E^a`` the initial ancestor ("Eve") of particle ``a``,
    ``var = p^2 - G^2 (N / (N - 1))^(n + 1) / N^2 sum_{E^a != E^b} phi_a
    phi_b``, an unbiased estimate of the variance for fixed levels.
    """
    import numpy as np

    N = len(phi)
    p = G * float(phi.mean())
    per_eve = np.bincount(eve, weights=phi, minlength=N)
    cross = float(phi.sum()) ** 2 - float((per_eve**2).sum())
    var = p**2 - G**2 * (N / (N - 1)) ** (n_steps + 1) * cross / N**2
    return p, math.sqrt(max(var, 0.0)), int((per_eve > 0).sum())


def _splitting_tail(
    initial,
    i,
    j,
    x_obs,
    *,
    rng,
    level_frac=0.1,
    n_moves=None,
    focus=0.5,
    block=0.2,
    fixed_levels=None,
):
    """Estimate ``P(C_ij >= x_obs)`` under the FDSM null by splitting.

    Parameters
    ----------
    initial : list of list of set
        ``N`` particles (row-set lists) drawn from the FDSM null, for example
        thinned states of the curveball chain.  They are copied.
    rng : numpy.random.Generator
        Source of randomness.
    level_frac : float
        Target survival fraction ``rho`` per level.  The next level is the
        largest integer ``l <= x_obs`` that at least ``rho * N`` particles
        reach, but at least one more than the current level.
    n_moves : int or None
        Moves (:func:`_move`) per particle after each resampling step
        (default: the number of rows).
    focus, block : float
        Passed to :func:`_move`.
    fixed_levels : sequence of int or None
        Use these levels (ending at ``x_obs``) instead of adaptive ones.
        The estimate is then unbiased when the initial particles are exact
        draws (Del Moral's Feynman-Kac unbiasedness, as in Lee & Whiteley
        2018, Proposition 1).

    Notes
    -----
    Resampling is multinomial and every particle keeps the index of its
    initial ancestor, so the variance estimate of Lee & Whiteley (2018)
    applies (:func:`_lee_whiteley`).  The plain estimate is the product of
    the survival fractions; the Rao-Blackwellized one replaces the last
    fraction by the mean of :func:`_rb_phi`, which has the same expectation
    because the one-trade kernel keeps the particles' law.  ``se_ideal`` is
    the asymptotic standard error of the idealized algorithm, ``p sqrt(sum_k
    (1 - p_k) / (N p_k))`` (Cerou et al. 2012, section 2.3).
    """
    import numpy as np

    prng = _Rand(rng)
    N = len(initial)
    particles = [[set(s) for s in rows] for rows in initial]
    n_rows = len(particles[0])
    others = [k for k in range(n_rows) if k != i and k != j]
    if n_moves is None:
        n_moves = n_rows
    eve = np.arange(N)
    scores = np.array([len(p[i] & p[j]) for p in particles])
    levels = []
    fractions = []
    current = -1
    step = 0
    fixed = list(fixed_levels) if fixed_levels is not None else None
    k_rho = min(max(math.ceil(level_frac * N), 1), N) - 1
    while True:
        if fixed is not None:
            nxt = fixed[step]
        else:
            q = int(np.partition(scores, N - 1 - k_rho)[N - 1 - k_rho])
            nxt = min(x_obs, max(q, current + 1))
        alive = scores >= nxt
        levels.append(int(nxt))
        fractions.append(float(alive.mean()))
        step += 1
        if nxt >= x_obs or not alive.any():
            break
        idx = np.flatnonzero(alive)
        pick = idx[rng.integers(0, len(idx), size=N)]
        particles = [[set(s) for s in particles[k]] for k in pick.tolist()]
        eve = eve[pick]
        for p in particles:
            _move(p, i, j, nxt, n_moves, prng, focus=focus, block=block, others=others)
        scores = np.array([len(p[i] & p[j]) for p in particles])
        current = nxt

    n_steps = len(fractions) - 1
    G = float(np.prod(fractions[:-1]))
    extinct = fractions[-1] == 0.0
    phi = (scores >= levels[-1]).astype(float)
    p_hat, se, eves = _lee_whiteley(phi, eve, G, n_steps)
    restrict = levels[-2] if n_steps else -1
    phi_rb = np.array([_rb_phi(p, i, j, restrict, levels[-1]) for p in particles])
    p_rb, se_rb, _ = _lee_whiteley(phi_rb, eve, G, n_steps)
    fr = np.array(fractions)
    se_ideal = (
        float("nan")
        if extinct
        else p_hat * math.sqrt(float(((1 - fr) / (N * fr)).sum()))
    )
    return SplittingResult(
        pvalue_rb=p_rb,
        se_rb=se_rb,
        pvalue=p_hat,
        se=se,
        se_ideal=se_ideal,
        levels=levels,
        fractions=fractions,
        n_moves=n_steps * N * n_moves,
        extinct=extinct,
        eve_survivors=eves,
    )


def _initial_particles(R, n_particles, rng, burn_in=None, thin=None):
    """Draw ``n_particles`` thinned states of the plain curveball chain."""
    from nx_backbone.bipartite import _fdsm_row_sets

    return [
        [set(s) for s in rows]
        for rows in _fdsm_row_sets(R, n_particles, rng, burn_in=burn_in, thin=thin)
    ]


def _refine_pairs(
    row_sets,
    keys,
    observed,
    n_agents,
    rng,
    *,
    n_particles,
    level_frac,
    n_moves,
    thin,
):
    """Splitting estimates of ``P(C_ij >= observed)`` for the pairs *keys*.

    ``row_sets`` is the last state of the plain FDSM curveball chain; the
    chain continues for ``n_particles`` more samples, ``thin`` trades apart,
    which are the initial particles of every refined pair.  A pair whose
    estimate is 0 (no particle reached the observed count) keeps its plain
    Monte Carlo p-value.

    Returns
    -------
    keys, pvalue, se, upper : ndarray
        Sorted pair keys ``i * n_agents + j`` and, for each, the estimate,
        its standard error and the one-sided upper confidence bound ``p
        exp(z se / p)`` (``z`` the standard normal quantile at
        :data:`nx_backbone._montecarlo.CONFIDENCE`), capped at 1.
    """
    import numpy as np
    from scipy.stats import norm

    from nx_backbone._montecarlo import CONFIDENCE
    from nx_backbone.bipartite import _curveball_trades

    keys = np.asarray(keys, dtype=np.int64)
    out_k, out_p, out_se, out_up = [], [], [], []
    if len(keys):
        state = [set(s) for s in row_sets]
        initial = []
        for _ in range(n_particles):
            _curveball_trades(state, thin, rng)
            initial.append([set(s) for s in state])
        z = float(norm.ppf(CONFIDENCE))
        for key, x in zip(keys.tolist(), np.asarray(observed).tolist()):
            i, j = divmod(key, n_agents)
            res = _splitting_tail(
                initial, i, j, int(x), rng=rng, level_frac=level_frac, n_moves=n_moves
            )
            p = min(res.pvalue_rb, 1.0)
            if not p > 0:
                continue
            out_k.append(key)
            out_p.append(p)
            out_se.append(min(res.se_rb, 1.0))
            out_up.append(min(1.0, p * math.exp(z * res.se_rb / p)))
    return (
        np.array(out_k, dtype=np.int64),
        np.array(out_p, dtype=float),
        np.array(out_se, dtype=float),
        np.array(out_up, dtype=float),
    )
