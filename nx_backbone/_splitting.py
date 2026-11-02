"""Adaptive multilevel splitting for FDSM co-occurrence tails (private).

Prototype for issue #77.  Nothing here is wired into :func:`fdsm` yet.

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
from dataclasses import dataclass, field

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
                new = dict()
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
                    acc + ((v, tv),) if tv else acc,
                )

        rec(0, t, 1, ())
        return out

    states = {(rows[i], rows[j], others): [1] + [0] * K}
    for k, c in enumerate(cols):
        left_cols = n_cols - k - 1
        nxt: dict = {}
        for (ri, rj, oth), poly in states.items():
            groups: dict = {}
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
                        if max((ni, nj) + key, default=0) > left_cols:
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
# Restricted curveball kernel
# =====================================================================


def _restricted_trade(rows, r1, r2, i, j, level, rng):
    """Heat-bath curveball trade between rows ``r1`` and ``r2`` on a level set.

    Rows ``r1`` and ``r2`` redistribute the columns they do not share,
    keeping their degrees, uniformly at random among the redistributions
    that keep ``C_ij = |rows[i] & rows[j]|`` at least ``level``.  Without the
    constraint this is the ordinary curveball trade (Strona et al. 2014),
    which is symmetric with the uniform law as its stationary law (Carstens
    2015); conditioning a heat-bath block update on the level set keeps the
    uniform law on the level set invariant and reversible.

    Only trades that involve exactly one of ``i`` and ``j`` can change
    ``C_ij``: a trade between ``i`` and ``j`` keeps their common columns.
    For a trade between ``i`` and ``r`` the number ``h`` of pool columns
    that row ``i`` receives from ``rows[j]`` is hypergeometric; it is drawn
    from that law truncated to ``h >= level - |common & rows[j]|``.
    Returns the new ``C_ij`` (or ``None`` when it cannot change).
    """
    s1 = rows[r1]
    s2 = rows[r2]
    only1 = s1 - s2
    if not only1:
        return None
    only2 = s2 - s1
    if not only2:
        return None
    common = s1 & s2
    k1 = len(only1)
    pool = list(only1 | only2)
    touches = (r1 == i or r1 == j) + (r2 == i or r2 == j)
    if touches != 1:
        rng.shuffle(pool)
        rows[r1] = common | set(pool[:k1])
        rows[r2] = common | set(pool[k1:])
        return None
    # Orient so that ``a`` is the constrained row (i or j) and ``b`` its partner.
    if r1 in (i, j):
        a, b, ka = r1, r2, k1
    else:
        a, b, ka = r2, r1, len(pool) - k1
    partner = rows[j] if a == i else rows[i]
    hot = [c for c in pool if c in partner]
    cold = [c for c in pool if c not in partner]
    base = len(common & partner)
    h_min = max(level - base, 0, ka - len(cold))
    h_max = min(len(hot), ka)
    if h_min > h_max:  # cannot happen when the state is in the level set
        raise AssertionError("particle left its level set")
    if h_min == h_max:
        h = h_min
    else:
        hs = list(range(h_min, h_max + 1))
        w = [math.comb(len(hot), h) * math.comb(len(cold), ka - h) for h in hs]
        tot = sum(w)
        u = rng.random() * tot
        acc = 0
        h = hs[-1]
        for hh, ww in zip(hs, w):
            acc += ww
            if u < acc:
                h = hh
                break
    rng.shuffle(hot)
    rng.shuffle(cold)
    new_a = set(hot[:h]) | set(cold[: ka - h])
    new_b = set(hot[h:]) | set(cold[ka - h :])
    rows[a] = common | new_a
    rows[b] = common | new_b
    return base + h


def _pick_pair(n, i, j, focus, rng):
    """A row pair from a fixed, state-independent distribution."""
    if rng.random() < focus:
        r1 = i if rng.random() < 0.5 else j
    else:
        r1 = int(rng.integers(0, n))
    r2 = int(rng.integers(0, n - 1))
    r2 += r2 >= r1
    return r1, r2


def _plain_trade(rows, r1, r2, rng):
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
    rng.shuffle(pool)
    k1 = len(only1)
    rows[r1] = common | set(pool[:k1])
    rows[r2] = common | set(pool[k1:])


def _double_trade(rows, i, j, level, focus, rng):
    """Metropolis move: two unconstrained trades, kept only if C_ij >= level.

    One curveball trade with a state-independent pair is a symmetric
    proposal, so two in a row are too; accepting exactly when the result is
    in the level set leaves the uniform law on it invariant.  The
    intermediate state may leave the level set, which connects level sets
    that single restricted trades cannot (for example the top level, where
    ``rows[i]`` and ``rows[j]`` must change together).
    """
    n = len(rows)
    a1, a2 = _pick_pair(n, i, j, focus, rng)
    b1, b2 = _pick_pair(n, i, j, focus, rng)
    touched = {a1, a2, b1, b2}
    saved = {r: rows[r] for r in touched}
    _plain_trade(rows, a1, a2, rng)
    _plain_trade(rows, b1, b2, rng)
    if len(rows[i] & rows[j]) < level:
        for r, s in saved.items():
            rows[r] = s
        return False
    return True


def _pair_block_trade(rows, i, j, rng):
    """Trade the shared columns of ``i`` and ``j`` with two other rows.

    For two other rows ``r`` and ``s`` (uniform, state-independent), the
    columns where rows ``(i, j, r, s)`` read ``(1, 1, 0, 0)`` or
    ``(0, 0, 1, 1)`` are redistributed uniformly, keeping how many read
    ``(1, 1, 0, 0)``.  Every row and column sum and ``C_ij`` are unchanged,
    and the set of redistributions is the same from each of its members, so
    this heat-bath move is symmetric.  It is the move that changes
    ``rows[i]`` and ``rows[j]`` together, which single trades cannot do
    without lowering ``C_ij`` when ``rows[i] == rows[j]``.
    """
    n = len(rows)
    if n < 4:
        return
    others = [k for k in range(n) if k != i and k != j]
    a, b = rng.choice(len(others), size=2, replace=False).tolist()
    r, s = others[a], others[b]
    si, sj, sr, ss = rows[i], rows[j], rows[r], rows[s]
    A = (si & sj) - sr - ss
    if not A:
        return
    D = (sr & ss) - si - sj
    if not D:
        return
    pool = list(A | D)
    rng.shuffle(pool)
    new_a = set(pool[: len(A)])
    new_d = set(pool[len(A) :])
    rows[i] = (si - A) | new_a
    rows[j] = (sj - A) | new_a
    rows[r] = (sr - D) | new_d
    rows[s] = (ss - D) | new_d


def _move(rows, i, j, level, n_trades, focus, rng, double=0.0, block=0.2):
    """Apply ``n_trades`` moves that keep ``C_ij >= level``.

    Each move is a :func:`_pair_block_trade` with probability ``block``, a
    :func:`_double_trade` with probability ``double``, and otherwise one
    heat-bath :func:`_restricted_trade`.  All pick rows from fixed,
    state-independent distributions (for trades: with probability ``focus``
    one of ``i``, ``j`` and a uniform partner, otherwise a uniform pair), and
    a state-independent mixture of kernels reversible for the uniform law on
    the level set is reversible for it.
    """
    n = len(rows)
    for _ in range(n_trades):
        u = rng.random()
        if u < block:
            _pair_block_trade(rows, i, j, rng)
        elif u < block + double:
            _double_trade(rows, i, j, level, focus, rng)
        else:
            r1, r2 = _pick_pair(n, i, j, focus, rng)
            _restricted_trade(rows, r1, r2, i, j, level, rng)


# =====================================================================
# Splitting estimator
# =====================================================================


@dataclass
class SplittingResult:
    """Output of :func:`_splitting_tail`."""

    pvalue: float
    se: float
    se_ideal: float
    levels: list[int]
    fractions: list[float]
    n_trades: int
    extinct: bool = False
    eve_survivors: int = 0
    pvalue_rb: float = float("nan")
    se_rb: float = float("nan")
    extra: dict = field(default_factory=dict)


def _cooc(rows, i, j):
    return len(rows[i] & rows[j])


def _rb_phi(rows, i, j, level, x):
    """``K 1{C_ij >= x}`` at one particle, for a restricted one-trade kernel.

    ``K`` picks ``a`` in ``{i, j}`` and a partner ``b`` outside ``{i, j}``
    uniformly and applies the heat-bath :func:`_restricted_trade` for the
    level set ``{C_ij >= level}``.  ``K`` leaves the uniform law on that
    level set invariant, so averaging ``K 1{C_ij >= x}`` over particles
    estimates the same probability as the indicator itself, with smaller
    variance (a Rao-Blackwellization of the last stage).  For each ``(a, b)``
    the new count is ``base + h`` with ``h`` hypergeometric truncated to the
    level set, so ``K 1{C_ij >= x}`` is a finite sum.
    """
    n = len(rows)
    c_now = len(rows[i] & rows[j])
    total = 0.0
    count = 0
    for a in (i, j):
        partner = rows[j] if a == i else rows[i]
        sa = rows[a]
        for b in range(n):
            if b == i or b == j:
                continue
            count += 1
            sb = rows[b]
            only_a = sa - sb
            only_b = sb - sa
            if not only_a or not only_b:
                total += c_now >= x
                continue
            common = sa & sb
            pool = only_a | only_b
            ka = len(only_a)
            hot = len(pool & partner)
            cold = len(pool) - hot
            base = len(common & partner)
            h_min = max(level - base, 0, ka - cold)
            h_max = min(hot, ka)
            num = 0
            den = 0
            for h in range(h_min, h_max + 1):
                w = math.comb(hot, h) * math.comb(cold, ka - h)
                den += w
                if base + h >= x:
                    num += w
            total += num / den
    return total / count


def _splitting_tail(
    initial,
    i,
    j,
    x_obs,
    *,
    level_frac=0.1,
    n_moves=None,
    focus=0.5,
    double=0.0,
    block=0.2,
    rng,
    fixed_levels=None,
    max_levels=10_000,
):
    """Estimate ``P(C_ij >= x_obs)`` under the FDSM null by splitting.

    Parameters
    ----------
    initial : list of list of set
        ``N`` particles (row-set lists) drawn from the FDSM null, for example
        thinned states of the curveball chain.  They are copied.
    level_frac : float
        Target survival fraction ``rho`` per level.  The next level is the
        largest integer ``l <= x_obs`` with at least ``rho * N`` particles at
        or above it, but at least one more than the current level.
    n_moves : int or None
        Restricted curveball trades per particle after each resampling
        (default ``2 * n_rows``).
    focus : float
        Share of trades that involve row ``i`` or ``j``.
    double, block : float
        Shares of moves that are :func:`_double_trade` proposals and
        :func:`_pair_block_trade` moves.
    fixed_levels : sequence of int or None
        Use these levels instead of adaptive ones (the estimator is then
        unbiased when the initial particles are exact draws).

    Notes
    -----
    Resampling is multinomial, and every particle keeps the index of its
    initial ancestor (its "Eve" index), so the single-run variance estimator
    of Lee & Whiteley (2018), eq. (4) with ``phi`` the final indicator, can
    be computed:  ``var = p^2 - G^2 (N/(N-1))^(n+1) / N^2 * sum_{E^a != E^b}
    phi_a phi_b`` where ``G`` is the product of the survival fractions
    before the last level.  ``se_ideal`` is the asymptotic standard error of
    the idealized algorithm with exact conditional resampling, ``p *
    sqrt(sum_k (1 - p_k) / (N p_k))`` (Cerou et al. 2012, section 2.3).
    """
    import numpy as np

    N = len(initial)
    particles = [[set(s) for s in rows] for rows in initial]
    n_rows = len(particles[0])
    if n_moves is None:
        n_moves = 2 * n_rows
    eve = np.arange(N)
    scores = np.array([_cooc(p, i, j) for p in particles])
    levels: list[int] = []
    fractions: list[float] = []
    current = -1
    trades = 0
    step = 0
    fixed = list(fixed_levels) if fixed_levels is not None else None
    while True:
        if fixed is not None:
            nxt = fixed[step]
        else:
            nxt = x_obs
            # largest level <= x_obs with at least rho*N particles at or above
            need = level_frac * N
            srt = np.sort(scores)[::-1]
            kq = int(math.ceil(need)) - 1
            q = int(srt[min(max(kq, 0), N - 1)])
            nxt = min(x_obs, max(q, current + 1))
        alive = scores >= nxt
        frac = float(alive.mean())
        levels.append(int(nxt))
        fractions.append(frac)
        step += 1
        if nxt >= x_obs or frac == 0.0:
            break
        if step >= max_levels:
            raise RuntimeError("too many levels")
        # multinomial resampling of the survivors
        idx = np.flatnonzero(alive)
        pick = idx[rng.integers(0, len(idx), size=N)]
        new_particles = []
        for k in pick.tolist():
            new_particles.append([set(s) for s in particles[k]])
        particles = new_particles
        eve = eve[pick]
        for p in particles:
            _move(p, i, j, nxt, n_moves, focus, rng, double, block)
        trades += N * n_moves
        scores = np.array([_cooc(p, i, j) for p in particles])
        current = nxt

    p_hat = float(np.prod(fractions))
    extinct = fractions[-1] == 0.0
    # Lee & Whiteley (2018) single-run variance estimate
    phi = scores >= levels[-1]
    n_steps = len(fractions) - 1  # resampling steps performed
    G = float(np.prod(fractions[:-1])) if n_steps else 1.0
    S = int(phi.sum())
    counts = np.bincount(eve[phi], minlength=N) if S else np.zeros(N, dtype=int)
    cross = S * S - int((counts.astype(np.int64) ** 2).sum())
    var = p_hat**2 - G**2 * (N / (N - 1)) ** (n_steps + 1) * cross / N**2
    se = math.sqrt(max(var, 0.0))
    # Rao-Blackwellized last stage (same particles, phi = K 1{C >= x})
    restrict = levels[-2] if n_steps else -1
    phi_rb = np.array([_rb_phi(p, i, j, restrict, levels[-1]) for p in particles])
    p_rb = G * float(phi_rb.mean())
    per_eve = np.bincount(eve, weights=phi_rb, minlength=N)
    cross_rb = float(phi_rb.sum()) ** 2 - float((per_eve**2).sum())
    var_rb = p_rb**2 - G**2 * (N / (N - 1)) ** (n_steps + 1) * cross_rb / N**2
    se_rb = math.sqrt(max(var_rb, 0.0))
    fr = np.array(fractions)
    if extinct:
        se_ideal = float("nan")
    else:
        se_ideal = p_hat * math.sqrt(float(((1 - fr) / (N * fr)).sum()))
    return SplittingResult(
        pvalue=p_hat,
        se=se,
        se_ideal=se_ideal,
        levels=levels,
        fractions=fractions,
        n_trades=trades,
        extinct=extinct,
        eve_survivors=int((counts > 0).sum()),
        pvalue_rb=p_rb,
        se_rb=se_rb,
    )


def _initial_particles(R, n_particles, rng, burn_in=None, thin=None):
    """Draw ``n_particles`` thinned states of the plain curveball chain."""
    from nx_backbone.bipartite import _fdsm_row_sets

    return [
        [set(s) for s in rows]
        for rows in _fdsm_row_sets(R, n_particles, rng, burn_in=burn_in, thin=thin)
    ]
