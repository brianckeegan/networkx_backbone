"""Validate ``fdsm(refine="splitting")`` against exact FDSM p-values (#77).

Each case is an incidence matrix whose rows 0 and 1 overlap strongly while
the other rows are sparse, so the exact null distribution of their
co-occurrence is cheap to enumerate
(:func:`nx_backbone._splitting._exact_cooccurrence_pmf`) and its upper tail
reaches 1e-9.  For every case and threshold ``x`` the script runs ``--reps``
independent splitting estimates of ``P(C_01 >= x)`` (each with its own
initial particles from the curveball chain) and reports the mean relative
bias, the relative standard deviation, the coverage of the 95% interval
``p +- 1.96 se`` and of the one-sided 95% upper bound ``p exp(1.645 se /
p)`` that ``fdsm`` stores, and the seconds per run.

Usage::

    PYTHONPATH=. python scripts/fdsm_splitting_validation.py --reps 100 --workers 3

The numbers in the ``fdsm`` Notes and in the changelog for #77 come from
this script with ``--reps 100`` and the default settings.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

from nx_backbone._splitting import (
    _exact_cooccurrence_pmf,
    _initial_particles,
    _splitting_tail,
)

# name: (other rows, columns, degree of rows 0/1, shared, other degrees, seed)
CASES = {
    "A": (8, 16, (7, 7), 7, (1, 4), 1),
    "B": (12, 24, (10, 9), 9, (1, 3), 2),
    "C": (16, 30, (12, 12), 12, (1, 3), 3),
    "D": (20, 30, (12, 10), 10, (1, 4), 4),
    "E": (24, 40, (14, 13), 13, (1, 3), 5),
    "F": (20, 32, (10, 13), 10, (1, 3), 6),
}
TARGETS = [
    ("A", 6),
    ("A", 7),
    ("B", 8),
    ("B", 9),
    ("C", 10),
    ("C", 11),
    ("C", 12),
    ("D", 9),
    ("D", 10),
    ("E", 11),
    ("E", 12),
    ("E", 13),
    ("F", 9),
    ("F", 10),
]


def case_matrix(name):
    n_others, n_cols, (d0, d1), shared, (lo, hi), seed = CASES[name]
    rng = np.random.default_rng(seed)
    R = np.zeros((2 + n_others, n_cols), dtype=int)
    perm = rng.permutation(n_cols)
    R[0, perm[:d0]] = 1
    R[1, perm[:shared]] = 1
    R[1, perm[d0 : d0 + d1 - shared]] = 1
    for r in range(2, 2 + n_others):
        R[r, rng.choice(n_cols, int(rng.integers(lo, hi + 1)), replace=False)] = 1
    return R


def exact_tail(R):
    pmf, _ = _exact_cooccurrence_pmf(R.sum(1), R.sum(0), 0, 1)
    return np.cumsum([float(f) for f in pmf][::-1])[::-1]


def one_run(job):
    name, x, rep, n_particles, level_frac = job
    R = case_matrix(name)
    rng = np.random.default_rng([ord(name), x, rep])
    t0 = time.perf_counter()
    init = _initial_particles(R, n_particles, rng)
    res = _splitting_tail(init, 0, 1, x, rng=rng, level_frac=level_frac)
    return name, x, res.pvalue_rb, res.se_rb, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--level-frac", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    exact = {}
    for name in sorted({n for n, _ in TARGETS}):
        exact[name] = exact_tail(case_matrix(name))
    jobs = [
        (n, x, r, args.particles, args.level_frac)
        for n, x in TARGETS
        for r in range(args.reps)
    ]
    with mp.Pool(args.workers) as pool:
        out = pool.map(one_run, jobs, chunksize=1)
    print(
        f"{'case':>4} {'x':>3} {'exact p':>9} {'bias':>12} {'rel. SD':>8} "
        f"{'95% CI':>7} {'upper':>6} {'s/run':>6}"
    )
    for name, x in TARGETS:
        rows = [o for o in out if o[0] == name and o[1] == x]
        p = np.array([o[2] for o in rows])
        se = np.array([o[3] for o in rows])
        ex = exact[name][x]
        rel = p / ex - 1
        bias_se = rel.std(ddof=1) / np.sqrt(len(rel))
        cover = np.mean(np.abs(p - ex) <= 1.96 * se)
        upper = np.where(p > 0, p * np.exp(1.6448536 * se / np.where(p > 0, p, 1)), 0)
        print(
            f"{name:>4} {x:>3} {ex:>9.2e} {100 * rel.mean():>+5.1f}%({100 * bias_se:3.1f})"
            f" {rel.std(ddof=1):>8.3f} {cover:>7.2f} {np.mean(upper >= ex):>6.2f}"
            f" {np.mean([o[4] for o in rows]):>6.2f}"
        )


if __name__ == "__main__":
    main()
