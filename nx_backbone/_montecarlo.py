"""Monte Carlo resolution of permutation-style p-values.

A Monte Carlo p-value estimated as ``(count + 1) / (trials + 1)`` from
``trials`` null samples (``count`` of them at least as extreme as the data)
can never be smaller than ``1 / (trials + 1)``, its **floor**; a two-sided
p-value ``min(1, 2 * min(upper, lower))`` built from two such tails has floor
``2 / (trials + 1)``.  These helpers describe that resolution:

- :func:`pvalue_floor` gives the floor;
- :func:`pvalue_resolution` gives, for each estimated p-value, the plug-in
  binomial standard error and a one-sided Clopper-Pearson upper confidence
  bound on the true p-value;
- :func:`min_adjusted_pvalue` and :func:`trials_needed` say whether, after a
  multiple-testing correction over ``m`` tests, any p-value at the floor can
  pass ``alpha``, and how many trials it would take;
- :func:`warn_if_unreachable` warns when it cannot.

:func:`~nx_backbone.fdsm` uses them (``fdsm_pvalue_se``,
``fdsm_pvalue_upper``, ``H.graph["fdsm_pvalue_floor"]``), as do
:func:`~nx_backbone.backbone_from_projection` and
:func:`~nx_backbone.threshold_filter`, which warn.  They take only numbers,
so a refined estimator (issue #77) can reuse them for the pairs it does not
refine.

Everything except :func:`pvalue_resolution` is pure Python, so
:func:`~nx_backbone.threshold_filter` keeps working without NumPy.
"""

from __future__ import annotations

import functools
import math
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = [
    "CONFIDENCE",
    "pvalue_floor",
    "pvalue_resolution",
    "min_adjusted_pvalue",
    "trials_needed",
    "warn_if_unreachable",
]

#: Confidence level of the one-sided upper bound from :func:`pvalue_resolution`.
CONFIDENCE = 0.95

# How each correction of ``adjust_pvalues`` scales the smallest p-value when
# all m p-values sit at the floor: by m (Bonferroni, Holm), by the harmonic
# number H_m (Benjamini-Yekutieli), or not at all (none, Hochberg, BH).
_FAMILYWISE = ("bonferroni", "holm")
_UNSCALED = ("none", "hochberg", "bh", "fdr")


def pvalue_floor(trials: int, sides: int = 1) -> float:
    """Smallest p-value a ``(count + 1) / (trials + 1)`` estimate can take.

    Parameters
    ----------
    trials : int
        Number of Monte Carlo samples (at least 1).
    sides : {1, 2}, optional (default=1)
        1 for a one-tailed p-value, 2 for the two-sided
        ``min(1, 2 * min(upper, lower))``.

    Returns
    -------
    floor : float
        ``sides / (trials + 1)``, computed as the estimators compute it (so
        a p-value at the floor compares equal to it).
    """
    return min(1.0, sides * (1 / (trials + 1)))


@functools.lru_cache(maxsize=8)
def _harmonic(m):
    """``sum(1 / k for k = 1..m)`` in the order ``adjust_pvalues`` sums it."""
    return sum(1.0 / k for k in range(1, m + 1))


def min_adjusted_pvalue(floor: float, m: int, mtc: str) -> float:
    """Smallest adjusted p-value attainable when no p-value is below *floor*.

    For every correction of :func:`~nx_backbone.adjust_pvalues` the adjusted
    p-values of ``m`` p-values that are all at least *floor* are at least
    this value, which is attained:

    - ``"bonferroni"`` and ``"holm"``: ``m * floor``, attained by one p-value
      at the floor;
    - ``"by"``: ``H_m * floor`` (``H_m`` the ``m``-th harmonic number),
      attained when all ``m`` p-values are at the floor;
    - ``"bh"``/``"fdr"`` and ``"hochberg"``: ``floor``, attained when all
      ``m`` p-values are at the floor;
    - ``"none"``: ``floor``.

    The arithmetic is the one ``adjust_pvalues`` uses, so the result is
    ``< alpha`` exactly when a p-value at the floor could pass ``alpha``.

    Parameters
    ----------
    floor : float
        Smallest possible p-value (see :func:`pvalue_floor`).
    m : int
        Number of tests the correction is applied to.
    mtc : str
        Correction name, as for :func:`~nx_backbone.adjust_pvalues`.

    Returns
    -------
    p : float
        The smallest attainable adjusted p-value, at most 1.

    Raises
    ------
    ValueError
        If *mtc* is not a known correction.
    """
    from nx_backbone.filters import _MTC_METHODS

    method = mtc.lower()
    if method not in _MTC_METHODS:
        raise ValueError(f"method must be one of {_MTC_METHODS}, got {mtc!r}")
    if m < 1 or method in _UNSCALED:
        scaled = floor
    elif method in _FAMILYWISE:
        scaled = m * floor
    else:  # "by"
        scaled = _harmonic(m) * floor
    return min(1.0, max(0.0, scaled))


def trials_needed(m: int, alpha: float, mtc: str, sides: int = 1) -> int | None:
    """Fewest trials whose floor lets a corrected p-value fall below *alpha*.

    This is the smallest ``trials >= 1`` with
    ``min_adjusted_pvalue(pvalue_floor(trials, sides), m, mtc) < alpha``,
    about ``m * sides / alpha`` for Bonferroni and Holm, ``H_m * sides /
    alpha`` for BY and ``sides / alpha`` otherwise.  It is a necessary
    condition only: a pair whose true p-value is near the corrected
    threshold needs many more trials to be classified reliably (see
    :func:`pvalue_resolution`).

    Parameters
    ----------
    m : int
        Number of tests the correction is applied to.
    alpha : float
        Significance level (pairs pass when the adjusted p-value is
        ``< alpha``).
    mtc : str
        Correction name, as for :func:`~nx_backbone.adjust_pvalues`.
    sides : {1, 2}, optional (default=1)
        As for :func:`pvalue_floor`.

    Returns
    -------
    trials : int or None
        The number of trials, or ``None`` if no number of trials suffices
        (``alpha <= 0``).
    """
    if not alpha > 0:
        return None
    method = mtc.lower()
    if m < 1 or method in _UNSCALED:
        factor = 1.0
    elif method in _FAMILYWISE:
        factor = float(m)
    else:
        factor = _harmonic(m)

    def ok(t):
        return min_adjusted_pvalue(pvalue_floor(t, sides), m, method) < alpha

    # factor * sides / (t + 1) < alpha  <=>  t > factor * sides / alpha - 1;
    # start there and step past floating-point rounding at the boundary.
    t = max(1, math.floor(factor * sides / alpha))
    while not ok(t):
        t += 1
    while t > 1 and ok(t - 1):
        t -= 1
    return t


def pvalue_resolution(
    pvalue: npt.ArrayLike, trials: int, sides: int = 1, level: float = CONFIDENCE
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo standard error and upper bound of estimated p-values.

    *pvalue* holds estimates ``(count + 1) / (trials + 1)`` (``sides=1``) or
    ``min(1, 2 * q)`` with ``q`` such an estimate of the smaller tail
    (``sides=2``).  The tail count behind each estimate is a binomial
    ``Binomial(trials, pi)`` draw if the samples are independent, where
    ``pi`` is the true tail probability.

    - The standard error is the plug-in binomial one, ``sqrt(q (1 - q) /
      trials)`` times *sides*, with ``q = pvalue / sides`` (``q = 1/2`` for
      a two-sided p-value capped at 1).
    - The upper bound is the one-sided Clopper-Pearson bound on ``pi`` at
      confidence *level*, the *level* quantile of ``Beta(count + 1,
      trials - count)`` (1 when ``count == trials``), times *sides* and
      capped at 1.  It bounds the true (two-sided) p-value with probability
      at least *level* for every ``pi``.

    Parameters
    ----------
    pvalue : array_like
        Estimated p-values.
    trials : int
        Number of Monte Carlo samples behind them.
    sides : {1, 2}, optional (default=1)
        As for :func:`pvalue_floor`.
    level : float, optional (default=0.95)
        Confidence level of the upper bound.

    Returns
    -------
    se, upper : ndarray
        Standard errors and upper confidence bounds, both in ``[0, 1]``.
    """
    import numpy as np
    from scipy.stats import beta

    p = np.asarray(pvalue, dtype=float)
    # A two-sided p-value capped at 1 has a smaller tail of at least 1/2.
    q_all = np.clip(p / sides, 0.0, 0.5 if sides == 2 else 1.0)
    q, inverse = np.unique(q_all, return_inverse=True)
    se = np.minimum(1.0, sides * np.sqrt(q * (1.0 - q) / trials))
    count = np.clip(np.rint(q * (trials + 1)).astype(np.int64) - 1, 0, trials)
    upper = np.ones(len(q))
    below = count < trials
    upper[below] = beta.ppf(level, count[below] + 1, trials - count[below])
    upper = np.minimum(1.0, sides * upper)
    shape = p.shape
    return se[inverse].reshape(shape), upper[inverse].reshape(shape)


def _fmt(x):
    return f"{x:.3g}"


def warn_if_unreachable(
    name: str,
    *,
    trials: int | None,
    floor: float,
    m: int,
    alpha: float,
    mtc: str,
) -> bool:
    """Warn when the floor leaves no test able to pass *alpha*.

    The ``RuntimeWarning`` fires exactly when
    ``min_adjusted_pvalue(floor, m, mtc) >= alpha`` (and ``alpha > 0``):
    then no p-value, whatever the data, can pass the correction.  It names
    the number of trials that would be needed when *trials* is known.

    Parameters
    ----------
    name : str
        What produced the p-values, for the message (e.g. ``"fdsm"``).
    trials : int or None
        Number of Monte Carlo samples, if known.
    floor : float
        Smallest possible p-value (see :func:`pvalue_floor`).
    m : int
        Number of tests the correction is applied to.
    alpha : float
        Significance level.
    mtc : str
        Correction name, as for :func:`~nx_backbone.adjust_pvalues`.

    Returns
    -------
    warned : bool
        Whether the warning was emitted.
    """
    from nx_backbone.statistical import _external_stacklevel

    if not alpha > 0:
        return False
    best = min_adjusted_pvalue(floor, m, mtc)
    if best < alpha:
        return False
    method = mtc.lower()
    if trials is not None:
        sides = max(1, round(floor * (trials + 1)))
        source = f"with trials={trials} no p-value is below {sides}/(trials + 1)"
    else:
        sides = 1
        source = "no p-value is below its Monte Carlo floor"
    if method == "none":
        how = f"which is not below alpha = {alpha:g}"
    else:
        how = (
            f"so after the {method!r} correction for {m} tests the smallest "
            f"attainable adjusted p-value is {_fmt(best)}, not below "
            f"alpha = {alpha:g}"
        )
    advice = ""
    needed = trials_needed(m, alpha, method, sides) if trials is not None else None
    if needed is not None:
        refine = (
            "refine='splitting', which re-estimates the pairs at the floor, "
            if name == "fdsm"
            else ""
        )
        advice = (
            f" Use at least trials={needed} for the floor to allow "
            "significance, and more to resolve p-values near the threshold, "
            f"or {refine}or sdsm, whose analytic p-values have no floor."
        )
    warnings.warn(
        f"{name}: no pair can be significant, whatever the data: {source} "
        f"= {_fmt(floor)}, {how}.{advice}",
        RuntimeWarning,
        stacklevel=_external_stacklevel(),
    )
    return True
