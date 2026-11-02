"""Two-state hidden Markov model for edge activity (array-only engine).

This private module is the numerical core of ``edge_persistence_hmm``
(issue #73).  It works on arrays only and knows nothing about graphs: the
public function in :mod:`nx_backbone.temporal` turns a temporal network
into an ``(E, T)`` observation array plus a mask and calls these functions.
numpy is imported lazily inside each function.

Model
-----
Each of ``E`` edges has a hidden state ``s_t`` in {0 (inactive), 1 (active)}
at each of ``T`` steps.  All edges share one set of parameters:

* **Transitions.** ``a01 = P(s_t = 1 | s_{t-1} = 0)`` and
  ``a10 = P(s_t = 0 | s_{t-1} = 1)``.
* **Initial distribution.** ``pi1 = P(s_1 = 1)``.  By default it is the
  stationary probability of the chain, ``pi1 = a01 / (a01 + a10)``, which
  treats the chain as having run long before the first step; when
  ``a01 = a10 = 0`` the chain is frozen, has no unique stationary
  distribution, and ``pi1 = 1/2`` is used.  ``fit_hmm(initial="fit")``
  estimates ``pi1`` as a free parameter instead, and a ``"pi1"`` entry in the
  parameters fixes it.
* **Emissions, ``observation="binary"``.**  A flag ``o_t`` in {0, 1} with
  ``P(o = 1 | s = 0) = alpha`` (false-positive rate) and
  ``P(o = 1 | s = 1) = 1 - beta`` (``beta`` is the miss rate).
* **Emissions, ``observation="pvalue"``.**  A p-value ``o_t`` in [0, 1] that
  is Uniform(0, 1) under ``s = 0`` and Beta(``a``, 1), with density
  ``a * p**(a - 1)``, under ``s = 1``.  With ``0 < a < 1`` the active state
  concentrates p-values near 0.  This beta-uniform emission is a modeling
  choice of this library; no source for it was checked, so none is cited.
* **Unknown steps.**  Where ``mask`` is False (the edge is not eligible at
  that step), the emission is 1 for both states: the hidden chain still runs
  through the step, but the step carries no evidence.  Observation values at
  unknown steps are ignored; they may be NaN.

Identifiability
---------------
Swapping the labels of the two states gives the same likelihood, so the
binary model imposes ``alpha < 1 - beta`` (an active edge is observed more
often than an inactive one).  When every swap-affected parameter (``a01``,
``a10``, ``alpha``, ``beta`` and a fitted ``pi1``) is free, an iterate that
violates the constraint is relabeled: ``(a01, a10, alpha, beta, pi1)``
becomes ``(a10, a01, 1 - beta, 1 - alpha, 1 - pi1)``.  When some of them are
fixed, relabeling would change a fixed value, so the free emission
parameters are instead pulled back inside the constraint, and the result
records ``constrained=True`` with a :class:`RuntimeWarning`.  The p-value
model has no label symmetry (state 0 is exactly uniform); there ``a`` is
kept in ``(0, 1)``.

Algorithms
----------
* :func:`forward_backward`: scaled forward-backward recursions, vectorized
  over edges (a Python loop over the ``T`` steps only).  It returns the
  posterior ``P(s_t = 1 | o_{1:T})`` and each edge's log-likelihood.  An edge
  whose observations have probability zero under the parameters gets a
  log-likelihood of ``-inf`` and NaN posteriors.
* :func:`viterbi`: the most likely state path, in log space and vectorized
  over edges.  Ties are broken deterministically toward the inactive state:
  among equally likely final states, and among equally likely predecessors
  in the backtrace, state 0 wins.  The returned path is therefore the most
  likely path that is smallest when paths are compared from the last step
  backward (0 before 1).
* :func:`fit_hmm`: Baum-Welch (EM) with the expected counts pooled over all
  edges, so every edge shares one parameter set.  Any subset of parameters
  can be fixed.  Convergence is declared when no free parameter moves by
  more than ``tol`` in one iteration; otherwise a :class:`RuntimeWarning` is
  emitted after ``max_iter`` iterations.  With the default stationary
  initial distribution the transition update is the usual Baum-Welch ratio
  of expected counts, which ignores the dependence of ``pi1`` on
  ``(a01, a10)``; the update is then not an exact M-step, so the
  log-likelihood is not guaranteed to increase at every iteration (the
  initial-step term is one of ``T`` and the effect is small).  With
  ``initial="fit"`` or a fixed ``pi1`` the M-step is exact.
"""

import math
import warnings

import networkx as nx

__all__: list[str] = []

OBSERVATIONS = ("binary", "pvalue")
_TRANSITION_NAMES = ("a01", "a10")
_EMISSION_NAMES = {"binary": ("alpha", "beta"), "pvalue": ("a",)}
_INITIALS = ("stationary", "fit")

# Distance kept from the identifiability boundary (alpha + beta = 1, a = 1)
# when a free parameter has to be pulled back inside it.
_MARGIN = 1e-6

# Starting values of free parameters when the caller gives none.
_DEFAULT_START = {
    "a01": 0.1,
    "a10": 0.1,
    "alpha": 0.1,
    "beta": 0.1,
    "a": 0.2,
    "pi1": 0.5,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_observation(observation):
    if observation not in OBSERVATIONS:
        raise nx.NetworkXError(
            f"observation must be one of {OBSERVATIONS}, not {observation!r}."
        )


def _param_names(observation):
    return _TRANSITION_NAMES + _EMISSION_NAMES[observation]


def _check_values(params, observation):
    """Raise unless every given parameter is a finite number in range."""
    allowed = (*_param_names(observation), "pi1")
    for name, value in params.items():
        if name not in allowed:
            raise nx.NetworkXError(
                f"Unknown HMM parameter {name!r} for observation="
                f"{observation!r}; expected some of {allowed}."
            )
        if name == "pi1" and value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise nx.NetworkXError(
                f"HMM parameter {name!r} must be a number, not {value!r}."
            ) from None
        if name == "a":
            if not 0.0 < value < 1.0:
                raise nx.NetworkXError(
                    f"HMM parameter 'a' must be in (0, 1), not {value}."
                )
        elif not 0.0 <= value <= 1.0:
            raise nx.NetworkXError(
                f"HMM parameter {name!r} must be in [0, 1], not {value}."
            )


def _check_complete(params, observation):
    """Return a float copy of *params*, which must name every parameter."""
    _check_values(params, observation)
    missing = [n for n in _param_names(observation) if n not in params]
    if missing:
        raise nx.NetworkXError(f"Missing HMM parameters: {missing}.")
    out = {n: float(params[n]) for n in _param_names(observation)}
    if params.get("pi1") is not None:
        out["pi1"] = float(params["pi1"])
    if observation == "binary" and not out["alpha"] < 1.0 - out["beta"]:
        raise nx.NetworkXError(
            "HMM parameters must satisfy alpha < 1 - beta (an active edge is "
            f"observed more often than an inactive one); got alpha="
            f"{out['alpha']}, beta={out['beta']}."
        )
    return out


def _prepare(obs, mask, observation):
    """Validate and return ``(obs, mask)`` as float and bool ``(E, T)`` arrays.

    Unknown steps (``mask`` False, or a non-finite value) get value 0 so that
    arithmetic on them is harmless; their emissions are set to 1 later.
    """
    import numpy as np

    _check_observation(observation)
    o = np.array(obs, dtype=float)
    if o.ndim != 2:
        raise nx.NetworkXError(
            f"obs must be a 2-D (edges, steps) array, not {o.ndim}-D."
        )
    if mask is None:
        m = np.ones(o.shape, dtype=bool)
    else:
        m = np.array(mask, dtype=bool)
        if m.shape != o.shape:
            raise nx.NetworkXError(
                f"mask has shape {m.shape}, but obs has shape {o.shape}."
            )
    m &= np.isfinite(o)
    o[~m] = 0.0
    known = o[m]
    if observation == "binary":
        if not np.all((known == 0.0) | (known == 1.0)):
            raise nx.NetworkXError(
                "observation='binary' needs 0/1 values at known steps."
            )
    elif not np.all((known >= 0.0) & (known <= 1.0)):
        raise nx.NetworkXError("observation='pvalue' needs values in [0, 1].")
    return o, m


# ---------------------------------------------------------------------------
# Model pieces
# ---------------------------------------------------------------------------


def _initial_pi1(p):
    if p.get("pi1") is not None:
        return p["pi1"]
    total = p["a01"] + p["a10"]
    return 0.5 if total == 0.0 else p["a01"] / total


def _transition_matrix(p):
    import numpy as np

    return np.array([[1.0 - p["a01"], p["a01"]], [p["a10"], 1.0 - p["a10"]]])


def _log_p(o, m):
    """``log(o)`` at known steps (0 elsewhere), with p = 0 clipped to tiny."""
    import numpy as np

    return np.where(m, np.log(np.clip(o, np.finfo(float).tiny, 1.0)), 0.0)


def _emissions(o, m, p, observation):
    """Return the ``(E, T, 2)`` emission likelihoods ``P(o_t | s_t = k)``."""
    import numpy as np

    B = np.ones((*o.shape, 2))
    if observation == "binary":
        B[..., 0] = np.where(o == 1.0, p["alpha"], 1.0 - p["alpha"])
        B[..., 1] = np.where(o == 1.0, 1.0 - p["beta"], p["beta"])
    else:
        a = p["a"]
        B[..., 1] = a * np.exp((a - 1.0) * _log_p(o, m))
    B[~m] = 1.0
    return B


def _forward_backward(B, A, pi, *, want_xi=False):
    """Scaled forward-backward on emission array *B* of shape ``(E, T, 2)``.

    Returns ``(gamma, loglik, xi)``: posterior state probabilities
    ``(E, T, 2)``, per-edge log-likelihoods ``(E,)`` and, if requested, the
    expected transition counts summed over steps ``(E, 2, 2)`` (else None).
    """
    import numpy as np

    E, T, _ = B.shape
    fwd = np.empty((E, T, 2))
    c = np.ones((E, T))
    bad = np.zeros(E, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for t in range(T):
            a = pi[None, :] * B[:, 0] if t == 0 else (fwd[:, t - 1] @ A) * B[:, t]
            ct = a.sum(axis=1)
            zero = ct <= 0.0
            bad |= zero
            ct[zero] = 1.0
            c[:, t] = ct
            fwd[:, t] = a / ct[:, None]
        bwd = np.ones((E, T, 2))
        for t in range(T - 2, -1, -1):
            bwd[:, t] = ((B[:, t + 1] * bwd[:, t + 1]) @ A.T) / c[:, t + 1, None]
        gamma = fwd * bwd
        # Guard against rounding: rows are probability vectors.
        gamma /= gamma.sum(axis=2, keepdims=True)
        loglik = np.log(c).sum(axis=1)
        xi = None
        if want_xi:
            if T > 1:
                nxt = (B[:, 1:] * bwd[:, 1:]) / c[:, 1:, None]
                xi = np.einsum("eti,ij,etj->eij", fwd[:, :-1], A, nxt)
            else:
                xi = np.zeros((E, 2, 2))
    loglik[bad] = -math.inf
    gamma[bad] = math.nan
    if xi is not None:
        xi[bad] = math.nan
    return gamma, loglik, xi


# ---------------------------------------------------------------------------
# Public (within the package) entry points
# ---------------------------------------------------------------------------


def forward_backward(obs, mask, params, *, observation="binary"):
    """Posterior activity probabilities and log-likelihoods.

    Parameters
    ----------
    obs : array_like, shape (E, T)
        Observations: 0/1 flags (``observation="binary"``) or p-values.
    mask : array_like of bool, shape (E, T), or None
        True where the step is known (eligible).  None means all known.
        Non-finite observations are treated as unknown as well.
    params : dict
        ``a01``, ``a10`` and the emission parameters (``alpha`` and ``beta``,
        or ``a``); optionally ``pi1`` (default: stationary).
    observation : {"binary", "pvalue"}
        Emission model.

    Returns
    -------
    posterior : ndarray, shape (E, T)
        ``P(s_t = 1 | o_{1:T})``.  NaN for an edge whose observations are
        impossible under *params*.
    loglik : ndarray, shape (E,)
        ``log P(o_{1:T})`` per edge (``-inf`` if impossible).
    """
    import numpy as np

    o, m = _prepare(obs, mask, observation)
    p = _check_complete(params, observation)
    pi1 = _initial_pi1(p)
    gamma, loglik, _ = _forward_backward(
        _emissions(o, m, p, observation),
        _transition_matrix(p),
        np.array([1.0 - pi1, pi1]),
    )
    return gamma[..., 1], loglik


def viterbi(obs, mask, params, *, observation="binary"):
    """Most likely hidden state path for every edge.

    Parameters
    ----------
    obs, mask, params, observation
        As in :func:`forward_backward`.

    Returns
    -------
    path : ndarray of int8, shape (E, T)
        The maximum a posteriori state sequence (1 = active).  Ties go to
        the inactive state (see the module docstring).
    """
    import numpy as np

    o, m = _prepare(obs, mask, observation)
    p = _check_complete(params, observation)
    pi1 = _initial_pi1(p)
    E, T = o.shape
    path = np.zeros((E, T), dtype=np.int8)
    if T == 0 or E == 0:
        return path
    with np.errstate(divide="ignore"):
        logB = np.log(_emissions(o, m, p, observation))
        logA = np.log(_transition_matrix(p))
        delta = np.log(np.array([1.0 - pi1, pi1]))[None, :] + logB[:, 0]
    psi = np.zeros((E, T, 2), dtype=np.int8)
    for t in range(1, T):
        cand = delta[:, :, None] + logA[None, :, :]  # (edge, from, to)
        # np.argmax returns the first maximum: ties go to state 0.
        psi[:, t] = np.argmax(cand, axis=1)
        delta = cand.max(axis=1) + logB[:, t]
    path[:, T - 1] = np.argmax(delta, axis=1)
    rows = np.arange(E)
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = psi[rows, t, path[:, t]]
    return path


def _canonicalize(p, observation, fixed, swappable):
    """Impose identifiability on *p* in place; return (swapped, constrained)."""
    if observation == "pvalue":
        if p["a"] >= 1.0 - _MARGIN:
            p["a"] = 1.0 - _MARGIN
            return False, True
        return False, False
    if p["alpha"] + p["beta"] <= 1.0 - _MARGIN:
        return False, False
    if swappable and p["alpha"] + p["beta"] > 1.0:
        p["a01"], p["a10"] = p["a10"], p["a01"]
        p["alpha"], p["beta"] = 1.0 - p["beta"], 1.0 - p["alpha"]
        if p.get("pi1") is not None:
            p["pi1"] = 1.0 - p["pi1"]
        if p["alpha"] + p["beta"] <= 1.0 - _MARGIN:
            return True, False
        swapped = True
    else:
        swapped = False
    # Pull the free emission parameters back inside alpha + beta < 1.
    limit = 1.0 - _MARGIN
    free = [n for n in ("alpha", "beta") if n not in fixed]
    if not free:
        # Both fixed by the caller (validated to satisfy alpha < 1 - beta).
        return swapped, False
    if len(free) == 2:
        scale = limit / (p["alpha"] + p["beta"])
        p["alpha"] *= scale
        p["beta"] *= scale
    else:
        (name,) = free
        other = "beta" if name == "alpha" else "alpha"
        p[name] = max(0.0, limit - p[other])
    return swapped, True


def _m_step(o, m, gamma, xi, p, observation, fit_pi1):
    """Pooled maximization step; returns the updated parameter dict."""
    new = dict(p)
    g0 = gamma[..., 0] * m
    g1 = gamma[..., 1] * m
    trans = xi.sum(axis=0)
    out0 = trans[0].sum()
    if out0 > 0.0:
        new["a01"] = float(trans[0, 1] / out0)
    out1 = trans[1].sum()
    if out1 > 0.0:
        new["a10"] = float(trans[1, 0] / out1)
    if observation == "binary":
        n0 = g0.sum()
        if n0 > 0.0:
            new["alpha"] = float((g0 * o).sum() / n0)
        n1 = g1.sum()
        if n1 > 0.0:
            new["beta"] = float((g1 * (1.0 - o)).sum() / n1)
    else:
        n1 = g1.sum()
        s = -(g1 * _log_p(o, m)).sum()
        if n1 > 0.0:
            new["a"] = float(n1 / s) if s > 0.0 else 1.0
    if fit_pi1 and gamma.shape[0] and gamma.shape[1]:
        new["pi1"] = float(gamma[:, 0, 1].mean())
    return new


def _em(o, m, start, fixed, observation, fit_pi1, max_iter, tol):
    """Run EM from *start*; return the final parameters and diagnostics."""
    import numpy as np

    names = list(_param_names(observation)) + (["pi1"] if fit_pi1 else [])
    swappable = not (set(fixed) & {"a01", "a10", "alpha", "beta", "pi1"})
    p = dict(start)
    swapped, constrained = _canonicalize(p, observation, fixed, swappable)
    converged = False
    n_iter = 0
    history = []
    for n_iter in range(1, max_iter + 1):  # noqa: B007 - used after the loop
        pi1 = _initial_pi1(p)
        gamma, loglik, xi = _forward_backward(
            _emissions(o, m, p, observation),
            _transition_matrix(p),
            np.array([1.0 - pi1, pi1]),
            want_xi=True,
        )
        total = float(loglik.sum())
        if not math.isfinite(total):
            raise nx.NetworkXError(
                "Some observations have probability zero under the HMM "
                f"parameters {p}; free or change the fixed parameters."
            )
        history.append(total)
        new = _m_step(o, m, gamma, xi, p, observation, fit_pi1)
        for name, value in fixed.items():
            new[name] = value
        s, c = _canonicalize(new, observation, fixed, swappable)
        swapped |= s
        constrained = c  # whether the final parameters sit on the constraint
        change = max(abs(new[n] - p[n]) for n in names) if names else 0.0
        p = new
        if change <= tol:
            converged = True
            break
    return {
        "params": p,
        "n_iter": n_iter,
        "converged": converged,
        "label_swapped": swapped,
        "constrained": constrained,
        "loglik_history": history,
    }


def fit_hmm(
    obs,
    mask=None,
    *,
    observation="binary",
    init=None,
    fixed=None,
    initial="stationary",
    max_iter=500,
    tol=1e-6,
    n_init=1,
    seed=None,
):
    """Fit the shared HMM parameters by pooled Baum-Welch (EM).

    Parameters
    ----------
    obs, mask, observation
        As in :func:`forward_backward`.
    init : dict or None
        Starting values for free parameters (defaults: ``a01 = a10 = 0.1``,
        ``alpha = beta = 0.1``, ``a = 0.2``, ``pi1 = 0.5``).
    fixed : dict or None
        Parameters held at the given values.  Any subset of ``a01``, ``a10``,
        the emission parameters and ``pi1`` may be fixed; fixing ``pi1``
        overrides *initial*.
    initial : {"stationary", "fit"}
        Initial distribution: the chain's stationary distribution (default)
        or a free parameter ``pi1`` estimated by EM.
    max_iter : int
        Maximum number of EM iterations per start.
    tol : float
        Convergence tolerance on the largest change of a free parameter.
    n_init : int
        Number of starts.  The first uses *init* (or the defaults); the
        others draw random starting values with *seed*.  The start with the
        highest final log-likelihood wins.
    seed : integer, random_state, or None
        Random number generation state for the extra starts.

    Returns
    -------
    result : dict
        ``params`` (fitted values, with ``pi1`` the initial probability
        actually used), ``loglik`` (total, at ``params``), ``n_iter``,
        ``converged``, ``label_swapped``, ``constrained``,
        ``loglik_history``, ``posterior`` ``(E, T)``, ``loglik_edges``
        ``(E,)`` and ``path`` ``(E, T)`` (Viterbi, int8).

    Warns
    -----
    RuntimeWarning
        If EM does not converge within *max_iter* iterations, or if the
        identifiability constraint had to be enforced by clipping.
    """
    import numpy as np

    o, m = _prepare(obs, mask, observation)
    if initial not in _INITIALS:
        raise nx.NetworkXError(f"initial must be one of {_INITIALS}, not {initial!r}.")
    if max_iter < 1:
        raise nx.NetworkXError(f"max_iter must be at least 1, not {max_iter}.")
    if not tol >= 0.0:
        raise nx.NetworkXError(f"tol must be non-negative, not {tol}.")
    if n_init < 1:
        raise nx.NetworkXError(f"n_init must be at least 1, not {n_init}.")
    fixed = {} if fixed is None else dict(fixed)
    init = {} if init is None else dict(init)
    _check_values(fixed, observation)
    _check_values(init, observation)
    fixed = {k: float(v) for k, v in fixed.items()}
    if (
        observation == "binary"
        and {"alpha", "beta"} <= set(fixed)
        and not fixed["alpha"] < 1.0 - fixed["beta"]
    ):
        raise nx.NetworkXError(
            "Fixed HMM parameters must satisfy alpha < 1 - beta; got "
            f"alpha={fixed['alpha']}, beta={fixed['beta']}."
        )
    for name in ("alpha", "beta"):
        if name in fixed and fixed[name] >= 1.0 - _MARGIN:
            raise nx.NetworkXError(f"Fixed {name!r} must be below 1.")
    fit_pi1 = initial == "fit" and "pi1" not in fixed
    names = list(_param_names(observation)) + (["pi1"] if fit_pi1 else [])

    def start_from(values):
        p = {n: float(values.get(n, _DEFAULT_START[n])) for n in names}
        p.update(fixed)
        if "pi1" not in fixed and not fit_pi1:
            p.pop("pi1", None)
        return p

    starts = [start_from(init)]
    if n_init > 1:
        from nx_backbone.bipartite import _as_generator

        rng = _as_generator(seed)
        ranges = {
            "a01": (0.01, 0.5),
            "a10": (0.01, 0.5),
            "alpha": (0.01, 0.3),
            "beta": (0.01, 0.3),
            "a": (0.05, 0.8),
            "pi1": (0.1, 0.9),
        }
        for _ in range(n_init - 1):
            starts.append(start_from({n: rng.uniform(*ranges[n]) for n in names}))

    best = None
    for start in starts:
        run = _em(o, m, start, fixed, observation, fit_pi1, max_iter, tol)
        p = run["params"]
        pi1 = _initial_pi1(p)
        gamma, loglik, _ = _forward_backward(
            _emissions(o, m, p, observation),
            _transition_matrix(p),
            np.array([1.0 - pi1, pi1]),
        )
        run["loglik"] = float(loglik.sum())
        run["posterior"] = gamma[..., 1]
        run["loglik_edges"] = loglik
        if best is None or run["loglik"] > best["loglik"]:
            best = run

    if not best["converged"]:
        warnings.warn(
            f"HMM EM did not converge in {max_iter} iterations "
            f"(tol={tol:g}); the fitted parameters may be inaccurate.",
            RuntimeWarning,
            stacklevel=3,
        )
    if best["constrained"]:
        if observation == "pvalue":
            message = (
                "HMM identifiability constraint a < 1 was enforced by clipping: "
                "the fitted active state does not concentrate p-values near 0, "
                "so the two states are nearly indistinguishable; try n_init > 1 "
                "or fix some parameters."
            )
        else:
            message = (
                "HMM identifiability constraint alpha < 1 - beta was enforced "
                "by clipping a free parameter: the two states are nearly "
                "indistinguishable, or fixed parameters push the fit onto the "
                "boundary; try n_init > 1 or other fixed parameters."
            )
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    params = dict(best["params"])
    params["pi1"] = _initial_pi1(params)
    best["params"] = params
    best["path"] = viterbi(o, m, params, observation=observation)
    return best
