"""
Statistical backbone extraction methods.

These methods evaluate edge significance using hypothesis testing against
a null model. Each method computes a p-value (or score) for every edge,
which can then be filtered using functions in :mod:`networkx_backbone.filters`.

Methods
-------
disparity_filter
    Serrano et al. (2009) -- uniform distribution null model.
noise_corrected_filter
    Coscia & Neffke (2017) -- Bayesian noise-corrected lift score.
marginal_likelihood_filter
    Dianati (2016) -- binomial null considering both endpoints.
ecm_filter
    Gemmetto et al. (2017) -- enhanced configuration model.
lans_filter
    Foti et al. (2011) -- nonparametric, empirical CDF-based.
multiple_linkage_analysis
    Van Nuffel (2007) -- dominant links by fitting ideal distributions.
polya_filter
    Marcaccioli & Livan (2019) -- Polya urn (beta-binomial) null model.
"""

import math
import warnings

import networkx as nx
from networkx.utils import not_implemented_for

from networkx_backbone._docstrings import append_complexity_docstrings

__all__ = [
    "disparity_filter",
    "noise_corrected_filter",
    "marginal_likelihood_filter",
    "ecm_filter",
    "lans_filter",
    "multiple_linkage_analysis",
    "polya_filter",
    # Short alias names
    "disparity",
    "mlf",
    "lans",
]


# =====================================================================
# Helper: validate positive weights
# =====================================================================


def _validate_weights(G, weight):
    """Check that all edges have a positive, finite weight attribute."""
    for u, v, data in G.edges(data=True):
        w = data.get(weight)
        # ``not (0 < w < inf)`` also rejects NaN, which fails every comparison.
        if w is None or not (0 < w < math.inf):
            raise nx.NetworkXError(
                f"Edge ({u}, {v}) has a non-positive, non-finite, or missing "
                f"{weight!r} attribute."
            )


def _combine_tails(p_hi, p_lo):
    """Combine upper/lower-tail p-values into a two-sided p-value and a sign.

    Returns ``(two_sided_pvalue, sign)`` where ``sign`` is ``+1`` for a
    significantly strong (over-represented) edge and ``-1`` for a significantly
    weak (under-represented) edge.  Used by the signed backbone variants.
    """
    p_hi = min(max(p_hi, 0.0), 1.0)
    p_lo = min(max(p_lo, 0.0), 1.0)
    pval = min(1.0, 2.0 * min(p_hi, p_lo))
    sign = 1 if p_hi <= p_lo else -1
    return pval, sign


# =====================================================================
# 1. Disparity filter -- Serrano et al. (2009)
# =====================================================================


@not_implemented_for("multigraph")
def disparity_filter(G, weight="weight", signed=False):
    r"""Compute disparity filter p-values for each edge.

    The disparity filter [1]_ tests whether an edge's weight is
    statistically significant compared to a null model where each node's
    total strength is uniformly distributed across its edges.

    For each node *u* with degree *k_u* and strength *s_u*, the normalised
    weight of edge (u, v) is ``p_uv = w_uv / s_u``.  Under the null the
    probability of observing a normalised weight >= p_uv is::

        alpha_uv = (1.0 - p_uv) ** (k_u - 1)

    For an undirected edge the p-value is the *minimum* of the values
    computed from each endpoint.  For a directed edge (u, v) the test is run
    from the source's outgoing links (out-strength and out-degree of *u*) and
    from the target's incoming links (in-strength and in-degree of *v*), and
    the p-value is the minimum of the two, following [1]_.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key for weights.  All weights must be positive.
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test: the stored p-value becomes two-sided
        and a ``"sign"`` edge attribute marks significantly strong (``+1``)
        versus significantly weak (``-1``) edges.

    Returns
    -------
    H : graph
        A copy of *G* (same type) with ``"disparity_pvalue"`` added as an
        edge attribute (and ``"sign"`` when ``signed=True``).

    Raises
    ------
    NetworkXError
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.  Collapse parallel edges first with
        :func:`~networkx_backbone.multigraph_to_weighted`.

    See Also
    --------
    threshold_filter : Keep edges whose p-value is below a threshold.
    marginal_likelihood_filter : Null model that uses both endpoints jointly.

    Notes
    -----
    An endpoint of degree 1 cannot reject the null model, so it contributes a
    p-value of 1 and the edge's p-value comes from the other endpoint.

    References
    ----------
    .. [1] Serrano, M. A., Boguna, M., & Vespignani, A. (2009).
       Extracting the multiscale backbone of complex weighted networks.
       *PNAS*, 106(16), 6483-6488.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import disparity_filter, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = disparity_filter(G)
    >>> backbone = threshold_filter(H, "disparity_pvalue", 0.05, mode="below")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    _validate_weights(H, weight)

    if G.is_directed():
        s_out = dict(G.out_degree(weight=weight))
        k_out = dict(G.out_degree())
        s_in = dict(G.in_degree(weight=weight))
        k_in = dict(G.in_degree())
    else:
        strength = dict(G.degree(weight=weight))
        degree = dict(G.degree())

    for u, v, data in H.edges(data=True):
        w = data[weight]
        if G.is_directed():
            hi_u, lo_u = _disparity_node_tails(w, s_out[u], k_out[u])
            hi_v, lo_v = _disparity_node_tails(w, s_in[v], k_in[v])
            p_hi = min(hi_u, hi_v)
            p_lo = min(lo_u, lo_v)
        else:
            hi_u, lo_u = _disparity_node_tails(w, strength[u], degree[u])
            hi_v, lo_v = _disparity_node_tails(w, strength[v], degree[v])
            # OR rule: strong/weak if significant from either endpoint.
            p_hi = min(hi_u, hi_v)
            p_lo = min(lo_u, lo_v)
        if signed:
            data["disparity_pvalue"], data["sign"] = _combine_tails(p_hi, p_lo)
        else:
            data["disparity_pvalue"] = p_hi

    return H


def _disparity_node_tails(w, s, k):
    """Upper- and lower-tail disparity p-values from one node's perspective."""
    if k <= 1:
        return 1.0, 1.0
    p = min(w / s, 1.0)
    try:
        p_hi = (1.0 - p) ** (k - 1)
    except (OverflowError, ValueError):
        p_hi = 0.0
    p_hi = max(p_hi, 0.0)
    return p_hi, max(1.0 - p_hi, 0.0)


# =====================================================================
# 2. Noise-corrected filter -- Coscia & Neffke (2017)
# =====================================================================


@not_implemented_for("multigraph")
def noise_corrected_filter(G, weight="weight", method="bayesian", signed=False):
    r"""Compute noise-corrected (NC) edge scores (Coscia & Neffke, 2017).

    The noise-corrected backbone [1]_ asks whether an edge carries more weight
    than expected from the strengths of its endpoints, and how sure we can be
    of that given that edge weights are noisy counts.  Write :math:`n_{ij}`
    for the weight of edge :math:`(i, j)`, :math:`n_{i.}` and :math:`n_{.j}`
    for the row and column sums of the weight matrix (out- and in-strength)
    and :math:`n_{..}` for its total.  The edge score is the *lift*
    :math:`L_{ij} = \kappa_{ij} n_{ij}`, with
    :math:`\kappa_{ij} = n_{..} / (n_{i.} n_{.j})`, mapped to
    :math:`(-1, 1)`:

    .. math::

        \text{nc\_score}_{ij} = \frac{\kappa_{ij} n_{ij} - 1}{\kappa_{ij} n_{ij} + 1}.

    A score of 0 means the edge has exactly the weight expected from its
    endpoints; positive scores mean more.  The standard deviation of the score
    comes from a Bayesian model of :math:`n_{ij}` (see Notes).  The paper keeps
    an edge when

    .. math::

        \text{nc\_score}_{ij} > \delta \cdot \text{nc\_sdev}_{ij},

    with :math:`\delta = 1.64` the usual choice (a one-tailed test at about
    the 5% level).  ``nc_pvalue`` is the matching normal-approximation p-value,
    :math:`P(Z \geq \text{nc\_score} / \text{nc\_sdev})`, so the rule
    ``score > delta * sdev`` is the same as ``nc_pvalue < 1 - Phi(delta)``
    (0.0505 for ``delta=1.64``) and the scores compose with
    :func:`~networkx_backbone.threshold_filter` and its multiple-testing
    corrections.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph.  The model treats weights as counts.
    weight : string, optional (default="weight")
        Edge attribute key for weights.  All weights must be positive.
    method : {"bayesian", "zscore"}, optional (default="bayesian")
        ``"bayesian"`` is the published noise-corrected backbone, matching the
        authors' reference implementation (``backboning.noise_corrected``).
        ``"zscore"`` is the binomial z-score computed by networkx-backbone
        0.2 and earlier, kept for backward compatibility (see Notes).
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test on ``nc_score / nc_sdev``:
        ``nc_pvalue`` becomes two-sided and a ``"sign"`` edge attribute marks
        significantly strong (``+1``) versus significantly weak (``-1``) edges.

    Returns
    -------
    H : graph
        A copy of *G* (same type).  Every edge gets ``"nc_score"`` and
        ``"nc_pvalue"``; with ``method="bayesian"`` also ``"nc_sdev"``; with
        ``signed=True`` also ``"sign"``.

    Raises
    ------
    NetworkXError
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.
    ValueError
        If *method* is not ``"bayesian"`` or ``"zscore"``.

    See Also
    --------
    threshold_filter : Keep edges with ``nc_pvalue`` below a threshold.
    marginal_likelihood_filter : Binomial null model with exact p-values.

    Notes
    -----
    **Prior.**  Under the null hypothesis the edge probability
    :math:`p_{ij}` has the mean and variance of drawing the :math:`n_{..}`
    units of weight without replacement (hypergeometric) given the margins:

    .. math::

        E[p_{ij}] = \frac{n_{i.} n_{.j}}{n_{..}^2}, \qquad
        V[p_{ij}] = \frac{n_{i.} n_{.j} (n_{..} - n_{i.}) (n_{..} - n_{.j})}
                         {n_{..}^4 (n_{..} - 1)}.

    A Beta(:math:`\alpha, \beta`) prior is fitted to these moments,
    :math:`\alpha = E^2 (1 - E) / V - E` and
    :math:`\beta = E (1 - E^2) / V - (1 - E)` (the latter exactly as in the
    reference code).

    **Posterior.**  Observing :math:`n_{ij}` successes in :math:`n_{..}`
    trials gives the posterior mean
    :math:`\hat p_{ij} = (\alpha + n_{ij}) / (\alpha + \beta + n_{..})` and
    the binomial variance
    :math:`V[n_{ij}] = n_{..} \hat p_{ij} (1 - \hat p_{ij})`.

    **Delta method.**  The variance of the score is
    :math:`V[n_{ij}] \, (\partial\, \text{nc\_score} / \partial n_{ij})^2`,
    where the derivative also accounts for :math:`n_{ij}` being part of
    :math:`n_{i.}`, :math:`n_{.j}` and :math:`n_{..}`:

    .. math::

        \frac{\partial\, \text{nc\_score}}{\partial n_{ij}}
        = \frac{2 (\kappa_{ij} + n_{ij} d_{ij})}{(\kappa_{ij} n_{ij} + 1)^2},
        \qquad
        d_{ij} = \frac{1}{n_{i.} n_{.j}}
               - n_{..} \frac{n_{i.} + n_{.j}}{(n_{i.} n_{.j})^2}.

    ``nc_sdev`` is the square root of this variance.  When the prior is
    degenerate (total weight of at most 1, or one node carrying all of it)
    the score cannot be tested: ``nc_sdev`` is ``inf`` and ``nc_pvalue`` 1.

    **Directed and undirected graphs.**  For a directed graph,
    :math:`n_{i.}` is the out-strength of *i*, :math:`n_{.j}` the in-strength
    of *j* and :math:`n_{..}` the total weight.  An undirected graph is
    treated as a symmetric weight matrix, as the reference implementation
    expects: every edge is counted in both directions, so :math:`n_{i.}` is
    the strength of *i* and :math:`n_{..} = 2W` for total edge weight *W*
    (a self-loop is a single diagonal entry, counted once).  The score is
    symmetric in *i* and *j*.

    **Legacy z-score.**  ``method="zscore"`` returns
    :math:`z = (w - W p) / \sqrt{W p (1 - p)}` with
    :math:`p = s_u s_v / W^2`, where *W* counts each undirected edge once but
    strengths count it at both endpoints.  That mismatch makes expected
    weights sum to about :math:`2W`, so on a unit-weight complete graph every
    edge gets a negative score.  Its ``nc_pvalue`` is :math:`P(Z \geq z)`.

    References
    ----------
    .. [1] Coscia, M. & Neffke, F. M. (2017). Network backboning with
       noisy data. *Proc. IEEE ICDE*, 425-436. arXiv:1701.07336.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import noise_corrected_filter, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = noise_corrected_filter(G)
    >>> kept = [(u, v) for u, v, d in H.edges(data=True)
    ...         if d["nc_score"] > 1.64 * d["nc_sdev"]]
    >>> len(kept)
    195
    >>> backbone = threshold_filter(H, "nc_pvalue", 0.05, mode="below")
    >>> backbone.number_of_edges()
    195
    """
    if method not in ("bayesian", "zscore"):
        raise ValueError(f"method must be 'bayesian' or 'zscore', got {method!r}")

    H = G.copy()
    _validate_weights(H, weight)

    if G.number_of_edges() == 0:
        return H

    if method == "zscore":
        _noise_corrected_zscore(G, H, weight, signed)
        return H

    # Row and column sums of the (symmetric, for undirected graphs) weight
    # matrix, counting each undirected non-loop edge in both directions.
    s_out, s_in = {}, {}
    total = 0.0
    for u, v, d in G.edges(data=True):
        w = d[weight]
        s_out[u] = s_out.get(u, 0.0) + w
        s_in[v] = s_in.get(v, 0.0) + w
        total += w
        if not G.is_directed() and u != v:
            s_out[v] = s_out.get(v, 0.0) + w
            s_in[u] = s_in.get(u, 0.0) + w
            total += w

    for u, v, data in H.edges(data=True):
        score, sdev = _nc_score_sdev(data[weight], s_out[u], s_in[v], total)
        if math.isinf(sdev):
            p_hi, p_lo = 1.0, 1.0  # degenerate prior: nothing to test
        elif sdev == 0.0:
            p_hi, p_lo = _normal_tails(math.copysign(math.inf, score) if score else 0.0)
        else:
            p_hi, p_lo = _normal_tails(score / sdev)
        data["nc_score"] = score
        data["nc_sdev"] = sdev
        if signed:
            data["nc_pvalue"], data["sign"] = _combine_tails(p_hi, p_lo)
        else:
            data["nc_pvalue"] = p_hi

    return H


def _nc_score_sdev(n_ij, n_i, n_j, n):
    """NC score and its standard deviation for one cell of the weight matrix.

    Mirrors ``backboning.noise_corrected`` of Coscia & Neffke (2017) term by
    term.  ``n_i`` and ``n_j`` are the row and column sums and ``n`` the total.
    """
    kappa = n / (n_i * n_j)
    lift = kappa * n_ij
    score = (lift - 1.0) / (lift + 1.0)
    if n <= 1.0:
        return score, math.inf
    mean_prior = (n_i * n_j / n) * (1.0 / n)
    var_prior = (1.0 / n**2) * (n_i * n_j * (n - n_i) * (n - n_j)) / (n**2 * (n - 1.0))
    if not var_prior > 0.0:
        return score, math.inf
    alpha_prior = (mean_prior**2 / var_prior) * (1.0 - mean_prior) - mean_prior
    beta_prior = (mean_prior / var_prior) * (1.0 - mean_prior**2) - (1.0 - mean_prior)
    alpha_post = alpha_prior + n_ij
    beta_post = n - n_ij + beta_prior
    expected_p = alpha_post / (alpha_post + beta_post)
    variance_n = expected_p * (1.0 - expected_p) * n
    d = 1.0 / (n_i * n_j) - n * ((n_i + n_j) / (n_i * n_j) ** 2)
    deriv = 2.0 * (kappa + n_ij * d) / (lift + 1.0) ** 2
    variance = variance_n * deriv**2
    if not (variance >= 0.0 and math.isfinite(variance)):
        return score, math.inf
    return score, math.sqrt(variance)


def _normal_tails(z):
    """Upper and lower standard-normal tail probabilities at ``z``."""
    return 0.5 * math.erfc(z / math.sqrt(2.0)), 0.5 * math.erfc(-z / math.sqrt(2.0))


def _noise_corrected_zscore(G, H, weight, signed):
    """Legacy binomial z-score (networkx-backbone <= 0.2), written into *H*."""
    W = sum(d[weight] for _, _, d in G.edges(data=True))
    if G.is_directed():
        s_out = dict(G.out_degree(weight=weight))
        s_in = dict(G.in_degree(weight=weight))
    else:
        s_out = dict(G.degree(weight=weight))
        s_in = s_out

    for u, v, data in H.edges(data=True):
        p_ij = (s_out[u] * s_in[v]) / (W * W)
        variance = W * p_ij * (1 - p_ij)
        if variance > 0:
            z = (data[weight] - W * p_ij) / math.sqrt(variance)
        else:
            z = 0.0
        data["nc_score"] = z
        p_hi, p_lo = _normal_tails(z)
        if signed:
            data["nc_pvalue"], data["sign"] = _combine_tails(p_hi, p_lo)
        else:
            data["nc_pvalue"] = p_hi


# =====================================================================
# 3. Marginal likelihood filter -- Dianati (2016)
# =====================================================================


@not_implemented_for("multigraph")
def marginal_likelihood_filter(G, weight="weight", signed=False):
    r"""Compute marginal likelihood p-values for each edge.

    The marginal likelihood filter [1]_ treats an integer edge weight as a
    count of unit edges.  Under the null model each of the *T* unit edges
    picks its two endpoints independently with probability proportional to
    their strengths, so the weight of the pair (u, v) is binomial,
    ``B(T, p_uv)``, with

    .. math::

        T = \frac{1}{2}\sum_i s_i, \qquad p_{uv} = \frac{s_u s_v}{2 T^2}

    for undirected graphs, and :math:`T = \sum_i s^{out}_i`,
    :math:`p_{uv} = s^{out}_u s^{in}_v / T^2` for directed graphs.  The
    p-value is the upper tail :math:`P(X \geq w_{uv})`.  This null preserves
    the strength sequence in expectation and is symmetric in *u* and *v*.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph.  Weights are rounded to the nearest integer, so the
        method is meant for count data.
    weight : string, optional (default="weight")
        Edge attribute key for weights.  All weights must be positive.
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test and add a ``"sign"`` edge attribute
        (``+1`` significantly strong, ``-1`` significantly weak).

    Returns
    -------
    H : graph
        A copy of *G* (same type) with ``"ml_pvalue"`` edge attribute.

    Raises
    ------
    NetworkXError
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.

    See Also
    --------
    disparity_filter : Tests each endpoint separately.
    ecm_filter : Maximum-entropy null that also fixes degrees.

    References
    ----------
    .. [1] Dianati, N. (2016). Unwinding the hairball graph: Pruning
       algorithms for weighted complex networks. *Physical Review E*,
       93, 012304.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import marginal_likelihood_filter, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = marginal_likelihood_filter(G)
    >>> backbone = threshold_filter(H, "ml_pvalue", 0.05, mode="below")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    from scipy import stats as sp_stats

    H = G.copy()
    _validate_weights(H, weight)

    if G.number_of_edges() == 0:
        return H

    # Strengths of the rounded (integer) weights.
    s_out, s_in = {}, {}
    T = 0
    for u, v, d in G.edges(data=True):
        w = round(d[weight])
        T += w
        s_out[u] = s_out.get(u, 0) + w
        s_in[v] = s_in.get(v, 0) + w
    if not G.is_directed():
        # Each undirected edge adds its weight to both endpoints.
        for node, val in s_in.items():
            s_out[node] = s_out.get(node, 0) + val
        s_in = s_out

    norm = T * T if G.is_directed() else 2 * T * T

    for u, v, data in H.edges(data=True):
        k = round(data[weight])
        if T > 0 and u != v:
            p_param = min(s_out[u] * s_in[v] / norm, 1.0)
            p_hi = float(sp_stats.binom.sf(k - 1, T, p_param))
            p_lo = float(sp_stats.binom.cdf(k, T, p_param))
        else:
            p_hi, p_lo = 1.0, 1.0

        if signed:
            data["ml_pvalue"], data["sign"] = _combine_tails(p_hi, p_lo)
        else:
            data["ml_pvalue"] = float(p_hi)

    return H


# =====================================================================
# 4. ECM filter -- Gemmetto et al. (2017)
# =====================================================================


@not_implemented_for("multigraph")
def ecm_filter(G, weight="weight", max_iter=1000, tol=1e-6, signed=False):
    r"""Compute ECM (Enhanced Configuration Model) p-values.

    The ECM [1]_ is the maximum-entropy ensemble of integer-weighted graphs
    whose expected degree *and* strength sequences equal the observed ones.
    Each node gets two Lagrange multipliers, :math:`x_i > 0` and
    :math:`0 < y_i < 1`.  The probability that *i* and *j* are linked and
    the distribution of the weight of an existing link are

    .. math::

        p_{ij} = \frac{x_i x_j y_i y_j}{1 - y_i y_j + x_i x_j y_i y_j},
        \qquad
        P(w_{ij} \geq w) = p_{ij} (y_i y_j)^{w - 1}, \; w \geq 1,

    and the p-value of an observed link is :math:`P(w_{ij} \geq w)`.  The
    multipliers are found by maximizing the likelihood (L-BFGS-B with the
    analytic gradient), which is equivalent to solving
    :math:`\sum_j p_{ij} = k_i` and :math:`\sum_j \langle w_{ij}\rangle = s_i`.

    For directed graphs, each node has separate out- and in-multipliers that
    reproduce its out- and in-degree and strength.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph.  Weights are rounded to the nearest integer, so the
        method is meant for count data.
    weight : string, optional (default="weight")
        Edge attribute key for weights.  All weights must be positive.
    max_iter : int, optional (default=1000)
        Maximum number of solver iterations.
    tol : float, optional (default=1e-6)
        Tolerance on the largest relative error of the reproduced degrees and
        strengths.  A :class:`RuntimeWarning` is raised if it is not met.
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test and add a ``"sign"`` edge attribute
        (``+1`` significantly strong, ``-1`` significantly weak).  The lower
        tail is :math:`P(w_{ij} \leq w) = 1 - p_{ij} (y_i y_j)^w`, using the
        same fitted multipliers.

    Returns
    -------
    H : graph
        A copy of *G* (same type) with ``"ecm_pvalue"`` edge attribute (and
        ``"sign"`` when ``signed=True``).

    Raises
    ------
    NetworkXError
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.

    Warns
    -----
    RuntimeWarning
        If the solver does not reproduce the degree and strength sequences to
        within ``tol``.

    See Also
    --------
    marginal_likelihood_filter : Binomial null that fixes only strengths.

    Notes
    -----
    Self-loops are ignored when fitting the null model and get a p-value
    of 1.  The fit costs :math:`O(n^2)` memory and time per iteration.

    References
    ----------
    .. [1] Gemmetto, V., Cardillo, A., & Garlaschelli, D. (2017).
       Irreducible network backbones: unbiased graph filtering via
       maximum entropy. arXiv:1706.00230.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import ecm_filter, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = ecm_filter(G)
    >>> backbone = threshold_filter(H, "ecm_pvalue", 0.05, mode="below")
    >>> backbone.number_of_edges() < H.number_of_edges()
    True
    """
    import numpy as np

    H = G.copy()
    _validate_weights(H, weight)

    nodes = list(G)
    n = len(nodes)
    if n == 0 or G.number_of_edges() == 0:
        for _, _, data in H.edges(data=True):
            data["ecm_pvalue"] = 1.0
            if signed:
                data["sign"] = 1
        return H

    idx = {v: i for i, v in enumerate(nodes)}
    k_out = np.zeros(n)
    k_in = np.zeros(n)
    s_out = np.zeros(n)
    s_in = np.zeros(n)
    for u, v, d in G.edges(data=True):
        if u == v:
            continue
        w = max(round(d[weight]), 1)
        i, j = idx[u], idx[v]
        k_out[i] += 1
        s_out[i] += w
        k_in[j] += 1
        s_in[j] += w
    if not G.is_directed():
        k_out = k_in = k_out + k_in
        s_out = s_in = s_out + s_in

    a_out, b_out, a_in, b_in, _act_out, _act_in, err = _fit_ecm(
        k_out, s_out, k_in, s_in, G.is_directed(), max_iter, tol
    )
    if err > tol:
        warnings.warn(
            f"ecm_filter: the ECM solver did not converge (largest relative "
            f"error {err:.2e} > tol={tol:g}); p-values may be inaccurate.",
            RuntimeWarning,
            stacklevel=2,
        )

    for u, v, data in H.edges(data=True):
        if u == v:
            data["ecm_pvalue"] = 1.0
            if signed:
                data["sign"] = 1
            continue
        i, j = idx[u], idx[v]
        # x_i x_j y_i y_j = exp(-(a_i + a_j)); y_i y_j = exp(-(b_i + b_j)).
        # Work in log space: x can be huge when y is tiny.
        log_xxyy = -float(a_out[i] + a_in[j])
        log_yy = -float(b_out[i] + b_in[j])
        log_one_minus_yy = math.log(-math.expm1(log_yy))
        hi = max(log_xxyy, log_one_minus_yy)
        log_denom = hi + math.log(
            math.exp(log_xxyy - hi) + math.exp(log_one_minus_yy - hi)
        )
        w = max(round(data[weight]), 1)
        # Upper tail P(w_ij >= w) = p_ij (y_i y_j)^(w - 1).
        log_p_hi = log_xxyy - log_denom + (w - 1) * log_yy
        p_hi = float(min(max(math.exp(log_p_hi), 0.0), 1.0))
        if signed:
            # Lower tail P(w_ij <= w) = 1 - P(w_ij >= w + 1).
            p_lo = -math.expm1(min(log_p_hi + log_yy, 0.0))
            data["ecm_pvalue"], data["sign"] = _combine_tails(p_hi, p_lo)
        else:
            data["ecm_pvalue"] = p_hi

    return H


# Largest parameter vector for which the dense Newton polish is used
# (2 parameters per node undirected, 4 directed).
_ECM_NEWTON_MAX_PARAMS = 4000


def _fit_ecm(k_out, s_out, k_in, s_in, directed, max_iter, tol):
    """Maximum-likelihood ECM multipliers.

    Works with ``x * y = exp(-c)`` and ``y = exp(-b)`` (``b > 0`` keeps
    ``y < 1``).  Parameterizing by ``x * y`` keeps the problem well
    conditioned for nodes whose links all have weight 1, where ``y -> 0``
    and ``x -> inf`` with finite product.  Nodes with zero degree are pinned
    to ``x = 0``.  Returns
    ``(x_out, y_out, x_in, y_in, max_relative_error)``.
    """
    import numpy as np
    from scipy.optimize import minimize

    n = len(k_out)
    act_out = k_out > 0
    act_in = k_in > 0
    off_diag = ~np.eye(n, dtype=bool)

    def unpack(theta):
        if directed:
            a_out, b_out, a_in, b_in = np.split(theta, 4)
        else:
            a_out, b_out = np.split(theta, 2)
            a_in, b_in = a_out, b_out
        return a_out, b_out, a_in, b_in

    def expectations(theta):
        a_out, b_out, a_in, b_in = unpack(theta)
        # log of x_i x_j y_i y_j and of y_i y_j, for every ordered pair.
        log_yy = -(b_out[:, None] + b_in[None, :])
        log_xxyy = -(a_out[:, None] + a_in[None, :])
        one_minus_yy = -np.expm1(log_yy)
        # q_ij = x_i x_j y_i y_j / (1 - y_i y_j); log Z_ij = log(1 + q_ij)
        log_q = log_xxyy - np.log(one_minus_yy)
        mask = off_diag & act_out[:, None] & act_in[None, :]
        log_q = np.where(mask, log_q, -np.inf)
        log_z = np.logaddexp(0.0, log_q)
        p = np.exp(log_q - log_z)  # link probability
        w_mean = p / one_minus_yy  # expected weight
        return p, w_mean, log_z

    def objective(theta):
        a_out, b_out, a_in, b_in = unpack(theta)
        p, w_mean, log_z = expectations(theta)
        # With x*y = exp(-c), the sufficient statistic of b is s - k.
        if directed:
            nll = (
                a_out @ k_out
                + b_out @ (s_out - k_out)
                + a_in @ k_in
                + b_in @ (s_in - k_in)
                + log_z.sum()
            )
            grad = np.concatenate(
                [
                    k_out - p.sum(1),
                    (s_out - k_out) - (w_mean - p).sum(1),
                    k_in - p.sum(0),
                    (s_in - k_in) - (w_mean - p).sum(0),
                ]
            )
        else:
            # Symmetric matrices count each unordered pair twice.
            nll = a_out @ k_out + b_out @ (s_out - k_out) + 0.5 * log_z.sum()
            grad = np.concatenate(
                [k_out - p.sum(1), (s_out - k_out) - (w_mean - p).sum(1)]
            )
        return nll, grad

    def start(k, s):
        total = max(k.sum(), 1.0)
        # Chung-Lu-like guess for x*y, and mean link weight 1/(1-y) ~ s/k.
        a = -np.log(np.where(k > 0, k / np.sqrt(total), 1.0))
        ratio = np.where(k > 0, s / np.maximum(k, 1.0), 2.0)
        b = -np.log(np.clip(1.0 - 1.0 / np.maximum(ratio, 1.0 + 1e-6), 1e-12, 1.0))
        return a, b

    a0, b0 = start(k_out, s_out)
    if directed:
        a1, b1 = start(k_in, s_in)
        theta0 = np.concatenate([a0, b0, a1, b1])
    else:
        theta0 = np.concatenate([a0, b0])
    b_floor = 1e-10
    bounds = []
    blocks = 2 if directed else 1
    for _ in range(blocks):
        bounds += [(None, None)] * n + [(b_floor, None)] * n

    def rel_err(obs, fit):
        ok = obs > 0
        return float(np.max(np.abs(fit[ok] - obs[ok]) / obs[ok])) if ok.any() else 0.0

    def max_error(theta):
        p, w_mean, _ = expectations(theta)
        if directed:
            return max(
                rel_err(k_out, p.sum(1)),
                rel_err(s_out, w_mean.sum(1)),
                rel_err(k_in, p.sum(0)),
                rel_err(s_in, w_mean.sum(0)),
            )
        return max(rel_err(k_out, p.sum(1)), rel_err(s_out, w_mean.sum(1)))

    def hessian(theta):
        p, w_mean, _ = expectations(theta)
        m = w_mean - p
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(p > 0, m / np.where(p > 0, p, 1.0), 0.0)
        A = p * (1.0 - p)  # -dp/dt
        B = A * g  # -dp/dr  (= -dm/dt)
        C = A * g * g + p * g * (1.0 + g)  # -dm/dr
        if directed:
            Hm = np.zeros((4 * n, 4 * n))
            # parameter order: c_out, b_out, c_in, b_in
            pairs = {(0, 0): A, (0, 1): B, (1, 1): C}
            for (r, c), M in pairs.items():
                Hm[r * n : (r + 1) * n, c * n : (c + 1) * n] += np.diag(M.sum(1))
                Hm[(r + 2) * n : (r + 3) * n, (c + 2) * n : (c + 3) * n] += np.diag(
                    M.sum(0)
                )
                Hm[r * n : (r + 1) * n, (c + 2) * n : (c + 3) * n] += M
                Hm[(c + 2) * n : (c + 3) * n, r * n : (r + 1) * n] += M.T
                if r != c:
                    Hm[c * n : (c + 1) * n, r * n : (r + 1) * n] += np.diag(M.sum(1))
                    Hm[(c + 2) * n : (c + 3) * n, (r + 2) * n : (r + 3) * n] += np.diag(
                        M.sum(0)
                    )
                    Hm[c * n : (c + 1) * n, (r + 2) * n : (r + 3) * n] += M
                    Hm[(r + 2) * n : (r + 3) * n, c * n : (c + 1) * n] += M.T
            return Hm
        # Undirected: symmetric pair matrices, each unordered pair once per row.
        Hcc = np.diag(A.sum(1)) + A
        Hcb = np.diag(B.sum(1)) + B
        Hbb = np.diag(C.sum(1)) + C
        return np.block([[Hcc, Hcb], [Hcb.T, Hbb]])

    lower = np.array([lo if lo is not None else -np.inf for lo, _ in bounds])

    theta = theta0
    use_newton = len(theta0) <= _ECM_NEWTON_MAX_PARAMS
    # With a Newton polish available, L-BFGS-B only needs to get close.
    iters_left = min(max_iter, 200) if use_newton else max_iter
    # Restarting L-BFGS-B resets its curvature estimate, which helps it
    # finish ill-conditioned problems; a few restarts are plenty.
    for _ in range(1 if use_newton else 3):
        res = minimize(
            objective,
            theta,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": iters_left,
                "maxfun": 4 * max_iter,
                "gtol": 1e-12,
                "ftol": 1e-15,
            },
        )
        theta = res.x
        iters_left -= res.nit
        if max_error(theta) <= tol or iters_left <= 0:
            break

    # Damped Newton polish with the analytic Hessian.  The negative
    # log-likelihood is convex, so this converges quadratically; it needs a
    # dense solve, so it is only used for moderately sized graphs.
    if use_newton and max_error(theta) > tol:
        f, grad = objective(theta)
        for _ in range(min(100, max_iter)):
            Hm = hessian(theta)
            # The directed model has a gauge freedom (shift out-parameters
            # up and in-parameters down), so the Hessian is singular; a small
            # scaled ridge keeps the step in the identifiable subspace.
            Hm[np.diag_indices_from(Hm)] += 1e-9 * max(np.abs(Hm).max(), 1.0)
            try:
                step = np.linalg.solve(Hm, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(Hm, grad, rcond=None)[0]
            t = 1.0
            while t > 1e-8:
                cand = np.maximum(theta - t * step, lower)
                f_new, grad_new = objective(cand)
                if f_new <= f + 1e-4 * t * (grad @ (cand - theta)) or f_new < f:
                    break
                t *= 0.5
            else:
                break
            theta, f, grad = cand, f_new, grad_new
            if max_error(theta) <= tol:
                break

    err = max_error(theta)

    a_out, b_out, a_in, b_in = unpack(theta)
    return a_out, b_out, a_in, b_in, act_out, act_in, err


# =====================================================================
# 5. LANS -- Foti et al. (2011)
# =====================================================================


@not_implemented_for("multigraph")
def lans_filter(G, weight="weight", signed=False):
    r"""Compute LANS (Locally Adaptive Network Sparsification) p-values.

    LANS [1]_ is a nonparametric method that makes no distributional
    assumptions.  For each edge (u, v), it computes the fraction of
    edges at node *u* (and node *v*) that have weight <= w_uv --- the
    empirical CDF value.  The edge p-value is one minus the maximum
    of the two eCDF values (so lower p means more significant).

    For directed graphs, the source node's out-edges and the target node's
    in-edges are both used, and the edge p-value is computed from the more
    significant of the two sides.

    The p-value is ``1 - F(w)``, where ``F`` is the empirical CDF of the
    node's weights.  The strongest link of each node therefore gets a p-value
    of 0, and so does the only link of a degree-1 node.  LANS always keeps
    these links; this is how the method guarantees that no node becomes
    isolated.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key for weights.  All weights must be positive.
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test and add a ``"sign"`` edge attribute
        (``+1`` significantly strong, ``-1`` significantly weak).

    Returns
    -------
    H : graph
        A copy of *G* (same type) with ``"lans_pvalue"`` edge attribute.

    Raises
    ------
    NetworkXError
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.

    See Also
    --------
    disparity_filter : Parametric counterpart with a uniform null model.

    References
    ----------
    .. [1] Foti, N. J., Hughes, J. M., & Rockmore, D. N. (2011).
       Nonparametric sparsification of complex multiscale networks.
       *PLoS ONE*, 6(2), e16431.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import lans_filter, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = lans_filter(G)
    >>> backbone = threshold_filter(H, "lans_pvalue", 0.05, mode="below")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    _validate_weights(H, weight)

    # Pre-compute sorted weight lists per node
    if G.is_directed():
        out_weights = {}
        in_weights = {}
        for u in G.nodes():
            out_weights[u] = sorted(d[weight] for _, _, d in G.out_edges(u, data=True))
            in_weights[u] = sorted(d[weight] for _, _, d in G.in_edges(u, data=True))
    else:
        node_weights = {}
        for u in G.nodes():
            ws = sorted(G[u][v][weight] for v in G[u])
            node_weights[u] = ws

    for u, v, data in H.edges(data=True):
        w = data[weight]

        if G.is_directed():
            ecdf_u = _empirical_cdf(w, out_weights[u])
            ecdf_v = _empirical_cdf(w, in_weights[v])
            # OR rule: strong/weak if significant from either side of the edge.
            p_hi = 1.0 - max(ecdf_u, ecdf_v)
            p_lo = min(ecdf_u, ecdf_v)
        else:
            ecdf_u = _empirical_cdf(w, node_weights[u])
            ecdf_v = _empirical_cdf(w, node_weights[v])
            # OR rule: strong/weak if significant from either endpoint.
            p_hi = 1.0 - max(ecdf_u, ecdf_v)
            p_lo = min(ecdf_u, ecdf_v)

        if signed:
            data["lans_pvalue"], data["sign"] = _combine_tails(p_hi, p_lo)
        else:
            data["lans_pvalue"] = max(p_hi, 0.0)

    return H


@not_implemented_for("multigraph")
def multiple_linkage_analysis(G, alpha=0.05, weight="weight", method="dominance"):
    r"""Extract a backbone using Multiple Linkage Analysis (MLA).

    With ``method="dominance"`` (the default) this is the multiple linkage
    analysis of Van Nuffel [1]_, which decides how many of a node's
    links are *dominant*.  For each node:

    1. Sort its link weights in decreasing order and divide by their sum,
       giving the profile :math:`p_1 \geq p_2 \geq \dots \geq p_n`.
    2. Compare the profile with the ideal distributions in which *k* links
       share the weight equally and the others have none,
       :math:`t^{(k)} = (1/k, \dots, 1/k, 0, \dots, 0)`, for
       :math:`k = 1, \dots, n`, using the squared deviation

       .. math::

           D_k = \sum_{r=1}^{n} \left(p_r - t^{(k)}_r\right)^2
               = \sum_{r=1}^{n} p_r^2 + \frac{1 - 2 \sum_{r \leq k} p_r}{k}.

    3. Keep the *k* strongest links of the best fit (smallest :math:`D_k`).

    Each node keeps at least one link.  For a directed graph the profile is
    built from each node's out-links (the setting of [1]_) and an edge is kept
    when it is dominant for its source.  For an undirected graph each node
    uses all of its links, and an edge is kept when it is dominant for
    *either* endpoint (OR rule).

    With ``method="lans"`` this reproduces the netbone [2]_ implementation,
    which thresholds the edge p-values of :func:`lans_filter` at ``alpha``.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph.
    alpha : float, optional (default=0.05)
        Significance threshold in [0, 1].  Only used by ``method="lans"``.
    weight : string, optional (default="weight")
        Edge attribute key for weights. All weights must be positive.
    method : {"dominance", "lans"}, optional (default="dominance")
        ``"dominance"`` fits ideal dominance distributions as in [1]_;
        ``"lans"`` keeps edges with LANS p-value <= ``alpha``, as netbone
        does and as networkx-backbone 0.2 and earlier did.

    Returns
    -------
    H : graph
        A copy of *G*.  With ``method="dominance"`` every edge gets a boolean
        ``"mla_keep"`` attribute and every node an integer ``"mla_rank"``
        attribute, the number of dominant links it keeps (0 for a node with
        no links, or no out-links in a directed graph).  With
        ``method="lans"`` every edge gets ``"lans_pvalue"``, ``"mla_pvalue"``
        and boolean ``"mla_keep"``.

    Raises
    ------
    ValueError
        If ``alpha`` is outside [0, 1] or *method* is unknown.
    NetworkXError
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.

    See Also
    --------
    lans_filter : Empirical-CDF p-values used by ``method="lans"``.
    primary_linkage_analysis : Keeps only each node's single strongest link.
    boolean_filter : Keep the edges flagged ``mla_keep``.

    Notes
    -----
    The procedure is the least-squares comparison with ideal distributions
    used in multiple linkage analysis, after Weaver's crop-combination
    method [3]_.  Here the deviation is summed over the node's whole profile,
    so links outside the dominant set count against a fit.  Weaver's original
    measure instead averages the squared deviation over the *k* links of the
    candidate combination only.  The exact variant used by Van Nuffel (2007)
    has not yet been checked against the paper (see issue #10), so results
    may differ from that paper's in borderline cases.  Ideal distributions
    that would split a group of tied weights are not considered (such a split
    never fits strictly better), so tied links are kept or dropped together and ``mla_rank`` is always the number
    of links the node keeps.  If two fits are equally good, the smaller *k*
    wins.

    Unlike LANS, which tests each link against the empirical distribution of
    its node's weights, the dominance fit chooses *how many* links each node
    keeps: a node with one overwhelming link keeps one, a node whose weight is
    spread evenly keeps all of them.

    References
    ----------
    .. [1] Van Nuffel, N. (2007). Determination of the number of significant
       flows in origin-destination specific analysis: the case of commuting in
       Flanders. *Regional Studies*, 41(4), 509-524.
       https://doi.org/10.1080/00343400701281808
    .. [3] Weaver, J. C. (1954). Crop-combination regions in the Middle West.
       *Geographical Review*, 44(2), 175-200. https://doi.org/10.2307/212355
    .. [2] Yassin, A., Haidar, A., Cherifi, H., Seba, H., & Togni, O. (2023).
       An evaluation tool for backbone extraction techniques in weighted
       complex networks. *Scientific Reports*, 13, 17000.

    Examples
    --------
    A node with one dominant link keeps one; a node with two equally strong
    links keeps both:

    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, multiple_linkage_analysis
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from([("a", "x", 10), ("a", "y", 1), ("a", "z", 1)])
    >>> G.add_weighted_edges_from([("b", "x", 10), ("b", "y", 10), ("b", "z", 1)])
    >>> H = multiple_linkage_analysis(G)
    >>> H.nodes["a"]["mla_rank"], H.nodes["b"]["mla_rank"]
    (1, 2)
    >>> sorted(boolean_filter(H, "mla_keep").edges())
    [('a', 'x'), ('b', 'x'), ('b', 'y')]

    The netbone variant thresholds LANS p-values:

    >>> G = nx.les_miserables_graph()
    >>> H = multiple_linkage_analysis(G, alpha=0.05, method="lans")
    >>> backbone = boolean_filter(H, "mla_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    if method not in ("dominance", "lans"):
        raise ValueError(f"method must be 'dominance' or 'lans', got {method!r}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    if method == "lans":
        scored = lans_filter(G, weight=weight)
        for _, _, data in scored.edges(data=True):
            pvalue = data["lans_pvalue"]
            data["mla_pvalue"] = pvalue
            data["mla_keep"] = pvalue <= alpha
        return scored

    H = G.copy()
    _validate_weights(H, weight)

    neighbors = G.succ if G.is_directed() else G.adj
    cutoff = {}
    for node in G:
        weights = sorted((d[weight] for d in neighbors[node].values()), reverse=True)
        k = _dominance_rank(weights)
        H.nodes[node]["mla_rank"] = k
        cutoff[node] = weights[k - 1] if k else math.inf

    for u, v, data in H.edges(data=True):
        w = data[weight]
        keep = w >= cutoff[u]
        if not G.is_directed():
            keep = keep or w >= cutoff[v]
        data["mla_keep"] = bool(keep)
    return H


def _dominance_rank(weights):
    """Number of dominant links for a node's weights sorted in decreasing order.

    Returns the *k* whose ideal distribution (*k* equal links, the rest zero)
    is closest in squared deviation to the normalized profile; *k* never
    splits a run of tied weights, and ties in the deviation go to the
    smaller *k*.  The constant ``sum(p_r**2)`` is left out of the deviation.
    """
    n = len(weights)
    if n == 0:
        return 0
    total = math.fsum(weights)
    best_k, best_dev = n, math.inf
    cum = 0.0
    for k in range(1, n + 1):
        cum += weights[k - 1]
        if k < n and weights[k] == weights[k - 1]:
            continue  # would split a group of tied weights
        dev = (1.0 - 2.0 * cum / total) / k
        if dev < best_dev - 1e-12:
            best_k, best_dev = k, dev
    return best_k


def _empirical_cdf(w, sorted_weights):
    """Compute the empirical CDF value F(w) from a sorted list."""
    n = len(sorted_weights)
    if n == 0:
        return 0.0
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_weights[mid] <= w:
            lo = mid + 1
        else:
            hi = mid
    return lo / n


# =====================================================================
# 7. Polya filter -- Marcaccioli & Livan (2019)
# =====================================================================


@not_implemented_for("multigraph")
def polya_filter(G, weight="weight", a=None, approximate=None):
    r"""Compute Pólya filter p-values for each edge.

    The Pólya filter [1]_ tests each edge's weight against a null model in
    which a node hands out its strength over its links through a Pólya urn,
    a self-reinforcing process: every unit of weight placed on a link makes
    the next unit more likely to land on the same link.  The reinforcement
    parameter *a* sets how much heterogeneity the null model tolerates.

    From the viewpoint of a node with degree *k* and strength *s*, the
    weight of one of its links follows a beta-binomial distribution with
    *s* draws and shape parameters :math:`1/a` and :math:`(k-1)/a`
    (Eq. (1) of [1]_):

    .. math::

        P(w \mid k, s, a) = \binom{s}{w}
        \frac{B\left(\frac{1}{a} + w, \frac{k-1}{a} + s - w\right)}
        {B\left(\frac{1}{a}, \frac{k-1}{a}\right)},

    which is the Pólya urn that starts with one red ball and *k* - 1 black
    balls and adds *a* balls of the drawn colour after each of the *s*
    draws.  The p-value is the upper tail (Eq. (2) of [1]_)

    .. math::

        \pi_P(w \mid k, s, a) = 1 - \sum_{x=0}^{w-1} P(x \mid k, s, a).

    At ``a=0`` the urn draws with simple replacement and the null is the
    binomial distribution with *s* trials and success probability *1/k*.

    Each edge is tested from both of its endpoints and ``polya_pvalue`` is
    the smaller of the two p-values: [1]_ keeps a link when at least one of
    its p-values is significant.  For a directed edge (u, v) the two tests
    use the out-degree and out-strength of *u* and the in-degree and
    in-strength of *v* (Supplementary Note 2 of [1]_).

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key for weights.  All weights must be positive.
        Integer weights (counts) are tested exactly; see *approximate*.
    a : float or None, optional (default=None)
        Reinforcement parameter, ``a >= 0``.  ``a=1`` is close to the
        disparity filter, larger values give sparser backbones.  If ``None``,
        *a* is set to its maximum-likelihood estimate :math:`a_{ML}`
        (Supplementary Note 5 of [1]_), which needs integer weights.
    approximate : bool or None, optional (default=None)
        If ``False``, compute the exact p-values of Eq. (2), which needs
        integer weights.  If ``True``, use the large-strength approximation
        of Eq. (6) of [1]_, which needs ``a > 0``.  If ``None``, use the exact
        p-values when every weight is an integer and Eq. (6) otherwise, as
        [1]_ prescribes for non-integer weights.

    Returns
    -------
    H : graph
        A copy of *G* (same type) with a ``"polya_pvalue"`` edge attribute.
        ``H.graph["polya_a"]`` holds the value of *a* used (the estimate
        when ``a=None``; ``nan`` if no endpoint has degree above 1, in
        which case every p-value is 1 whatever *a* is).

    Raises
    ------
    NetworkXError
        If any edge has a non-positive, non-finite, or missing weight, or if
        a weight is not an integer when ``approximate=False`` or ``a=None``.
    NetworkXNotImplemented
        If *G* is a multigraph.
    ValueError
        If *a* is negative or not finite, if *approximate* is not ``None``,
        ``True`` or ``False``, or if ``approximate=True`` with ``a=0``.

    See Also
    --------
    disparity_filter : Large-strength limit of the Pólya filter at ``a=1``.
    threshold_filter : Keep edges whose p-value is below a threshold.

    Notes
    -----
    **Choosing a.**  [1]_ gives three criteria: sweeping *a* until the
    backbone reaches a target sparsity (backbones shrink monotonically as
    *a* grows, Eq. (3)), maximum likelihood, and a salience-based optimum.
    The default is the maximum-likelihood one, the only one that needs no
    further choice.  It maximizes the log-likelihood of Eq. (15) in the
    Supplementary Information of [1]_,

    .. math::

        \mathcal{L}(a) = \sum_{i,j} \log P(w_{ij} \mid k_i, s_i, a),

    over :math:`a \in [0, \infty)`, summing over every link from each of its
    endpoints.  The maximum is bracketed on a log-spaced grid from
    :math:`10^{-4}` to :math:`10^{4}` and refined by solving the score
    equation (16) of the Supplementary Information with Brent's method.  An
    endpoint of degree 1 contributes nothing to the likelihood
    (:math:`P = 1`).  For a directed graph Eq. (15) is applied as written to
    the non-symmetric weight matrix, that is with out-degrees and
    out-strengths; [1]_ does not spell out the directed likelihood.

    **Non-integer weights.**  Eq. (1) is defined for integer weights only.
    For other weights [1]_ uses the approximation, valid for
    :math:`s \gg k/a` and :math:`w \gg 1`,

    .. math::

        \pi_P(w \mid k, s, a) \approx \frac{1}{\Gamma(1/a)}
        \left(1 - \frac{w}{s}\right)^{(k-1)/a}
        \left(\frac{w k}{s a}\right)^{1/a - 1},

    which tests the rate of interaction *w/s* rather than the weight itself.
    At ``a=1`` it equals the disparity filter p-value
    :math:`(1 - w/s)^{k-1}` (Eq. (8)).  The approximation can exceed 1 for
    small *w*; such values are clipped to 1.

    **Degree-1 endpoints.**  An endpoint of degree 1 holds all of its
    strength on the one link, so its p-value is 1 and the edge is tested
    from the other endpoint only, as [1]_ prescribes.

    **Multiple testing.**  [1]_ applies a Bonferroni correction over
    :math:`L = 2m` tests for *m* undirected links, i.e. it keeps a link when
    ``polya_pvalue < alpha / (2 * m)``.  ``threshold_filter(..., mtc=...)``
    corrects over the *m* edge p-values instead.

    [1]_ defines only this upper-tail test, so there is no ``signed``
    variant.

    References
    ----------
    .. [1] Marcaccioli, R., & Livan, G. (2019). A Pólya urn approach to
       information filtering in complex networks. *Nature Communications*,
       10, 745. https://doi.org/10.1038/s41467-019-08667-3

    Examples
    --------
    The toy network of Fig. 1 of [1]_: node A has degree 3 and strength 4,
    and its link to B has weight 2.  At ``a=1`` the p-value is 6/15.

    >>> import networkx as nx
    >>> from networkx_backbone import polya_filter, threshold_filter
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([("A", "B", 2), ("A", "C", 1), ("A", "D", 1)])
    >>> H = polya_filter(G, a=1)
    >>> round(H["A"]["B"]["polya_pvalue"], 12)
    0.4

    With ``a=None`` the reinforcement parameter is fitted by maximum
    likelihood:

    >>> G = nx.les_miserables_graph()
    >>> H = polya_filter(G)
    >>> round(H.graph["polya_a"], 4)
    0.3397
    >>> backbone = threshold_filter(H, "polya_pvalue", 0.05, mode="below")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    if approximate not in (None, True, False):
        raise ValueError(
            f"approximate must be None, True or False, got {approximate!r}."
        )
    if a is not None:
        a = float(a)
        if not (0.0 <= a < math.inf):
            raise ValueError(f"a must be a finite number >= 0, got {a!r}.")

    H = G.copy()
    _validate_weights(H, weight)
    integer = all(float(d[weight]).is_integer() for _, _, d in H.edges(data=True))
    if approximate is None:
        approximate = not integer
    if not integer and (approximate is False or a is None):
        why = "approximate=False" if approximate is False else "a=None"
        raise nx.NetworkXError(
            f"polya_filter with {why} needs integer weights (Eq. (1) of "
            "Marcaccioli & Livan, 2019, is defined for counts); pass a "
            "value for a and approximate=None or True."
        )
    if approximate and a == 0.0:
        raise ValueError("approximate=True (Eq. (6)) needs a > 0.")

    def num(w):
        return int(w) if integer else float(w)

    # Two tests per edge: (k, s) of the endpoint the test is run from.
    directed = G.is_directed()
    k_u, k_v = (G.out_degree, G.in_degree) if directed else (G.degree, G.degree)
    s_u, s_v = {}, {}
    for u, v, d in H.edges(data=True):
        w = num(d[weight])
        s_u[u] = s_u.get(u, 0) + w
        s_v[v] = s_v.get(v, 0) + w
    if not directed:
        for node, val in s_v.items():
            s_u[node] = s_u.get(node, 0) + val
        s_v = s_u
    tests = [
        (
            (num(d[weight]), k_u[u], s_u[u]),
            (num(d[weight]), k_v[v], s_v[v]),
        )
        for u, v, d in H.edges(data=True)
    ]

    if a is None:
        # Eq. (15) sums over the rows of the weight matrix: both endpoints of
        # an undirected link, the source (out-) side of a directed one.
        rows = [t for pair in tests for t in (pair if not directed else pair[:1])]
        a = _polya_ml_estimate([t for t in rows if t[1] > 1])
        if approximate and a == 0.0:
            raise ValueError(
                "The maximum-likelihood estimate of a is 0, where "
                "approximate=True (Eq. (6)) is undefined; use approximate=False."
            )
    H.graph["polya_a"] = a

    informative = {t for pair in tests for t in pair if t[1] > 1}
    if approximate:
        pvalue = {t: _polya_approx_pvalue(*t, a) for t in informative}
    else:
        pvalue = _polya_exact_pvalues(informative, a)

    for (t_u, t_v), (_, _, data) in zip(tests, H.edges(data=True), strict=True):
        data["polya_pvalue"] = min(pvalue.get(t_u, 1.0), pvalue.get(t_v, 1.0))
    return H


def _polya_logpmf(x, k, s, a):
    """Log of Eq. (1) of Marcaccioli & Livan (2019) at array *x*; binomial at a=0."""
    import numpy as np
    from scipy.special import betaln, gammaln

    log_binom = gammaln(s + 1) - gammaln(x + 1) - gammaln(s - x + 1)
    if a == 0:
        return log_binom + x * np.log(1.0 / k) + (s - x) * np.log1p(-1.0 / k)
    alpha, beta = 1.0 / a, (k - 1) / a
    return log_binom + betaln(alpha + x, beta + s - x) - betaln(alpha, beta)


def _polya_exact_pvalues(tests, a):
    """Eq. (2) for every ``(w, k, s)`` test, one tail sum per distinct (k, s)."""
    import numpy as np

    by_node = {}
    for w, k, s in tests:
        by_node.setdefault((k, s), []).append(w)
    pvalue = {}
    for (k, s), ws in by_node.items():
        w_min = min(ws)
        logpmf = _polya_logpmf(np.arange(w_min, s + 1, dtype=float), k, s, a)
        # Upper tail P(X >= w) summed from the smallest terms up, in logs.
        log_tail = np.logaddexp.accumulate(logpmf[::-1])[::-1]
        for w in ws:
            pvalue[(w, k, s)] = min(float(np.exp(log_tail[w - w_min])), 1.0)
    return pvalue


def _polya_approx_pvalue(w, k, s, a):
    """Eq. (6) of Marcaccioli & Livan (2019), clipped to [0, 1]."""
    from scipy.special import gammaln

    if w >= s:
        return 0.0
    log_p = (
        -gammaln(1.0 / a)
        + (k - 1) / a * math.log1p(-w / s)
        + (1.0 / a - 1.0) * math.log(w * k / (s * a))
    )
    return min(math.exp(log_p), 1.0)


def _polya_ml_estimate(rows):
    """Maximum-likelihood *a* (Supplementary Note 5 of Marcaccioli & Livan, 2019).

    *rows* are the ``(w, k, s)`` terms of Eq. (15) with ``k > 1``.  Returns
    ``nan`` when there are none (the likelihood does not depend on *a*).
    """
    import numpy as np
    from scipy.optimize import brentq, minimize_scalar
    from scipy.special import betaln, digamma

    if not rows:
        return math.nan
    # Distinct terms in sorted order: the sums do not depend on edge order.
    terms, counts = np.unique(np.array(rows, dtype=float), axis=0, return_counts=True)
    w, k, s = terms.T

    def loglik(a):
        # Eq. (15) without the binomial coefficient, which does not depend on a.
        if a == 0:
            ll = w * np.log(1.0 / k) + (s - w) * np.log1p(-1.0 / k)
        else:
            ll = betaln(1 / a + w, (k - 1) / a + s - w) - betaln(1 / a, (k - 1) / a)
        return float(np.dot(counts, ll))

    def score(a):
        # Left-hand side of Eq. (16), which equals a**2 * dL/da.
        lhs = (
            -(k - 1) * digamma((k + a * s - a * w - 1) / a)
            + k * digamma(k / a + s)
            + (k - 1) * digamma((k - 1) / a)
            - k * digamma(k / a)
            - digamma(w + 1 / a)
            + digamma(1 / a)
        )
        return float(np.dot(counts, lhs))

    grid = np.logspace(-4, 4, 81)
    values = [loglik(x) for x in grid]
    i = int(np.argmax(values))
    if i == len(grid) - 1:
        warnings.warn(
            "polya_filter: the likelihood of a is still increasing at "
            f"a = {grid[-1]:g}; using that value.",
            RuntimeWarning,
            stacklevel=3,
        )
        return float(grid[-1])
    if i == 0:
        # Maximum in [0, 1e-4]: compare the a = 0 (binomial) end with the
        # best interior point.
        res = minimize_scalar(
            lambda x: -loglik(x),
            bounds=(0.0, grid[1]),
            method="bounded",
            options={"xatol": 1e-12},
        )
        return 0.0 if loglik(0.0) >= -res.fun else float(res.x)
    lo, hi = grid[i - 1], grid[i + 1]
    if score(lo) > 0 > score(hi):
        return float(brentq(score, lo, hi, xtol=1e-14, rtol=1e-12))
    res = minimize_scalar(
        lambda t: -loglik(math.exp(t)),
        bounds=(math.log(lo), math.log(hi)),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return float(math.exp(res.x))


def disparity(G, weight="weight", signed=False):
    """Alias for :func:`disparity_filter`."""
    return disparity_filter(G, weight=weight, signed=signed)


def mlf(G, weight="weight", signed=False):
    """Alias for :func:`marginal_likelihood_filter`."""
    return marginal_likelihood_filter(G, weight=weight, signed=signed)


def lans(G, weight="weight", signed=False):
    """Alias for :func:`lans_filter`."""
    return lans_filter(G, weight=weight, signed=signed)


_COMPLEXITY = {
    "disparity_filter": {
        "time": "O(n + m)",
        "space": "O(n + m)",
        "notes": "n=|V|, m=|E|.",
    },
    "noise_corrected_filter": {
        "time": "O(n + m)",
        "space": "O(n + m)",
        "notes": "n=|V|, m=|E|.",
    },
    "marginal_likelihood_filter": {
        "time": "O(n + m)",
        "space": "O(n + m)",
        "notes": "n=|V|, m=|E|.",
    },
    "ecm_filter": {
        "time": "O(I * n^2 + m)",
        "space": "O(n^2 + m)",
        "notes": "I=solver iterations; a dense O(n^3) Newton polish is used for n <= 2000.",
    },
    "lans_filter": {
        "time": "O(m log n)",
        "space": "O(n + m)",
        "notes": "Worst-case over node-local sorted edge-weight lookups.",
    },
    "multiple_linkage_analysis": {
        "time": "O(m log n)",
        "space": "O(n + m)",
        "notes": "Dominated by sorting each node's link weights (or lans_filter).",
    },
    "polya_filter": {
        "time": "O(n + m + S)",
        "space": "O(n + m + s_max)",
        "notes": "S=sum of distinct node strengths (exact tails); "
        "a=None adds O(m) per likelihood evaluation.",
    },
    "disparity": {
        "time": "O(n + m)",
        "space": "O(n + m)",
        "notes": "Alias for disparity_filter.",
    },
    "mlf": {
        "time": "O(n + m)",
        "space": "O(n + m)",
        "notes": "Alias for marginal_likelihood_filter.",
    },
    "lans": {
        "time": "O(m log n)",
        "space": "O(n + m)",
        "notes": "Alias for lans_filter.",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
