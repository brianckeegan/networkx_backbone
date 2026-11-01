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
    multipliers are found by maximizing the likelihood (damped Newton with
    the analytic gradient and Hessian), which is equivalent to solving
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
    of 1.

    Nodes with the same degree and strength (for directed graphs, the same
    out- and in-degree and strength) have the same multipliers, so the
    likelihood is maximized over these :math:`C` classes rather than over
    the :math:`n` nodes.  This is exact, and :math:`C` is usually much
    smaller than :math:`n` (520 classes for a 2,000-node Barabási-Albert
    graph with weights 1-20, 1,695 for 20,000 nodes; directed graphs have
    more classes).  A node whose links all have weight 1 gets
    :math:`y_i = 0`, the maximum-likelihood value.

    The fit uses damped Newton steps.  Each costs :math:`O(C^2)` time, with
    pair sums computed in fixed-size chunks so that their memory does not
    grow with :math:`C`.  The step is a dense solve, :math:`O(C^3)` time and
    :math:`O(C^2)` memory, for up to 4,000 parameters (2 per class, 4 for
    directed graphs), and conjugate gradients with Hessian-vector products,
    :math:`O(C^2)` time each, above that.

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
        # Upper tail P(w_ij >= w) = p_ij (y_i y_j)^(w - 1).  y is 0 (b and
        # log_yy infinite) for nodes whose links all have weight 1.
        log_p_hi = log_xxyy - log_denom + ((w - 1) * log_yy if w > 1 else 0.0)
        p_hi = float(min(max(math.exp(log_p_hi), 0.0), 1.0))
        if signed:
            # Lower tail P(w_ij <= w) = 1 - P(w_ij >= w + 1).
            p_lo = -math.expm1(min(log_p_hi + log_yy, 0.0))
            data["ecm_pvalue"], data["sign"] = _combine_tails(p_hi, p_lo)
        else:
            data["ecm_pvalue"] = p_hi

    return H


# Largest number of class parameters for which Newton steps use a dense
# Hessian solve; larger problems use Newton-CG with Hessian-vector products.
_ECM_NEWTON_MAX_PARAMS = 4000
# Number of class pairs evaluated at once; bounds the solver's working memory.
_ECM_CHUNK = 1 << 19
# Newton-CG keeps the per-pair curvature between Hessian-vector products when
# there are at most this many class pairs.
_ECM_CACHE_PAIRS = 1 << 22


def _fit_ecm(k_out, s_out, k_in, s_in, directed, max_iter, tol):
    """Maximum-likelihood ECM multipliers.

    Works with ``x * y = exp(-a)`` and ``y = exp(-b)``.  Parameterizing by
    ``x * y`` keeps the problem well conditioned for nodes whose links all
    have weight 1: their ``y`` is exactly 0 (``b = inf``) at the optimum,
    with ``x * y`` finite.

    The likelihood is strictly convex up to the directed gauge, so nodes
    that the constraints cannot tell apart share their multipliers at the
    optimum: nodes with the same degree and strength (undirected), or the
    same out- and in-degree and strength (directed).  The fit runs over
    these classes, which is exact.  For directed graphs it is warm-started
    from the cheaper fit over separate out- and in-classes, which is exact
    except for the excluded self-pairs.

    Nodes with zero degree are left out of the fit.  Returns
    ``(a_out, b_out, a_in, b_in, active_out, active_in, max_relative_error)``
    with one entry per node.
    """
    import numpy as np

    n = len(k_out)
    act_out = k_out > 0
    act_in = k_in > 0

    def unique_rows(*columns):
        keys, inverse, counts = np.unique(
            np.stack(columns, axis=1),
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        return keys, inverse.reshape(-1), counts.astype(float)

    def self_pairs(row, col, counts, n_rows, n_cols):
        # Aggregate (row, col, count) triples; result sorted by row.
        flat, inverse = np.unique(row * n_cols + col, return_inverse=True)
        total = np.bincount(inverse.reshape(-1), weights=counts, minlength=len(flat))
        r, c = np.divmod(flat, n_cols)
        return r, c, total

    if not directed:
        keys, node_class, mult = unique_rows(k_out, s_out)
        rows = np.flatnonzero(keys[:, 0] > 0)
        pos = np.full(len(keys), -1)
        pos[rows] = np.arange(len(rows))
        k, s, m = keys[rows, 0], keys[rows, 1], mult[rows]
        diag = np.arange(len(rows))
        a, b, _, _, err = _ecm_newton(
            (k, s, m), (k, s, m), (diag, diag, m), False, None, max_iter, tol
        )
        out_pos = in_pos = pos[node_class]
        a_o, b_o, a_i, b_i = a, b, a, b
    else:
        keys, node_class, mult = unique_rows(k_out, s_out, k_in, s_in)
        rows = np.flatnonzero(keys[:, 0] > 0)
        cols = np.flatnonzero(keys[:, 2] > 0)
        row_pos = np.full(len(keys), -1)
        row_pos[rows] = np.arange(len(rows))
        col_pos = np.full(len(keys), -1)
        col_pos[cols] = np.arange(len(cols))
        both = np.flatnonzero((row_pos >= 0) & (col_pos >= 0))
        out_side = (keys[rows, 0], keys[rows, 1], mult[rows])
        in_side = (keys[cols, 2], keys[cols, 3], mult[cols])
        selfp = (row_pos[both], col_pos[both], mult[both])

        # Warm start: classes of (k_out, s_out) and of (k_in, s_in) alone.
        okeys, ocls, _ = unique_rows(*out_side[:2])
        ikeys, icls, _ = unique_rows(*in_side[:2])
        theta0 = None
        if len(okeys) < len(rows) or len(ikeys) < len(cols):
            coarse = _ecm_newton(
                (okeys[:, 0], okeys[:, 1], np.bincount(ocls, weights=out_side[2])),
                (ikeys[:, 0], ikeys[:, 1], np.bincount(icls, weights=in_side[2])),
                self_pairs(
                    ocls[selfp[0]], icls[selfp[1]], selfp[2], len(okeys), len(ikeys)
                ),
                True,
                None,
                max_iter,
                tol,
            )
            theta0 = (
                coarse[0][ocls],
                coarse[1][ocls],
                coarse[2][icls],
                coarse[3][icls],
            )
        a_o, b_o, a_i, b_i, err = _ecm_newton(
            out_side, in_side, selfp, True, theta0, max_iter, tol
        )
        out_pos = row_pos[node_class]
        in_pos = col_pos[node_class]

    def per_node(values, pos, default):
        out = np.full(n, default)
        out[pos >= 0] = values[pos[pos >= 0]]
        return out

    return (
        per_node(a_o, out_pos, 0.0),
        per_node(b_o, out_pos, 1.0),
        per_node(a_i, in_pos, 0.0),
        per_node(b_i, in_pos, 1.0),
        act_out,
        act_in,
        err,
    )


def _ecm_newton(out_side, in_side, self_pairs, directed, theta0, max_iter, tol):
    """Damped Newton fit of the class-reduced ECM likelihood.

    ``out_side = (k, s, m)`` holds the out-degree, out-strength and number of
    nodes of each out-class (row), and ``in_side`` the same for the
    in-classes (columns).  Class pair ``(c, d)`` stands for
    ``m_out[c] * m_in[d]`` ordered node pairs, less the self-pairs: the
    ``self_pairs = (row, col, count)`` triples, sorted by row, count the
    nodes in out-class ``row`` and in-class ``col``.  For undirected graphs
    the two sides are the same classes and share their multipliers, and
    every unordered pair is counted twice.

    Pair sums are computed in chunks of about :data:`_ECM_CHUNK` class pairs.
    Newton steps are solved densely for up to
    :data:`_ECM_NEWTON_MAX_PARAMS` parameters and by block-Jacobi-preconditioned
    conjugate gradients (Hessian-vector products) above.

    Returns ``(a_out, b_out, a_in, b_in, max_relative_error)`` per class.
    """
    import numpy as np
    from scipy.linalg import LinAlgWarning, solve

    ko, so, mo = out_side
    ki, si, mi = in_side
    self_row, self_col, self_counts = self_pairs
    n_out, n_in = len(ko), len(ki)

    step_rows = max(1, _ECM_CHUNK // max(n_in, 1))
    chunks = []
    for r0 in range(0, n_out, step_rows):
        r1 = min(r0 + step_rows, n_out)
        lo, hi = np.searchsorted(self_row, [r0, r1])
        chunks.append((r0, r1, lo, hi))

    def pair_counts(r0, r1, lo, hi):
        W = mo[r0:r1, None] * mi[None, :]
        W[self_row[lo:hi] - r0, self_col[lo:hi]] -= self_counts[lo:hi]
        return W

    # Parameter layout.  Undirected graphs share out- and in-multipliers;
    # their objective is half the directed one, which counts each unordered
    # pair twice.
    ao = np.arange(n_out)
    bo = ao + n_out
    if directed:
        ai = np.arange(n_in) + 2 * n_out
        bi = ai + n_in
        n_par = 2 * (n_out + n_in)
        scale = 1.0
    else:
        ai, bi = ao, bo
        n_par = 2 * n_out
        scale = 0.5

    def pairs(theta, r0, r1):
        # log of x_i x_j y_i y_j and of y_i y_j for out-class rows r0:r1.
        log_yy = -(theta[bo[r0:r1], None] + theta[bi][None, :])
        log_xxyy = -(theta[ao[r0:r1], None] + theta[ai][None, :])
        one_minus_yy = -np.expm1(log_yy)
        # q = x_i x_j y_i y_j / (1 - y_i y_j); log Z = log(1 + q)
        log_q = log_xxyy - np.log(one_minus_yy)
        log_z = np.logaddexp(0.0, log_q)
        p = np.exp(log_q - log_z)  # link probability
        return p, one_minus_yy, log_yy, log_z

    def evaluate(theta):
        """Objective, gradient and largest relative error."""
        # With x*y = exp(-a), the sufficient statistic of b is s - k.
        # Pinned classes (b = inf) have s = k and contribute nothing.
        b_o = np.where(free[bo], theta[bo], 0.0)
        b_i = np.where(free[bi], theta[bi], 0.0)
        nll = mo @ (theta[ao] * ko + b_o * (so - ko)) + mi @ (
            theta[ai] * ki + b_i * (si - ki)
        )
        k_o, s_o = np.zeros(n_out), np.zeros(n_out)
        k_i, s_i = np.zeros(n_in), np.zeros(n_in)
        for r0, r1, lo, hi in chunks:
            W = pair_counts(r0, r1, lo, hi)
            p, one_minus_yy, _, log_z = pairs(theta, r0, r1)
            nll += np.sum(W * log_z)
            Wp = W * p
            Ww = Wp / one_minus_yy  # times the expected weight
            k_o[r0:r1] = Wp.sum(1)
            s_o[r0:r1] = Ww.sum(1)
            k_i += Wp.sum(0)
            s_i += Ww.sum(0)
        grad = np.zeros(n_par)
        grad[ao] += mo * ko - k_o
        grad[bo] += mo * (so - ko) - (s_o - k_o)
        grad[ai] += mi * ki - k_i
        grad[bi] += mi * (si - ki) - (s_i - k_i)

        def rel(obs, total, mult):
            return np.max(np.abs(total / mult - obs) / obs, initial=0.0)

        err = max(
            rel(ko, k_o, mo), rel(so, s_o, mo), rel(ki, k_i, mi), rel(si, s_i, mi)
        )
        return scale * nll, scale * grad, float(err)

    def curvature(theta, r0, r1, lo, hi):
        """Pair-count-weighted second derivatives of log Z in (a, b)."""
        W = pair_counts(r0, r1, lo, hi)
        p, one_minus_yy, log_yy, _ = pairs(theta, r0, r1)
        g = np.exp(log_yy) / one_minus_yy  # (expected weight - p) / p
        A = W * p * (1.0 - p)  # d2/da2
        B = A * g  # d2/da db
        C = B * g + W * p * g * (1.0 + g)  # d2/db2
        return A, B, C

    def dense_hessian(theta):
        # Each parameter block is a contiguous range, so the pair blocks are
        # added through slices.  Only one triangle of each symmetric pair of
        # blocks is filled (diagonal terms halved), then H += H.T.
        Hm = np.zeros((n_par, n_par))
        i_a = slice(ai[0], ai[-1] + 1) if n_in else slice(0, 0)
        i_b = slice(bi[0], bi[-1] + 1) if n_in else slice(0, 0)
        for r0, r1, lo, hi in chunks:
            A, B, C = curvature(theta, r0, r1, lo, hi)
            ro, rb = ao[r0:r1], bo[r0:r1]
            Hm[ro, ro] += 0.5 * A.sum(1)
            Hm[ro, rb] += B.sum(1)
            Hm[rb, rb] += 0.5 * C.sum(1)
            Hm[ai, ai] += 0.5 * A.sum(0)
            Hm[ai, bi] += B.sum(0)
            Hm[bi, bi] += 0.5 * C.sum(0)
            o_a = slice(ao[r0], ao[r0] + r1 - r0)
            o_b = slice(bo[r0], bo[r0] + r1 - r0)
            Hm[o_a, i_a] += A
            Hm[o_a, i_b] += B
            Hm[o_b, i_a] += B
            Hm[o_b, i_b] += C
        Hm += Hm.T
        Hm *= scale
        return Hm

    def newton_cg_step(theta, grad, eta):
        """Approximately solve H d = grad by block-Jacobi-preconditioned CG."""
        if n_out * n_in <= _ECM_CACHE_PAIRS:
            cached = [(c, curvature(theta, *c)) for c in chunks]

            def curvatures():
                return cached
        else:

            def curvatures():
                return ((c, curvature(theta, *c)) for c in chunks)

        # H = (2x2 block per class, from the pair-curvature row and column
        # sums) + (pair-curvature matrices coupling out- and in-parameters).
        a_idx = np.concatenate([ao, ai]) if directed else ao
        b_idx = np.concatenate([bo, bi]) if directed else bo
        d_aa, d_ab, d_bb = np.zeros(n_par), np.zeros(n_par), np.zeros(n_par)
        for (r0, r1, _, _), (A, B, C) in curvatures():
            for M, d in ((A, d_aa), (B, d_ab), (C, d_bb)):
                d[ao[r0:r1]] += M.sum(1)
                d[ai] += M.sum(0)
        d_aa, d_ab, d_bb = scale * d_aa[a_idx], scale * d_ab[a_idx], scale * d_bb[a_idx]
        b_free = free[b_idx]
        det = np.where(b_free, d_aa * d_bb - d_ab**2, 1.0)

        def hessp(v):
            out = np.zeros(n_par)
            for (r0, r1, _, _), (A, B, C) in curvatures():
                va, vb = v[ao[r0:r1]], v[bo[r0:r1]]
                out[ao[r0:r1]] += A @ v[ai] + B @ v[bi]
                out[bo[r0:r1]] += B @ v[ai] + C @ v[bi]
                out[ai] += va @ A + vb @ B
                out[bi] += va @ B + vb @ C
            out *= scale
            va, vb = v[a_idx], v[b_idx]
            out[a_idx] += d_aa * va + d_ab * vb
            out[b_idx] += d_ab * va + d_bb * vb
            return out

        def precondition(r):
            # Inverse of the 2x2 diagonal blocks (just d_aa for pinned b).
            z = np.zeros(n_par)
            ra, rb = r[a_idx], r[b_idx]
            z[a_idx] = np.where(b_free, (d_bb * ra - d_ab * rb) / det, ra / d_aa)
            z[b_idx] = np.where(b_free, (d_aa * rb - d_ab * ra) / det, 0.0)
            return z

        target = eta * np.linalg.norm(grad)
        x = np.zeros(n_par)
        r = grad.copy()
        z = precondition(r)
        d = z.copy()
        rz = r @ z
        for _ in range(max(100, n_par // 20)):
            Hd = hessp(d)
            curv = d @ Hd
            if curv <= 0:
                break
            alpha = rz / curv
            x += alpha * d
            r -= alpha * Hd
            if np.linalg.norm(r) <= target:
                break
            z = precondition(r)
            rz_new = r @ z
            d = z + (rz_new / rz) * d
            rz = rz_new
        return x if x.any() else precondition(grad)

    theta = np.zeros(n_par)
    if theta0 is None:
        total = max(mo @ ko, 1.0)

        def start(k, s):
            # Chung-Lu-like guess for x*y, and mean link weight 1/(1-y) ~ s/k.
            a = -np.log(k / np.sqrt(total))
            ratio = np.maximum(s / k, 1.0 + 1e-3)
            return a, -np.log1p(-1.0 / ratio)

        theta[ao], theta[bo] = start(ko, so)
        theta[ai], theta[bi] = start(ki, si)
    else:
        theta[ao], theta[bo], theta[ai], theta[bi] = theta0
    # A class whose links all have weight 1 (s = k) has y = 0 at the optimum:
    # its b is pinned to infinity and left out of the Newton system.
    free = np.ones(n_par, dtype=bool)
    free[bo[so == ko]] = False
    free[bi[si == ki]] = False
    theta[~free] = np.inf
    idx = np.flatnonzero(free)

    # The directed model has a gauge freedom: shifting every out-parameter
    # up and every in-parameter down by the same amount changes nothing, so
    # the Hessian is singular along these directions.  Adding them as
    # rank-one terms makes it invertible without changing the Newton step,
    # because the gradient is orthogonal to them.
    gauge = []
    if directed:
        for out_idx, in_idx in ((ao, ai), (bo, bi)):
            u = np.zeros(n_par)
            u[out_idx] = 1.0
            u[in_idx] = -1.0
            u[~free] = 0.0
            if u.any():
                gauge.append(u[idx] / np.linalg.norm(u))

    # Damped Newton.  The negative log-likelihood is convex and grows
    # without bound as any y_i y_j -> 1, so with a backtracking line search
    # (which rejects steps leaving that domain) it converges globally and,
    # near the optimum, quadratically.
    dense = len(idx) <= _ECM_NEWTON_MAX_PARAMS
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        f, grad, err = evaluate(theta)
        polish = True
        for _ in range(max_iter):
            # Once within tol, one more (quadratically convergent) step makes
            # the result essentially independent of where the solver stopped.
            if err <= tol:
                if not polish or err <= 1e-3 * tol:
                    break
                polish = False
            if dense:
                Hm = dense_hessian(theta)
                if len(idx) < n_par:
                    Hm = Hm[np.ix_(idx, idx)]
                shift = np.trace(Hm) / max(len(idx), 1)
                for u in gauge:
                    Hm += shift * np.outer(u, u)
                step = np.zeros(n_par)
                try:
                    # Positive definite: a Cholesky solve.  An ill-conditioned
                    # Hessian is fine here; the line search guards the step.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", LinAlgWarning)
                        step[idx] = solve(Hm, grad[idx], assume_a="pos")
                except (np.linalg.LinAlgError, ValueError):
                    step[idx] = np.linalg.lstsq(Hm, grad[idx], rcond=None)[0]
            else:
                # Inexact Newton: solve more accurately as the margins
                # converge, for superlinear convergence.
                step = newton_cg_step(theta, grad, min(0.5, np.sqrt(err)))
            step[~free] = 0.0
            t = 1.0
            while t > 1e-10:
                cand = theta - t * step
                f_new, grad_new, err_new = evaluate(cand)
                # Near the optimum f is flat to rounding error; then accept
                # a step that reduces the error in the margins.
                if np.isfinite(f_new) and (
                    f_new <= f - 1e-4 * t * (grad @ step)
                    or (f_new - f <= 1e-12 * abs(f) and err_new < err)
                ):
                    break
                t *= 0.5
            else:
                break
            theta, f, grad, err = cand, f_new, grad_new, err_new

    return theta[ao], theta[bo], theta[ai], theta[bi], err


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
        "time": "O(I * C^2 + n log n + m)",
        "space": "O(C^2 + n + m)",
        "notes": (
            "C=distinct (degree, strength) classes (directed: (k_out, s_out, "
            "k_in, s_in)), I=Newton iterations; a dense O(C^3) Newton solve "
            "is used up to 4000 parameters, Newton-CG above."
        ),
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
