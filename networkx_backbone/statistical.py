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
gloss_filter
    Radicchi et al. (2011) -- global null with the empirical weight distribution.
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
    "gloss_filter",
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
# Helper: missing_as_zero (absent dyads scored as weight-0 edges)
# =====================================================================

# Above this many absent dyads, ``missing_as_zero=True`` warns: every one of
# them is stored as an edge of the returned graph (O(n^2) memory).
_MISSING_AS_ZERO_WARN_DYADS = 1_000_000


def _external_stacklevel():
    """``warnings.warn`` stacklevel of the first frame outside this package.

    Frame 1 (the caller of this helper) is stacklevel 1.  Frames in
    networkx-backbone and in NetworkX's decorator machinery are skipped, so
    the warning points at the user's call whether they called a scorer
    directly or through a wrapper such as ``backbone_from_weighted``.
    """
    import os
    import sys

    package_dir = os.path.dirname(os.path.abspath(__file__))
    decorators_file = nx.utils.decorators.__file__
    frame = sys._getframe(1)
    level = 1
    while frame is not None:
        filename = frame.f_code.co_filename
        if not (
            filename.startswith(package_dir)
            or filename == decorators_file
            or "argmap" in filename
        ):
            break
        frame = frame.f_back
        level += 1
    return level


def _absent_dyads(G, func_name):
    """List the node pairs of *G* that have no edge (``missing_as_zero``).

    Self-pairs are never included.  Undirected graphs give each unordered
    pair once; directed graphs give every ordered pair ``(u, v)`` without an
    edge ``u -> v``.  Warns when the list is larger than
    ``_MISSING_AS_ZERO_WARN_DYADS``.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges() - nx.number_of_selfloops(G)
    n_pairs = n * (n - 1) if G.is_directed() else n * (n - 1) // 2
    n_absent = n_pairs - m
    if n_absent > _MISSING_AS_ZERO_WARN_DYADS:
        warnings.warn(
            f"{func_name}: missing_as_zero=True adds {n_absent:,} weight-0 "
            "edges, one for every absent node pair; the result needs O(n^2) "
            "memory.",
            RuntimeWarning,
            stacklevel=_external_stacklevel(),
        )
    nodes = list(G)
    if G.is_directed():
        succ = G.succ
        return [(u, v) for u in nodes for v in nodes if u != v and v not in succ[u]]
    adj = G.adj
    return [
        (u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :] if v not in adj[u]
    ]


def _add_zero_edges(H, dyads, weight, tails, pvalue_attr, signed):
    """Add each scored absent dyad to *H* as a weight-0 edge.

    ``tails(u, v)`` returns ``(p_hi, p_lo, extra_attrs)`` for a dyad of
    weight 0, or ``None`` when the null model cannot test that dyad, in which
    case the pair is not added.
    """
    for u, v in dyads:
        scored = tails(u, v)
        if scored is None:
            continue
        p_hi, p_lo, extra = scored
        attrs = {weight: 0, **extra}
        if signed:
            attrs[pvalue_attr], attrs["sign"] = _combine_tails(p_hi, p_lo)
        else:
            attrs[pvalue_attr] = min(max(p_hi, 0.0), 1.0)
        H.add_edge(u, v, **attrs)


# =====================================================================
# 1. Disparity filter -- Serrano et al. (2009)
# =====================================================================


@not_implemented_for("multigraph")
def disparity_filter(G, weight="weight", signed=False, missing_as_zero=False):
    r"""Compute disparity filter p-values for each edge.

    The disparity filter [1]_ tests whether an edge's weight is
    statistically significant compared to a null model where each node's
    total strength is uniformly distributed across its edges.

    For each node *u* with degree *k_u* and strength *s_u*, the normalized
    weight of edge (u, v) is ``p_uv = w_uv / s_u``.  Under the null the
    probability of observing a normalized weight >= p_uv is::

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
    missing_as_zero : bool, optional (default=False)
        If ``True``, also test every pair of distinct nodes that has no edge,
        as an observed edge of weight 0, following ``missing_as_zero=TRUE``
        in the R backbone package (version 3.0).  Tested pairs are added to
        the result as edges with ``weight`` 0.  Meant for ``signed=True``;
        see Notes.

    Returns
    -------
    H : graph
        A copy of *G* (same type) with ``"disparity_pvalue"`` added as an
        edge attribute (and ``"sign"`` when ``signed=True``).  With
        ``missing_as_zero=True`` it also has the tested weight-0 edges.

    Raises
    ------
    NetworkXError
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.  Collapse parallel edges first with
        :func:`~networkx_backbone.multigraph_to_weighted`.

    Warns
    -----
    RuntimeWarning
        If ``missing_as_zero=True`` would add more than one million edges.

    See Also
    --------
    threshold_filter : Keep edges whose p-value is below a threshold.
    marginal_likelihood_filter : Null model that uses both endpoints jointly.

    Notes
    -----
    An endpoint of degree 1 cannot reject the null model, so it contributes a
    p-value of 1 and the edge's p-value comes from the other endpoint.

    **Comparison with R backbone.**  The one-tailed p-values equal R
    ``backbone::disparity`` and netbone ``disparity``.  For ``signed=True`` R
    takes the lower tail as ``1 - upper``, the complement of the combined
    upper tail ``min(alpha_u, alpha_v)``; here the OR rule is applied to each
    tail, giving ``min(1 - alpha_u, 1 - alpha_v)``.  The lower tail is
    therefore never larger than R's, so an edge can be significantly weak
    here and not in R (``tests/test_reference_implementations.py``).

    **Absent pairs** (``missing_as_zero=True``).  A weight-0 pair does not
    count toward any node's degree or strength; the null model of every
    existing edge is unchanged, as in R.  From an endpoint of degree
    :math:`k \geq 2` a weight of 0 has upper tail
    :math:`(1 - 0)^{k - 1} = 1` and lower tail 0, because the null
    distribution of the normalized weight is continuous.  So with
    ``signed=True`` every absent pair with such an endpoint gets a two-sided
    p-value of 0 and ``sign`` -1, as in R.  (R also gives a lower tail of 0
    when both endpoints have degree 1; here both endpoints then contribute
    1, by the degree-1 rule above.)  A pair with an endpoint that has no
    links (for a directed pair ``(u, v)``: *u* has no out-links or *v* no
    in-links) cannot be tested, because its normalized weight is
    :math:`0/0`; R gives it a missing p-value, and it is not added here.
    With ``signed=False`` the added pairs all have p-value 1.

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
        s_out = s_in = dict(G.degree(weight=weight))
        k_out = k_in = dict(G.degree())

    def tails(u, v, w):
        hi_u, lo_u = _disparity_node_tails(w, s_out[u], k_out[u])
        hi_v, lo_v = _disparity_node_tails(w, s_in[v], k_in[v])
        # OR rule: strong/weak if significant from either endpoint.
        return min(hi_u, hi_v), min(lo_u, lo_v)

    dyads = _absent_dyads(G, "disparity_filter") if missing_as_zero else []

    for u, v, data in H.edges(data=True):
        p_hi, p_lo = tails(u, v, data[weight])
        if signed:
            data["disparity_pvalue"], data["sign"] = _combine_tails(p_hi, p_lo)
        else:
            data["disparity_pvalue"] = p_hi

    def zero_tails(u, v):
        if k_out[u] == 0 or k_in[v] == 0:
            return None  # 0/0 normalized weight: not testable
        return (*tails(u, v, 0.0), {})

    _add_zero_edges(H, dyads, weight, zero_tails, "disparity_pvalue", signed)
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
def noise_corrected_filter(
    G, weight="weight", method="bayesian", signed=False, missing_as_zero=False
):
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
    missing_as_zero : bool, optional (default=False)
        If ``True``, also score every pair of distinct nodes that has no
        edge, as an observed edge of weight 0, and add the scored pairs to
        the result as edges with ``weight`` 0.  This mirrors
        ``missing_as_zero=TRUE`` in the R backbone package (version 3.0),
        which has no noise-corrected model.  Meant for ``signed=True``;
        see Notes.

    Returns
    -------
    H : graph
        A copy of *G* (same type).  Every edge gets ``"nc_score"`` and
        ``"nc_pvalue"``; with ``method="bayesian"`` also ``"nc_sdev"``; with
        ``signed=True`` also ``"sign"``.  With ``missing_as_zero=True`` it
        also has the scored weight-0 edges.

    Raises
    ------
    NetworkXError
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.
    ValueError
        If *method* is not ``"bayesian"`` or ``"zscore"``.

    Warns
    -----
    RuntimeWarning
        If ``missing_as_zero=True`` would add more than one million edges.

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

    **Comparison with netbone.**  netbone's ``noise_corrected`` uses the same
    score and standard deviation formulas but sums each undirected edge once
    for :math:`n_{..}` while counting it at both endpoints for the strengths,
    i.e. :math:`n_{..} = W`.  Its values equal ``nc_score`` and ``nc_sdev``
    computed with :math:`n_{..} = W` instead of :math:`2W`.

    **Absent pairs** (``missing_as_zero=True``).  The margins
    :math:`n_{i.}`, :math:`n_{.j}` and :math:`n_{..}` are computed from the
    existing edges only and are not changed by the weight-0 pairs.  A pair
    is scored with :math:`n_{ij} = 0` in the formulas above: its lift is 0,
    so ``nc_score`` is -1, and ``nc_sdev`` comes from the posterior with
    :math:`n_{ij} = 0`.  With ``method="zscore"`` the score is
    :math:`-\sqrt{W p / (1 - p)}`.  A pair with an endpoint whose margin is
    0 (no links; for a directed pair ``(i, j)``, no out-links of *i* or no
    in-links of *j*) has an undefined lift and is not added.  The R backbone
    package has no noise-corrected model to compare against; this is the
    model of the paper evaluated at a weight of 0.

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

    dyads = _absent_dyads(G, "noise_corrected_filter") if missing_as_zero else []

    if method == "zscore":
        _noise_corrected_zscore(G, H, weight, signed, dyads)
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

    def tails(n_ij, n_i, n_j):
        score, sdev = _nc_score_sdev(n_ij, n_i, n_j, total)
        if math.isinf(sdev):
            p_hi, p_lo = 1.0, 1.0  # degenerate prior: nothing to test
        elif sdev == 0.0:
            p_hi, p_lo = _normal_tails(math.copysign(math.inf, score) if score else 0.0)
        else:
            p_hi, p_lo = _normal_tails(score / sdev)
        return p_hi, p_lo, {"nc_score": score, "nc_sdev": sdev}

    for u, v, data in H.edges(data=True):
        p_hi, p_lo, extra = tails(data[weight], s_out[u], s_in[v])
        data.update(extra)
        if signed:
            data["nc_pvalue"], data["sign"] = _combine_tails(p_hi, p_lo)
        else:
            data["nc_pvalue"] = p_hi

    def zero_tails(u, v):
        n_i, n_j = s_out.get(u, 0.0), s_in.get(v, 0.0)
        if n_i == 0.0 or n_j == 0.0:
            return None  # zero margin: the lift is undefined
        return tails(0.0, n_i, n_j)

    _add_zero_edges(H, dyads, weight, zero_tails, "nc_pvalue", signed)
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


def _noise_corrected_zscore(G, H, weight, signed, dyads):
    """Legacy binomial z-score (networkx-backbone <= 0.2), written into *H*.

    *dyads* are absent pairs to score as weight-0 edges (``missing_as_zero``).
    """
    W = sum(d[weight] for _, _, d in G.edges(data=True))
    if G.is_directed():
        s_out = dict(G.out_degree(weight=weight))
        s_in = dict(G.in_degree(weight=weight))
    else:
        s_out = dict(G.degree(weight=weight))
        s_in = s_out

    def tails(w, s_u, s_v):
        p_ij = (s_u * s_v) / (W * W)
        variance = W * p_ij * (1 - p_ij)
        if variance > 0:
            z = (w - W * p_ij) / math.sqrt(variance)
        else:
            z = 0.0
        p_hi, p_lo = _normal_tails(z)
        return p_hi, p_lo, {"nc_score": z}

    for u, v, data in H.edges(data=True):
        p_hi, p_lo, extra = tails(data[weight], s_out[u], s_in[v])
        data.update(extra)
        if signed:
            data["nc_pvalue"], data["sign"] = _combine_tails(p_hi, p_lo)
        else:
            data["nc_pvalue"] = p_hi

    def zero_tails(u, v):
        s_u, s_v = s_out[u], s_in[v]
        if s_u == 0 or s_v == 0:
            return None  # zero margin: the lift is undefined
        return tails(0.0, s_u, s_v)

    _add_zero_edges(H, dyads, weight, zero_tails, "nc_pvalue", signed)


# =====================================================================
# 3. Marginal likelihood filter -- Dianati (2016)
# =====================================================================


@not_implemented_for("multigraph")
def marginal_likelihood_filter(G, weight="weight", signed=False, missing_as_zero=False):
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
    missing_as_zero : bool, optional (default=False)
        If ``True``, also test every pair of distinct nodes that has no edge,
        as an observed edge of weight 0, following ``missing_as_zero=TRUE``
        in the R backbone package (version 3.0).  Tested pairs are added to
        the result as edges with ``weight`` 0.  Meant for ``signed=True``;
        see Notes.

    Returns
    -------
    H : graph
        A copy of *G* (same type) with ``"ml_pvalue"`` edge attribute.  With
        ``missing_as_zero=True`` it also has the tested weight-0 edges.

    Raises
    ------
    NetworkXError
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.

    Warns
    -----
    RuntimeWarning
        If ``missing_as_zero=True`` would add more than one million edges.

    See Also
    --------
    disparity_filter : Tests each endpoint separately.
    ecm_filter : Maximum-entropy null that also fixes degrees.

    Notes
    -----
    The p-values, one-tailed and signed, equal R ``backbone::mlf`` and netbone's
    marginal likelihood filter on Les Misérables and karate
    (``tests/test_reference_implementations.py``).

    **Absent pairs** (``missing_as_zero=True``).  *T* and the strengths are
    computed from the existing edges only, so the null model of every
    existing edge is unchanged, as in R.  An absent pair is tested as
    :math:`X = 0` under :math:`B(T, p_{uv})`: the upper tail is
    :math:`P(X \geq 0) = 1` and the lower tail is
    :math:`P(X \leq 0) = (1 - p_{uv})^T`, so with ``signed=True`` a pair
    of strong nodes that never interact can be significantly weak.  As in R,
    every pair is tested, including pairs with a node of strength 0, whose
    p-value is 1.

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

    def tails(u, v, k):
        if T > 0 and u != v:
            p_param = min(s_out.get(u, 0) * s_in.get(v, 0) / norm, 1.0)
            p_hi = float(sp_stats.binom.sf(k - 1, T, p_param))
            p_lo = float(sp_stats.binom.cdf(k, T, p_param))
            return p_hi, p_lo
        return 1.0, 1.0

    for u, v, data in H.edges(data=True):
        p_hi, p_lo = tails(u, v, round(data[weight]))
        if signed:
            data["ml_pvalue"], data["sign"] = _combine_tails(p_hi, p_lo)
        else:
            data["ml_pvalue"] = float(p_hi)

    if missing_as_zero and T > 0:
        dyads = _absent_dyads(G, "marginal_likelihood_filter")
        _add_zero_edges(
            H, dyads, weight, lambda u, v: (*tails(u, v, 0), {}), "ml_pvalue", signed
        )
    return H


# =====================================================================
# 4. ECM filter -- Gemmetto et al. (2017)
# =====================================================================


@not_implemented_for("multigraph")
def ecm_filter(
    G, weight="weight", max_iter=1000, tol=1e-6, signed=False, missing_as_zero=False
):
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
    missing_as_zero : bool, optional (default=False)
        If ``True``, also test every pair of distinct nodes that has no edge,
        as an observed edge of weight 0, and add the tested pairs to the
        result as edges with ``weight`` 0.  This mirrors
        ``missing_as_zero=TRUE`` in the R backbone package (version 3.0),
        which has no ECM model.  Meant for ``signed=True``; see Notes.

    Returns
    -------
    H : graph
        A copy of *G* (same type) with ``"ecm_pvalue"`` edge attribute (and
        ``"sign"`` when ``signed=True``).  With ``missing_as_zero=True`` it
        also has the tested weight-0 edges.

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
        within ``tol``, or if ``missing_as_zero=True`` would add more than
        one million edges.

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

    **Absent pairs** (``missing_as_zero=True``).  The multipliers are fitted
    to the degrees and strengths of the existing edges only, so the null
    model of every existing edge is unchanged.  The ECM gives a weight of 0
    probability :math:`1 - p_{ij}`, so an absent pair has upper tail
    :math:`P(w_{ij} \geq 0) = 1` and lower tail
    :math:`P(w_{ij} \leq 0) = 1 - p_{ij}`.  A pair with a node of degree 0
    (for a directed pair ``(i, j)``, out-degree of *i* or in-degree of *j*)
    has :math:`p_{ij} = 0` and p-value 1.  The R backbone package has no ECM
    model to compare against; this is the ECM evaluated at a weight of 0.

    The p-values agree with netbone's ``ecm`` to about ``1e-5``, the
    tolerance of the two solvers (``tests/test_reference_implementations.py``).

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

    dyads = _absent_dyads(G, "ecm_filter") if missing_as_zero else []
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

    a_out, b_out, a_in, b_in, act_out, act_in, err = _fit_ecm(
        k_out, s_out, k_in, s_in, G.is_directed(), max_iter, tol
    )
    if err > tol:
        warnings.warn(
            f"ecm_filter: the ECM solver did not converge (largest relative "
            f"error {err:.2e} > tol={tol:g}); p-values may be inaccurate.",
            RuntimeWarning,
            stacklevel=2,
        )

    def tails(i, j, w):
        # x_i x_j y_i y_j = exp(-(a_i + a_j)); y_i y_j = exp(-(b_i + b_j)).
        # Work in log space: x can be huge when y is tiny.
        log_xxyy = -float(a_out[i] + a_in[j])
        log_yy = -float(b_out[i] + b_in[j])
        log_one_minus_yy = math.log(-math.expm1(log_yy))
        hi = max(log_xxyy, log_one_minus_yy)
        log_denom = hi + math.log(
            math.exp(log_xxyy - hi) + math.exp(log_one_minus_yy - hi)
        )
        log_p_link = log_xxyy - log_denom  # log p_ij = log P(w_ij >= 1)
        # y is 0 (b and log_yy infinite) for nodes whose links all have
        # weight 1, so (y_i y_j)^e is only formed for positive exponents e.
        if w == 0:
            p_hi = 1.0
        else:
            # Upper tail P(w_ij >= w) = p_ij (y_i y_j)^(w - 1) for w >= 1.
            log_p_hi = log_p_link + ((w - 1) * log_yy if w > 1 else 0.0)
            p_hi = float(min(max(math.exp(log_p_hi), 0.0), 1.0))
        # Lower tail P(w_ij <= w) = 1 - P(w_ij >= w + 1) = 1 - p_ij (y_i y_j)^w.
        log_p_next = log_p_link + (w * log_yy if w > 0 else 0.0)
        p_lo = -math.expm1(min(log_p_next, 0.0))
        return p_hi, p_lo

    for u, v, data in H.edges(data=True):
        if u == v:
            data["ecm_pvalue"] = 1.0
            if signed:
                data["sign"] = 1
            continue
        p_hi, p_lo = tails(idx[u], idx[v], max(round(data[weight]), 1))
        if signed:
            data["ecm_pvalue"], data["sign"] = _combine_tails(p_hi, p_lo)
        else:
            data["ecm_pvalue"] = p_hi

    def zero_tails(u, v):
        i, j = idx[u], idx[v]
        if not (act_out[i] and act_in[j]):
            return 1.0, 1.0, {}  # x = 0: the pair is never linked
        return (*tails(i, j, 0), {})

    _add_zero_edges(H, dyads, weight, zero_tails, "ecm_pvalue", signed)
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
def lans_filter(G, weight="weight", signed=False, missing_as_zero=False):
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
    missing_as_zero : bool, optional (default=False)
        If ``True``, also test every pair of distinct nodes that has no edge,
        as an observed edge of weight 0, following ``missing_as_zero=TRUE``
        in the R backbone package (version 3.0).  Tested pairs are added to
        the result as edges with ``weight`` 0.  Meant for ``signed=True``;
        see Notes.

    Returns
    -------
    H : graph
        A copy of *G* (same type) with ``"lans_pvalue"`` edge attribute.  With
        ``missing_as_zero=True`` it also has the tested weight-0 edges.

    Raises
    ------
    NetworkXError
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.

    Warns
    -----
    RuntimeWarning
        If ``missing_as_zero=True`` would add more than one million edges.

    See Also
    --------
    disparity_filter : Parametric counterpart with a uniform null model.

    Notes
    -----
    **Comparison with R backbone.**  The one-tailed p-values equal R
    ``backbone::lans`` and netbone ``lans``.  For ``signed=True`` R's lower tail
    is the mirror image of the upper tail, ``#{weights < w} / k``, so the weakest
    link of every node gets 0.  Here it is the empirical CDF
    ``#{weights <= w} / k``, which counts the link itself, so it is at least
    ``1 / k`` and a link is significantly weak only when ``1 / k < alpha / 2``.
    When both tails are 0 (for example the only link of a degree-1 node) R
    signs the link ``-1`` and this function ``+1``.  On Les Misérables at
    alpha = 0.05 R's signed backbone has 68 positive and 126 negative links
    (40 of them such ties); this one has 108 positive and none negative.  The
    two tails here are not symmetric (the upper one excludes the link, the
    lower one includes it), which may be worth revisiting.

    **Absent pairs** (``missing_as_zero=True``).  As in R, each node's
    empirical CDF is built from its existing links only; weight-0 pairs do
    not enter it, so the p-values of existing edges are unchanged.  A weight
    of 0 is below every link of a node, so an absent pair has upper tail 1
    and lower tail :math:`F(0) = 0`: with ``signed=True`` every absent pair
    gets a two-sided p-value of 0 and ``sign`` -1, as in R.  A pair with an
    endpoint that has no links (for a directed pair ``(u, v)``: *u* has no
    out-links or *v* no in-links) has no empirical CDF to compare against;
    R gives it a missing p-value, and it is not added here.

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
        out_weights = in_weights = {
            u: sorted(G[u][v][weight] for v in G[u]) for u in G.nodes()
        }

    def tails(u, v, w):
        ecdf_u = _empirical_cdf(w, out_weights[u])
        ecdf_v = _empirical_cdf(w, in_weights[v])
        # OR rule: strong/weak if significant from either side of the edge.
        return 1.0 - max(ecdf_u, ecdf_v), min(ecdf_u, ecdf_v)

    dyads = _absent_dyads(G, "lans_filter") if missing_as_zero else []

    for u, v, data in H.edges(data=True):
        p_hi, p_lo = tails(u, v, data[weight])
        if signed:
            data["lans_pvalue"], data["sign"] = _combine_tails(p_hi, p_lo)
        else:
            data["lans_pvalue"] = max(p_hi, 0.0)

    def zero_tails(u, v):
        if not out_weights[u] or not in_weights[v]:
            return None  # no links to compare against
        return (*tails(u, v, 0.0), {})

    _add_zero_edges(H, dyads, weight, zero_tails, "lans_pvalue", signed)
    return H


@not_implemented_for("multigraph")
def multiple_linkage_analysis(
    G, alpha=0.05, weight="weight", method="dominance", deviation="cod"
):
    r"""Extract a backbone using Multiple Linkage Analysis (MLA).

    With ``method="dominance"`` (the default) this is the multiple linkage
    analysis of Van Nuffel [1]_, [2]_, which decides how many of a node's
    links are *dominant*.  For each node:

    1. Sort its link weights in decreasing order and divide by their sum,
       giving the profile :math:`p_1 \geq p_2 \geq \dots \geq p_n`.  With
       the default ``deviation="cod"`` the profile also holds a zero for
       every other node the node has no link to, so :math:`n` is the number
       of other nodes in *G*.
    2. Compare the profile with the ideal distributions in which *k* links
       share the weight equally and the others have none,
       :math:`t^{(k)} = (1/k, \dots, 1/k, 0, \dots, 0)`, for *k* from 1 to
       the number of links.
    3. Keep the *k* strongest links of the best-fitting ideal distribution.

    The goodness of fit is set by *deviation*:

    ``"cod"`` (default)
        The coefficient of determination of [2]_, the squared Pearson
        correlation between :math:`p` and :math:`t^{(k)}` over all :math:`n`
        entries.  The largest value wins.
    ``"profile"``
        The squared deviation summed over the whole profile,
        :math:`D_k = \sum_{r=1}^{n} (p_r - t^{(k)}_r)^2`, so links outside
        the dominant set count against a fit.  The smallest value wins.
        This was the default in the development versions of 0.2.0.
    ``"weaver"``
        The mean squared deviation over the *k* links of the candidate set
        only, :math:`V_k = \frac{1}{k} \sum_{r=1}^{k} (p_r - 1/k)^2`, the
        variance of Weaver's crop-combination method [3]_.  The smallest
        value wins.

    Each node keeps at least one link.  For a directed graph the profile is
    built from each node's out-links (the setting of [1]_) and an edge is kept
    when it is dominant for its source.  For an undirected graph each node
    uses all of its links, and an edge is kept when it is dominant for
    *either* endpoint (OR rule).

    With ``method="lans"`` the edge p-values of :func:`lans_filter` are
    thresholded at ``alpha`` instead, as networkx-backbone 0.2 and earlier
    did.

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
        ``"lans"`` keeps edges with LANS p-value <= ``alpha``, as
        networkx-backbone 0.2 and earlier did.
    deviation : {"cod", "profile", "weaver"}, optional (default="cod")
        How ``method="dominance"`` measures the fit between a node's profile
        and an ideal distribution; see above.  Ignored by ``method="lans"``.

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
        If ``alpha`` is outside [0, 1], or *method* or *deviation* is
        unknown.
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
    The 2007 paper [1]_ was not available to us.  The default follows its
    restatement by the same first author in [2]_ (Section "Determination of
    the number of significant connections", and Table 1): the flows of a
    spatial unit, including zero flows to the units it is not linked to, are
    compared with each ideal distribution by the coefficient of
    determination, and "the number of significant flows corresponds to the
    jth cycle with the highest COD value".  The tests reproduce every COD
    value of Table 1 of [2]_.  A footnote of [2]_ describes MLA as an
    extension of Weaver's method [3]_, which "uses the sum of the squared
    differences (instead of the coefficient of determination) between the
    fictive and the real percentages".  ``deviation="weaver"`` implements
    Weaver's measure in its usual form, averaged over the *k* candidate
    links; this form has not been checked against [3]_ itself.

    Because the coefficient of determination depends on the number of zero
    entries, ``deviation="cod"`` results depend on the number of nodes in
    *G*, as in [2]_.  The other two measures do not.  If a node is linked to
    every other node, the ideal distribution with all :math:`n` links equal
    is constant, so its correlation is undefined and it is used only when
    all of the node's weights are tied.

    Ideal distributions that would split a group of tied weights are not
    considered, so tied links are kept or dropped together and ``mla_rank``
    is always the number of links the node keeps.  For ``"cod"`` and
    ``"profile"`` such a split never fits strictly better; for ``"weaver"``
    it sometimes does, so there this rule can change the result.  If two
    fits are equally good, the smaller *k* wins.

    Unlike LANS, which tests each link against the empirical distribution of
    its node's weights, the dominance fit chooses *how many* links each node
    keeps: a node with one overwhelming link keeps one, a node whose weight is
    spread evenly keeps all of them.

    netbone [2]_ (version 0.2.3, ``mlam``) fits the same ideal distributions
    by the squared correlation (rounded to two decimals) and stops at the
    first fit that does not improve; neither method here reproduces it.  It
    keeps fewer links than ``method="dominance"``: 134 on Les Misérables
    versus 205 with the default ``deviation="cod"`` (223 with ``"profile"``,
    253 with ``"weaver"``), all of them also kept here (``method="lans"``
    keeps 109).

    References
    ----------
    .. [1] Van Nuffel, N. (2007). Determination of the number of significant
       flows in origin-destination specific analysis: the case of commuting in
       Flanders. *Regional Studies*, 41(4), 509-524.
       https://doi.org/10.1080/00343400701281808
    .. [2] Van Nuffel, N., Derudder, B., & Witlox, F. (2010). Even important
       connections are not always meaningful: on the use of a polarisation
       measure in a typology of European cities in air transport networks.
       *Tijdschrift voor Economische en Sociale Geografie*, 101(3), 333-348.
       https://doi.org/10.1111/j.1467-9663.2009.00547.x
    .. [3] Weaver, J. C. (1954). Crop-combination regions in the Middle West.
       *Geographical Review*, 44(2), 175-200. https://doi.org/10.2307/212355
    .. [4] Yassin, A., Haidar, A., Cherifi, H., Seba, H., & Togni, O. (2023).
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

    The deviation measures can disagree.  For flows (50, 43, 42, 21, 12, 7)
    to six of eight other units (Table 1 of [2]_), the coefficient of
    determination picks three links, the whole-profile deviation four:

    >>> G = nx.DiGraph()
    >>> G.add_nodes_from("abcdefgh")
    >>> G.add_weighted_edges_from(
    ...     [("X", "h", 50), ("X", "a", 43), ("X", "d", 42),
    ...      ("X", "g", 21), ("X", "c", 12), ("X", "e", 7)]
    ... )
    >>> multiple_linkage_analysis(G).nodes["X"]["mla_rank"]
    3
    >>> multiple_linkage_analysis(G, deviation="profile").nodes["X"]["mla_rank"]
    4

    The LANS variant thresholds LANS p-values:

    >>> G = nx.les_miserables_graph()
    >>> H = multiple_linkage_analysis(G, alpha=0.05, method="lans")
    >>> backbone = boolean_filter(H, "mla_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    if method not in ("dominance", "lans"):
        raise ValueError(f"method must be 'dominance' or 'lans', got {method!r}")
    if deviation not in ("cod", "profile", "weaver"):
        raise ValueError(
            f"deviation must be 'cod', 'profile' or 'weaver', got {deviation!r}"
        )
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
    others = G.number_of_nodes() - 1
    cutoff = {}
    for node in G:
        weights = sorted((d[weight] for d in neighbors[node].values()), reverse=True)
        k = _dominance_rank(weights, deviation, max(others, len(weights)))
        H.nodes[node]["mla_rank"] = k
        cutoff[node] = weights[k - 1] if k else math.inf

    for u, v, data in H.edges(data=True):
        w = data[weight]
        keep = w >= cutoff[u]
        if not G.is_directed():
            keep = keep or w >= cutoff[v]
        data["mla_keep"] = bool(keep)
    return H


def _dominance_rank(weights, deviation="cod", n=None):
    """Number of dominant links for a node's weights sorted in decreasing order.

    Returns the *k* whose ideal distribution (*k* equal links, the rest zero)
    fits the normalized profile best under *deviation*; *k* never splits a
    run of tied weights, and ties in the fit go to the smaller *k*.  *n* is
    the length of the profile once padded with zero flows (``"cod"`` only;
    defaults to ``len(weights)``).

    Each measure is reduced to a loss that is minimized:

    - ``"profile"``: ``sum_r (p_r - t_r)**2`` minus the constant
      ``sum_r p_r**2``, i.e. ``(1 - 2 S_k) / k``.
    - ``"weaver"``: ``sum_{r<=k} (p_r - 1/k)**2 / k``.
    - ``"cod"``: minus ``(S_k / k - 1/n)**2 / (1/k - 1/n)``, which is the
      squared correlation times the constant ``n * var(p)``.

    Here ``S_k`` is the sum of the *k* largest shares.
    """
    m = len(weights)
    if m == 0:
        return 0
    if n is None:
        n = m
    total = math.fsum(weights)
    best_k, best_loss = m, math.inf
    cum = sq = 0.0
    for k in range(1, m + 1):
        share = weights[k - 1] / total
        cum += share
        sq += share * share
        if k < m and weights[k] == weights[k - 1]:
            continue  # would split a group of tied weights
        if deviation == "profile":
            loss = (1.0 - 2.0 * cum) / k
        elif deviation == "weaver":
            loss = (sq - 2.0 * cum / k + 1.0 / k) / k
        else:
            if k >= n:
                continue  # constant ideal: correlation undefined
            loss = -((cum / k - 1.0 / n) ** 2) / (1.0 / k - 1.0 / n)
        if loss < best_loss - 1e-12:
            best_k, best_loss = k, loss
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
    balls and adds *a* balls of the drawn color after each of the *s*
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


# =====================================================================
# 8. GloSS filter -- Radicchi, Ramasco & Fortunato (2011)
# =====================================================================


@not_implemented_for("multigraph")
def gloss_filter(G, weight="weight", resolution=None):
    r"""Compute Global Statistical Significance (GloSS) p-values for each edge.

    The GloSS filter [1]_ tests every edge weight against a *global* null
    model: the topology of *G* is kept as it is and every edge draws its
    weight independently from :math:`P_{obs}(w)`, the empirical
    distribution of all the edge weights of *G*.  The test of edge
    :math:`(i, j)` conditions on the degrees :math:`k_i, k_j` and strengths
    :math:`s_i, s_j` of its endpoints.  The weights of the other
    :math:`k_i - 1` edges of *i* must add up to :math:`s_i - w`, and those of
    *j* to :math:`s_j - w`, so the weight of :math:`(i, j)` has the
    posterior (Eqs. 1-2 of [1]_)

    .. math::

        P(w \mid s_i, k_i, s_j, k_j) \propto
        P_{obs}(w)\, F(s_i - w, k_i - 1)\, F(s_j - w, k_j - 1),

    where :math:`F(s, k)` is the probability that *k* independent draws from
    :math:`P_{obs}` sum to *s*, the *k*-fold convolution of
    :math:`P_{obs}` (Eq. 3; :math:`F(s, 0)` is 1 at :math:`s = 0` and 0
    elsewhere).  The p-value is the posterior probability of a weight at
    least as large as the observed one (Eq. 4):

    .. math::

        \alpha_{ij} = \frac{\sum_{w \geq w_{ij}} P_{obs}(w)\,
                            F(s_i - w, k_i - 1)\, F(s_j - w, k_j - 1)}
                           {\sum_{w} P_{obs}(w)\,
                            F(s_i - w, k_i - 1)\, F(s_j - w, k_j - 1)}.

    For a directed edge :math:`i \to j` the source's out-degree and
    out-strength replace :math:`k_i, s_i`, and the target's in-degree and
    in-strength replace :math:`k_j, s_j`, as in [1]_.  The p-values are
    computed from these formulas; nothing is simulated.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key for weights.  All weights must be positive.
    resolution : float or None, optional (default=None)
        Spacing of the lattice on which the weights are placed to compute
        the convolutions.  If ``None``, integer weights use the largest
        spacing that represents them exactly (the greatest common divisor of
        their differences from the smallest weight), and other weights use
        the smallest gap between two distinct weights, as in Appendix A of
        [1]_.  A coarser lattice is faster but rounds the weights more.

    Returns
    -------
    H : graph
        A copy of *G* (same type) with ``"gloss_pvalue"`` added as an edge
        attribute.

    Raises
    ------
    NetworkXError
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.  Collapse parallel edges first with
        :func:`~networkx_backbone.multigraph_to_weighted`.
    ValueError
        If *resolution* is not a positive finite number.

    See Also
    --------
    threshold_filter : Keep edges whose p-value is below a threshold.
    disparity_filter : Local null model that tests each node's edges.

    Notes
    -----
    In a simple graph the other edges of *i* and the other edges of *j* are
    disjoint sets, so :math:`\alpha_{ij}` is exactly the probability, under
    the null model, that edge :math:`(i, j)` gets a weight of at least
    :math:`w_{ij}` given the strengths of its endpoints.  The tail includes
    the observed weight, which makes it a valid p-value for discrete
    weights.  If an endpoint has degree 1, its strength fixes the weight of
    the edge (Sec. II A of [1]_) and the p-value is 1.  Self-loops are left
    out of the null model (of :math:`P_{obs}`, the degrees and the
    strengths) and get a p-value of 1.

    The weights are placed on the lattice :math:`w_{min} + n\,\delta` before
    convolving.  Every endpoint has a fixed number of free weights, so
    shifting all weights by the same constant only shifts each
    :math:`F(s, k)` by a known amount and leaves :math:`\alpha_{ij}`
    unchanged; integer weights are therefore represented exactly.
    Non-integer weights are rounded to the nearest lattice point, which
    approximates the p-values as the binning in [1]_ does.

    [1]_ evaluates the convolutions with fast Fourier transforms.  FFT
    round-off is absolute, about :math:`10^{-16}` times the largest
    probability, while the probabilities that matter for an endpoint whose
    strength is far from typical can be much smaller than that.  This
    implementation convolves directly instead, rescaling at every step and
    recomputing in log space any probability too small for that, so every
    term keeps full relative precision for any degree.  It takes
    :math:`O(k_{max}^2 L^2)` time, where *L* is the number of lattice steps
    between the smallest and the largest weight; pass a coarser
    *resolution* to reduce *L*.

    References
    ----------
    .. [1] Radicchi, F., Ramasco, J. J., & Fortunato, S. (2011).
       Information filtering in complex weighted networks.
       *Physical Review E*, 83(4), 046101.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import gloss_filter, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = gloss_filter(G)
    >>> backbone = threshold_filter(H, "gloss_pvalue", 0.05, mode="below")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True

    An edge whose endpoint has degree 1 is fully determined by that
    endpoint's strength, so its p-value is 1:

    >>> P = nx.path_graph(3)
    >>> nx.set_edge_attributes(P, {(0, 1): 1, (1, 2): 3}, "weight")
    >>> gloss_filter(P)[0][1]["gloss_pvalue"]
    1.0
    """
    if resolution is not None and not (0 < resolution < math.inf):
        raise ValueError(
            f"resolution must be a positive finite number, got {resolution!r}."
        )
    H = G.copy()
    _validate_weights(H, weight)
    for _, _, data in nx.selfloop_edges(H, data=True):
        data["gloss_pvalue"] = 1.0
    edges = [(u, v) for u, v in H.edges() if u != v]
    if not edges:
        return H

    import numpy as np

    w = np.array([H[u][v][weight] for u, v in edges], dtype=float)
    x = _gloss_lattice(w, resolution)
    values, counts = np.unique(x, return_counts=True)
    logp = np.log(counts / counts.sum())

    # Each edge (u, v) is tested from a row for u and a row for v.  A row
    # holds the degree and lattice strength of a node (undirected), or of its
    # out-side (rows 0..n-1) or in-side (rows n..2n-1) in a directed graph.
    index = {node: i for i, node in enumerate(H)}
    n = len(index)
    a = np.array([index[u] for u, _ in edges])
    b = np.array([index[v] for _, v in edges])
    if H.is_directed():
        b += n
    n_rows = 2 * n if H.is_directed() else n
    degree = np.bincount(np.concatenate([a, b]), minlength=n_rows)
    strength = np.zeros(n_rows, dtype=np.int64)
    np.add.at(strength, a, x)
    np.add.at(strength, b, x)

    # log F(s_r - w, k_r - 1) for every row r and every distinct weight w.
    log_f = _gloss_log_windows(degree - 1, strength, values, logp)

    pos = np.searchsorted(values, x)
    pvals = np.empty(len(edges))
    chunk = max(1, 2**22 // len(values))
    for start in range(0, len(edges), chunk):
        sl = slice(start, start + chunk)
        terms = logp + log_f[a[sl]] + log_f[b[sl]]
        # tail[:, t] = log of the sum of exp(terms[:, t:]) (Eq. 4 numerator).
        tail = np.logaddexp.accumulate(terms[:, ::-1], axis=1)[:, ::-1]
        num = tail[np.arange(len(tail)), pos[sl]]
        pvals[sl] = np.exp(num - tail[:, 0])
    np.clip(pvals, 0.0, 1.0, out=pvals)

    for (u, v), p in zip(edges, pvals):
        H[u][v]["gloss_pvalue"] = float(p)
    return H


def _gloss_lattice(w, resolution):
    """Integer lattice coordinates ``round((w - min(w)) / resolution)``.

    With ``resolution=None``, integer weights use the gcd of their offsets
    from the minimum (exact); other weights use the smallest gap between
    distinct weights (Radicchi et al., 2011, Appendix A).
    """
    import numpy as np

    offset = w - w.min()
    if not offset.any():
        return np.zeros(len(w), dtype=np.int64)
    if resolution is None:
        if np.all(w == np.floor(w)) and w.max() < 2**53:
            ints = offset.astype(np.int64)
            return ints // np.gcd.reduce(ints)
        resolution = float(np.diff(np.unique(w)).min())
    return np.floor(offset / resolution + 0.5).astype(np.int64)


def _gloss_log_windows(free, strength, values, logp):
    """``log F(strength[r] - values[u], free[r])`` for every row *r* and *u*.

    ``F(., d)`` is the d-fold convolution of the lattice pmf ``exp(logp)``
    supported on ``values``, kept in log space and built one convolution at a
    time.  Each step convolves directly (no FFT, so every output has full
    relative precision) after scaling by the maximum; outputs below
    ``_GLOSS_TINY`` of that maximum, where underflow could have cost precision,
    are recomputed as log-sum-exps.  Rows with ``free < 0`` stay at ``-inf``.
    """
    import numpy as np

    out = np.full((len(free), len(values)), -np.inf)
    active = np.flatnonzero(free >= 0)
    if active.size == 0:
        return out
    order = active[np.argsort(free[active], kind="stable")]
    d_max = int(free[order[-1]])
    bounds = np.searchsorted(free[order], np.arange(d_max + 2))
    pmf = np.zeros(int(values[-1]) + 1)
    pmf[values] = np.exp(logp)
    chunk = max(1, 2**22 // len(values))
    log_f = np.zeros(1)  # F(., 0): all the mass at 0
    for d in range(d_max + 1):
        if d:
            top = log_f.max()
            conv = np.convolve(np.exp(log_f - top), pmf)
            with np.errstate(divide="ignore"):
                nxt = np.log(conv) + top
            bad = np.flatnonzero(conv < _GLOSS_TINY)
            for i in range(0, len(bad), chunk):
                pos = bad[i : i + chunk]
                terms = _gloss_lookup(log_f, pos[:, None] - values) + logp
                nxt[pos] = np.logaddexp.reduce(terms, axis=1)
            log_f = nxt
        rows = order[bounds[d] : bounds[d + 1]]
        if rows.size:
            out[rows] = _gloss_lookup(log_f, strength[rows, None] - values)
    return out


# Relative size below which a directly convolved probability is recomputed in
# log space: far above the ~1e-308 at which the scaled inputs underflow.
_GLOSS_TINY = 1e-250


def _gloss_lookup(log_f, idx):
    """``log_f[idx]``, with ``-inf`` where *idx* is outside the array."""
    import numpy as np

    ok = (idx >= 0) & (idx < len(log_f))
    return np.where(ok, log_f[np.clip(idx, 0, len(log_f) - 1)], -np.inf)


def disparity(G, weight="weight", signed=False, missing_as_zero=False):
    """Alias for :func:`disparity_filter`."""
    return disparity_filter(
        G, weight=weight, signed=signed, missing_as_zero=missing_as_zero
    )


def mlf(G, weight="weight", signed=False, missing_as_zero=False):
    """Alias for :func:`marginal_likelihood_filter`."""
    return marginal_likelihood_filter(
        G, weight=weight, signed=signed, missing_as_zero=missing_as_zero
    )


def lans(G, weight="weight", signed=False, missing_as_zero=False):
    """Alias for :func:`lans_filter`."""
    return lans_filter(G, weight=weight, signed=signed, missing_as_zero=missing_as_zero)


_COMPLEXITY = {
    "disparity_filter": {
        "time": "O(n + m)",
        "space": "O(n + m)",
        "notes": "n=|V|, m=|E|. missing_as_zero=True adds O(n^2) time and space.",
    },
    "noise_corrected_filter": {
        "time": "O(n + m)",
        "space": "O(n + m)",
        "notes": "n=|V|, m=|E|. missing_as_zero=True adds O(n^2) time and space.",
    },
    "marginal_likelihood_filter": {
        "time": "O(n + m)",
        "space": "O(n + m)",
        "notes": "n=|V|, m=|E|. missing_as_zero=True adds O(n^2) time and space.",
    },
    "ecm_filter": {
        "time": "O(I * C^2 + n log n + m)",
        "space": "O(C^2 + n + m)",
        "notes": (
            "C=distinct (degree, strength) classes (directed: (k_out, s_out, "
            "k_in, s_in)), I=Newton iterations; a dense O(C^3) Newton solve "
            "is used up to 4000 parameters, Newton-CG above. "
            "missing_as_zero=True adds O(n^2) time and space."
        ),
    },
    "lans_filter": {
        "time": "O(m log n)",
        "space": "O(n + m)",
        "notes": "Worst-case over node-local sorted edge-weight lookups. missing_as_zero=True adds O(n^2) time and space.",
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
    "gloss_filter": {
        "time": "O(k_max^2 * L^2 + m * u)",
        "space": "O(k_max * L + n * u + m)",
        "notes": (
            "k_max=max degree, L=lattice steps spanned by the weights, "
            "u=distinct weights."
        ),
    },
    "disparity": {
        "time": "O(n + m)",
        "space": "O(n + m)",
        "notes": "Alias for disparity_filter. missing_as_zero=True adds O(n^2) time and space.",
    },
    "mlf": {
        "time": "O(n + m)",
        "space": "O(n + m)",
        "notes": "Alias for marginal_likelihood_filter. missing_as_zero=True adds O(n^2) time and space.",
    },
    "lans": {
        "time": "O(m log n)",
        "space": "O(n + m)",
        "notes": "Alias for lans_filter. missing_as_zero=True adds O(n^2) time and space.",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
