"""
Bipartite projection backbone methods.

These methods operate on bipartite networks and extract significant edges
in the one-mode projection among "agent" nodes.

Methods
-------
simple_projection
    Shared-neighbor weighted projection.
hyper_projection
    Hyperbolic weighted projection.
probs_projection
    ProbS random-walk weighted projection.
ycn_projection
    YCN stationary-flow weighted projection.
sdsm
    Stochastic Degree Sequence Model (Neal 2014).
fdsm
    Fixed Degree Sequence Model (Neal et al. 2021).
"""

import networkx as nx

__all__ = [
    # Bipartite projection methods
    "simple_projection",
    "hyper_projection",
    "probs_projection",
    "ycn_projection",
    "bipartite_projection",
    # Projection methods
    "sdsm",
    "fdsm",
    "fixedfill",
    "fixedrow",
    "fixedcol",
    # Utilities
    "bicm",
    "fastball",
    # High-level wrappers
    "backbone_from_projection",
    "backbone_from_weighted",
    "backbone_from_unweighted",
    "backbone",
]


def _validate_bipartite(B, agent_nodes):
    """Validate that B is bipartite and agent_nodes is a valid partition."""
    if not nx.is_bipartite(B):
        raise nx.NetworkXError("Input graph B is not bipartite.")
    all_nodes = set(B.nodes())
    agent_set = set(agent_nodes)
    if not agent_set.issubset(all_nodes):
        raise nx.NetworkXError("agent_nodes contains nodes not in B.")


def _bipartite_projection_matrix(B, agent_nodes):
    """Build the co-occurrence matrix for agent nodes.

    Returns
    -------
    agents : list
        Ordered list of agent nodes.
    artifacts : list
        Ordered list of artifact nodes.
    R : np.ndarray of shape (n_agents, n_artifacts)
        Binary incidence matrix (agents x artifacts).
    observed : np.ndarray of shape (n_agents, n_agents)
        Observed co-occurrence counts (symmetric).
    """
    import numpy as np

    agent_set = set(agent_nodes)
    agents = sorted(agent_set, key=str)
    artifacts = sorted(set(B.nodes()) - agent_set, key=str)

    a_idx = {v: i for i, v in enumerate(agents)}
    f_idx = {v: i for i, v in enumerate(artifacts)}

    na = len(agents)
    nf = len(artifacts)
    R = np.zeros((na, nf), dtype=int)

    for u, v in B.edges():
        if u in a_idx and v in f_idx:
            R[a_idx[u], f_idx[v]] = 1
        elif v in a_idx and u in f_idx:
            R[a_idx[v], f_idx[u]] = 1

    # Observed co-occurrence matrix
    observed = R @ R.T
    np.fill_diagonal(observed, 0)

    return agents, artifacts, R, observed


def _infer_agent_nodes(B):
    """Infer a candidate agent partition from bipartite node attributes or coloring."""
    attr_agents = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]
    if attr_agents:
        return sorted(attr_agents, key=str)

    colors = nx.algorithms.bipartite.color(B)
    color0 = [n for n, c in colors.items() if c == 0]
    return sorted(color0, key=str)


def _projection_weight_matrix(R, method="simple", symmetrize=True, max_iter=1000, tol=1e-12):
    """Compute projection weights from a binary incidence matrix."""
    import numpy as np

    method_l = method.lower()
    if method_l == "hyperbolic":
        method_l = "hyper"

    if method_l == "simple":
        W = (R @ R.T).astype(float)
        np.fill_diagonal(W, 0.0)
        return W

    artifact_degree = R.sum(axis=0).astype(float)
    inv_artifact_degree = np.zeros_like(artifact_degree, dtype=float)
    nonzero_artifact = artifact_degree > 0
    inv_artifact_degree[nonzero_artifact] = 1.0 / artifact_degree[nonzero_artifact]
    H = (R * inv_artifact_degree) @ R.T

    if method_l == "hyper":
        W = H.astype(float)
        np.fill_diagonal(W, 0.0)
        return W

    agent_degree = R.sum(axis=1).astype(float)
    W_dir = np.divide(
        H,
        agent_degree[:, None],
        out=np.zeros_like(H, dtype=float),
        where=agent_degree[:, None] > 0,
    )

    if method_l == "probs":
        np.fill_diagonal(W_dir, 0.0)
        if symmetrize:
            W = 0.5 * (W_dir + W_dir.T)
            np.fill_diagonal(W, 0.0)
            return W
        return W_dir

    if method_l == "ycn":
        # Transition matrix for a random walk on the projected layer.
        T = W_dir.copy()
        zero_rows = np.where(agent_degree <= 0)[0]
        if zero_rows.size:
            T[zero_rows, :] = 0.0
            T[zero_rows, zero_rows] = 1.0

        n_agents = T.shape[0]
        if n_agents == 0:
            return np.zeros((0, 0), dtype=float)

        pi = np.full(n_agents, 1.0 / n_agents, dtype=float)
        for _ in range(max_iter):
            next_pi = pi @ T
            if np.linalg.norm(next_pi - pi, ord=1) <= tol:
                pi = next_pi
                break
            pi = next_pi

        total = pi.sum()
        if total > 0:
            pi = pi / total

        W_dir = pi[:, None] * T
        np.fill_diagonal(W_dir, 0.0)
        if symmetrize:
            W = 0.5 * (W_dir + W_dir.T)
            np.fill_diagonal(W, 0.0)
            return W
        return W_dir

    raise ValueError(
        "Unknown projection method. Choose one of: "
        "'simple', 'hyper', 'probs', 'ycn'."
    )


def _projection_graph_from_matrix(agents, W, weight="weight", directed=False):
    """Create a weighted projected graph from a dense weight matrix."""
    projection = nx.DiGraph() if directed else nx.Graph()
    projection.add_nodes_from(agents)

    n_agents = len(agents)
    if directed:
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    continue
                wij = float(W[i, j])
                if wij > 0:
                    projection.add_edge(agents[i], agents[j], **{weight: wij})
        return projection

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            wij = float(W[i, j])
            if wij > 0:
                projection.add_edge(agents[i], agents[j], **{weight: wij})
    return projection


def simple_projection(B, agent_nodes, weight="weight"):
    """Project a bipartite graph using shared-neighbor counts.

    For agent nodes ``u`` and ``v``, the weight is
    ``|Gamma(u) ∩ Gamma(v)|``.

    References
    ----------
    .. [1] Coscia, M. & Neffke, F. (2017). Network backboning with noisy
       data. arXiv:1906.09081.
    """
    _validate_bipartite(B, agent_nodes)
    agents, _, R, _ = _bipartite_projection_matrix(B, agent_nodes)
    W = _projection_weight_matrix(R, method="simple")
    return _projection_graph_from_matrix(agents, W, weight=weight, directed=False)


def hyper_projection(B, agent_nodes, weight="weight"):
    """Project a bipartite graph using hyperbolic weighting.

    For agent nodes ``u`` and ``v``, the weight is
    ``sum_{z in Gamma(u) ∩ Gamma(v)} 1 / d(z)`` where ``d(z)`` is the
    artifact degree.

    References
    ----------
    .. [1] Coscia, M. & Neffke, F. (2017). Network backboning with noisy
       data. arXiv:1906.09081.
    """
    _validate_bipartite(B, agent_nodes)
    agents, _, R, _ = _bipartite_projection_matrix(B, agent_nodes)
    W = _projection_weight_matrix(R, method="hyper")
    return _projection_graph_from_matrix(agents, W, weight=weight, directed=False)


def probs_projection(B, agent_nodes, weight="weight", directed=False):
    """Project a bipartite graph using ProbS random-walk weights.

    The directed score from ``u`` to ``v`` is
    ``(1 / d(u)) * sum_{z in Gamma(u) ∩ Gamma(v)} 1 / d(z)``.

    References
    ----------
    .. [1] Coscia, M. & Neffke, F. (2017). Network backboning with noisy
       data. arXiv:1906.09081.
    """
    _validate_bipartite(B, agent_nodes)
    agents, _, R, _ = _bipartite_projection_matrix(B, agent_nodes)
    W = _projection_weight_matrix(R, method="probs", symmetrize=not directed)
    return _projection_graph_from_matrix(agents, W, weight=weight, directed=directed)


def ycn_projection(B, agent_nodes, weight="weight", directed=False, max_iter=1000, tol=1e-12):
    """Project a bipartite graph using YCN random-walk stationary flow.

    This method estimates a random-walk transition matrix on the projected
    layer and scores pairs by stationary flow ``pi_u * T_uv``.

    References
    ----------
    .. [1] Coscia, M. & Neffke, F. (2017). Network backboning with noisy
       data. arXiv:1906.09081.
    """
    _validate_bipartite(B, agent_nodes)
    agents, _, R, _ = _bipartite_projection_matrix(B, agent_nodes)
    W = _projection_weight_matrix(
        R, method="ycn", symmetrize=not directed, max_iter=max_iter, tol=tol
    )
    return _projection_graph_from_matrix(agents, W, weight=weight, directed=directed)


def bipartite_projection(
    B,
    agent_nodes,
    method="simple",
    weight="weight",
    directed=False,
    max_iter=1000,
    tol=1e-12,
):
    """Dispatch weighted bipartite projections by method name.

    Supported methods are ``"simple"``, ``"hyper"``, ``"probs"``, and
    ``"ycn"``.
    """
    method_l = method.lower()
    if method_l == "simple":
        if directed:
            raise ValueError("simple projection does not support directed=True")
        return simple_projection(B, agent_nodes=agent_nodes, weight=weight)
    if method_l in ("hyper", "hyperbolic"):
        if directed:
            raise ValueError("hyper projection does not support directed=True")
        return hyper_projection(B, agent_nodes=agent_nodes, weight=weight)
    if method_l == "probs":
        return probs_projection(B, agent_nodes=agent_nodes, weight=weight, directed=directed)
    if method_l == "ycn":
        return ycn_projection(
            B,
            agent_nodes=agent_nodes,
            weight=weight,
            directed=directed,
            max_iter=max_iter,
            tol=tol,
        )
    raise ValueError(
        "Unknown projection method. Choose one of: "
        "'simple', 'hyper', 'probs', 'ycn'."
    )


def _apply_projection_weights(
    graph,
    B,
    agent_nodes,
    projection="simple",
    projection_weight="weight",
    projection_directed=False,
    projection_max_iter=1000,
    projection_tol=1e-12,
):
    """Annotate graph edges with weights from a selected bipartite projection."""
    if projection_directed:
        raise ValueError("projection_directed=True is not supported for undirected backbones")

    projected = bipartite_projection(
        B,
        agent_nodes=agent_nodes,
        method=projection,
        weight=projection_weight,
        directed=False,
        max_iter=projection_max_iter,
        tol=projection_tol,
    )
    for u, v in graph.edges():
        if projected.has_edge(u, v):
            graph[u][v][projection_weight] = float(projected[u][v][projection_weight])
        else:
            graph[u][v][projection_weight] = 0.0

    return graph


def bicm(B, agent_nodes, return_labels=False):
    """Estimate Bipartite Configuration Model edge probabilities.

    Parameters
    ----------
    B : graph
        A bipartite NetworkX graph.
    agent_nodes : iterable
        Nodes in the "agent" partition.
    return_labels : bool, optional (default=False)
        If ``True``, also return ``(agents, artifacts)`` labels.

    Returns
    -------
    P : np.ndarray
        Probability matrix of shape ``(n_agents, n_artifacts)`` where
        ``P[i, k]`` is the estimated probability of an edge between
        agent ``i`` and artifact ``k``.
    (P, agents, artifacts) : tuple
        Returned when ``return_labels=True``.
    """
    import numpy as np

    _validate_bipartite(B, agent_nodes)
    agents, artifacts, R, _ = _bipartite_projection_matrix(B, agent_nodes)
    row_sums = R.sum(axis=1).astype(float)
    col_sums = R.sum(axis=0).astype(float)
    total = R.sum()

    if total == 0:
        P = np.zeros_like(R, dtype=float)
    else:
        P = np.outer(row_sums, col_sums) / total
        P = np.clip(P, 0, 1)

    if return_labels:
        return P, agents, artifacts
    return P


def fastball(matrix, n_swaps=None, seed=None):
    """Randomize a binary matrix while preserving row and column sums.

    This is a Curveball-style randomizer commonly used for bipartite
    null-model generation.

    Parameters
    ----------
    matrix : array-like
        Binary matrix with shape ``(n_rows, n_cols)``.
    n_swaps : int or None, optional
        Number of row-pair trades. Defaults to ``5 * n_rows``.
    seed : integer, random_state, or None
        Random seed for reproducibility.

    Returns
    -------
    randomized : np.ndarray
        Randomized binary matrix with preserved row/column sums.
    """
    import numpy as np

    M = np.asarray(matrix)
    if M.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    R = (M > 0).astype(int)
    n_rows, n_cols = R.shape
    if n_rows == 0 or n_cols == 0:
        return R.copy()

    if n_swaps is None:
        n_swaps = max(1, n_rows * 5)
    if n_swaps < 0:
        raise ValueError("n_swaps must be >= 0")

    rng = np.random.default_rng(seed)
    row_sets = [set(np.flatnonzero(R[i])) for i in range(n_rows)]

    for _ in range(n_swaps):
        if n_rows < 2:
            break
        r1, r2 = rng.choice(n_rows, size=2, replace=False)
        s1 = row_sets[r1]
        s2 = row_sets[r2]
        only1 = list(s1 - s2)
        only2 = list(s2 - s1)
        if len(only1) == 0 or len(only2) == 0:
            continue

        common = s1 & s2
        trade_pool = only1 + only2
        rng.shuffle(trade_pool)
        keep1 = len(only1)

        row_sets[r1] = common | set(trade_pool[:keep1])
        row_sets[r2] = common | set(trade_pool[keep1:])

    out = np.zeros((n_rows, n_cols), dtype=int)
    for i, cols in enumerate(row_sets):
        if cols:
            out[i, list(cols)] = 1
    return out


def fixedfill(
    B,
    agent_nodes,
    alpha=0.05,
    projection="simple",
    projection_weight="weight",
    projection_directed=False,
    projection_max_iter=1000,
    projection_tol=1e-12,
):
    """Backbone from a fixed-fill null model for bipartite projections.

    Under the null, each bipartite cell is occupied independently with
    probability equal to the observed fill rate.
    """
    from scipy import stats as sp_stats

    _validate_bipartite(B, agent_nodes)
    agents, artifacts, R, observed = _bipartite_projection_matrix(B, agent_nodes)
    na = len(agents)
    nf = len(artifacts)
    total = R.sum()

    backbone = nx.Graph()
    backbone.add_nodes_from(agents)

    if na < 2 or nf == 0:
        return _apply_projection_weights(
            backbone,
            B,
            agent_nodes,
            projection=projection,
            projection_weight=projection_weight,
            projection_directed=projection_directed,
            projection_max_iter=projection_max_iter,
            projection_tol=projection_tol,
        )

    fill = total / (na * nf)
    q = min(max(fill * fill, 0.0), 1.0)

    for i in range(na):
        for j in range(i + 1, na):
            obs = int(observed[i, j])
            pval = float(sp_stats.binom.sf(obs - 1, nf, q))
            if pval < alpha:
                backbone.add_edge(agents[i], agents[j], fixedfill_pvalue=pval)

    return _apply_projection_weights(
        backbone,
        B,
        agent_nodes,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
    )


def fixedrow(
    B,
    agent_nodes,
    alpha=0.05,
    projection="simple",
    projection_weight="weight",
    projection_directed=False,
    projection_max_iter=1000,
    projection_tol=1e-12,
):
    """Backbone from a fixed-row null model for bipartite projections."""
    from scipy import stats as sp_stats

    _validate_bipartite(B, agent_nodes)
    agents, artifacts, R, observed = _bipartite_projection_matrix(B, agent_nodes)
    na = len(agents)
    nf = len(artifacts)
    row_sums = R.sum(axis=1).astype(int)

    backbone = nx.Graph()
    backbone.add_nodes_from(agents)

    if na < 2 or nf == 0:
        return _apply_projection_weights(
            backbone,
            B,
            agent_nodes,
            projection=projection,
            projection_weight=projection_weight,
            projection_directed=projection_directed,
            projection_max_iter=projection_max_iter,
            projection_tol=projection_tol,
        )

    for i in range(na):
        for j in range(i + 1, na):
            obs = int(observed[i, j])
            di = int(row_sums[i])
            dj = int(row_sums[j])
            pval = float(sp_stats.hypergeom.sf(obs - 1, nf, di, dj))
            if pval < alpha:
                backbone.add_edge(agents[i], agents[j], fixedrow_pvalue=pval)

    return _apply_projection_weights(
        backbone,
        B,
        agent_nodes,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
    )


def fixedcol(
    B,
    agent_nodes,
    alpha=0.05,
    projection="simple",
    projection_weight="weight",
    projection_directed=False,
    projection_max_iter=1000,
    projection_tol=1e-12,
):
    """Backbone from a fixed-column null model for bipartite projections."""
    import numpy as np
    from scipy import stats as sp_stats

    _validate_bipartite(B, agent_nodes)
    agents, artifacts, R, observed = _bipartite_projection_matrix(B, agent_nodes)
    na = len(agents)
    col_sums = R.sum(axis=0).astype(float)

    backbone = nx.Graph()
    backbone.add_nodes_from(agents)

    if na < 2 or len(artifacts) == 0:
        return _apply_projection_weights(
            backbone,
            B,
            agent_nodes,
            projection=projection,
            projection_weight=projection_weight,
            projection_directed=projection_directed,
            projection_max_iter=projection_max_iter,
            projection_tol=projection_tol,
        )

    pair_prob = np.clip(col_sums * (col_sums - 1) / (na * (na - 1)), 0.0, 1.0)
    mu = pair_prob.sum()
    sigma2 = (pair_prob * (1.0 - pair_prob)).sum()

    for i in range(na):
        for j in range(i + 1, na):
            obs = float(observed[i, j])
            if sigma2 > 0:
                z = (obs - mu) / np.sqrt(sigma2)
                pval = float(1.0 - sp_stats.norm.cdf(z))
            else:
                pval = 0.0 if obs > mu else 1.0
            pval = float(max(min(pval, 1.0), 0.0))
            if pval < alpha:
                backbone.add_edge(agents[i], agents[j], fixedcol_pvalue=pval)

    return _apply_projection_weights(
        backbone,
        B,
        agent_nodes,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
    )


def backbone_from_projection(
    B,
    agent_nodes=None,
    method="sdsm",
    alpha=0.05,
    projection="simple",
    projection_weight="weight",
    projection_directed=False,
    projection_max_iter=1000,
    projection_tol=1e-12,
    **kwargs,
):
    """Extract a projection backbone using a method-name dispatch wrapper.

    Parameters
    ----------
    projection : {"simple", "hyper", "probs", "ycn"}, optional
        Bipartite projection weighting to assign to returned edges.
    projection_weight : str, optional
        Edge attribute name used for projection weights.
    projection_directed : bool, optional
        Whether to use directed projection weights. Undirected backbones do
        not support directed projections and will raise an error if ``True``.
    projection_max_iter : int, optional
        Maximum iterations for ``projection="ycn"`` stationary distribution
        estimation.
    projection_tol : float, optional
        Convergence tolerance for ``projection="ycn"``.
    """
    if agent_nodes is None:
        agent_nodes = _infer_agent_nodes(B)

    method_l = method.lower()
    if method_l == "sdsm":
        return sdsm(
            B,
            agent_nodes=agent_nodes,
            alpha=alpha,
            projection=projection,
            projection_weight=projection_weight,
            projection_directed=projection_directed,
            projection_max_iter=projection_max_iter,
            projection_tol=projection_tol,
            **kwargs,
        )
    if method_l == "fdsm":
        return fdsm(
            B,
            agent_nodes=agent_nodes,
            alpha=alpha,
            projection=projection,
            projection_weight=projection_weight,
            projection_directed=projection_directed,
            projection_max_iter=projection_max_iter,
            projection_tol=projection_tol,
            **kwargs,
        )
    if method_l == "fixedfill":
        return fixedfill(
            B,
            agent_nodes=agent_nodes,
            alpha=alpha,
            projection=projection,
            projection_weight=projection_weight,
            projection_directed=projection_directed,
            projection_max_iter=projection_max_iter,
            projection_tol=projection_tol,
        )
    if method_l == "fixedrow":
        return fixedrow(
            B,
            agent_nodes=agent_nodes,
            alpha=alpha,
            projection=projection,
            projection_weight=projection_weight,
            projection_directed=projection_directed,
            projection_max_iter=projection_max_iter,
            projection_tol=projection_tol,
        )
    if method_l == "fixedcol":
        return fixedcol(
            B,
            agent_nodes=agent_nodes,
            alpha=alpha,
            projection=projection,
            projection_weight=projection_weight,
            projection_directed=projection_directed,
            projection_max_iter=projection_max_iter,
            projection_tol=projection_tol,
        )
    raise ValueError(
        "Unknown projection method. Choose one of: "
        "'sdsm', 'fdsm', 'fixedfill', 'fixedrow', 'fixedcol'."
    )


def backbone_from_weighted(
    G,
    method="disparity",
    weight="weight",
    alpha=0.05,
    collapse_multiedges=True,
    edge_type_attr=None,
    **kwargs,
):
    """Extract a weighted-network backbone using method-name dispatch.

    Multi-edge inputs can be collapsed automatically before scoring.

    Parameters
    ----------
    collapse_multiedges : bool, optional (default=True)
        If ``True`` and ``G`` is a ``MultiGraph`` or ``MultiDiGraph``,
        collapse parallel edges using
        :func:`~networkx_backbone.multigraph_to_weighted`.
    edge_type_attr : string or None, optional (default=None)
        Edge attribute used when collapsing multi-edges. If provided, edge
        weights count distinct edge types per node pair; otherwise they count
        parallel edges.
    """
    from networkx_backbone.filters import multigraph_to_weighted, threshold_filter
    from networkx_backbone.statistical import (
        disparity_filter,
        lans_filter,
        marginal_likelihood_filter,
    )
    from networkx_backbone.structural import global_threshold_filter

    if collapse_multiedges and G.is_multigraph():
        G = multigraph_to_weighted(G, weight=weight, edge_type_attr=edge_type_attr)

    method_l = method.lower()

    if method_l in ("disparity", "disparity_filter"):
        scored = disparity_filter(G, weight=weight)
        return threshold_filter(scored, "disparity_pvalue", alpha, mode="below")

    if method_l in ("mlf", "marginal_likelihood", "marginal_likelihood_filter"):
        scored = marginal_likelihood_filter(G, weight=weight)
        return threshold_filter(scored, "ml_pvalue", alpha, mode="below")

    if method_l in ("lans", "lans_filter"):
        scored = lans_filter(G, weight=weight)
        return threshold_filter(scored, "lans_pvalue", alpha, mode="below")

    if method_l in ("global", "global_threshold", "global_threshold_filter"):
        threshold = kwargs.get("threshold")
        if threshold is None:
            raise ValueError("global threshold method requires `threshold=...`")
        return global_threshold_filter(G, threshold=threshold, weight=weight)

    raise ValueError(
        "Unknown weighted method. Choose one of: "
        "'disparity', 'mlf', 'lans', 'global'."
    )


def backbone_from_unweighted(G, method="sparsify", **kwargs):
    """Extract an unweighted-network backbone using method-name dispatch."""
    from networkx_backbone.unweighted import local_degree, lspar, sparsify

    method_l = method.lower()
    if method_l == "sparsify":
        return sparsify(G, **kwargs)
    if method_l == "lspar":
        return lspar(G, **kwargs)
    if method_l in ("local_degree", "localdegree"):
        return local_degree(G, **kwargs)

    raise ValueError(
        "Unknown unweighted method. Choose one of: "
        "'sparsify', 'lspar', 'local_degree'."
    )


def backbone(G, method=None, **kwargs):
    """Unified dispatch wrapper across projection, weighted, and unweighted methods."""
    method_l = method.lower() if isinstance(method, str) else None

    projection_methods = {"sdsm", "fdsm", "fixedfill", "fixedrow", "fixedcol"}
    weighted_methods = {
        "disparity",
        "disparity_filter",
        "mlf",
        "marginal_likelihood",
        "marginal_likelihood_filter",
        "lans",
        "lans_filter",
        "global",
        "global_threshold",
        "global_threshold_filter",
    }
    unweighted_methods = {"sparsify", "lspar", "local_degree", "localdegree"}

    if method_l in projection_methods:
        return backbone_from_projection(G, method=method_l, **kwargs)
    if method_l in weighted_methods:
        return backbone_from_weighted(G, method=method_l, **kwargs)
    if method_l in unweighted_methods:
        return backbone_from_unweighted(G, method=method_l, **kwargs)
    if method is not None:
        raise ValueError(f"Unknown method {method!r}")

    has_bipartite_attr = any("bipartite" in data for _, data in G.nodes(data=True))
    if has_bipartite_attr and nx.is_bipartite(G):
        return backbone_from_projection(G, **kwargs)

    weighted = any("weight" in data for _, _, data in G.edges(data=True))
    if weighted:
        return backbone_from_weighted(G, **kwargs)
    return backbone_from_unweighted(G, **kwargs)


# =====================================================================
# 1. SDSM -- Neal (2014)
# =====================================================================


def sdsm(
    B,
    agent_nodes,
    alpha=0.05,
    weight=None,
    projection="simple",
    projection_weight="weight",
    projection_directed=False,
    projection_max_iter=1000,
    projection_tol=1e-12,
):
    """Extract a backbone from a bipartite projection using the SDSM.

    The Stochastic Degree Sequence Model [1]_ generates random bipartite
    networks that preserve the row and column sums *on average* (as
    probabilities).  The p-value for each pair of agents is computed
    analytically using a Poisson-binomial approximation (normal).

    Parameters
    ----------
    B : graph
        A bipartite NetworkX graph.
    agent_nodes : iterable
        Nodes in the "agent" partition.
    alpha : float, optional (default=0.05)
        Retained for API compatibility. This function returns the full
        projected graph with p-values on all edges; apply
        :func:`networkx_backbone.filters.threshold_filter` to select
        significant edges.
    weight : None or string, optional (default=None)
        Not used for SDSM (bipartite is unweighted); reserved for API
        consistency.
    projection : {"simple", "hyper", "probs", "ycn"}, optional
        Projection weighting assigned to each returned edge.
    projection_weight : str, optional
        Edge attribute name for the projection weight.
    projection_directed : bool, optional
        Whether to use directed projection scores. Not supported for the
        undirected backbone output.
    projection_max_iter : int, optional
        Maximum iterations for ``projection="ycn"``.
    projection_tol : float, optional
        Convergence tolerance for ``projection="ycn"``.

    Returns
    -------
    backbone : graph
        Full unipartite projection among agent nodes.  Each edge has a
        ``"sdsm_pvalue"`` attribute and projection weights.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* contains nodes not in *B*.

    References
    ----------
    .. [1] Neal, Z. P. (2014). The backbone of bipartite projections:
       Inferring relationships from co-authorship, co-sponsorship,
       co-attendance and other co-behaviors. *Social Networks*, 39, 84-97.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import sdsm
    >>> B = nx.davis_southern_women_graph()
    >>> women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    >>> scored = sdsm(B, agent_nodes=women)
    >>> from networkx_backbone import threshold_filter
    >>> backbone = threshold_filter(scored, "sdsm_pvalue", 0.05, mode="below")
    >>> all("sdsm_pvalue" in data for _, _, data in backbone.edges(data=True))
    True
    """
    import numpy as np
    from scipy import stats as sp_stats

    _validate_bipartite(B, agent_nodes)
    agents, artifacts, R, observed = _bipartite_projection_matrix(B, agent_nodes)

    na = len(agents)

    # Row and column sums
    row_sums = R.sum(axis=1).astype(float)  # agent degrees
    col_sums = R.sum(axis=0).astype(float)  # artifact degrees

    # Probability matrix: P[i,k] = prob that agent i connects to artifact k
    total = R.sum()
    backbone = nx.Graph()
    backbone.add_nodes_from(agents)
    if total == 0:
        return _apply_projection_weights(
            backbone,
            B,
            agent_nodes,
            projection=projection,
            projection_weight=projection_weight,
            projection_directed=projection_directed,
            projection_max_iter=projection_max_iter,
            projection_tol=projection_tol,
        )

    P = np.outer(row_sums, col_sums) / total
    P = np.clip(P, 0, 1)

    for i in range(na):
        for j in range(i + 1, na):
            obs = observed[i, j]

            # For each artifact k, probability both agents connect:
            probs = P[i, :] * P[j, :]

            # Mean and variance of the sum of independent Bernoullis
            mu = probs.sum()
            sigma2 = (probs * (1 - probs)).sum()

            if sigma2 > 0:
                z = (obs - mu) / np.sqrt(sigma2)
                pval = 1.0 - sp_stats.norm.cdf(z)
            else:
                pval = 0.0 if obs > mu else 1.0

            pval = float(max(min(pval, 1.0), 0.0))
            backbone.add_edge(agents[i], agents[j], sdsm_pvalue=pval)

    return _apply_projection_weights(
        backbone,
        B,
        agent_nodes,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
    )


# =====================================================================
# 2. FDSM -- Neal et al. (2021)
# =====================================================================


def fdsm(
    B,
    agent_nodes,
    alpha=0.05,
    trials=1000,
    seed=None,
    projection="simple",
    projection_weight="weight",
    projection_directed=False,
    projection_max_iter=1000,
    projection_tol=1e-12,
):
    """Extract a backbone from a bipartite projection using the FDSM.

    The Fixed Degree Sequence Model [1]_ uses Monte Carlo simulation to
    estimate p-values.  Each trial generates a random bipartite graph that
    *exactly* preserves both the row sums (agent degrees) and column sums
    (artifact degrees), then computes the co-occurrence matrix.

    Parameters
    ----------
    B : graph
        A bipartite NetworkX graph.
    agent_nodes : iterable
        Nodes in the "agent" partition.
    alpha : float, optional (default=0.05)
        Retained for API compatibility. This function returns the full
        projected graph with p-values on all edges; apply
        :func:`networkx_backbone.filters.threshold_filter` to select
        significant edges.
    trials : int, optional (default=1000)
        Number of Monte Carlo randomisations.
    seed : integer, random_state, or None (default)
        Random seed for reproducibility.
    projection : {"simple", "hyper", "probs", "ycn"}, optional
        Projection weighting assigned to each returned edge.
    projection_weight : str, optional
        Edge attribute name for the projection weight.
    projection_directed : bool, optional
        Whether to use directed projection scores. Not supported for the
        undirected backbone output.
    projection_max_iter : int, optional
        Maximum iterations for ``projection="ycn"``.
    projection_tol : float, optional
        Convergence tolerance for ``projection="ycn"``.

    Returns
    -------
    backbone : graph
        Full unipartite projection among agent nodes.  Each edge has an
        ``"fdsm_pvalue"`` attribute and projection weights.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* contains nodes not in *B*.

    References
    ----------
    .. [1] Neal, Z. P., Domagalski, R., & Sagan, B. (2021). Comparing
       alternatives to the fixed degree sequence model. *Scientific
       Reports*, 11, 23929.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fdsm
    >>> B = nx.davis_southern_women_graph()
    >>> women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    >>> scored = fdsm(B, agent_nodes=women, trials=100, seed=42)
    >>> from networkx_backbone import threshold_filter
    >>> backbone = threshold_filter(scored, "fdsm_pvalue", 0.05, mode="below")
    >>> all("fdsm_pvalue" in data for _, _, data in backbone.edges(data=True))
    True
    """
    import numpy as np

    if trials < 1:
        raise ValueError("trials must be >= 1")

    _validate_bipartite(B, agent_nodes)
    agents, artifacts, R, observed = _bipartite_projection_matrix(B, agent_nodes)

    na = len(agents)
    rng = np.random.default_rng(seed)

    row_sums = R.sum(axis=1)
    col_sums = R.sum(axis=0)

    # Count how many times the random co-occurrence >= observed
    exceed_count = np.zeros((na, na), dtype=int)

    for _ in range(trials):
        R_rand = _random_bipartite_matrix(row_sums, col_sums, rng)
        co_rand = R_rand @ R_rand.T
        np.fill_diagonal(co_rand, 0)
        exceed_count += (co_rand >= observed).astype(int)

    backbone = nx.Graph()
    backbone.add_nodes_from(agents)

    for i in range(na):
        for j in range(i + 1, na):
            pval = exceed_count[i, j] / trials
            backbone.add_edge(agents[i], agents[j], fdsm_pvalue=float(pval))

    return _apply_projection_weights(
        backbone,
        B,
        agent_nodes,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
    )


def _random_bipartite_matrix(row_sums, col_sums, rng):
    """Generate a random binary matrix with given row and column sums.

    Uses a greedy fill followed by Curveball swaps for randomisation.
    """
    import numpy as np

    nrows = len(row_sums)
    ncols = len(col_sums)

    # Initialise with a valid matrix using a greedy fill
    R = np.zeros((nrows, ncols), dtype=int)
    col_remaining = col_sums.copy().astype(float)

    for i in range(nrows):
        k = int(row_sums[i])
        if k <= 0:
            continue
        avail = np.where(col_remaining > 0)[0]
        if len(avail) == 0:
            continue
        k = min(k, len(avail))
        probs = col_remaining[avail]
        total = probs.sum()
        if total <= 0:
            continue
        probs = probs / total
        chosen = rng.choice(avail, size=k, replace=False, p=probs)
        R[i, chosen] = 1
        col_remaining[chosen] -= 1
        col_remaining = np.maximum(col_remaining, 0)

    # Curveball swaps to improve randomisation
    n_swaps = nrows * 5
    for _ in range(n_swaps):
        if nrows < 2:
            break
        r1, r2 = rng.choice(nrows, size=2, replace=False)
        cols1 = set(np.where(R[r1] == 1)[0])
        cols2 = set(np.where(R[r2] == 1)[0])
        only1 = list(cols1 - cols2)
        only2 = list(cols2 - cols1)
        if len(only1) == 0 or len(only2) == 0:
            continue
        n_swap = rng.integers(1, min(len(only1), len(only2)) + 1)
        swap1 = rng.choice(only1, size=n_swap, replace=False)
        swap2 = rng.choice(only2, size=n_swap, replace=False)
        R[r1, swap1] = 0
        R[r1, swap2] = 1
        R[r2, swap2] = 0
        R[r2, swap1] = 1

    return R
