"""
Structural backbone extraction methods.

These methods use the network's topology (edge weights, shortest paths,
spanning trees, degree, neighborhood overlap) to identify the most
important substructure.

Methods
-------
global_threshold_filter
    Simple weight cutoff.
strongest_n_ties
    Per-node strongest edges.
global_sparsification
    Keep top-weight edges globally by fraction.
primary_linkage_analysis
    Nystuen & Dacey (1961) -- strongest outgoing edge per node.
edge_betweenness_filter
    Girvan & Newman (2002) -- keep top edge-betweenness edges.
node_degree_filter
    Keep nodes above a minimum degree threshold.
high_salience_skeleton
    Grady et al. (2012) -- shortest-path tree participation.
metric_backbone
    Simas et al. (2021) -- edges on shortest paths (sum distances).
ultrametric_backbone
    Shortest paths using max distance.
doubly_stochastic_filter
    Slater (2009) -- Sinkhorn-Knopp normalization.
h_backbone
    Zhang et al. (2018) -- h-index inspired.
modularity_backbone
    Rajeh et al. (2022) -- node vitality index.
planar_maximally_filtered_graph
    Tumminello et al. (2005) -- planar constraint.
maximum_spanning_tree_backbone
    Maximum spanning tree.
"""

import heapq
import math

import networkx as nx
from networkx.utils import not_implemented_for

from networkx_backbone._docstrings import append_complexity_docstrings

__all__ = [
    "global_threshold_filter",
    "strongest_n_ties",
    "global_sparsification",
    "primary_linkage_analysis",
    "edge_betweenness_filter",
    "node_degree_filter",
    "high_salience_skeleton",
    "metric_backbone",
    "ultrametric_backbone",
    "doubly_stochastic_filter",
    "h_backbone",
    "modularity_backbone",
    "planar_maximally_filtered_graph",
    "maximum_spanning_tree_backbone",
]


def _edge_key(G, u, v):
    """Return a deterministic edge key that is direction-aware."""
    if G.is_directed():
        return (u, v)
    ku = (type(u).__name__, repr(u))
    kv = (type(v).__name__, repr(v))
    return (u, v) if ku <= kv else (v, u)


def _mark_kept_edges(H, keep_edges, attr):
    """Annotate every edge in H with a boolean keep flag."""
    for u, v, data in H.edges(data=True):
        data[attr] = _edge_key(H, u, v) in keep_edges


# =====================================================================
# 1. Global threshold filter
# =====================================================================


def global_threshold_filter(G, threshold, weight="weight"):
    """Score edges against a global weight threshold.

    For each edge, mark whether its *weight* attribute is greater than
    or equal to *threshold*.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    threshold : float
        Minimum edge weight required.  Edges with
        ``data[weight] >= threshold`` are kept.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    H : graph
        A copy of *G* with a boolean ``"global_threshold_keep"`` edge
        attribute.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, global_threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = global_threshold_filter(G, threshold=3.0)
    >>> backbone = boolean_filter(H, "global_threshold_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    for _, _, data in H.edges(data=True):
        data["global_threshold_keep"] = data.get(weight, 0) >= threshold
    return H


# =====================================================================
# 2. Per-node strongest-N ties
# =====================================================================


def strongest_n_ties(G, n=1, weight="weight"):
    """Score edges by strongest-*n* membership per node.

    For each node, select the *n* incident edges with the largest
    *weight* value.  An edge is kept if *either* endpoint selects it
    (OR semantics).  For directed graphs the selection is based on
    out-edges.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    n : int, optional (default=1)
        Number of strongest edges to keep per node.  Must be at least 1.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    H : graph
        A copy of *G* with a boolean ``"strongest_n_ties_keep"`` edge
        attribute.

    Raises
    ------
    ValueError
        If *n* < 1.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, strongest_n_ties
    >>> G = nx.les_miserables_graph()
    >>> H = strongest_n_ties(G, n=1)
    >>> backbone = boolean_filter(H, "strongest_n_ties_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    kept_edges = set()
    if G.is_directed():
        for u in G.nodes():
            out = [(data.get(weight, 0), v) for v, data in G[u].items()]
            for _, v in heapq.nlargest(n, out):
                kept_edges.add(_edge_key(G, u, v))
    else:
        for u in G.nodes():
            nbrs = [(G[u][v].get(weight, 0), v) for v in G[u]]
            for _, v in heapq.nlargest(n, nbrs):
                kept_edges.add(_edge_key(G, u, v))

    H = G.copy()
    _mark_kept_edges(H, kept_edges, "strongest_n_ties_keep")
    return H


# =====================================================================
# 2b. Global sparsification
# =====================================================================


def global_sparsification(G, s=0.5, weight="weight"):
    """Score edges by global top-fraction membership.

    This method ranks all edges by weight and keeps the top ``s`` fraction.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    s : float, optional (default=0.5)
        Fraction of edges to retain. Must be in (0, 1].
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    H : graph
        A copy of *G* with a boolean ``"global_sparsification_keep"``
        edge attribute.

    Raises
    ------
    ValueError
        If ``s`` is outside (0, 1].

    References
    ----------
    .. [1] Satuluri, V., Parthasarathy, S., & Ruan, Y. (2011). Local graph
       sparsification for scalable clustering. *SIGMOD*, 721-732.
    """
    if not 0.0 < s <= 1.0:
        raise ValueError("s must be in (0, 1]")

    H = G.copy()
    m = G.number_of_edges()
    if m == 0:
        return H

    keep = max(1, int(math.ceil(s * m)))
    ranked_edges = sorted(
        G.edges(data=True),
        key=lambda x: (x[2].get(weight, 0.0), str(x[0]), str(x[1])),
        reverse=True,
    )
    keep_edges = {_edge_key(G, u, v) for u, v, _ in ranked_edges[:keep]}
    _mark_kept_edges(H, keep_edges, "global_sparsification_keep")
    return H


# =====================================================================
# 2c. Primary linkage analysis
# =====================================================================


def primary_linkage_analysis(G, weight="weight"):
    """Score edges by primary-linkage membership.

    Each node keeps only its strongest outgoing connection.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input weighted graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    H : graph
        A copy of *G* with a boolean ``"primary_linkage_keep"`` edge
        attribute.

    References
    ----------
    .. [1] Nystuen, J. D., & Dacey, M. F. (1961). A graph theory
       interpretation of nodal regions. *Papers in Regional Science*,
       7(1), 29-42.
    """
    H = G.copy()
    kept_edges = set()

    if G.is_directed():
        for u in G.nodes():
            out_edges = list(G.out_edges(u, data=True))
            if not out_edges:
                continue
            _, v, _ = max(
                out_edges, key=lambda x: (x[2].get(weight, 0.0), str(x[1]))
            )
            kept_edges.add(_edge_key(G, u, v))
        _mark_kept_edges(H, kept_edges, "primary_linkage_keep")
        return H

    for u in G.nodes():
        neighbors = list(G[u].items())
        if not neighbors:
            continue
        v, _ = max(neighbors, key=lambda x: (x[1].get(weight, 0.0), str(x[0])))
        kept_edges.add(_edge_key(G, u, v))

    _mark_kept_edges(H, kept_edges, "primary_linkage_keep")
    return H


# =====================================================================
# 2d. Edge betweenness filter
# =====================================================================


def edge_betweenness_filter(G, s=0.5, weight="weight"):
    """Score edges by edge betweenness and top-fraction membership.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    s : float, optional (default=0.5)
        Fraction of edges to retain. Must be in (0, 1].
    weight : string, optional (default="weight")
        Edge attribute key used as the weight. If present, distance is
        interpreted as ``1 / weight``.

    Returns
    -------
    H : graph
        A copy of *G* with ``"edge_betweenness"`` scores and a boolean
        ``"edge_betweenness_keep"`` edge attribute.

    Raises
    ------
    ValueError
        If ``s`` is outside (0, 1].

    References
    ----------
    .. [1] Girvan, M., & Newman, M. E. J. (2002). Community structure in
       social and biological networks. *PNAS*, 99(12), 7821-7826.
    """
    if not 0.0 < s <= 1.0:
        raise ValueError("s must be in (0, 1]")

    scored = G.copy()
    dist_attr = "__eb_dist__"
    use_weight = any(weight in data for _, _, data in scored.edges(data=True))

    if use_weight:
        for _, _, data in scored.edges(data=True):
            w = data.get(weight, 1.0)
            data[dist_attr] = 1.0 / w if w > 0 else float("inf")
        eb = nx.edge_betweenness_centrality(scored, weight=dist_attr, normalized=True)
    else:
        eb = nx.edge_betweenness_centrality(scored, weight=None, normalized=True)

    for u, v, data in scored.edges(data=True):
        data["edge_betweenness"] = float(eb.get((u, v), eb.get((v, u), 0.0)))
        data.pop(dist_attr, None)

    m = scored.number_of_edges()
    keep = max(1, int(math.ceil(s * m))) if m > 0 else 0
    ranked_edges = sorted(
        scored.edges(data=True),
        key=lambda x: (x[2]["edge_betweenness"], str(x[0]), str(x[1])),
        reverse=True,
    )
    keep_edges = {_edge_key(scored, u, v) for u, v, _ in ranked_edges[:keep]}
    _mark_kept_edges(scored, keep_edges, "edge_betweenness_keep")
    return scored


# =====================================================================
# 2e. Node-degree filter
# =====================================================================


def node_degree_filter(G, min_degree=1):
    """Score nodes and edges by a minimum node-degree criterion.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    min_degree : int, optional (default=1)
        Minimum degree required for a node to be retained.

    Returns
    -------
    H : graph
        A copy of *G* with boolean ``"node_degree_keep"`` on nodes and
        edges.

    Raises
    ------
    ValueError
        If ``min_degree`` is negative.

    References
    ----------
    .. [1] Freeman, L. C. (1978). Centrality in social networks:
       conceptual clarification. *Social Networks*, 1(3), 215-239.
    """
    if min_degree < 0:
        raise ValueError("min_degree must be >= 0")
    H = G.copy()
    keep_nodes = {n for n, deg in G.degree() if deg >= min_degree}
    for n in H.nodes():
        H.nodes[n]["node_degree_keep"] = n in keep_nodes
    for u, v, data in H.edges(data=True):
        data["node_degree_keep"] = (u in keep_nodes) and (v in keep_nodes)
    return H


# =====================================================================
# 3. High salience skeleton -- Grady et al. (2012)
# =====================================================================


@not_implemented_for("directed")
def high_salience_skeleton(G, weight="weight"):
    """Compute edge salience from shortest-path tree participation.

    For every node *r*, a shortest-path tree rooted at *r* is computed
    (using inverse weights as distances).  The salience of an edge is the
    fraction of these trees that contain it.  The salience is stored as
    the ``"salience"`` edge attribute on the returned graph.

    Parameters
    ----------
    G : graph
        A NetworkX graph.  All weights must be positive.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    H : graph
        A copy of *G* with an additional ``"salience"`` edge attribute
        whose value lies in [0, 1].

    References
    ----------
    .. [1] Grady, D., Thiemann, C., & Brockmann, D. (2012). Robust
       classification of salient links in complex networks. *Nature
       Communications*, 3, 864.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import high_salience_skeleton, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = high_salience_skeleton(G)
    >>> backbone = threshold_filter(H, "salience", 0.5, mode="above")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    n = G.number_of_nodes()
    salience_count = {(min(u, v), max(u, v)): 0 for u, v in G.edges()}

    dist_attr = "__hss_dist__"
    for u, v, data in H.edges(data=True):
        data[dist_attr] = 1.0 / data[weight]

    for root in G.nodes():
        pred = nx.dijkstra_predecessor_and_distance(H, root, weight=dist_attr)[0]
        for node, preds in pred.items():
            for p in preds:
                key = (min(node, p), max(node, p))
                if key in salience_count:
                    salience_count[key] += 1

    for u, v, data in H.edges(data=True):
        key = (min(u, v), max(u, v))
        data["salience"] = salience_count[key] / n
        data.pop(dist_attr, None)

    return H


# =====================================================================
# 4/5. Metric & ultrametric backbones -- Simas et al. (2021)
# =====================================================================


@not_implemented_for("directed")
def metric_backbone(G, weight="weight"):
    """Extract the metric backbone using sum-of-distances shortest paths.

    An edge *(u, v)* is in the metric backbone if and only if its direct
    distance (the inverse of its weight) equals the shortest-path
    distance between *u* and *v* computed over the full graph.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.  All weights must be
        positive.

    Returns
    -------
    H : graph
        A copy of *G* with boolean ``"metric_keep"`` on each edge.

    References
    ----------
    .. [1] Simas, T., Correia, R. B., & Rocha, L. M. (2021). The distance
       backbone of complex networks. *J. Complex Netw.*, 9, cnab021.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, metric_backbone
    >>> G = nx.les_miserables_graph()
    >>> H = metric_backbone(G)
    >>> backbone = boolean_filter(H, "metric_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    return _distance_backbone(G, weight=weight, ultrametric=False)


@not_implemented_for("directed")
def ultrametric_backbone(G, weight="weight"):
    """Extract the ultrametric backbone using max-distance (minimax) paths.

    An edge *(u, v)* is in the ultrametric backbone if its direct
    distance is no greater than the minimax path distance between *u*
    and *v* (the path that minimises the maximum edge distance along
    the path).

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.  All weights must be
        positive.

    Returns
    -------
    H : graph
        A copy of *G* with boolean ``"ultrametric_keep"`` on each edge.

    References
    ----------
    .. [1] Simas, T., Correia, R. B., & Rocha, L. M. (2021). The distance
       backbone of complex networks. *J. Complex Netw.*, 9, cnab021.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, ultrametric_backbone
    >>> G = nx.les_miserables_graph()
    >>> H = ultrametric_backbone(G)
    >>> backbone = boolean_filter(H, "ultrametric_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    return _distance_backbone(G, weight=weight, ultrametric=True)


def _distance_backbone(G, weight="weight", ultrametric=False):
    """Shared helper for metric and ultrametric backbones."""
    H = G.copy()

    dist_G = nx.Graph()
    for u, v, data in G.edges(data=True):
        dist_G.add_edge(u, v, dist=1.0 / data[weight])

    keep_edges = set()
    if ultrametric:
        mst = nx.minimum_spanning_tree(dist_G, weight="dist")
        for u, v, data in G.edges(data=True):
            direct_dist = 1.0 / data[weight]
            path = nx.shortest_path(mst, u, v, weight=None)
            max_edge = max(
                dist_G[path[i]][path[i + 1]]["dist"] for i in range(len(path) - 1)
            )
            if abs(direct_dist - max_edge) < 1e-12 or direct_dist <= max_edge:
                keep_edges.add(_edge_key(G, u, v))
    else:
        sp = dict(nx.all_pairs_dijkstra_path_length(dist_G, weight="dist"))
        for u, v, data in G.edges(data=True):
            direct_dist = 1.0 / data[weight]
            if abs(direct_dist - sp[u][v]) < 1e-12:
                keep_edges.add(_edge_key(G, u, v))

    keep_attr = "ultrametric_keep" if ultrametric else "metric_keep"
    _mark_kept_edges(H, keep_edges, keep_attr)
    return H


# =====================================================================
# 6. Doubly stochastic filter -- Slater (2009)
# =====================================================================


@not_implemented_for("directed")
def doubly_stochastic_filter(G, weight="weight", max_iter=1000, tol=1e-8):
    """Compute the doubly-stochastic backbone via Sinkhorn-Knopp normalization.

    Transforms the adjacency matrix into a doubly stochastic matrix via
    iterative Sinkhorn-Knopp normalization, then stores the normalised
    weight as the ``"ds_weight"`` edge attribute on the returned graph.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.
    max_iter : int, optional (default=1000)
        Maximum number of Sinkhorn-Knopp iterations.
    tol : float, optional (default=1e-8)
        Convergence tolerance.

    Returns
    -------
    H : graph
        A copy of *G* with an additional ``"ds_weight"`` edge attribute
        (doubly-stochastic normalised weight).

    References
    ----------
    .. [1] Slater, P. B. (2009). A two-stage algorithm for extracting
       the multiscale backbone of complex weighted networks. *PNAS*,
       106(26), E66.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import doubly_stochastic_filter, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = doubly_stochastic_filter(G)
    >>> backbone = threshold_filter(H, "ds_weight", 0.1, mode="above")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    import numpy as np

    nodes = sorted(G.nodes())
    n = len(nodes)
    if n == 0:
        return G.copy()

    idx = {v: i for i, v in enumerate(nodes)}

    # Build adjacency matrix
    A = np.zeros((n, n), dtype=float)
    for u, v, data in G.edges(data=True):
        w = data.get(weight, 0)
        A[idx[u], idx[v]] = w
        A[idx[v], idx[u]] = w

    # Sinkhorn-Knopp iteration
    D = A.copy()
    for _ in range(max_iter):
        # Row normalise
        row_sums = D.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        D = D / row_sums
        # Column normalise
        col_sums = D.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        D = D / col_sums
        # Check convergence
        rs = D.sum(axis=1)
        cs = D.sum(axis=0)
        if np.allclose(rs, 1, atol=tol) and np.allclose(cs, 1, atol=tol):
            break

    H = G.copy()
    for u, v, data in H.edges(data=True):
        data["ds_weight"] = float(D[idx[u], idx[v]])

    return H


# =====================================================================
# 7. h-backbone -- Zhang et al. (2018)
# =====================================================================


@not_implemented_for("directed")
def h_backbone(G, weight="weight"):
    r"""Extract the h-backbone of a weighted graph.

    The h-backbone [1]_ is composed of two parts:

    1. **h-strength network**: find *h* such that there are at least *h*
       edges with weight >= *h*.  Keep those edges.
    2. **h-bridge network**: among the remaining edges, keep those whose
       edge betweenness centrality is in the top-*h*.

    The union of both subsets forms the h-backbone.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    H : graph
        A copy of *G* with boolean ``"h_backbone_keep"`` on each edge.

    References
    ----------
    .. [1] Zhang, R. J., Stanley, H. E., & Ye, F. Y. (2018). Extracting
       h-backbone as a core structure in weighted networks. *Scientific
       Reports*, 8, 14356.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, h_backbone
    >>> G = nx.les_miserables_graph()
    >>> H = h_backbone(G)
    >>> backbone = boolean_filter(H, "h_backbone_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    # Compute h-index of the weight sequence
    weights = sorted([data[weight] for _, _, data in G.edges(data=True)], reverse=True)
    h = 0
    for i, w in enumerate(weights):
        if w >= i + 1:
            h = i + 1
        else:
            break

    # h-strength network: edges with weight >= h
    h_strength_edges = set()
    for u, v, data in G.edges(data=True):
        if data[weight] >= h:
            h_strength_edges.add(_edge_key(G, u, v))

    # h-bridge network: top-h edges by betweenness among NON-h-strength edges
    remaining = G.copy()
    for u, v in list(h_strength_edges):
        if remaining.has_edge(u, v):
            remaining.remove_edge(u, v)

    h_bridge_edges = set()
    if remaining.number_of_edges() > 0 and h > 0:
        eb = nx.edge_betweenness_centrality(remaining, weight=weight)
        top_h = sorted(eb.items(), key=lambda x: x[1], reverse=True)[:h]
        for (u, v), _ in top_h:
            h_bridge_edges.add(_edge_key(G, u, v))

    # Union
    backbone_edges = h_strength_edges | h_bridge_edges
    H = G.copy()
    _mark_kept_edges(H, backbone_edges, "h_backbone_keep")
    return H


# =====================================================================
# 8. Modularity backbone -- Rajeh et al. (2022)
# =====================================================================


@not_implemented_for("directed")
def modularity_backbone(G, weight="weight"):
    """Compute the modularity vitality index for each node.

    The vitality index [1]_ measures the change in modularity when a
    node is removed.  Nodes that contribute positively to community
    structure have positive vitality.  The vitality is stored as the
    ``"vitality"`` node attribute on the returned graph.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    H : graph
        A copy of *G* with ``"vitality"`` node scores and boolean
        ``"modularity_keep"`` on edges.

    References
    ----------
    .. [1] Rajeh, S., Savonnet, M., Leclercq, E., & Cherifi, H. (2022).
       Modularity-based backbone extraction in weighted complex networks.
       *NetSci-X 2022*, 67-79.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, modularity_backbone
    >>> G = nx.les_miserables_graph()
    >>> H = modularity_backbone(G)
    >>> backbone = boolean_filter(H, "modularity_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()

    # Baseline modularity
    try:
        communities_full = nx.community.louvain_communities(G, weight=weight, seed=42)
    except Exception:
        communities_full = [set(G.nodes())]
    mod_full = nx.community.modularity(G, communities_full, weight=weight)

    for node in G.nodes():
        G_reduced = G.copy()
        G_reduced.remove_node(node)
        if G_reduced.number_of_nodes() == 0:
            H.nodes[node]["vitality"] = 0.0
            continue
        try:
            communities_reduced = nx.community.louvain_communities(
                G_reduced, weight=weight, seed=42
            )
        except Exception:
            communities_reduced = [set(G_reduced.nodes())]
        mod_reduced = nx.community.modularity(
            G_reduced, communities_reduced, weight=weight
        )
        H.nodes[node]["vitality"] = mod_full - mod_reduced

    for u, v, data in H.edges(data=True):
        data["modularity_keep"] = (
            H.nodes[u].get("vitality", 0.0) > 0.0
            and H.nodes[v].get("vitality", 0.0) > 0.0
        )

    return H


# =====================================================================
# 9. Planar Maximally Filtered Graph -- Tumminello et al. (2005)
# =====================================================================


@not_implemented_for("directed")
def planar_maximally_filtered_graph(G, weight="weight"):
    """Construct the Planar Maximally Filtered Graph (PMFG).

    Iteratively adds edges from heaviest to lightest, keeping only
    those whose addition preserves planarity [1]_.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    H : graph
        A copy of *G* with boolean ``"pmfg_keep"`` on edges.

    References
    ----------
    .. [1] Tumminello, M., Aste, T., Di Matteo, T., & Mantegna, R. N.
       (2005). A tool for filtering information in complex systems.
       *PNAS*, 102(30), 10421-10426.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, planar_maximally_filtered_graph
    >>> G = nx.les_miserables_graph()
    >>> H = planar_maximally_filtered_graph(G)
    >>> backbone = boolean_filter(H, "pmfg_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    n = G.number_of_nodes()
    max_edges = 3 * (n - 2) if n >= 3 else n - 1

    # Sort edges by weight descending
    edges = sorted(G.edges(data=True), key=lambda e: e[2].get(weight, 0), reverse=True)

    candidate = G.__class__()
    candidate.add_nodes_from(G.nodes(data=True))

    for u, v, data in edges:
        if candidate.number_of_edges() >= max_edges:
            break
        candidate.add_edge(u, v, **data)
        is_planar, _ = nx.check_planarity(candidate)
        if not is_planar:
            candidate.remove_edge(u, v)

    keep_edges = {_edge_key(candidate, u, v) for u, v in candidate.edges()}
    H = G.copy()
    _mark_kept_edges(H, keep_edges, "pmfg_keep")
    return H


# =====================================================================
# 10. Maximum spanning tree backbone
# =====================================================================


@not_implemented_for("directed")
def maximum_spanning_tree_backbone(G, weight="weight"):
    """Extract the maximum spanning tree as a backbone.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    H : graph
        A copy of *G* with boolean ``"mst_keep"`` on edges.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, maximum_spanning_tree_backbone
    >>> G = nx.les_miserables_graph()
    >>> H = maximum_spanning_tree_backbone(G)
    >>> backbone = boolean_filter(H, "mst_keep")
    >>> backbone.number_of_edges() == G.number_of_nodes() - 1
    True
    """
    tree = nx.maximum_spanning_tree(G, weight=weight)
    keep_edges = {_edge_key(tree, u, v) for u, v in tree.edges()}
    H = G.copy()
    _mark_kept_edges(H, keep_edges, "mst_keep")
    return H


_COMPLEXITY = {
    "global_threshold_filter": {
        "time": "O(m)",
        "space": "O(n + m)",
        "notes": "n=|V|, m=|E|.",
    },
    "strongest_n_ties": {
        "time": "O(m log n)",
        "space": "O(n + m)",
        "notes": "Worst-case across per-node strongest-edge selection.",
    },
    "global_sparsification": {
        "time": "O(m log m)",
        "space": "O(n + m)",
    },
    "primary_linkage_analysis": {
        "time": "O(m)",
        "space": "O(n + m)",
    },
    "edge_betweenness_filter": {
        "time": "O(nm + n^2 log n)",
        "space": "O(n + m)",
        "notes": "Dominated by weighted Brandes edge-betweenness.",
    },
    "node_degree_filter": {
        "time": "O(n + m)",
        "space": "O(n + m)",
    },
    "high_salience_skeleton": {
        "time": "O(nm + n^2 log n)",
        "space": "O(n + m)",
        "notes": "Runs a shortest-path tree from each root.",
    },
    "metric_backbone": {
        "time": "O(nm + n^2 log n)",
        "space": "O(n^2 + m)",
        "notes": "All-pairs weighted shortest paths.",
    },
    "ultrametric_backbone": {
        "time": "O(mn + m log n)",
        "space": "O(n + m)",
        "notes": "MST plus per-edge minimax path checks.",
    },
    "doubly_stochastic_filter": {
        "time": "O(I * n^2 + m)",
        "space": "O(n^2 + m)",
        "notes": "I=max_iter for Sinkhorn-Knopp normalization.",
    },
    "h_backbone": {
        "time": "O(nm + n^2 log n + m log m)",
        "space": "O(n + m)",
        "notes": "Dominated by edge-betweenness on residual graph.",
    },
    "modularity_backbone": {
        "time": "O(nm)",
        "space": "O(n + m)",
        "notes": "Practical/heuristic bound due to repeated Louvain runs.",
    },
    "planar_maximally_filtered_graph": {
        "time": "O(mn + m log m)",
        "space": "O(n + m)",
    },
    "maximum_spanning_tree_backbone": {
        "time": "O(m log n)",
        "space": "O(n + m)",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
