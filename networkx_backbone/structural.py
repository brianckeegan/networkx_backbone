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

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "global_threshold_filter",
    "strongest_n_ties",
    "high_salience_skeleton",
    "metric_backbone",
    "ultrametric_backbone",
    "doubly_stochastic_filter",
    "h_backbone",
    "modularity_backbone",
    "planar_maximally_filtered_graph",
    "maximum_spanning_tree_backbone",
]


# =====================================================================
# 1. Global threshold filter
# =====================================================================


def global_threshold_filter(G, threshold, weight="weight"):
    """Return a subgraph containing only edges whose weight meets a threshold.

    Construct a new graph of the same type as *G* that keeps every node
    but retains only edges whose *weight* attribute is greater than or
    equal to *threshold*.

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
        A new graph of the same type as *G*.  All original nodes are
        preserved; only edges meeting the threshold are included.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import global_threshold_filter
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 3.0), (0, 2, 1.0)])
    >>> H = global_threshold_filter(G, threshold=3.0)
    >>> sorted(H.edges())
    [(0, 1), (1, 2)]
    """
    H = G.__class__()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if data.get(weight, 0) >= threshold:
            H.add_edge(u, v, **data)
    return H


# =====================================================================
# 2. Per-node strongest-N ties
# =====================================================================


def strongest_n_ties(G, n=1, weight="weight"):
    """Retain the *n* strongest edges for every node.

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
        A new graph of the same type as *G* containing only the
        selected edges.  All original nodes are preserved.

    Raises
    ------
    ValueError
        If *n* < 1.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import strongest_n_ties
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 3.0), (0, 2, 1.0)])
    >>> H = strongest_n_ties(G, n=1)
    >>> sorted(H.edges())
    [(0, 1), (1, 2)]
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    kept_edges = set()
    if G.is_directed():
        for u in G.nodes():
            out = [(data.get(weight, 0), v) for v, data in G[u].items()]
            for _, v in heapq.nlargest(n, out):
                kept_edges.add((u, v))
    else:
        for u in G.nodes():
            nbrs = [(G[u][v].get(weight, 0), v) for v in G[u]]
            for _, v in heapq.nlargest(n, nbrs):
                kept_edges.add((u, v) if u <= v else (v, u))

    H = G.__class__()
    H.add_nodes_from(G.nodes(data=True))
    for u, v in kept_edges:
        H.add_edge(u, v, **G[u][v])
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
    >>> from networkx_backbone import high_salience_skeleton
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 3.0), (0, 2, 1.0)])
    >>> H = high_salience_skeleton(G)
    >>> H[0][1]["salience"] >= 0
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
        A new graph of the same type as *G* containing only edges that
        lie on shortest paths.  All original nodes are preserved.

    References
    ----------
    .. [1] Simas, T., Correia, R. B., & Rocha, L. M. (2021). The distance
       backbone of complex networks. *J. Complex Netw.*, 9, cnab021.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import metric_backbone
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 3.0), (0, 2, 1.0)])
    >>> H = metric_backbone(G)
    >>> H.number_of_edges() <= G.number_of_edges()
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
        A new graph of the same type as *G* containing only edges that
        satisfy the ultrametric condition.  All original nodes are
        preserved.

    References
    ----------
    .. [1] Simas, T., Correia, R. B., & Rocha, L. M. (2021). The distance
       backbone of complex networks. *J. Complex Netw.*, 9, cnab021.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import ultrametric_backbone
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 3.0), (0, 2, 1.0)])
    >>> H = ultrametric_backbone(G)
    >>> H.number_of_edges() <= G.number_of_edges()
    True
    """
    return _distance_backbone(G, weight=weight, ultrametric=True)


def _distance_backbone(G, weight="weight", ultrametric=False):
    """Shared helper for metric and ultrametric backbones."""
    H = G.__class__()
    H.add_nodes_from(G.nodes(data=True))

    dist_G = nx.Graph()
    for u, v, data in G.edges(data=True):
        dist_G.add_edge(u, v, dist=1.0 / data[weight])

    if ultrametric:
        mst = nx.minimum_spanning_tree(dist_G, weight="dist")
        for u, v, data in G.edges(data=True):
            direct_dist = 1.0 / data[weight]
            path = nx.shortest_path(mst, u, v, weight=None)
            max_edge = max(
                dist_G[path[i]][path[i + 1]]["dist"] for i in range(len(path) - 1)
            )
            if abs(direct_dist - max_edge) < 1e-12 or direct_dist <= max_edge:
                H.add_edge(u, v, **data)
    else:
        sp = dict(nx.all_pairs_dijkstra_path_length(dist_G, weight="dist"))
        for u, v, data in G.edges(data=True):
            direct_dist = 1.0 / data[weight]
            if abs(direct_dist - sp[u][v]) < 1e-12:
                H.add_edge(u, v, **data)

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
    >>> from networkx_backbone import doubly_stochastic_filter
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 3.0), (0, 2, 1.0)])
    >>> H = doubly_stochastic_filter(G)
    >>> "ds_weight" in H[0][1]
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
        The h-backbone subgraph.  All original nodes are preserved.

    References
    ----------
    .. [1] Zhang, R. J., Stanley, H. E., & Ye, F. Y. (2018). Extracting
       h-backbone as a core structure in weighted networks. *Scientific
       Reports*, 8, 14356.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import h_backbone
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 3.0), (0, 2, 1.0)])
    >>> H = h_backbone(G)
    >>> H.number_of_edges() <= G.number_of_edges()
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
            h_strength_edges.add((min(u, v), max(u, v)))

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
            h_bridge_edges.add((min(u, v), max(u, v)))

    # Union
    backbone_edges = h_strength_edges | h_bridge_edges
    H = G.__class__()
    H.add_nodes_from(G.nodes(data=True))
    for u, v in backbone_edges:
        if G.has_edge(u, v):
            H.add_edge(u, v, **G[u][v])
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
        A copy of *G* with an additional ``"vitality"`` node attribute
        (float).

    References
    ----------
    .. [1] Rajeh, S., Savonnet, M., Leclercq, E., & Cherifi, H. (2022).
       Modularity-based backbone extraction in weighted complex networks.
       *NetSci-X 2022*, 67-79.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import modularity_backbone
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 3.0), (0, 2, 1.0)])
    >>> H = modularity_backbone(G)
    >>> "vitality" in H.nodes[0]
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
        A planar subgraph of *G* with maximum total weight.  All
        original nodes are preserved.

    References
    ----------
    .. [1] Tumminello, M., Aste, T., Di Matteo, T., & Mantegna, R. N.
       (2005). A tool for filtering information in complex systems.
       *PNAS*, 102(30), 10421-10426.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import planar_maximally_filtered_graph
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 3.0), (0, 2, 1.0)])
    >>> H = planar_maximally_filtered_graph(G)
    >>> H.number_of_edges() <= G.number_of_edges()
    True
    """
    n = G.number_of_nodes()
    max_edges = 3 * (n - 2) if n >= 3 else n - 1

    # Sort edges by weight descending
    edges = sorted(G.edges(data=True), key=lambda e: e[2].get(weight, 0), reverse=True)

    H = G.__class__()
    H.add_nodes_from(G.nodes(data=True))

    for u, v, data in edges:
        if H.number_of_edges() >= max_edges:
            break
        H.add_edge(u, v, **data)
        is_planar, _ = nx.check_planarity(H)
        if not is_planar:
            H.remove_edge(u, v)

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
        The maximum spanning tree of *G*.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import maximum_spanning_tree_backbone
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 3.0), (0, 2, 1.0)])
    >>> H = maximum_spanning_tree_backbone(G)
    >>> H.number_of_edges()
    2
    """
    return nx.maximum_spanning_tree(G, weight=weight)
