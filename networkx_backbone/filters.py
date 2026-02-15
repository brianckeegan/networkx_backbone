"""
Filtering utilities for backbone extraction.

After a backbone method annotates edges (or nodes) with scores or p-values,
these functions extract the final subgraph.
"""

import networkx as nx

from networkx_backbone._docstrings import append_complexity_docstrings


__all__ = [
    "multigraph_to_weighted",
    "threshold_filter",
    "fraction_filter",
    "boolean_filter",
    "consensus_backbone",
]


def multigraph_to_weighted(G, weight="weight", edge_type_attr=None):
    """Convert a MultiGraph/MultiDiGraph into a weighted simple graph.

    Parallel edges between the same node pair are collapsed into a single
    weighted edge.

    Parameters
    ----------
    G : graph
        A NetworkX graph. If ``G`` is a ``MultiGraph`` or ``MultiDiGraph``,
        parallel edges are aggregated. For simple graphs, a copy is returned
        with missing weights filled as 1.
    weight : string, optional (default="weight")
        Edge attribute name used to store the aggregated weight.
    edge_type_attr : string or None, optional (default=None)
        If provided, weight is the count of distinct values of this edge
        attribute among parallel edges. If missing on all parallel edges,
        falls back to the number of parallel edges.

    Returns
    -------
    H : graph
        ``nx.Graph`` for undirected input and ``nx.DiGraph`` for directed
        input. Each collapsed edge includes:

        - ``weight``: aggregated weight
        - ``"edge_count"``: number of parallel edges collapsed
        - ``"edge_type_count"``: number of distinct edge types (only when
          ``edge_type_attr`` is provided)

    Examples
    --------
    >>> import networkx as nx

    >>> from networkx_backbone import multigraph_to_weighted
    >>> B = nx.davis_southern_women_graph()
    >>> MG = nx.MultiGraph()
    >>> MG.add_nodes_from(B.nodes(data=True))
    >>> u, v = next(iter(B.edges()))
    >>> _ = MG.add_edge(u, v, edge_type="attendance_a")
    >>> _ = MG.add_edge(u, v, edge_type="attendance_b")
    >>> H = multigraph_to_weighted(MG, edge_type_attr="edge_type")
    >>> H[u][v]["weight"]
    2
    """
    if not G.is_multigraph():
        H = G.copy()
        for _, _, data in H.edges(data=True):
            data.setdefault(weight, 1)
        return H

    H = nx.DiGraph() if G.is_directed() else nx.Graph()
    H.add_nodes_from(G.nodes(data=True))

    grouped = {}
    for u, v, _, data in G.edges(keys=True, data=True):
        key = (u, v)
        grouped.setdefault(key, []).append(data)

    for (u, v), edge_datas in grouped.items():
        edge_count = len(edge_datas)
        out_data = {"edge_count": edge_count}

        if edge_type_attr is None:
            out_data[weight] = edge_count
        else:
            present_types = [
                data[edge_type_attr] for data in edge_datas if edge_type_attr in data
            ]
            type_count = len(set(present_types)) if present_types else edge_count
            out_data[weight] = type_count
            out_data["edge_type_count"] = type_count

        H.add_edge(u, v, **out_data)

    return H


def threshold_filter(
    G, score, threshold, mode="below", filter_on="edges", include_all_nodes=True
):
    """Retain edges or nodes whose score passes a threshold test.

    Parameters
    ----------
    G : graph
        A NetworkX graph with a computed score attribute on edges or nodes.
    score : string
        Attribute name to filter on.
    threshold : float
        Cutoff value.
    mode : {"below", "above"}, optional (default="below")
        ``"below"`` keeps elements with ``score < threshold`` (typical for
        p-values).  ``"above"`` keeps elements with ``score >= threshold``
        (typical for salience or importance scores).
    filter_on : {"edges", "nodes"}, optional (default="edges")
        Whether to filter edges or nodes.
    include_all_nodes : bool, optional (default=True)
        Controls isolate retention in the output.

        - If ``filter_on="edges"`` and ``True``, all input nodes are kept.
          If ``False``, only nodes incident to retained edges are kept.
        - If ``filter_on="nodes"`` and ``True``, all retained nodes are kept
          even if isolated in the induced subgraph. If ``False``, retained
          nodes with degree 0 are removed.

    Returns
    -------
    H : graph
        Filtered subgraph of the same type as *G*.  When filtering edges,
        all original nodes are preserved by default.  When filtering nodes,
        only retained nodes and their mutual edges are preserved.

    Raises
    ------
    ValueError
        If *mode* is not ``"below"`` or ``"above"``, or if *filter_on* is
        not ``"edges"`` or ``"nodes"``.

    Examples
    --------
    >>> import networkx as nx

    >>> from networkx_backbone import disparity_filter, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = disparity_filter(G)
    >>> filtered = threshold_filter(H, "disparity_pvalue", 0.5)
    >>> filtered.number_of_edges() <= H.number_of_edges()
    True
    """
    if mode not in ("below", "above"):
        raise ValueError(f"mode must be 'below' or 'above', got {mode!r}")

    if filter_on == "edges":
        H = G.__class__()
        if include_all_nodes:
            H.add_nodes_from(G.nodes(data=True))
        for u, v, data in G.edges(data=True):
            val = data.get(score)
            if val is None:
                continue
            if (mode == "below" and val < threshold) or (
                mode == "above" and val >= threshold
            ):
                H.add_edge(u, v, **data)
        return H

    elif filter_on == "nodes":
        keep = set()
        for node, data in G.nodes(data=True):
            val = data.get(score)
            if val is None:
                continue
            if (mode == "below" and val < threshold) or (
                mode == "above" and val >= threshold
            ):
                keep.add(node)
        H = G.subgraph(keep).copy()
        if not include_all_nodes:
            isolates = list(nx.isolates(H))
            if isolates:
                H.remove_nodes_from(isolates)
        return H

    else:
        raise ValueError(f"filter_on must be 'edges' or 'nodes', got {filter_on!r}")


def fraction_filter(G, score, fraction, ascending=True, filter_on="edges"):
    """Retain the top or bottom fraction of edges or nodes by score.

    Parameters
    ----------
    G : graph
        A NetworkX graph with a computed score attribute.
    score : string
        Attribute name to sort on.
    fraction : float
        Fraction of elements to retain, in (0, 1].
    ascending : bool, optional (default=True)
        If ``True``, keep the elements with the *smallest* scores (e.g.
        lowest p-values).  If ``False``, keep the *largest*.
    filter_on : {"edges", "nodes"}, optional (default="edges")
        Whether to filter edges or nodes.

    Returns
    -------
    H : graph
        Filtered subgraph of the same type as *G*.

    Raises
    ------
    ValueError
        If *fraction* is not in (0, 1] or *filter_on* is invalid.

    Examples
    --------
    >>> import networkx as nx

    >>> from networkx_backbone import disparity_filter, fraction_filter
    >>> G = nx.les_miserables_graph()
    >>> H = disparity_filter(G)
    >>> filtered = fraction_filter(H, "disparity_pvalue", 0.5)
    >>> filtered.number_of_edges() <= H.number_of_edges()
    True
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    if filter_on == "edges":
        scored = [
            (u, v, data, data.get(score, float("inf") if ascending else float("-inf")))
            for u, v, data in G.edges(data=True)
        ]
        scored.sort(key=lambda x: x[3], reverse=not ascending)
        n_keep = max(1, int(len(scored) * fraction))
        keep = scored[:n_keep]

        H = G.__class__()
        H.add_nodes_from(G.nodes(data=True))
        for u, v, data, _ in keep:
            H.add_edge(u, v, **data)
        return H

    elif filter_on == "nodes":
        scored = [
            (node, data, data.get(score, float("inf") if ascending else float("-inf")))
            for node, data in G.nodes(data=True)
        ]
        scored.sort(key=lambda x: x[2], reverse=not ascending)
        n_keep = max(1, int(len(scored) * fraction))
        keep = {s[0] for s in scored[:n_keep]}
        return G.subgraph(keep).copy()

    else:
        raise ValueError(f"filter_on must be 'edges' or 'nodes', got {filter_on!r}")


def boolean_filter(G, score):
    """Retain edges whose boolean score attribute is truthy.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    score : string
        Edge attribute name (should contain boolean values).

    Returns
    -------
    H : graph
        A new graph of the same type as *G* containing only edges for
        which ``data[score]`` is truthy.  All original nodes are preserved.

    Examples
    --------
    >>> import networkx as nx

    >>> from networkx_backbone import boolean_filter, global_threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> scored = global_threshold_filter(G, threshold=2)
    >>> H = boolean_filter(scored, "global_threshold_keep")
    >>> H.number_of_edges() <= scored.number_of_edges()
    True
    """
    H = G.__class__()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if data.get(score):
            H.add_edge(u, v, **data)
    return H


def consensus_backbone(*backbones):
    """Return the intersection of multiple backbone graphs.

    An edge is in the consensus backbone if and only if it appears in
    **every** input backbone.  Only nodes that are endpoints of consensus
    edges are retained.

    Parameters
    ----------
    *backbones : graph
        Two or more backbone graphs (must share the same node identifiers).

    Returns
    -------
    H : graph
        A new graph of the same type as the first backbone, containing
        only edges present in every input.

    Raises
    ------
    ValueError
        If fewer than 2 backbones are provided.

    Examples
    --------
    >>> import networkx as nx

    >>> from networkx_backbone import (
    ...     consensus_backbone,
    ...     jaccard_backbone,
    ...     cosine_backbone,
    ...     fraction_filter,
    ... )
    >>> G = nx.les_miserables_graph()
    >>> B1 = fraction_filter(jaccard_backbone(G), "jaccard", 0.3, ascending=False)
    >>> B2 = fraction_filter(cosine_backbone(G), "cosine", 0.3, ascending=False)
    >>> H = consensus_backbone(B1, B2)
    >>> H.number_of_edges() <= min(B1.number_of_edges(), B2.number_of_edges())
    True
    """
    if len(backbones) < 2:
        raise ValueError("consensus_backbone requires at least 2 graphs")

    edge_sets = []
    for bb in backbones:
        if bb.is_directed():
            edge_sets.append(set(bb.edges()))
        else:
            edge_sets.append({(min(u, v), max(u, v)) for u, v in bb.edges()})

    common = edge_sets[0]
    for es in edge_sets[1:]:
        common &= es

    ref = backbones[0]
    H = ref.__class__()
    for u, v in common:
        if ref.has_edge(u, v):
            H.add_edge(u, v, **ref[u][v])
        elif not ref.is_directed() and ref.has_edge(v, u):
            H.add_edge(v, u, **ref[v][u])

    return H


_COMPLEXITY = {
    "multigraph_to_weighted": {
        "time": "O(m)",
        "space": "O(n + m)",
        "notes": "m counts input edge instances for multigraphs.",
    },
    "threshold_filter": {
        "time": "O(n + m)",
        "space": "O(n + m)",
    },
    "fraction_filter": {
        "time": "O(m log m)",
        "space": "O(n + m)",
        "notes": "Edge mode; node mode is O(n log n).",
    },
    "boolean_filter": {
        "time": "O(n + m)",
        "space": "O(n + m)",
    },
    "consensus_backbone": {
        "time": "O(km)",
        "space": "O(m)",
        "notes": "k=number of input backbones, m=edges per backbone.",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
