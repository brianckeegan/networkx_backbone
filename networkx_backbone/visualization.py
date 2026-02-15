"""
Visualization utilities for backbone comparisons.

These helpers compare an original graph with a backbone graph and render
the differences using consistent defaults:

- removed nodes in red
- retained edges in thicker black
"""

import networkx as nx

from networkx_backbone._docstrings import append_complexity_docstrings

__all__ = [
    "graph_difference",
    "compare_graphs",
    "save_graph_comparison",
]


def _edge_key(u, v, directed):
    """Build a comparable edge key for directed or undirected matching."""
    if directed:
        return (u, v)
    a, b = sorted((u, v), key=str)
    return (a, b)


def graph_difference(original, backbone):
    """Compute node/edge overlap and removals between two graphs.

    Parameters
    ----------
    original : graph
        Reference graph.
    backbone : graph
        Candidate backbone graph to compare against ``original``.

    Returns
    -------
    diff : dict
        Dictionary containing:

        - ``"kept_nodes"``
        - ``"removed_nodes"``
        - ``"kept_edges"``
        - ``"removed_edges"``
    """
    directed = original.is_directed()

    original_nodes = set(original.nodes())
    backbone_nodes = set(backbone.nodes())
    kept_nodes = original_nodes & backbone_nodes
    removed_nodes = original_nodes - backbone_nodes

    original_edges = {_edge_key(u, v, directed) for u, v in original.edges()}
    backbone_edges = {_edge_key(u, v, directed) for u, v in backbone.edges()}
    kept_edges = original_edges & backbone_edges
    removed_edges = original_edges - kept_edges

    return {
        "kept_nodes": kept_nodes,
        "removed_nodes": removed_nodes,
        "kept_edges": kept_edges,
        "removed_edges": removed_edges,
    }


def compare_graphs(
    original,
    backbone,
    pos=None,
    ax=None,
    with_labels=False,
    node_size=220,
    kept_node_color="#f2f2f2",
    removed_node_color="red",
    base_edge_color="#c9c9c9",
    kept_edge_color="black",
    base_edge_width=0.8,
    kept_edge_width=2.6,
    title=None,
    return_diff=False,
):
    """Visualize differences between an original graph and a backbone graph.

    Parameters
    ----------
    original : graph
        Reference graph.
    backbone : graph
        Backbone graph to compare against ``original``.
    pos : dict or None, optional (default=None)
        Node-position dictionary. If ``None``, uses ``spring_layout``.
    ax : matplotlib Axes or None, optional (default=None)
        Axis to draw on. If ``None``, a new figure and axis are created.
    with_labels : bool, optional (default=False)
        Whether to draw node labels.
    node_size : int, optional (default=220)
        Node size.
    kept_node_color : color, optional
        Color for nodes retained in the backbone.
    removed_node_color : color, optional (default="red")
        Color for nodes removed from the backbone.
    base_edge_color : color, optional
        Base color for original edges.
    kept_edge_color : color, optional (default="black")
        Color for retained backbone edges.
    base_edge_width : float, optional
        Width for base/original edge layer.
    kept_edge_width : float, optional (default=2.6)
        Width for retained backbone edges (drawn on top).
    title : str or None, optional
        Plot title.
    return_diff : bool, optional (default=False)
        If ``True``, return the computed graph-difference dictionary.

    Returns
    -------
    (fig, ax) or (fig, ax, diff)
        Matplotlib figure/axes, and optionally the difference dictionary.
    """
    import matplotlib.pyplot as plt

    if pos is None:
        pos = nx.spring_layout(original, seed=42)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    diff = graph_difference(original, backbone)
    directed = original.is_directed()

    original_edge_tuples = list(original.edges())
    kept_edge_tuples = [
        (u, v)
        for u, v in original_edge_tuples
        if _edge_key(u, v, directed) in diff["kept_edges"]
    ]

    kept_nodes = [n for n in original.nodes() if n in diff["kept_nodes"]]
    removed_nodes = [n for n in original.nodes() if n in diff["removed_nodes"]]

    nx.draw_networkx_edges(
        original,
        pos,
        ax=ax,
        edgelist=original_edge_tuples,
        edge_color=base_edge_color,
        width=base_edge_width,
        alpha=0.9,
        arrows=directed,
    )
    nx.draw_networkx_edges(
        original,
        pos,
        ax=ax,
        edgelist=kept_edge_tuples,
        edge_color=kept_edge_color,
        width=kept_edge_width,
        alpha=1.0,
        arrows=directed,
    )
    nx.draw_networkx_nodes(
        original,
        pos,
        ax=ax,
        nodelist=kept_nodes,
        node_color=kept_node_color,
        node_size=node_size,
        linewidths=0.8,
        edgecolors="black",
    )
    if removed_nodes:
        nx.draw_networkx_nodes(
            original,
            pos,
            ax=ax,
            nodelist=removed_nodes,
            node_color=removed_node_color,
            node_size=node_size,
            linewidths=0.8,
            edgecolors="black",
        )

    if with_labels:
        nx.draw_networkx_labels(original, pos, ax=ax, font_size=7)

    if title:
        ax.set_title(title)
    ax.set_axis_off()

    if return_diff:
        return fig, ax, diff
    return fig, ax


def save_graph_comparison(
    original,
    backbone,
    output_path,
    pos=None,
    dpi=180,
    bbox_inches="tight",
    **kwargs,
):
    """Save a comparison visualization to disk."""
    fig, _ = compare_graphs(original, backbone, pos=pos, **kwargs)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    return output_path


_COMPLEXITY = {
    "graph_difference": {
        "time": "O(n + m)",
        "space": "O(n + m)",
    },
    "compare_graphs": {
        "time": "O(n + m) without layout; plus layout cost if pos is None",
        "space": "O(n + m)",
        "notes": "Spring layout adds iterative graph-drawing overhead.",
    },
    "save_graph_comparison": {
        "time": "O(n + m) without layout; plus rendering/write cost",
        "space": "O(n + m)",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
