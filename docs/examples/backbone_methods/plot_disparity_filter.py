"""
Disparity Filter Visualization
==============================

This example applies the disparity filter to ``nx.les_miserables_graph()``
using a strict score-then-filter workflow:

1. Score all edges on the full graph with ``disparity_filter``.
2. Filter by ``disparity_pvalue`` with ``threshold_filter``.
3. Visualize removed and retained structure with ``compare_graphs``.
"""

import warnings
from pathlib import Path
import tempfile

import networkx as nx

import networkx_backbone as nb
from networkx_backbone.visualization import (
    compare_graphs,
    graph_difference,
    save_graph_comparison,
)

G = nx.les_miserables_graph()

# Score on the full graph.
scored = nb.disparity_filter(G)

# Then filter scored edges to extract the backbone.
backbone = nb.threshold_filter(
    scored,
    score="disparity_pvalue",
    threshold=0.05,
    mode="below",
    include_all_nodes=False,
)

if backbone.number_of_edges() == G.number_of_edges():
    warnings.warn(
        "Disparity filter returned the same number of edges as the original "
        "graph. Re-test and validate threshold settings.",
        UserWarning,
    )

pos = nx.spring_layout(G, seed=42, weight="weight")
fig, ax, diff = compare_graphs(
    G,
    backbone,
    pos=pos,
    title="Disparity Filter on Les Miserables",
    return_diff=True,
)

diff_summary = graph_difference(G, backbone)
_ = save_graph_comparison(
    G,
    backbone,
    output_path=Path(tempfile.gettempdir()) / "disparity_filter_example.png",
    pos=pos,
    title="Disparity Filter on Les Miserables",
)

print(f"Original edges: {G.number_of_edges()}")
print(f"Backbone edges: {len(diff['kept_edges'])}")
print(f"Removed nodes: {len(diff['removed_nodes'])}")
print(f"Removed edges: {len(diff_summary['removed_edges'])}")
