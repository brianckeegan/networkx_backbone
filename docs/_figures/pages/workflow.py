"""Figures of ``docs/user_guide/workflow.rst``."""

from __future__ import annotations

import functools
import math

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.collections import LineCollection

import nx_backbone as nb
from _figures import style
from _figures.pages import figure

ALPHA = 0.05


@functools.cache
def _scored():
    """Les Misérables, its disparity scores and the backbone at ``ALPHA``."""
    G = nx.les_miserables_graph()
    H = nb.disparity_filter(G)
    backbone = nb.threshold_filter(H, "disparity_pvalue", ALPHA)
    return G, H, backbone


def _counts() -> style.BackboneCounts:
    G, _, backbone = _scored()
    return style.backbone_counts(G, backbone)


def _alt() -> str:
    c = _counts()
    return (
        f"Two drawings of the Les Misérables network, {c.nodes_total} "
        f"characters and {c.edges_total} edges, on the same layout. Left, "
        "every edge shaded by its disparity-filter p-value, thicker and "
        "stronger in color for smaller p-values. Right, the backbone: the "
        f"{c.edges_kept} edges with p below {ALPHA} drawn solid over the "
        "removed edges in faint gray."
    )


def _caption() -> str:
    c = _counts()
    return (
        "Score, then filter. Step 1, :func:`~nx_backbone.disparity_filter`, "
        "gives every edge a ``disparity_pvalue`` (left; the thicker the line, "
        "the smaller the p-value). "
        "Step 2, :func:`~nx_backbone.threshold_filter` at "
        f"``{ALPHA}``, keeps the edges below it (right). {c.caption}"
    )


@figure("score-then-filter", alt=_alt, caption=_caption, fmt="png")
def draw_score_then_filter(mode):
    G, H, backbone = _scored()
    p = style.palette(mode)
    pos = style.layout(G, "les_miserables")
    fig, (ax_score, ax_filter) = plt.subplots(1, 2, figsize=(8, 3.6))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.14, top=0.92, wspace=0.04)

    # Step 1: every edge, shaded and sized by its evidence -log10(p).
    cmap = style.sequential_cmap(mode)
    edges = sorted(H.edges(data="disparity_pvalue"), key=lambda e: -e[2])
    evidence = [-math.log10(max(pv, 1e-12)) for *_, pv in edges]
    top = max(evidence) or 1.0
    shade = [e / top for e in evidence]
    ax_score.add_collection(
        LineCollection(
            [(pos[u], pos[v]) for u, v, _ in edges],
            colors=[cmap(s) for s in shade],
            linewidths=[0.4 + 1.6 * s for s in shade],
            capstyle="round",
        )
    )
    xy = [pos[n] for n in G]
    ax_score.scatter(
        [x for x, _ in xy],
        [y for _, y in xy],
        s=style.NODE_SIZE * 0.6,
        c=p.node_removed,
        linewidths=0,
        zorder=3,
    )
    ax_score.set_aspect("equal", adjustable="datalim")
    ax_score.margins(0.04)
    ax_score.autoscale_view()
    ax_score.set_axis_off()
    ax_score.set_title("1. Score: disparity p-value")

    # Step 2: the edges that pass the filter, over the original graph.
    style.draw_backbone(ax_filter, G, backbone, pos, mode=mode)
    ax_filter.set_title(f"2. Filter: p < {ALPHA}")
    return fig
