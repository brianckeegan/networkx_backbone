"""Figures of ``docs/tutorials/hypergraph_backbone.rst``."""

from __future__ import annotations

import functools
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import nx_backbone as nb
from _figures import style
from _figures.pages import figure

# The page's local-MDL example: a core, two nested sub-interactions and two
# satellites, each joining part of the core to three nodes of its own.
H2 = [
    (0, 1, 2, 3, 4, 5),
    (0, 1, 2, 3, 4),
    (0, 1, 2, 3, 5),
    (0, 1, 2, 3, 6, 7, 8),
    (2, 3, 4, 5, 9, 10, 11),
]
ROLES = ["core", "nested", "nested", "satellite", "satellite"]


@functools.cache
def _scored():
    """The global and local (majority) MDL backbones and the clique expansion."""
    edges = [frozenset(h) for h in H2]
    kept_global = set(nb.mdl_hypergraph_backbone(H2).backbone)
    kept_local = set(nb.local_mdl_hypergraph_backbone(H2).backbone)
    # The page's claims: both drop the nested sub-interactions, and only the
    # local backbone keeps the first satellite.
    assert kept_global == {edges[0], edges[4]}
    assert kept_local == {edges[0], edges[3], edges[4]}
    G = nx.Graph()
    G.add_nodes_from(sorted(set().union(*edges)))
    for h in edges:
        for u, v in combinations(sorted(h), 2):
            if G.has_edge(u, v):
                G[u][v]["hyperedges"] += 1
            else:
                G.add_edge(u, v, hyperedges=1)
    return edges, kept_global, kept_local, G


def _span(h) -> str:
    """``{0, 1, 2, 3, 6, 7, 8}`` as ``{0-3, 6-8}``."""
    nodes = sorted(h)
    runs, start = [], nodes[0]
    for a, b in zip(nodes, [*nodes[1:], None], strict=True):
        if b != a + 1:
            runs.append(f"{start}" if start == a else f"{start}–{a}")
            start = b
    return "{" + ", ".join(runs) + "}"


def _stats() -> dict:
    edges, kept_global, kept_local, G = _scored()
    top = max(m for *_, m in G.edges(data="hyperedges"))
    busiest = [tuple(sorted(e)) for e in G.edges if G.edges[e]["hyperedges"] == top]
    return {
        "hyperedges": len(edges),
        "nodes": G.number_of_nodes(),
        "global": len(kept_global),
        "local": len(kept_local),
        "pairs": G.number_of_edges(),
        "top": top,
        "busiest": busiest,
    }


def _alt() -> str:
    s = _stats()
    return (
        f"Left, a grid of the {s['hyperedges']} hyperedges (rows) by the "
        f"{s['nodes']} nodes (columns), a filled square where a node is in a "
        "hyperedge, with two columns of text saying whether the global and the "
        "local MDL backbone keep each row. The global backbone keeps the core "
        "and the second satellite; the local backbone also keeps the first "
        "satellite; neither keeps the two nested hyperedges. Right, the "
        f"pairwise projection: {s['pairs']} edges on the same {s['nodes']} "
        "nodes, thicker and darker where a pair lies in more hyperedges, up to "
        f"{s['top']}."
    )


def _caption() -> str:
    s = _stats()
    pairs = " and ".join("{" + f"{u}, {v}" + "}" for u, v in s["busiest"])
    return (
        f"A sub-hypergraph, not a projection. Left, the {s['hyperedges']} hyperedges of "
        "``H2`` and which backbone keeps each: "
        f":func:`~nx_backbone.mdl_hypergraph_backbone` keeps {s['global']}, "
        ":func:`~nx_backbone.local_mdl_hypergraph_backbone` (majority) keeps "
        f"{s['local']}, and both drop the nested sub-interactions. Right, the "
        f"pairwise projection of the same hyperedges: {s['pairs']} edges, "
        f"and the pair {pairs} lies in all {s['top']} hyperedges. The "
        "projection cannot tell the six-node core from its nested parts, so a "
        "pairwise backbone of it has no hyperedges to keep or drop."
    )


@figure("hyperedges-vs-projection", alt=_alt, caption=_caption, fmt="svg")
def draw_hyperedges_vs_projection(mode):
    edges, kept_global, kept_local, G = _scored()
    p = style.palette(mode)
    nodes = list(G)
    fig, (ax_h, ax_p) = plt.subplots(
        1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1.45, 1]}
    )
    fig.subplots_adjust(left=0.2, right=0.99, bottom=0.2, top=0.88, wspace=0.12)

    # Left: hyperedges x nodes; squares in the kept color where the global
    # backbone keeps the row, and the keep calls written out on the right.
    n = len(nodes)
    for row, h in enumerate(edges):
        color = p.kept if h in kept_global else p.node_removed
        xs = [nodes.index(v) for v in sorted(h)]
        ax_h.scatter(range(n), [row] * n, marker="s", s=6, c=p.hairline, linewidths=0)
        ax_h.scatter(xs, [row] * len(xs), marker="s", s=52, c=color, linewidths=0)
        for col, kept in ((n + 0.6, kept_global), (n + 2.4, kept_local)):
            ax_h.text(
                col,
                row,
                "kept" if h in kept else "–",
                ha="center",
                va="center",
                color=p.ink if h in kept else p.muted,
            )
    for col, label in ((n + 0.6, "global"), (n + 2.4, "local")):
        ax_h.text(col, -0.9, label, ha="center", va="bottom", color=p.muted)
    ax_h.set_yticks(
        range(len(edges)),
        [f"{role} {_span(h)}" for role, h in zip(ROLES, edges, strict=True)],
    )
    ax_h.set_xticks(range(n), nodes)
    ax_h.set_xlim(-0.6, n + 3.2)
    ax_h.set_ylim(len(edges) - 0.5, -1.2)
    ax_h.set_xlabel("Node")
    ax_h.tick_params(length=0)
    for side in ("left", "bottom"):
        ax_h.spines[side].set_visible(False)
    ax_h.set_title("Hyperedges and the MDL backbones", pad=10)
    ax_h.legend(
        handles=[
            Line2D(
                [], [], marker="s", ls="", color=p.kept, label="in the global backbone"
            ),
            Line2D([], [], marker="s", ls="", color=p.node_removed, label="not in it"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.16),
        ncols=2,
        handletextpad=0.3,
    )

    # Right: the clique expansion, edges by how many hyperedges hold the pair.
    pos = style.layout(G, "hypergraph_h2")
    top = max(m for *_, m in G.edges(data="hyperedges"))
    cmap = style.sequential_cmap(mode)
    lines = sorted(G.edges(data="hyperedges"), key=lambda e: e[2])
    ax_p.add_collection(
        LineCollection(
            [(pos[u], pos[v]) for u, v, _ in lines],
            colors=[cmap(m / top) for *_, m in lines],
            linewidths=[0.5 + 2.0 * (m - 1) / (top - 1) for *_, m in lines],
            capstyle="round",
        )
    )
    xy = [pos[v] for v in nodes]
    ax_p.scatter(
        [x for x, _ in xy],
        [y for _, y in xy],
        s=150,
        c=p.surface,
        edgecolors=p.muted,
        linewidths=1.0,
        zorder=3,
    )
    for v in nodes:
        ax_p.text(*pos[v], str(v), ha="center", va="center", fontsize=7.5, zorder=4)
    ax_p.set_aspect("equal", adjustable="datalim")
    ax_p.margins(0.08)
    ax_p.autoscale_view()
    ax_p.set_axis_off()
    ax_p.set_title("Pairwise projection", pad=10)
    ax_p.legend(
        handles=[
            Line2D(
                [],
                [],
                color=cmap(m / top),
                lw=0.5 + 2.0 * (m - 1) / (top - 1),
                label=f"{m}",
            )
            for m in (1, (1 + top) // 2, top)
        ],
        title="Hyperedges holding the pair",
        title_fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncols=3,
    )
    return fig
