"""Figures of ``docs/user_guide/directed_graphs.rst``.

The page's own example graph, and the two one-sided tests that the disparity
filter runs on each of its directed edges: from the source's out-links and
from the target's in-links.  The edge's ``disparity_pvalue`` is the smaller
of the two; the build checks that against the library.
"""

from __future__ import annotations

import functools
import math

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

import nx_backbone as nb
from _figures import style
from _figures.pages import figure

#: The weighted edges of the page's example.
EDGES = [
    (0, 1, 5.0),
    (0, 2, 3.0),
    (0, 3, 1.0),
    (1, 0, 2.0),
    (1, 2, 4.0),
    (2, 3, 6.0),
]
POS = {0: (0.0, 1.0), 1: (1.0, 1.0), 2: (1.0, 0.0), 3: (0.0, 0.0)}


def _side_pvalue(w: float, s: float, k: int) -> float:
    """Disparity p-value from one node's links: ``(1 - w/s)^(k-1)``; 1 if k <= 1."""
    return 1.0 if k <= 1 else (1.0 - w / s) ** (k - 1)


@functools.cache
def _tests():
    """Per edge: (u, v, w, out-test p, in-test p, disparity_pvalue)."""
    G = nx.DiGraph()
    G.add_weighted_edges_from(EDGES)
    H = nb.disparity_filter(G)
    s_out, k_out = dict(G.out_degree(weight="weight")), dict(G.out_degree())
    s_in, k_in = dict(G.in_degree(weight="weight")), dict(G.in_degree())
    rows = []
    for u, v, w in EDGES:
        p_out = _side_pvalue(w, s_out[u], k_out[u])
        p_in = _side_pvalue(w, s_in[v], k_in[v])
        p = H.edges[u, v]["disparity_pvalue"]
        # The page's claim: the edge p-value is the smaller one-sided p-value.
        assert math.isclose(p, min(p_out, p_in), rel_tol=1e-12), (u, v, p)
        rows.append((u, v, w, p_out, p_in, p))
    return G, rows


def _decided_by_in():
    """Edges whose out-test gives 1 but whose in-test does not."""
    _, rows = _tests()
    return [r for r in rows if r[3] == 1.0 > r[4]]


def _alt() -> str:
    G, _ = _tests()
    (u, v, w, p_out, p_in, _), *_ = _decided_by_in()
    return (
        f"Left, a directed graph of {G.number_of_nodes()} nodes and "
        f"{G.number_of_edges()} weighted edges drawn as arrows with their "
        "weights. Right, one row per edge with two markers on a p-value axis "
        "from 0 to 1: a circle for the test from the source's out-links and a "
        "square for the test from the target's in-links, and the edge's "
        f"p-value written at the right. For edge {u} to {v}, weight {w:g}, "
        f"the out-test gives {p_out:.2f} and the in-test {p_in:.2f}."
    )


def _caption() -> str:
    G, rows = _tests()
    u, v, w, p_out, p_in, _ = _decided_by_in()[0]
    s_in = G.in_degree(v, weight="weight")
    n_one = sum(1 for r in rows if 1.0 in (r[3], r[4]))
    return (
        ":func:`~nx_backbone.disparity_filter` on the example below tests each "
        "edge *u* → *v* twice: against *u*'s out-strength and out-degree (circle) "
        "and against *v*'s in-strength and in-degree (square). "
        "``disparity_pvalue`` is the smaller of the two. "
        f"Edge {u} → {v} (weight {w:g}) is node {u}'s only out-link, so its "
        f"out-test gives {p_out:g}; it carries {w:g} of the {s_in:g} units "
        f"reaching node {v}, so its in-test gives {p_in:.2f}. "
        f"{n_one} of the {len(rows)} edges have a side with a single link, "
        "whose test always gives 1."
    )


@figure("out-in-tests", alt=_alt, caption=_caption)
def draw_out_in_tests(mode):
    G, rows = _tests()
    p = style.palette(mode)
    c_out, c_in = style.categorical(2, mode)
    fig, (ax_g, ax_p) = plt.subplots(
        1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1, 2.1]}
    )
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.25, top=0.88, wspace=0.55)

    # The graph: arrows with their weights; reciprocal edges curve apart.
    nx.draw_networkx_nodes(
        G, POS, ax=ax_g, node_size=380, node_color=p.surface, edgecolors=p.ink
    )
    nx.draw_networkx_labels(G, POS, ax=ax_g, font_size=9, font_color=p.ink)
    for u, v, w in EDGES:
        rad = 0.25 if G.has_edge(v, u) else 0.0
        nx.draw_networkx_edges(
            G,
            POS,
            edgelist=[(u, v)],
            ax=ax_g,
            node_size=380,
            edge_color=p.muted,
            width=1.2,
            arrowsize=12,
            connectionstyle=f"arc3,rad={rad}",
        )
        (x0, y0), (x1, y1) = POS[u], POS[v]
        # Label at the midpoint, pushed toward the arc's bulge.
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        nx_, ny_ = (y1 - y0), -(x1 - x0)
        off = rad * 0.5 + (0.0 if rad else 0.09)
        ax_g.text(
            mx + nx_ * off,
            my + ny_ * off,
            f"{w:g}",
            fontsize=8,
            color=p.ink,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.12", "fc": p.surface, "ec": "none"},
        )
    ax_g.set_xlim(-0.3, 1.3)
    ax_g.set_ylim(-0.35, 1.35)
    ax_g.set_aspect("equal")
    ax_g.set_axis_off()
    ax_g.set_title("Example graph (weights on arrows)")

    # The two one-sided tests per edge, and the edge's p-value.
    order = list(reversed(rows))
    for y, (_u, _v, _w, p_out, p_in, pv) in enumerate(order):
        ax_p.plot([p_out, p_in], [y, y], color=p.hairline, lw=2.0, zorder=1)
        ax_p.scatter(
            p_out, y, s=46, marker="o", c=c_out, edgecolors=p.surface, zorder=3
        )
        ax_p.scatter(p_in, y, s=46, marker="s", c=c_in, edgecolors=p.surface, zorder=2)
        ax_p.annotate(
            f"p = {pv:.2f}",
            (1.0, y),
            xytext=(14, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=p.ink,
            annotation_clip=False,
        )
    ax_p.set_yticks(
        range(len(order)), [f"{u} → {v}  (w = {w:g})" for u, v, w, *_ in order]
    )
    ax_p.tick_params(axis="y", length=0, labelcolor=p.ink)
    ax_p.set_xlim(-0.03, 1.03)
    ax_p.set_ylim(-0.6, len(order) - 0.4)
    ax_p.set_xlabel("One-sided disparity p-value")
    ax_p.set_title("Two tests per edge; the edge keeps the smaller")
    ax_p.grid(True, axis="x")
    ax_p.spines["left"].set_visible(False)
    handles = [
        Line2D([], [], marker="o", ls="", color=c_out, label="From u's out-links"),
        Line2D([], [], marker="s", ls="", color=c_in, label="From v's in-links"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.62, 0.0),
        ncols=2,
        frameon=False,
    )
    return fig
