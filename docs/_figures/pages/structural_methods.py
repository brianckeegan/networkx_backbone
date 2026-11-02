"""Figures of ``docs/tutorials/structural_methods.rst``."""

from __future__ import annotations

import functools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import nx_backbone as nb
from _figures import style
from _figures.pages import figure


@functools.cache
def _compared():
    """The page's "Comparing structural methods" block, verbatim."""
    G = nx.les_miserables_graph()
    backbones = {
        "metric": nb.boolean_filter(nb.metric_backbone(G), "metric_keep"),
        "ultrametric": nb.boolean_filter(
            nb.ultrametric_backbone(G), "ultrametric_keep"
        ),
        "h_backbone": nb.boolean_filter(nb.h_backbone(G), "h_backbone_keep"),
        "global_sparsification": nb.boolean_filter(
            nb.global_sparsification(G, s=0.4), "global_sparsification_keep"
        ),
        "edge_betweenness": nb.boolean_filter(
            nb.edge_betweenness_filter(G, s=0.3), "edge_betweenness_keep"
        ),
        "pmfg": nb.boolean_filter(nb.planar_maximally_filtered_graph(G), "pmfg_keep"),
        "mst": nb.boolean_filter(nb.maximum_spanning_tree_backbone(G), "mst_keep"),
    }
    results = nb.compare_backbones(G, backbones)
    rows = []
    for name, metrics in results.items():
        B = backbones[name]
        # Nodes that keep at least one edge, as compare_backbones counts them.
        touched = B.subgraph(n for n in B if B.degree(n) > 0)
        rows.append(
            {
                "name": name,
                "edges": B.number_of_edges(),
                "nodes": round(metrics["node_fraction"] * G.number_of_nodes()),
                "connected": touched.number_of_nodes() == G.number_of_nodes()
                and nx.is_connected(touched),
            }
        )
    rows.sort(key=lambda r: -r["edges"])
    return G, rows


def _alt() -> str:
    G, rows = _compared()
    listed = "; ".join(
        f"{r['name']} {r['edges']} edges, {r['nodes']} nodes"
        + (", connected" if r["connected"] else "")
        for r in rows
    )
    return (
        "Two horizontal bar charts sharing one list of seven structural "
        "backbones of Les Misérables, sorted by edges kept. Left, edges kept "
        f"of {G.number_of_edges()}; right, nodes that keep at least one edge, "
        f"of {G.number_of_nodes()}; backbones that keep the whole graph "
        f"connected are marked. Values: {listed}."
    )


def _caption() -> str:
    G, rows = _compared()
    connected = [r["name"] for r in rows if r["connected"]]
    split = [r for r in rows if not r["connected"]]
    fewest = min(split, key=lambda r: r["nodes"]) if split else None
    tail = (
        f" The other {len(split)} drop nodes; ``{fewest['name']}`` keeps "
        f"{fewest['edges']} edges on {fewest['nodes']} of "
        f"{G.number_of_nodes()} nodes."
        if fewest
        else ""
    )
    return (
        "Edges and nodes each backbone of the comparison above keeps (the "
        "``edge_fraction`` and ``node_fraction`` of "
        ":func:`~nx_backbone.compare_backbones`, as counts). "
        f"{len(connected)} of {len(rows)} keep every node connected "
        f"(marked ●): {', '.join(f'``{n}``' for n in connected)}." + tail
    )


@figure("compared", alt=_alt, caption=_caption)
def draw_compared(mode):
    G, rows = _compared()
    pal = style.palette(mode)
    y = np.arange(len(rows))
    fig, (ax_e, ax_n) = plt.subplots(
        1,
        2,
        figsize=(6.6, 0.32 * len(rows) + 0.9),
        sharey=True,
        gridspec_kw={"width_ratios": (1.6, 1.0), "wspace": 0.12},
    )
    panels = (
        (ax_e, "edges", G.number_of_edges(), "edges kept"),
        (ax_n, "nodes", G.number_of_nodes(), "nodes kept"),
    )
    for ax, key, total, label in panels:
        values = [r[key] for r in rows]
        ax.barh(y, values, height=0.62, color=pal.kept)
        for yi, v in zip(y, values, strict=True):
            ax.text(
                v + total * 0.015,
                yi,
                str(v),
                va="center",
                color=pal.ink,
                fontsize=8.5,
            )
        ax.set_xlim(0, total * 1.12)
        ax.set_title(f"{label} (of {total})", fontsize=9, fontweight="normal")
        ax.grid(axis="x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax_e.set_yticks(y, [r["name"] + ("  ●" if r["connected"] else "") for r in rows])
    ax_e.tick_params(axis="y", labelcolor=pal.ink)
    ax_e.invert_yaxis()
    fig.text(
        0.02,
        -0.02,
        f"● every one of the {G.number_of_nodes()} nodes kept, in one "
        "connected backbone",
        color=pal.muted,
        fontsize=8.5,
        va="top",
    )
    return fig
