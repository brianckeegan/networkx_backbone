"""Figures of ``docs/tutorials/unweighted_sparsification.rst``."""

from __future__ import annotations

import functools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import nx_backbone as nb
from _figures import style
from _figures.pages import figure

S_VALUES = np.round(np.arange(0.0, 1.0001, 0.05), 2)

# (label, call, marker, linestyle): at most style.ALL_PAIRS_CAP series.
SERIES = (
    ("lspar", lambda G, s: nb.lspar(G, s=s), "o", "-"),
    ("local_degree", lambda G, s: nb.local_degree(G, s=s), "s", "-"),
    # sparsify's defaults are LSpar's (Jaccard, rank, degree filter).
    ("sparsify, umst=True", lambda G, s: nb.sparsify(G, s=s, umst=True), "^", "--"),
)


@functools.cache
def _sweep():
    """Edges each sparsifier keeps on unweighted Les Misérables, per ``s``."""
    G_full = nx.les_miserables_graph()
    G = nx.Graph()
    G.add_nodes_from(G_full.nodes(data=True))
    G.add_edges_from(G_full.edges())
    counts = {
        label: [
            nb.boolean_filter(call(G, float(s)), "sparsify_keep").number_of_edges()
            for s in S_VALUES
        ]
        for label, call, *_ in SERIES
    }
    return G, counts


def _at(s: float) -> int:
    return int(np.flatnonzero(np.isclose(S_VALUES, s))[0])


def _first_equal() -> int:
    """First ``s`` index from which LSpar and LSpar + UMST agree."""
    _, counts = _sweep()
    same = [
        a == b
        for a, b in zip(counts["lspar"], counts["sparsify, umst=True"], strict=True)
    ]
    return next(i for i in range(len(same)) if all(same[i:]))


def _alt() -> str:
    G, counts = _sweep()
    lo, mid, hi = _at(0.0), _at(0.5), _at(1.0)
    parts = "; ".join(
        f"{label}: {c[lo]} edges at s = 0, {c[mid]} at s = 0.5, {c[hi]} at s = 1"
        for label, c in counts.items()
    )
    return (
        "Line chart of the number of edges kept against the sparsification "
        f"parameter s from 0 to 1, for three sparsifiers on the unweighted Les "
        f"Misérables graph ({G.number_of_edges()} edges, {G.number_of_nodes()} "
        f"nodes), with a different marker per line. {parts}."
    )


def _caption() -> str:
    _, counts = _sweep()
    lo, hi = _at(0.0), _at(1.0)
    lspar, umst = counts["lspar"], counts["sparsify, umst=True"]
    return (
        "The ``s`` dial. In :func:`~nx_backbone.lspar` each node keeps its "
        r"top :math:`\lceil d^s \rceil` edges, so the edge count rises "
        f"from {lspar[lo]} at ``s=0`` (every node's top-ranked edges, ties "
        f"included) to all {lspar[hi]} at ``s=1``. ``sparsify(..., umst=True)``, "
        "LSpar plus the union of "
        "minimum spanning trees of the scores, connects the backbone and "
        f"so raises the sparsest ones to at least {min(umst)} edges; from "
        f"``s={S_VALUES[_first_equal()]:g}`` on, LSpar already contains those "
        "trees and the two lines meet."
    )


@figure("edges-vs-s", alt=_alt, caption=_caption)
def draw_edges_vs_s(mode):
    G, counts = _sweep()
    pal = style.palette(mode)
    colors = style.categorical(len(SERIES), mode)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    total = G.number_of_edges()
    ax.axhline(total, color=pal.muted, linewidth=0.75)
    ax.text(
        0.01, total + 5, f"original graph: {total} edges", color=pal.muted, va="bottom"
    )
    for (label, _, marker, linestyle), color in zip(SERIES, colors, strict=True):
        ax.plot(
            S_VALUES,
            counts[label],
            color=color,
            marker=marker,
            markersize=5 if linestyle == "-" else 6.5,
            markerfacecolor=color if linestyle == "-" else "none",
            linestyle=linestyle,
            label=label,
        )
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, total * 1.12)
    ax.set_xlabel("s")
    ax.set_ylabel("edges kept")
    ax.grid(True)
    return fig
