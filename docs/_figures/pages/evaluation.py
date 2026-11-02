"""Figures of ``docs/user_guide/evaluation.rst``.

Every measure of :mod:`nx_backbone.measures` on one backbone: what it counts
or which two distributions it compares.  The drawn quantities are recomputed
here and checked against the library's values.
"""

from __future__ import annotations

import functools
import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

import nx_backbone as nb
from _figures import style
from _figures.pages import figure

ALPHA = 0.05
METHOD = "noise_corrected_filter"


@functools.cache
def _example():
    """Les Misérables and its noise-corrected backbone at ALPHA."""
    G = nx.les_miserables_graph()
    backbone = nb.threshold_filter(nb.noise_corrected_filter(G), "nc_pvalue", ALPHA)
    return G, backbone


def _ecdf_gap(a, b):
    """KS statistic of two samples and where it is reached: (D, x, F_a(x), F_b(x))."""
    a, b = np.sort(a), np.sort(b)
    xs = np.union1d(a, b)
    fa = np.searchsorted(a, xs, side="right") / len(a)
    fb = np.searchsorted(b, xs, side="right") / len(b)
    i = int(np.argmax(np.abs(fa - fb)))
    return float(abs(fa[i] - fb[i])), float(xs[i]), float(fa[i]), float(fb[i])


@functools.cache
def _measures():
    """The library's values, and the parts each one is computed from."""
    G, B = _example()

    def active(H):
        return sum(1 for n in H if H.degree(n) > 0)

    def total(H):
        return math.fsum(w for *_, w in H.edges(data="weight"))

    n = B.number_of_nodes()
    pairs = sum(len(c) * (len(c) - 1) for c in nx.connected_components(B))
    parts = {
        "node_fraction": (active(B), active(G), "nodes with an edge"),
        "edge_fraction": (B.number_of_edges(), G.number_of_edges(), "edges"),
        "weight_fraction": (total(B), total(G), "total weight"),
        "reachability": (pairs, n * (n - 1), "node pairs joined by a path"),
    }
    values = {
        "node_fraction": nb.node_fraction(G, B),
        "edge_fraction": nb.edge_fraction(G, B),
        "weight_fraction": nb.weight_fraction(G, B),
        "reachability": nb.reachability(B),
    }
    for name, (num, den, _) in parts.items():
        assert math.isclose(values[name], num / den), name
    degrees = ([d for _, d in G.degree()], [d for _, d in B.degree()])
    weights = (
        [w for *_, w in G.edges(data="weight")],
        [w for *_, w in B.edges(data="weight")],
    )
    ks = {"ks_degree": _ecdf_gap(*degrees), "ks_weight": _ecdf_gap(*weights)}
    assert math.isclose(ks["ks_degree"][0], nb.ks_degree(G, B))
    assert math.isclose(ks["ks_weight"][0], nb.ks_weight(G, B))
    return parts, values, {"ks_degree": degrees, "ks_weight": weights}, ks


def _alt() -> str:
    parts, values, _, ks = _measures()
    bars = "; ".join(
        f"{name} {values[name]:.2f}, {num:,.0f} of {den:,.0f} {what}"
        for name, (num, den, what) in parts.items()
    )
    return (
        "Three panels about one backbone of Les Misérables. Top, four "
        f"horizontal bars, each the kept part of a whole: {bars}. Bottom left "
        "and right, two step curves each, the cumulative distributions of node "
        "degree and of edge weight in the original graph (dashed) and the "
        "backbone (solid), with a vertical bracket at their largest gap: "
        f"ks_degree {ks['ks_degree'][0]:.2f} and ks_weight "
        f"{ks['ks_weight'][0]:.2f}."
    )


def _caption() -> str:
    _, B = _example()
    *_, ks = _measures()
    _, x_deg, f_g, f_b = ks["ks_degree"]
    isolated = sum(1 for n in B if B.degree(n) == 0)
    return (
        f"The measures on one backbone: :func:`~nx_backbone.{METHOD}` on Les "
        f"Misérables at p < {ALPHA}, filtered with the default "
        "``include_all_nodes=True``. The fractions divide what the backbone "
        "keeps by what the original has (top); "
        ":func:`~nx_backbone.reachability` looks at the backbone alone, the "
        "share of ordered node pairs joined by a path. "
        ":func:`~nx_backbone.ks_degree` and :func:`~nx_backbone.ks_weight` "
        "are the largest vertical gap between the original's and the "
        "backbone's cumulative distributions (bottom). "
        f"The {isolated} nodes the backbone leaves without an edge count with "
        f"degree 0 in ``ks_degree``: {f_b:.0%} of the backbone's nodes have "
        f"degree {x_deg:.0f} or less, against {f_g:.0%} of the original's."
    )


@figure("measures", alt=_alt, caption=_caption)
def draw_measures(mode):
    parts, values, samples, ks = _measures()
    p = style.palette(mode)
    fig = plt.figure(figsize=(6.8, 5.6), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.15])
    ax = top.subplots()
    axes = [ax, *bottom.subplots(1, 2, sharey=True)]

    # Top: each fraction as the kept part of a whole bar.
    names = list(parts)
    for y, name in enumerate(names):
        num, den, what = parts[name]
        ax.barh(y, 1.0, height=0.5, color=p.removed, linewidth=0)
        ax.barh(y, num / den, height=0.5, color=p.kept, linewidth=0)
        ax.annotate(
            f"{values[name]:.2f}",
            (1.0, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=p.ink,
        )
        ax.annotate(
            f"{num:,.0f} of {den:,.0f} {what}",
            (0, y - 0.27),
            va="bottom",
            fontsize=8,
            color=p.muted,
            annotation_clip=False,
        )
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(len(names) - 0.4, -0.85)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.tick_params(axis="y", length=0, labelcolor=p.ink, labelsize=8)
    ax.spines["left"].set_visible(False)
    ax.set_title("Fractions kept")

    # Bottom: the two cumulative distributions and the KS gap.
    for ax, name, xlabel in (
        (axes[1], "ks_degree", "Node degree"),
        (axes[2], "ks_weight", "Edge weight"),
    ):
        original, backbone = samples[name]
        for data, color, ls in ((original, p.muted, "--"), (backbone, p.kept, "-")):
            xs = np.sort(data)
            ax.step(
                np.r_[0, xs],
                np.r_[0, np.arange(1, len(xs) + 1) / len(xs)],
                where="post",
                color=color,
                lw=1.5,
                ls=ls,
            )
        d, x, f_orig, f_bb = ks[name]
        ax.plot([x, x], [f_orig, f_bb], color=p.ink, lw=1.0, marker="_", ms=8)
        ax.annotate(
            f"{name} = {d:.2f}",
            (x, (f_orig + f_bb) / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=p.ink,
            bbox={"boxstyle": "round,pad=0.15", "fc": p.surface, "ec": "none"},
        )
        ax.set_xscale("symlog", linthresh=1)
        top = max(max(original), max(backbone))
        ticks = [t for t in (0, 1, 2, 5, 10, 20, 50) if t <= top]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(xlabel + " (log scale)")
        ax.set_title(name)
        ax.grid(True)
    axes[1].set_ylabel("Share at or below")
    handles = [
        Line2D([], [], color=p.muted, lw=1.5, ls="--", label="Original"),
        Line2D([], [], color=p.kept, lw=1.5, label="Backbone"),
    ]
    axes[2].legend(handles=handles, loc="upper left")
    return fig
