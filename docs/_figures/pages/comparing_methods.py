"""Figures of ``docs/tutorials/comparing_methods.rst``."""

from __future__ import annotations

import functools
import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import nx_backbone as nb
from _figures import style
from _figures.pages import figure


@functools.cache
def _backbones():
    """The page's "Creating multiple backbones" block, verbatim."""
    G = nx.les_miserables_graph()
    backbones = {
        "disparity": nb.threshold_filter(
            nb.disparity_filter(G), "disparity_pvalue", 0.05
        ),
        "noise_corrected": nb.threshold_filter(
            nb.noise_corrected_filter(G), "nc_pvalue", 0.05
        ),
        "metric": nb.boolean_filter(nb.metric_backbone(G), "metric_keep"),
        "mst": nb.boolean_filter(nb.maximum_spanning_tree_backbone(G), "mst_keep"),
        "strongest_3": nb.boolean_filter(
            nb.strongest_n_ties(G, n=3), "strongest_n_ties_keep"
        ),
    }
    return G, backbones


def _edge_set(B):
    return {frozenset(e) for e in B.edges()}


@functools.cache
def _agreement():
    """Jaccard index of the edge sets of every pair of backbones."""
    _, backbones = _backbones()
    names = list(backbones)
    sets = [_edge_set(backbones[n]) for n in names]
    J = np.ones((len(names), len(names)))
    for i, j in itertools.combinations(range(len(names)), 2):
        union = sets[i] | sets[j]
        J[i, j] = J[j, i] = len(sets[i] & sets[j]) / len(union) if union else 1.0
    return names, [len(s) for s in sets], J


def _pairs():
    names, _, J = _agreement()
    pairs = [
        (J[i, j], names[i], names[j])
        for i, j in itertools.combinations(range(len(names)), 2)
    ]
    return sorted(pairs, reverse=True)


def _agreement_alt() -> str:
    names, sizes, _ = _agreement()
    pairs = _pairs()
    listed = "; ".join(f"{a} and {b} {v:.2f}" for v, a, b in pairs)
    return (
        "Lower-triangular matrix of the Jaccard index between the edge sets "
        f"of {len(names)} backbones of Les Misérables ("
        + ", ".join(f"{n} with {k} edges" for n, k in zip(names, sizes, strict=True))
        + f"), each cell shaded and labeled with its value. By pair: {listed}."
    )


def _agreement_caption() -> str:
    pairs = _pairs()
    (hi, a, b), (lo, c, d) = pairs[0], pairs[-1]
    return (
        "How far the methods agree: the Jaccard index of two backbones' edge "
        "sets, the edges they share over the edges either keeps (1 means the "
        f"same backbone, 0 no shared edge). ``{a}`` and ``{b}`` agree most "
        f"({hi:.2f}); ``{c}`` and ``{d}`` least ({lo:.2f}). A backbone much "
        "smaller than another caps their index, however nested they are."
    )


def _luminance(rgb) -> float:
    def channel(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _text_color(cell, pal) -> str:
    """Whichever of the ink and the surface contrasts more with *cell*."""
    from matplotlib.colors import to_rgb

    bg = _luminance(cell)

    def contrast(color):
        fg = _luminance(to_rgb(color))
        return (max(fg, bg) + 0.05) / (min(fg, bg) + 0.05)

    return max((pal.ink, pal.surface), key=contrast)


@figure("agreement", alt=_agreement_alt, caption=_agreement_caption)
def draw_agreement(mode):
    names, sizes, J = _agreement()
    pal = style.palette(mode)
    cmap = style.sequential_cmap(mode)
    n = len(names)
    mask = np.triu(np.ones_like(J, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.imshow(np.ma.masked_array(J, mask), cmap=cmap, vmin=0, vmax=1)
    for i in range(n):
        for j in range(i + 1):
            value = J[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color=_text_color(cmap(value), pal),
            )
    labels = [f"{name} ({k})" for name, k in zip(names, sizes, strict=True)]
    ax.set_xticks(range(n), labels, rotation=35, ha="right")
    ax.set_yticks(range(n), labels)
    ax.tick_params(length=0, labelcolor=pal.ink)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


# -- the consensus backbone -------------------------------------------------------


NAMES = {"b1": "disparity", "b2": "metric", "b3": "maximum spanning tree"}


@functools.cache
def _consensus():
    """The page's "Consensus backbone" block, verbatim."""
    G = nx.les_miserables_graph()
    b1 = nb.threshold_filter(nb.disparity_filter(G), "disparity_pvalue", 0.05)
    b2 = nb.boolean_filter(nb.metric_backbone(G), "metric_keep")
    b3 = nb.boolean_filter(nb.maximum_spanning_tree_backbone(G), "mst_keep")
    consensus = nb.consensus_backbone(b1, b2, b3)
    inputs = {NAMES["b1"]: b1, NAMES["b2"]: b2, NAMES["b3"]: b3}
    return G, consensus, inputs


def _consensus_alt() -> str:
    G, consensus, inputs = _consensus()
    c = style.backbone_counts(G, consensus)
    sizes = ", ".join(f"{n} {b.number_of_edges()}" for n, b in inputs.items())
    return (
        f"The Les Misérables network, {c.nodes_total} characters, with the "
        f"{c.edges_kept} edges that three backbones (edges kept: {sizes}) all "
        f"keep drawn solid, over the other {c.edges_total - c.edges_kept} "
        f"edges in faint gray; {c.nodes_kept} characters touch a consensus "
        "edge."
    )


def _consensus_caption() -> str:
    G, consensus, inputs = _consensus()
    c = style.backbone_counts(G, consensus)
    smallest = min(inputs, key=lambda n: inputs[n].number_of_edges())
    return (
        "The edges the disparity, metric and maximum spanning tree backbones "
        "all keep, from :func:`~nx_backbone.consensus_backbone`. A consensus "
        "is never larger than its smallest input, here the "
        f"{smallest} backbone with {inputs[smallest].number_of_edges()} "
        f"edges. {c.caption}"
    )


@figure("consensus", alt=_consensus_alt, caption=_consensus_caption, fmt="png")
def draw_consensus(mode):
    G, consensus, _ = _consensus()
    fig, ax = plt.subplots(figsize=(6, 4))
    style.draw_backbone(ax, G, consensus, style.layout(G, "les_miserables"), mode=mode)
    return fig
