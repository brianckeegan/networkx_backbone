"""Figures of ``docs/tutorials/statistical_backbone.rst``."""

from __future__ import annotations

import functools
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import nx_backbone as nb
from _figures import style
from _figures.pages import figure

ALPHA = 0.05
BIN = 0.05  # histogram bin width: the first bin is exactly p < ALPHA


@functools.cache
def _pvalues():
    """Les Misérables and its disparity-filter p-values, one per edge."""
    G = nx.les_miserables_graph()
    H = nb.disparity_filter(G)
    p = np.array([pv for *_, pv in H.edges(data="disparity_pvalue")])
    kept = nb.threshold_filter(H, "disparity_pvalue", ALPHA).number_of_edges()
    assert kept == int((p < ALPHA).sum())
    return G, p


@functools.cache
def _histogram():
    _, p = _pvalues()
    counts, edges = np.histogram(p, bins=np.arange(0.0, 1.0 + BIN / 2, BIN))
    return counts, edges


def _pvalue_alt() -> str:
    _, p = _pvalues()
    counts, edges = _histogram()
    kept = int((p < ALPHA).sum())
    peak = int(np.argmax(counts))
    return (
        f"Histogram of the disparity-filter p-values of the {p.size} edges of "
        f"Les Misérables, in bins of width {BIN}. The first bar, p below "
        f"{ALPHA}, holds the {kept} kept edges and is highlighted; a vertical "
        f"line marks alpha = {ALPHA}. The other {p.size - kept} edges spread "
        f"from {ALPHA} up to {p.max():.2f}, most often between "
        f"{edges[peak]:.2f} and {edges[peak + 1]:.2f} ({counts[peak]} edges); "
        f"the median p-value is {np.median(p):.2f}."
    )


def _pvalue_caption() -> str:
    _, p = _pvalues()
    kept = int((p < ALPHA).sum())
    return (
        "Most edges are nowhere near significant. The "
        ":func:`~nx_backbone.disparity_filter` p-values of all "
        f"{p.size} edges (median {np.median(p):.2f}); "
        f":func:`~nx_backbone.threshold_filter` at ``{ALPHA}`` keeps only the "
        f"{kept} edges left of the line ({kept / p.size:.0%}). Under this "
        "null a node spreads its strength at random over its edges, so an "
        "edge stands out only when it carries a much larger share of an "
        "endpoint's strength than chance would give it."
    )


@figure("pvalues", alt=_pvalue_alt, caption=_pvalue_caption)
def draw_pvalues(mode):
    _, p = _pvalues()
    counts, edges = _histogram()
    pal = style.palette(mode)
    kept = int((p < ALPHA).sum())
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    colors = [pal.kept if lo < ALPHA else pal.removed for lo in edges[:-1]]
    ax.bar(
        edges[:-1],
        counts,
        width=BIN,
        align="edge",
        color=colors,
        edgecolor=pal.surface,
        linewidth=1.0,
    )
    ax.axvline(ALPHA, color=pal.ink, linewidth=1.0)
    top = counts.max()
    ax.annotate(
        f"α = {ALPHA}: {kept} edges kept",
        xy=(ALPHA, counts[0]),
        xytext=(ALPHA + 0.012, top * 1.02),
        color=pal.ink,
        fontsize=8.5,
        va="bottom",
    )
    ax.text(
        0.62,
        top * 0.8,
        f"{p.size - kept} edges removed",
        color=pal.muted,
        fontsize=8.5,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, top * 1.18)
    ax.set_xlabel("disparity p-value")
    ax.set_ylabel("edges")
    ax.grid(axis="y")
    return fig


# -- the eight methods, compared ------------------------------------------------


@functools.cache
def _compared():
    """The page's "Comparing all statistical methods" block, verbatim."""
    G = nx.les_miserables_graph()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # fit and convergence warnings
        backbones = {
            "disparity": nb.threshold_filter(
                nb.disparity_filter(G), "disparity_pvalue", ALPHA
            ),
            "noise_corrected": nb.threshold_filter(
                nb.noise_corrected_filter(G), "nc_pvalue", ALPHA
            ),
            "marginal_likelihood": nb.threshold_filter(
                nb.marginal_likelihood_filter(G), "ml_pvalue", ALPHA
            ),
            "ecm": nb.threshold_filter(nb.ecm_filter(G), "ecm_pvalue", ALPHA),
            "lans": nb.threshold_filter(nb.lans_filter(G), "lans_pvalue", ALPHA),
            "polya": nb.threshold_filter(nb.polya_filter(G), "polya_pvalue", ALPHA),
            "gloss": nb.threshold_filter(nb.gloss_filter(G), "gloss_pvalue", ALPHA),
            "mla": nb.boolean_filter(nb.multiple_linkage_analysis(G), "mla_keep"),
        }
    results = nb.compare_backbones(G, backbones)
    rows = sorted(
        (
            (name, backbones[name].number_of_edges(), m["edge_fraction"])
            for name, m in results.items()
        ),
        key=lambda row: -row[1],
    )
    return G, rows


def _compared_alt() -> str:
    G, rows = _compared()
    listed = "; ".join(f"{name} {n} ({share:.0%})" for name, n, share in rows)
    return (
        "Horizontal bar chart of the number of edges each of eight statistical "
        f"backbones keeps of the {G.number_of_edges()} edges of Les Misérables, "
        f"largest first: {listed}."
    )


def _compared_caption() -> str:
    G, rows = _compared()
    (most, n_most, _), (least, n_least, _) = rows[0], rows[-1]
    return (
        f"Same graph, same α = {ALPHA}, very different backbones: the edges "
        "each method keeps, from the "
        ":func:`~nx_backbone.compare_backbones` call above (``mla`` uses its "
        "keep flag, not a p-value). They range from "
        f"{n_most} ({most}) to {n_least} ({least}) of "
        f"{G.number_of_edges()}, because each null model asks a different "
        "question of the same weights."
    )


@figure("methods", alt=_compared_alt, caption=_compared_caption)
def draw_methods(mode):
    G, rows = _compared()
    pal = style.palette(mode)
    total = G.number_of_edges()
    names = [name for name, *_ in rows]
    counts = [n for _, n, _ in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.4, 0.3 * len(rows) + 0.7))
    ax.barh(y, counts, height=0.62, color=pal.kept)
    for yi, (_, n, share) in zip(y, rows, strict=True):
        ax.text(
            n + total * 0.012,
            yi,
            f"{n} ({share:.0%})",
            va="center",
            color=pal.ink,
            fontsize=8.5,
        )
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", length=0, labelcolor=pal.ink)
    ax.invert_yaxis()
    ax.set_xlim(0, total)
    ax.set_xlabel(f"edges kept (of {total})")
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    return fig
