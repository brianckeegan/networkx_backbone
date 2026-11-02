"""Figures of ``docs/tutorials/proximity_scoring.rst``."""

from __future__ import annotations

import functools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import spearmanr

import nx_backbone as nb
from _figures import style
from _figures.pages import figure

FRACTION = 0.3  # the page's fraction_filter(..., 0.3, ascending=False)

# (title, scorer, attribute): normalized, neighbor-weighted, degree only.
SCORES = (
    ("Jaccard", nb.jaccard_backbone, "jaccard"),
    ("Adamic-Adar", nb.adamic_adar_index, "adamic_adar"),
    ("Preferential attachment", nb.preferential_attachment_score, "pa"),
)


@functools.cache
def _scored():
    """Per score: degree product, score and top-30% flag of every edge."""
    G = nx.les_miserables_graph()
    panels = []
    for title, scorer, attr in SCORES:
        H = scorer(G)
        top = nb.fraction_filter(H, attr, FRACTION, ascending=False)
        edges = list(H.edges(data=attr))
        degree = np.array([G.degree(u) * G.degree(v) for u, v, _ in edges])
        score = np.array([s for *_, s in edges], dtype=float)
        kept = np.array([top.has_edge(u, v) for u, v, _ in edges])
        rho = float(spearmanr(degree, score).statistic)
        panels.append((title, attr, degree, score, kept, rho))
    return G, panels


def _alt() -> str:
    G, panels = _scored()
    parts = "; ".join(
        f"{title}: Spearman correlation with the degree product {rho:.2f}, "
        f"{int(kept.sum())} edges kept"
        for title, _, _, _, kept, rho in panels
    )
    return (
        f"Three scatter plots, one per proximity score, of the "
        f"{G.number_of_edges()} edges of Les Misérables: the score against the "
        "product of the endpoint degrees, on a log scale. Filled markers are "
        f"the top {FRACTION:.0%} of edges by that score, which "
        f"fraction_filter keeps; hollow markers are removed. {parts}."
    )


def _caption() -> str:
    _, panels = _scored()
    rho = {title: r for title, *_, r in panels}
    return (
        "What each score rewards, edge by edge. "
        f"Jaccard divides out the degrees (Spearman ρ = {rho['Jaccard']:.2f} "
        "with the degree product), Adamic-Adar counts shared neighbors and "
        f"so grows with degree (ρ = {rho['Adamic-Adar']:.2f}), and "
        "preferential attachment *is* the degree product. Filled markers: the "
        f"top {FRACTION:.0%} that :func:`~nx_backbone.fraction_filter` keeps."
    )


@figure("score-vs-degree", alt=_alt, caption=_caption)
def draw_score_vs_degree(mode):
    _, panels = _scored()
    pal = style.palette(mode)
    fig, axes = plt.subplots(1, len(panels), figsize=(6.6, 2.6), sharex=True)
    fig.subplots_adjust(wspace=0.38)
    for ax, (title, attr, degree, score, kept, rho) in zip(axes, panels, strict=True):
        ax.scatter(
            degree[~kept],
            score[~kept],
            s=14,
            facecolors="none",
            edgecolors=pal.removed,
            linewidths=0.8,
            label="removed",
        )
        ax.scatter(
            degree[kept],
            score[kept],
            s=14,
            c=pal.kept,
            edgecolors=pal.surface,
            linewidths=0.5,
            label=f"kept: top {FRACTION:.0%} by the score",
        )
        ax.set_xscale("log")
        if attr == "pa":
            ax.set_yscale("log")
        ax.set_title(f"{title}\nρ = {rho:.2f}", fontsize=9)
        ax.set_ylabel(attr)
        ax.grid(True)
    axes[1].set_xlabel("degree(u) × degree(v)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        labels[::-1],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
    )
    return fig
