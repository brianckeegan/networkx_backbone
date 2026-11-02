"""Figures of ``docs/tutorials/validating_backbones.rst``."""

from __future__ import annotations

import functools
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import nx_backbone as nb
from _figures import style
from _figures.pages import figure

ALPHA = 0.05
CORRECTIONS = {"none": "uncorrected", "holm": "Holm", "bh": "BH"}
ALPHAS = np.geomspace(1e-4, 0.2, 61)
VOTERS = ["mlf (BH)", "lans (BH)", "metric", "salience"]


@functools.cache
def _graph():
    return nx.les_miserables_graph()


@functools.cache
def _scored():
    """The page's two scored graphs: disparity and marginal likelihood."""
    G = _graph()
    return {
        "disparity_filter": (nb.disparity_filter(G), "disparity_pvalue"),
        "marginal_likelihood_filter": (
            nb.marginal_likelihood_filter(G),
            "ml_pvalue",
        ),
    }


@functools.cache
def _sweep():
    """Edges kept at each alpha of ``ALPHAS``, per method and correction."""
    sizes = {}
    for name, (H, score) in _scored().items():
        for mtc in CORRECTIONS:
            sizes[name, mtc] = [
                nb.threshold_filter(H, score, a, mtc=mtc).number_of_edges()
                for a in [*ALPHAS, ALPHA]
            ]
    return sizes


def _at_alpha(name: str, mtc: str) -> int:
    return _sweep()[name, mtc][-1]


def _corrected_max() -> int:
    sizes = _sweep()
    return max(max(sizes["disparity_filter", m]) for m in ("holm", "bh"))


def _sweep_alt() -> str:
    d = {m: _at_alpha("disparity_filter", m) for m in CORRECTIONS}
    ml = {m: _at_alpha("marginal_likelihood_filter", m) for m in CORRECTIONS}
    return (
        "Two line charts side by side with a shared vertical axis, edges "
        f"kept out of {_graph().number_of_edges()}, against alpha on a "
        f"logarithmic axis from {ALPHAS[0]:g} to {ALPHAS[-1]:g}; each has a "
        "solid line for no correction, a dashed line for Holm and a dotted line "
        "for BH, and a vertical mark at alpha 0.05. Left, the disparity "
        f"filter: at 0.05 it keeps {d['none']} edges uncorrected, {d['holm']} "
        f"with Holm and {d['bh']} with BH, and with either correction it keeps "
        f"at most {_corrected_max()} edges anywhere on the axis. Right, the "
        "marginal likelihood filter: at "
        f"0.05 it keeps {ml['none']} uncorrected, {ml['holm']} with Holm and "
        f"{ml['bh']} with BH."
    )


def _sweep_caption() -> str:
    d = {m: _at_alpha("disparity_filter", m) for m in CORRECTIONS}
    ml = {m: _at_alpha("marginal_likelihood_filter", m) for m in CORRECTIONS}
    return (
        "Edges kept against :math:`\\alpha`, for each correction of "
        ":func:`~nx_backbone.threshold_filter`. At :math:`\\alpha = 0.05` "
        f"(the vertical mark), the disparity filter keeps {d['none']} edges "
        f"uncorrected and {d['holm']} with Holm or {d['bh']} with BH; the "
        f"marginal likelihood filter keeps {ml['none']}, {ml['holm']} and "
        f"{ml['bh']}. How far the lines move with :math:`\\alpha` is the "
        "sensitivity to report."
    )


@figure("edges-vs-alpha", alt=_sweep_alt, caption=_sweep_caption)
def draw_edges_vs_alpha(mode):
    p = style.palette(mode)
    sizes = _sweep()
    styles = {"none": "-", "holm": "--", "bh": ":"}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.87, bottom=0.17, top=0.86, wspace=0.42)
    for ax, name in zip(axes, _scored(), strict=True):
        ax.axvline(ALPHA, color=p.hairline, lw=0.75)
        ends = {}
        for mtc, label in CORRECTIONS.items():
            ys = sizes[name, mtc][:-1]
            ax.step(ALPHAS, ys, where="post", color=p.kept, ls=styles[mtc], lw=1.5)
            ends.setdefault(ys[-1], []).append(label)
        # Direct labels at the right end; lines ending together share one.
        for y, labels in ends.items():
            ax.annotate(
                ", ".join(labels),
                (ALPHAS[-1], y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                color=p.ink,
                fontsize=8,
            )
        ax.set_xscale("log")
        ax.set_xlim(ALPHAS[0], ALPHAS[-1])
        ax.set_title(name)
        ax.set_xlabel("alpha")
        ax.grid(axis="y")
        ax.text(
            ALPHA,
            1.0,
            f" {ALPHA}",
            transform=ax.get_xaxis_transform(),
            color=p.muted,
            fontsize=8,
            va="top",
        )
    axes[0].set_ylabel("Edges kept")
    axes[0].set_ylim(0, None)
    return fig


@functools.cache
def _votes():
    """How many of the four backbones of the page's §3 keep each edge."""
    G = _graph()
    ml = nb.marginal_likelihood_filter(G)
    la = nb.lans_filter(G)
    backbones = {
        "mlf (BH)": nb.threshold_filter(ml, "ml_pvalue", ALPHA, mtc="bh"),
        "lans (BH)": nb.threshold_filter(la, "lans_pvalue", ALPHA, mtc="bh"),
        "metric": nb.boolean_filter(nb.metric_backbone(G), "metric_keep"),
        "salience": nb.threshold_filter(
            nb.high_salience_skeleton(G), "salience", 0.5, mode="above"
        ),
    }
    votes = Counter()
    for name in VOTERS:
        votes.update(frozenset(e) for e in backbones[name].edges())
    counts = Counter(votes.values())
    counts[0] = G.number_of_edges() - sum(counts.values())
    # The page's doctest output.
    assert sorted((k, v) for k, v in counts.items() if k) == [
        (1, 19),
        (2, 41),
        (3, 70),
        (4, 34),
    ]
    return [counts[k] for k in range(len(VOTERS) + 1)]


def _votes_alt() -> str:
    c = _votes()
    parts = ", ".join(f"{k}: {n} edges" for k, n in enumerate(c))
    return (
        f"A bar chart of the {sum(c)} edges of Les Misérables by how many of "
        f"the {len(VOTERS)} backbones keep them, from 0 to {len(VOTERS)}, "
        f"each bar labeled with its count. {parts}."
    )


def _votes_caption() -> str:
    c = _votes()
    k = len(VOTERS)
    return (
        f"Vote counts. Of the {sum(c)} edges, {c[k]} are kept by all {k} "
        f"backbones (marginal likelihood and LANS with BH, metric, salience), "
        f"{c[1]} by only one, and {c[0]} by none."
    )


@figure("vote-counts", alt=_votes_alt, caption=_votes_caption)
def draw_vote_counts(mode):
    p = style.palette(mode)
    c = _votes()
    ks = range(len(c))
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    bars = ax.bar(ks, c, color=p.kept, width=0.62)
    ax.bar_label(bars, padding=2, color=p.ink, fontsize=8.5)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel(f"Backbones keeping the edge (of {len(VOTERS)})")
    ax.set_ylabel("Edges")
    ax.set_ylim(0, max(c) * 1.15)
    ax.grid(axis="y")
    ax.set_title("How many backbones keep each edge")
    return fig
