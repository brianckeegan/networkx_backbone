"""
Comparing, validating, and reporting backbones
==============================================

You have run a backbone method, perhaps several, and they disagree. Which
backbone should you trust, and how do you report it so that a reader can
judge it too?

The :doc:`Start here </auto_examples/start_here/plot_hairball_to_backbone>`
tutorial already takes one backbone through a quick comparison, a consensus
and a methods paragraph (its sections 4 to 6). This page goes deeper and
asks what each of those checks can and cannot tell you:

1. :ref:`what compare_backbones measures, and why agreement between methods
   is not correctness <cv-compare>`;
2. :ref:`validation against a planted ground truth: precision and recall
   <cv-truth>`;
3. :ref:`robustness to the significance level and the correction
   <cv-alpha>`;
4. :ref:`robustness to the order in which the network was built
   <cv-order>`;
5. :ref:`consensus and vote counts <cv-votes>`;
6. :ref:`the Les Misérables benchmark, explained <cv-benchmark>`;
7. :ref:`reporting: the methods paragraph, the citations and a checklist
   <cv-report>`.

It assumes you know the score-then-filter workflow and what a p-value and a
multiple-testing correction are; "Start here" introduces both. Every number
and figure below is computed when the documentation is built.
"""

# sphinx_gallery_thumbnail_path = "_static/example_figures/validation/thumbnail.png"

# %%
# Setup
# -----
#
# Besides ``networkx`` and ``nx_backbone``, the analysis uses NumPy and
# SciPy, and the figures Matplotlib (``pip install "nx-backbone[full,viz]"``
# installs them all). The plotting code is left out of the page; it is in
# the script and notebook you can download at the end.

import itertools
import random
from collections import Counter

import networkx as nx
import numpy as np

import nx_backbone as nb

# sphinx_gallery_start_ignore
# isort: split
import warnings

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from _figures import style
from _figures.tutorial import TutorialFigures
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from benchmarks.benchmarks import les_miserables as lm

figures = TutorialFigures("validation")
# sphinx_gallery_end_ignore


# %%
# A network with a known answer
# -----------------------------
#
# To ask whether a backbone is *right*, we need a network in which we know
# which edges are real. Real data never tells us that, so we simulate one
# and **plant** the structure ourselves. The planted edges are the
# **ground truth**: the answer a perfect method would return.
#
# The simulation mimics interaction counts, such as messages, joint papers
# or co-appearances:
#
# - 120 nodes in four groups. Each node has an **activity level** drawn from
#   a heavy-tailed (lognormal) distribution, so a few nodes are very active
#   and most are not.
# - **Background noise:** 3,000 single interactions, each between a pair of
#   nodes chosen with probability proportional to the product of the two
#   nodes' activity levels. Active nodes interact often with everyone, by
#   chance alone.
# - **Planted ties:** each pair of nodes in the same group gets, with
#   probability 0.15, a tie with extra interactions (one plus a Poisson
#   number with mean 4) on top of any background noise.
#
# An edge's **weight** is its total count. Every planted tie is an edge;
# every other edge is noise, however heavy.


def planted_network(n=120, groups=4, noise=3000, p_tie=0.15, extra=4, seed=7):
    """Weighted network with planted ties (edge attribute "planted")."""
    rng = np.random.default_rng(seed)
    activity = rng.lognormal(0.0, 1.0, n)
    group = np.arange(n) % groups
    pairs = np.array(list(itertools.combinations(range(n), 2)))
    chance = activity[pairs[:, 0]] * activity[pairs[:, 1]]
    background = rng.multinomial(noise, chance / chance.sum())
    same_group = group[pairs[:, 0]] == group[pairs[:, 1]]
    tie = same_group & (rng.random(len(pairs)) < p_tie)
    boost = np.where(tie, 1 + rng.poisson(extra, len(pairs)), 0)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, group=int(group[i]), activity=float(activity[i]))
    for (u, v), w, planted in zip(pairs.tolist(), background + boost, tie):
        if w > 0:
            G.add_edge(u, v, weight=int(w), planted=bool(planted))
    return G


def edge_set(G):
    """The edges of G as a set of node pairs, ignoring direction."""
    return {frozenset(e) for e in G.edges()}


P = planted_network()
TRUTH = {frozenset((u, v)) for u, v, t in P.edges(data="planted") if t}
noise_weights = [
    w for u, v, w in P.edges(data="weight") if frozenset((u, v)) not in TRUTH
]
print(
    f"{P.number_of_nodes()} nodes, {P.number_of_edges()} edges, "
    f"{len(TRUTH)} of them planted ({len(TRUTH) / P.number_of_edges():.0%})"
)
typical_tie = np.median([P.edges[tuple(e)]["weight"] for e in TRUTH])
heavy_noise = sum(w > typical_tie for w in noise_weights)
print(f"median weight of a planted tie: {typical_tie:g}")
print(
    f"noise edges heavier than that: {heavy_noise}; "
    f"heaviest noise edge: {max(noise_weights)}"
)

# %%
# The noise is not weak: dozens of noise edges are heavier than the typical
# planted tie, and the heaviest noise edge outweighs every planted one. A
# method that keeps the heaviest edges will keep noise.
#
# One caution before using this network as a referee. The background noise
# is close to the null model of the **marginal likelihood filter** (Dianati,
# 2016), in which each unit of weight picks its two endpoints in proportion
# to their strengths. A planted experiment favors the methods whose null
# model matches the way the data were simulated, so simulate something that
# resembles *your* data before trusting a ranking from it.
#
# Six backbones
# ~~~~~~~~~~~~~
#
# We extract six backbones built on different ideas:
#
# - the **marginal likelihood filter** and **LANS** (locally adaptive
#   network sparsification; Foti, Hughes & Rockmore, 2011), which compares
#   each edge with the empirical distribution of its endpoints' weights,
#   both at :math:`\alpha = 0.05` with the Benjamini-Hochberg (BH)
#   correction;
# - a **global weight cut** keeping the heaviest edges, sized to keep as
#   many edges as the marginal likelihood backbone (ties at the threshold
#   can add a few);
# - the **strongest three ties** of every node;
# - the **metric backbone** (Simas, Correia & Rocha, 2021), the edges that
#   are themselves the shortest path between their endpoints when the
#   distance of an edge is :math:`1/w`;
# - the **maximum spanning tree**.

ml_P = nb.marginal_likelihood_filter(P)
ml_backbone = nb.threshold_filter(ml_P, "ml_pvalue", 0.05, mtc="bh")
k = ml_backbone.number_of_edges()
cut = sorted((w for _, _, w in P.edges(data="weight")), reverse=True)[k - 1]

backbones = {
    "Marginal likelihood": ml_backbone,
    "LANS": nb.threshold_filter(nb.lans_filter(P), "lans_pvalue", 0.05, mtc="bh"),
    "Global weight cut": nb.boolean_filter(
        nb.global_threshold_filter(P, cut), "global_threshold_keep"
    ),
    "Strongest 3 ties": nb.boolean_filter(
        nb.strongest_n_ties(P, n=3), "strongest_n_ties_keep"
    ),
    "Metric backbone": nb.boolean_filter(nb.metric_backbone(P), "metric_keep"),
    "Spanning tree": nb.boolean_filter(
        nb.maximum_spanning_tree_backbone(P), "mst_keep"
    ),
}
for name, H in backbones.items():
    print(f"{name:<20} {H.number_of_edges():>4} edges")

# %%
# .. _cv-compare:
#
# 1. What ``compare_backbones`` measures
# --------------------------------------
#
# :func:`~nx_backbone.compare_backbones` takes the original graph, a
# dictionary of backbones and a list of **measures**, and applies every
# measure to every backbone. A measure is any function of ``(original,
# backbone)`` that returns a number. The defaults are the shares of nodes,
# edges and total weight that a backbone keeps; here we add the share of
# node pairs still joined by a path, wrapped to take two arguments.


def reachability(original, backbone):
    return nb.reachability(backbone)


measures = [nb.node_fraction, nb.edge_fraction, nb.weight_fraction, reachability]
table = nb.compare_backbones(P, backbones, measures=measures)
print(f"{'':<20}{'nodes':>8}{'edges':>8}{'weight':>8}{'reach':>8}")
for name, row in table.items():
    print(f"{name:<20}" + "".join(f"{v:>8.2f}" for v in row.values()))

# %%
# Notice what this table is: each backbone measured *against the original
# network*, one row at a time. It describes each backbone (how sparse, how
# heavy, how connected) but it never compares two backbones with each
# other, and it says nothing about which edges are real.
#
# To see whether two methods keep the *same* edges, measure their
# **agreement**. The usual measure is the **Jaccard similarity** of their
# edge sets: the number of edges both keep divided by the number either
# keeps, from 0 (nothing shared) to 1 (identical).


def jaccard(a, b):
    a, b = edge_set(a), edge_set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


names = list(backbones)
agreement = np.array(
    [[jaccard(backbones[a], backbones[b]) for b in names] for a in names]
)
pairs = sorted(
    itertools.combinations(range(len(names)), 2), key=lambda ij: -agreement[ij]
)
for i, j in pairs[:3]:
    print(f"{names[i]} and {names[j]}: Jaccard {agreement[i, j]:.2f}")

# sphinx_gallery_start_ignore
SHORT = {
    "Marginal likelihood": "Marginal\nlikelihood",
    "LANS": "LANS",
    "Global weight cut": "Global\nweight cut",
    "Strongest 3 ties": "Strongest\n3 ties",
    "Metric backbone": "Metric\nbackbone",
    "Spanning tree": "Spanning\ntree",
}


def _luminance(color):
    from matplotlib.colors import to_rgb

    c = np.array(to_rgb(color))
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _readable_on(background, candidates):
    """The candidate text color with the highest contrast on *background*."""
    lb = _luminance(background)

    def contrast(color):
        hi, lo = sorted([lb, _luminance(color)], reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    return max(candidates, key=contrast)


def draw_agreement(mode):
    pal = style.palette(mode)
    cmap = style.sequential_cmap(mode)
    n = len(names)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
    shown = np.ma.masked_array(agreement, mask=np.eye(n, dtype=bool))
    cmap = cmap.with_extremes(bad=pal.hairline)
    image = ax.imshow(shown, cmap=cmap, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            value = agreement[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=_readable_on(cmap(value), [pal.ink, pal.surface]),
            )
    labels = [SHORT[name] for name in names]
    ax.set_xticks(range(n), labels, fontsize=8.5)
    ax.set_yticks(range(n), labels, fontsize=8.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=pal.surface, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    bar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.02)
    bar.set_label("Jaccard similarity of the edge sets")
    bar.outline.set_visible(False)
    ax.set_title("Agreement between the six backbones")
    return fig


_top = pairs[0]
figures.show(
    draw_agreement,
    "agreement",
    width="80%",
    alt=lambda mode: (
        "A six-by-six grid of Jaccard similarities between the backbones, "
        f"each cell labeled with its value; {'darker' if mode == 'light' else 'brighter'} "
        "cells mean more agreement. The highest value, "
        f"{agreement[_top]:.2f}, is between the {names[_top[0]]} and the "
        f"{names[_top[1]]} backbones."
    ),
    caption=lambda mode: (
        "Pairwise agreement (Jaccard similarity of the edge sets) between the "
        "six backbones of the planted network. "
        f"{'Darker' if mode == 'light' else 'Brighter'} cells agree more; "
        "every value is printed in its cell."
    ),
)
# sphinx_gallery_end_ignore

# %%
# Some methods agree closely. Is that evidence that they are right? Only if
# they could not agree for any other reason, and here they could:
#
# - The most similar pair, **LANS** and the **spanning tree**, share a rule
#   rather than a discovery. Both keep, for every node, one of its heaviest
#   edges: LANS gives each node's strongest edge a p-value of 0, and a
#   maximum spanning tree always contains a heaviest edge of every node.
# - The **global weight cut** and the **strongest three ties** both select
#   edges by weight alone, so whatever heavy noise there is, both keep it.
#
# Agreement measures **shared assumptions** as much as truth. To see which
# backbone is right, we need the truth itself.

# sphinx_gallery_start_ignore
# Both rules hold: LANS keeps every edge of maximum weight at a node, and the
# spanning tree contains at least one such edge at every node.
_lans, _tree = edge_set(backbones["LANS"]), edge_set(backbones["Spanning tree"])
for _v in P:
    _wmax = max(w for _, _, w in P.edges(_v, data="weight"))
    _heaviest = {frozenset((_v, u)) for u in P[_v] if P[_v][u]["weight"] == _wmax}
    assert _heaviest <= _lans and _heaviest & _tree
# sphinx_gallery_end_ignore

# %%
# .. _cv-truth:
#
# 2. Validating against a planted truth
# -------------------------------------
#
# With a known answer, two numbers describe a backbone:
#
# - **precision**, the share of the backbone's edges that are planted (how
#   much of what it keeps is real);
# - **recall**, the share of the planted edges that the backbone keeps (how
#   much of what is real it finds).
#
# One minus the precision is the **false discovery proportion**, the share
# of kept edges that are noise. Both measures take ``(original, backbone)``
# and read the truth from the original graph's ``planted`` attribute, so they
# plug straight into :func:`~nx_backbone.compare_backbones`.


def precision(original, backbone):
    kept = edge_set(backbone)
    true = sum(original.edges[tuple(e)]["planted"] for e in kept)
    return true / len(kept) if kept else float("nan")


def recall(original, backbone):
    truth = {frozenset((u, v)) for u, v, t in original.edges(data="planted") if t}
    return len(truth & edge_set(backbone)) / len(truth)


scores = nb.compare_backbones(P, backbones, measures=[precision, recall])
print(f"{'':<20}{'edges':>7}{'precision':>11}{'recall':>8}")
for name, row in scores.items():
    print(
        f"{name:<20}{backbones[name].number_of_edges():>7}"
        f"{row['precision']:>11.2f}{row['recall']:>8.2f}"
    )

# sphinx_gallery_start_ignore
_p = {name: row["precision"] for name, row in scores.items()}
_r = {name: row["recall"] for name, row in scores.items()}
_idx = {name: i for i, name in enumerate(names)}
_ml_row = [agreement[0, j] for j in range(1, len(names))]
# The most similar pair is LANS and the spanning tree, both of low recall;
# the next is the global cut and the strongest ties, both less precise than
# the marginal likelihood backbone, which agrees with no method as closely.
assert {names[i] for i in pairs[0]} == {"LANS", "Spanning tree"}
assert {names[i] for i in pairs[1]} == {"Global weight cut", "Strongest 3 ties"}
assert _r["Spanning tree"] < 0.5 and _r["LANS"] < 0.5
for _a in ["Global weight cut", "Strongest 3 ties"]:
    assert _p[_a] < _p["Marginal likelihood"] - 0.2
assert max(_ml_row) < agreement[pairs[1]] < agreement[pairs[0]]
# The marginal likelihood backbone: most precise; BH keeps the false
# discovery proportion below alpha; it finds most planted edges.
assert max(_p, key=_p.get) == "Marginal likelihood"
assert 1 - _p["Marginal likelihood"] < 0.05
assert _r["Marginal likelihood"] > 0.7
assert _r["Metric backbone"] == max(_r.values())
assert _p["Metric backbone"] == min(_p.values())
# The setup's claims about heavy noise.
assert heavy_noise > 24
assert max(noise_weights) > max(P.edges[tuple(e)]["weight"] for e in TRUTH)
# sphinx_gallery_end_ignore

# %%
# The ranking by agreement and the ranking by correctness are different:
#
# - The **metric backbone** finds the most planted edges but keeps the most
#   noise: it has the highest recall and the lowest precision.
# - The **spanning tree** and **LANS**, the most similar pair, are fairly
#   precise but find less than half the planted edges: both spend much of
#   their budget on giving every node an edge.
# - The **global cut** and the **strongest ties**, the next most similar
#   pair, are both clearly less precise than the marginal likelihood filter.
# - The **marginal likelihood** backbone is the most precise, with a false
#   discovery proportion below the 5% that BH promises, and still finds most
#   planted edges, yet it agrees with no other method as closely as those
#   two pairs agree. As cautioned above, this simulation suits its null
#   model.
#
# Where does the marginal likelihood backbone miss? Compare the weights of
# the planted ties it finds with those it misses:

found = TRUTH & edge_set(ml_backbone)
missed = TRUTH - found
for label, edges in [("found", found), ("missed", missed)]:
    w = np.array([P.edges[tuple(e)]["weight"] for e in edges])
    print(
        f"{label:>6}: {len(edges):>3} planted ties, median weight {np.median(w):g}, "
        f"{np.mean(w <= 4):.0%} with weight 4 or less"
    )

# sphinx_gallery_start_ignore
_wf = np.array([P.edges[tuple(e)]["weight"] for e in found])
_wm = np.array([P.edges[tuple(e)]["weight"] for e in missed])
assert np.median(_wm) < np.median(_wf)
assert np.mean(_wm <= 4) > 0.75 > 0.5 > np.mean(_wf <= 4)
activity = np.array([P.nodes[n]["activity"] for n in P])


def draw_precision_recall(mode):
    pal = style.palette(mode)
    halo = [pe.withStroke(linewidth=3, foreground=pal.surface)]
    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    for name in names:
        x, y = _r[name], _p[name]
        ax.scatter(
            x, y, s=60, color=pal.kept, edgecolors=pal.surface, linewidths=1.5, zorder=3
        )
        right = x < 0.75
        ax.annotate(
            name,
            (x, y),
            xytext=(8 if right else -8, 0),
            textcoords="offset points",
            ha="left" if right else "right",
            va="center",
            fontsize=9,
            color=pal.ink,
            path_effects=halo,
        )
    ax.set(
        xlim=(0, 1.02),
        ylim=(0, 1.02),
        xlabel="Recall: share of planted edges kept",
        ylabel="Precision: share of kept edges planted",
        title="Validation against the planted truth",
    )
    ax.grid(True)
    return fig


figures.show(
    draw_precision_recall,
    "precision_recall",
    width="80%",
    alt=(
        "A scatter of precision against recall with one labeled point per "
        "backbone. The marginal likelihood backbone sits highest, at precision "
        f"{_p['Marginal likelihood']:.2f} and recall "
        f"{_r['Marginal likelihood']:.2f}. The metric backbone is farthest right "
        f"and lowest, at recall {_r['Metric backbone']:.2f} and precision "
        f"{_p['Metric backbone']:.2f}. The spanning tree and LANS sit at the "
        "left, with recall below one half."
    ),
    caption=(
        "Precision against recall for the six backbones of the planted "
        "network. Up is fewer false edges, right is more planted edges found; "
        "the values are printed above."
    ),
)
# sphinx_gallery_end_ignore

# %%
# Most of the missed ties drew only a few extra interactions. A tie seen
# two or three times looks like chance, and no method can tell it from
# noise using the weights alone without keeping much noise too: that is
# the recall a method trades for its precision.
#
# The drawing below shows each of the four groups as a ring of nodes, so
# that planted ties are chords inside a ring. It compares the marginal
# likelihood backbone with the global cut of about the same size.

# sphinx_gallery_start_ignore
# The groups are known, so draw each one as a ring: planted ties are chords
# inside a ring, and edges between groups cross the gaps.
pos_P = {}
for _g in range(4):
    _members = [n for n in P if P.nodes[n]["group"] == _g]
    _center = 0.62 * np.array(
        [np.cos(np.pi / 4 + _g * np.pi / 2), np.sin(np.pi / 4 + _g * np.pi / 2)]
    )
    for _k, _node in enumerate(_members):
        _angle = 2 * np.pi * _k / len(_members)
        pos_P[_node] = _center + 0.32 * np.array([np.cos(_angle), np.sin(_angle)])
RECOVERY = ["Marginal likelihood", "Global weight cut"]


def recovery_sets(name):
    kept = edge_set(backbones[name])
    return kept & TRUTH, kept - TRUTH, TRUTH - kept


def draw_recovery(mode):
    pal = style.palette(mode)
    false_color = style.categorical(2, mode)[1]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.8), layout="constrained")
    sizes = 10 + 70 * np.sqrt(activity / activity.max())
    for ax, name in zip(axes, RECOVERY):
        hits, false, misses = recovery_sets(name)
        layers = [
            (misses, pal.removed, ":", 0.9),
            (hits, pal.kept, "-", 1.1),
            (false, false_color, (0, (3.0, 2.0)), 1.1),
        ]
        for z, (edges, color, ls, lw) in enumerate(layers):
            ax.add_collection(
                LineCollection(
                    [(pos_P[u], pos_P[v]) for u, v in map(tuple, edges)],
                    colors=color,
                    linestyles=ls,
                    linewidths=lw,
                    capstyle="round",
                    zorder=1 + z,
                )
            )
        xy = np.array([pos_P[n] for n in P])
        ax.scatter(
            *xy.T,
            s=sizes,
            c=pal.node,
            edgecolors=pal.surface,
            linewidths=0.8,
            zorder=5,
        )
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_axis_off()
        ax.set_title(
            f"{name}\n{len(hits)} planted found, {len(false)} noise kept, "
            f"{len(misses)} missed",
            fontsize=10,
        )
    handles = [
        Line2D([], [], color=pal.kept, lw=1.6, label="Planted, kept"),
        Line2D(
            [], [], color=false_color, lw=1.6, ls=(0, (3.0, 2.0)), label="Noise, kept"
        ),
        Line2D([], [], color=pal.removed, lw=1.6, ls=":", label="Planted, missed"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    return fig


_ml_sets = [len(s) for s in recovery_sets("Marginal likelihood")]
_gc_sets = [len(s) for s in recovery_sets("Global weight cut")]
assert _gc_sets[1] > 3 * _ml_sets[1]
# Most of the global cut's noise edges join two groups, between active nodes.
_gc_false = recovery_sets("Global weight cut")[1]
_across = [e for e in _gc_false if len({P.nodes[n]["group"] for n in e}) == 2]
assert len(_across) > len(_gc_false) / 2
_fp_activity = [activity[n] for e in _gc_false for n in e]
assert np.median(_fp_activity) > np.percentile(activity, 90)
figures.show(
    draw_recovery,
    "recovery",
    fmt="png",
    width=None,
    thumbnail=True,
    alt=(
        "Two drawings of the planted network, nodes sized by activity and "
        "arranged in four rings, one per group. Left, the marginal likelihood backbone: "
        f"{_ml_sets[0]} planted edges kept as solid blue chords inside the "
        f"rings, {_ml_sets[1]} noise edges as dashed orange lines and "
        f"{_ml_sets[2]} missed planted edges as dotted gray lines. Right, the "
        f"global weight cut: {_gc_sets[0]} planted edges kept, {_gc_sets[1]} "
        "noise edges, many of them running between the large, active nodes "
        f"of different rings, and {_gc_sets[2]} missed."
    ),
    caption=(
        "Two backbones of about the same size on the planted network. Solid: "
        "planted edges kept; dashed: noise edges kept; dotted: planted edges "
        "missed. Noise edges that neither backbone keeps are not drawn. Larger "
        "nodes are more active; each ring is one group."
    ),
)
# sphinx_gallery_end_ignore

# %%
# The global cut spends much of its budget on heavy noise between the most
# active nodes, across groups. The marginal likelihood filter expects those
# pairs to be heavy and does not keep them.
#
# Validation without a planted truth
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Real networks come without labels, but the same logic applies to any
# property you know or care about. If the network has known groups,
# measure the share of kept edges within a group, as "Start here" does. If
# the analysis depends on paths, measure reachability. Write the check as a
# function of ``(original, backbone)`` and pass it to
# :func:`~nx_backbone.compare_backbones`. A planted experiment like this
# one, simulated to resemble your data, is the closest you can get to a
# precision and recall for a real study.
#
# .. _cv-alpha:
#
# 3. Robustness to alpha and the correction
# -----------------------------------------
#
# The significance level :math:`\alpha` and the multiple-testing correction
# (``mtc``) are choices, and the backbone changes with both. "Start here"
# shows *how much* the size changes. With a known truth we can see *what*
# changes: which kind of error each choice trades for the other.
#
# The correction is applied to all the p-values at once, so we compute each
# correction's adjusted p-values once, with
# :func:`~nx_backbone.adjust_pvalues`, and count the edges below every
# :math:`\alpha`; :func:`~nx_backbone.threshold_filter` with ``mtc=``
# keeps exactly those edges.

CORRECTIONS = {"none": "No correction", "bonferroni": "Bonferroni", "bh": "BH"}
ALPHAS = np.logspace(-4, np.log10(0.5), 100)
pvalues = np.array([p for _, _, p in ml_P.edges(data="ml_pvalue")])
is_planted = np.array([t for _, _, t in ml_P.edges(data="planted")])


def sweep(mtc, alphas=ALPHAS):
    """Edges kept, precision and recall at every alpha, for one correction."""
    adjusted = pvalues if mtc == "none" else np.asarray(nb.adjust_pvalues(pvalues, mtc))
    kept = adjusted[None, :] < alphas[:, None]
    n_kept = kept.sum(axis=1)
    hits = (kept & is_planted).sum(axis=1)
    with np.errstate(invalid="ignore"):
        return n_kept, hits / n_kept, hits / is_planted.sum()


curves = {mtc: sweep(mtc) for mtc in CORRECTIONS}
LEVELS = np.array([0.001, 0.01, 0.05, 0.1, 0.2])
print(f"{'alpha':>6} {'correction':<14}{'edges':>6}{'precision':>11}{'recall':>8}")
for mtc, label in CORRECTIONS.items():
    for alpha, n, prec, rec in zip(LEVELS, *sweep(mtc, LEVELS)):
        print(f"{alpha:>6} {label:<14}{n:>6}{prec:>11.2f}{rec:>8.2f}")

# sphinx_gallery_start_ignore
_lv = {mtc: sweep(mtc, LEVELS) for mtc in CORRECTIONS}
_i05 = int(np.flatnonzero(LEVELS == 0.05)[0])
for _mtc in CORRECTIONS:
    _bb = nb.threshold_filter(ml_P, "ml_pvalue", 0.05, mtc=_mtc)
    assert _bb.number_of_edges() == _lv[_mtc][0][_i05]
    assert np.isclose(precision(P, _bb), _lv[_mtc][1][_i05])
# BH: false discovery proportion below alpha at every level in the table.
assert np.all(1 - _lv["bh"][1] < LEVELS)
# Uncorrected: it exceeds alpha at the usual levels.
assert np.all(1 - _lv["none"][1][2:] > LEVELS[2:])
# Bonferroni: no false edge at any level in the table, but recall < 0.5.
assert np.all(_lv["bonferroni"][1] == 1.0) and np.all(_lv["bonferroni"][2] < 0.5)
# Recall ordering at 0.05: none > BH > Bonferroni.
assert _lv["none"][2][_i05] > _lv["bh"][2][_i05] > _lv["bonferroni"][2][_i05]


def draw_alpha_sweep(mode):
    pal = style.palette(mode)
    colors = style.categorical(3, mode)
    styles = ["-", "--", ":"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5), layout="constrained")
    titles = ["Edges kept", "Precision", "Recall"]
    for panel, (ax, title) in enumerate(zip(axes, titles)):
        for (mtc, label), color, ls in zip(CORRECTIONS.items(), colors, styles):
            ax.plot(ALPHAS, curves[mtc][panel], color=color, ls=ls, lw=1.6, label=label)
        ax.axvline(0.05, color=pal.muted, lw=0.8)
        ax.set(xscale="log", xlabel="Significance level α", title=title)
        ax.grid(True)
    axes[0].axhline(len(TRUTH), color=pal.ink, lw=0.8)
    axes[0].text(
        1.2e-4,
        len(TRUTH) * 1.04,
        "planted edges",
        fontsize=8,
        color=pal.ink,
        va="bottom",
    )
    axes[0].set_ylim(0, 2.5 * len(TRUTH))
    axes[1].plot(ALPHAS, 1 - ALPHAS, color=pal.ink, lw=0.8)
    axes[1].text(
        ALPHAS[-1],
        1 - ALPHAS[-1] - 0.02,
        "1 − α",
        fontsize=8,
        color=pal.ink,
        ha="center",
        va="top",
    )
    axes[1].set_ylim(0.4, 1.02)
    axes[2].set_ylim(0, 1.02)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=3
    )
    return fig


figures.show(
    draw_alpha_sweep,
    "alpha_sweep",
    width=None,
    alt=(
        "Three line charts against the significance level alpha on a log axis, "
        "for no correction (solid), Bonferroni (dashed) and Benjamini-Hochberg "
        "(dotted). Left, edges kept: the uncorrected line climbs past the number "
        "of planted edges, Bonferroni stays low and flat, BH lies between. "
        "Middle, precision: Bonferroni stays at 1, BH stays above the 1 minus "
        "alpha reference line, and the uncorrected line falls well below it as "
        "alpha grows. Right, recall: highest without correction, lowest with "
        "Bonferroni."
    ),
    caption=(
        "The marginal likelihood backbone of the planted network as "
        ":math:`\\alpha` grows: edges kept, precision and recall, for each "
        "correction. The thin line in the middle panel is :math:`1 - \\alpha`, "
        "the precision BH aims to stay above; the vertical lines mark "
        ":math:`\\alpha = 0.05`. The table above gives the values at five "
        "levels."
    ),
)
# sphinx_gallery_end_ignore

# %%
# Each correction controls a different error:
#
# - **No correction** controls nothing about the backbone as a whole. At
#   the usual levels, the share of noise among the kept edges is larger
#   than :math:`\alpha`, and it grows as :math:`\alpha` does.
# - **Bonferroni** controls the chance of keeping even one noise edge. Here
#   it keeps no noise edge at any level in the table, and finds less than
#   half of the planted ties.
# - **BH** controls the **false discovery rate**, the expected share of
#   noise among the kept edges (Benjamini & Hochberg, 1995). Here the
#   observed share stays below :math:`\alpha` at every level in the table,
#   while recall stays far above Bonferroni's.
#
# So the choice is not a technicality: it decides which error your
# conclusions can tolerate. Choose it, and :math:`\alpha`, before you look
# at the backbone, then report the backbone at a few levels as a
# sensitivity analysis.
#
# .. _cv-order:
#
# 4. Robustness to input order
# ----------------------------
#
# A network is a set of nodes and edges, so building it in a different
# order should not change the backbone. For most methods it does not. Three
# structural methods are exceptions, as their Notes say:
#
# - the **maximum spanning tree**: when weights tie, a graph has many
#   maximum spanning trees, and which one is returned depends on the order
#   in which edges were added;
# - the **planar maximally filtered graph** (PMFG; Tumminello et al.,
#   2005), which adds edges from heaviest to lightest while the graph stays
#   planar, trying tied edges in insertion order;
# - the **modularity backbone** (Rajeh et al., 2022), whose community
#   detection visits nodes in graph order even with a fixed ``seed``.
#
# Ties are common in count data. The Les Misérables co-appearance network
# that ships with NetworkX has small integer weights:

L = nx.les_miserables_graph()
weight_counts = Counter(w for _, _, w in L.edges(data="weight"))
print(
    f"{L.number_of_edges()} edges, {len(weight_counts)} distinct weights; "
    f"{weight_counts[1]} edges have weight 1"
)

# %%
# To measure the effect, rebuild the network several times with its nodes
# and edges added in a random order (and each edge's endpoints in a random
# order), extract each backbone from every rebuild, and compare it with the
# backbone of the original by Jaccard similarity. The disparity filter
# (Serrano, Boguñá & Vespignani, 2009) and the ultrametric backbone serve as
# controls.


def rebuilt(G, seed):
    """The same network, with nodes and edges added in a shuffled order."""
    rng = random.Random(seed)
    nodes, edges = list(G.nodes(data=True)), list(G.edges(data=True))
    rng.shuffle(nodes)
    rng.shuffle(edges)
    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_edges_from(
        (v, u, d) if rng.random() < 0.5 else (u, v, d) for u, v, d in edges
    )
    return H


ORDER_METHODS = {
    "Disparity filter": lambda G: nb.threshold_filter(
        nb.disparity_filter(G), "disparity_pvalue", 0.05
    ),
    "Ultrametric backbone": lambda G: nb.boolean_filter(
        nb.ultrametric_backbone(G), "ultrametric_keep"
    ),
    "Spanning tree": lambda G: nb.boolean_filter(
        nb.maximum_spanning_tree_backbone(G), "mst_keep"
    ),
    "PMFG": lambda G: nb.boolean_filter(
        nb.planar_maximally_filtered_graph(G), "pmfg_keep"
    ),
    "Modularity backbone": lambda G: nb.boolean_filter(
        nb.modularity_backbone(G, seed=42), "modularity_keep"
    ),
}
REBUILDS = [rebuilt(L, seed) for seed in range(10)]
original_order = {name: method(L) for name, method in ORDER_METHODS.items()}
shuffled_order = {
    name: [method(H) for H in REBUILDS] for name, method in ORDER_METHODS.items()
}

print(f"{'':<22}{'original':>9}{'rebuilds':>10}{'Jaccard':>14}")
for name, H in original_order.items():
    sizes = [B.number_of_edges() for B in shuffled_order[name]]
    sims = [jaccard(H, B) for B in shuffled_order[name]]
    print(
        f"{name:<22}{H.number_of_edges():>9}{min(sizes):>6}-{max(sizes):<3}"
        f"{min(sims):>9.2f}-{max(sims):.2f}"
    )

# sphinx_gallery_start_ignore
_sims = {
    name: [jaccard(original_order[name], B) for B in shuffled_order[name]]
    for name in ORDER_METHODS
}
for _name in ["Disparity filter", "Ultrametric backbone"]:
    assert all(s == 1.0 for s in _sims[_name])
for _name in ["Spanning tree", "PMFG", "Modularity backbone"]:
    assert min(_sims[_name]) < 0.9
assert all(B.number_of_edges() == 76 for B in shuffled_order["Spanning tree"])
assert min(_sims["Modularity backbone"]) == min(min(s) for s in _sims.values())
assert min(_sims["Modularity backbone"]) < 0.5
for _name in ["PMFG", "Modularity backbone"]:
    _sizes = {B.number_of_edges() for B in shuffled_order[_name]}
    assert _sizes != {original_order[_name].number_of_edges()}
assert weight_counts[1] > L.number_of_edges() / 3


def draw_order(mode):
    pal = style.palette(mode)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.4, 3.6), layout="constrained")
    order = list(ORDER_METHODS)[::-1]
    for row, name in enumerate(order):
        sims = np.array(_sims[name])
        jitter = rng.uniform(-0.18, 0.18, len(sims))
        ax.scatter(
            sims,
            row + jitter,
            s=36,
            color=pal.kept,
            edgecolors=pal.surface,
            linewidths=1.2,
            zorder=3,
        )
    labels = []
    for name in order:
        sizes = [B.number_of_edges() for B in shuffled_order[name]]
        span = (
            f"{min(sizes)}"
            if min(sizes) == max(sizes)
            else f"{min(sizes)}-{max(sizes)}"
        )
        labels.append(f"{name}\n{span} edges")
    ax.set_yticks(range(len(order)), labels, fontsize=8.5)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Jaccard similarity to the backbone of the original order")
    ax.set_title(f"{len(REBUILDS)} shuffled rebuilds of Les Misérables")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    return fig


figures.show(
    draw_order,
    "input_order",
    width="85%",
    alt=(
        "A dot chart with one row per method and one dot per shuffled rebuild, "
        "placed at the Jaccard similarity of its backbone to the original "
        "one. The disparity filter and the ultrametric backbone have every dot "
        "at 1. The spanning tree's dots spread from "
        f"{min(_sims['Spanning tree']):.2f} to {max(_sims['Spanning tree']):.2f}, "
        f"the PMFG's from {min(_sims['PMFG']):.2f} to {max(_sims['PMFG']):.2f}, "
        "and the modularity backbone's from "
        f"{min(_sims['Modularity backbone']):.2f} to "
        f"{max(_sims['Modularity backbone']):.2f}."
    ),
    caption=(
        "How much each backbone of Les Misérables changes when the same network "
        "is built in a different order. One dot per rebuild; 1 means the same "
        "edges as the original order. The row labels give the range of edge "
        "counts over the rebuilds."
    ),
)
# sphinx_gallery_end_ignore

# %%
# The two controls never change. The spanning tree always has one edge
# fewer than the number of nodes, but *which* edges varies. The PMFG and
# the modularity backbone change in size as well as content, and one
# rebuild of the modularity backbone shares less than half its edges with
# the original. None of this is a bug: with tied weights, the order is
# simply the tie-breaker. But a reader who rebuilds your network from a
# file sorted differently will get a different backbone. Two remedies:
#
# - **Use an order-free equivalent** where one exists. For positive
#   weights, the ultrametric backbone is exactly the union of all maximum
#   spanning trees, so every rebuilt tree lies inside it:

ultra = edge_set(original_order["Ultrametric backbone"])
print(
    "every rebuilt spanning tree inside the ultrametric backbone:",
    all(edge_set(B) <= ultra for B in shuffled_order["Spanning tree"]),
)

# %%
# - **Keep the edges that every rebuild agrees on**, with
#   :func:`~nx_backbone.consensus_backbone`, and report how many there are:

for name in ["Spanning tree", "PMFG", "Modularity backbone"]:
    stable = nb.consensus_backbone(original_order[name], *shuffled_order[name])
    print(
        f"{name:<20} {stable.number_of_edges():>3} edges kept by all "
        f"{len(REBUILDS) + 1} orders (original: "
        f"{original_order[name].number_of_edges()})"
    )

# sphinx_gallery_start_ignore
assert all(edge_set(B) <= ultra for B in shuffled_order["Spanning tree"])
# sphinx_gallery_end_ignore

# %%
# Otherwise, report the order: say that the edges were added in the order of
# the input file, and publish the file.
#
# .. _cv-votes:
#
# 5. Consensus and vote counts
# ----------------------------
#
# :func:`~nx_backbone.consensus_backbone` keeps the edges present in every
# backbone it is given. A softer version counts **votes**: how many
# backbones keep each edge. Keeping the edges with at least *v* votes gives
# a family of backbones, from the union of all (*v* = 1) to the consensus
# (*v* = all of them). Back on the planted network, with the six backbones
# of section 1:

votes = Counter(e for H in backbones.values() for e in edge_set(H))
consensus = nb.consensus_backbone(*backbones.values())
print(
    f"{'votes':>5}{'edges':>7}{'planted':>9}{'at least':>10}{'precision':>11}{'recall':>8}"
)
by_votes = {}
for v in range(1, len(backbones) + 1):
    exactly = {e for e, c in votes.items() if c == v}
    at_least = {e for e, c in votes.items() if c >= v}
    hits = len(at_least & TRUTH)
    by_votes[v] = (
        len(exactly),
        len(exactly & TRUTH),
        hits / len(at_least),
        hits / len(TRUTH),
    )
    print(
        f"{v:>5}{len(exactly):>7}{len(exactly & TRUTH):>9}{len(at_least):>10}"
        f"{by_votes[v][2]:>11.2f}{by_votes[v][3]:>8.2f}"
    )

# sphinx_gallery_start_ignore
_n = len(backbones)
assert edge_set(consensus) == {e for e, c in votes.items() if c == _n}
_prec = [by_votes[v][2] for v in range(1, _n + 1)]
_rec = [by_votes[v][3] for v in range(1, _n + 1)]
assert all(a > b for a, b in itertools.pairwise(_rec))  # recall falls with v
assert _prec[-1] > _prec[0] + 0.3  # precision rises overall
# The consensus is about as precise as the marginal likelihood backbone, but
# finds far fewer planted edges.
assert abs(_prec[-1] - _p["Marginal likelihood"]) < 0.02
assert _rec[-1] < _r["Marginal likelihood"] / 2
# A single-vote edge is mostly noise; an edge every method keeps is planted.
assert by_votes[1][1] / by_votes[1][0] < 0.25
assert by_votes[_n][1] / by_votes[_n][0] > 0.95


def draw_votes(mode):
    pal = style.palette(mode)
    halo = [pe.withStroke(linewidth=3, foreground=pal.surface)]
    colors = style.categorical(2, mode)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8), layout="constrained")
    vs = np.arange(1, _n + 1)
    planted = np.array([by_votes[v][1] for v in vs])
    noise = np.array([by_votes[v][0] - by_votes[v][1] for v in vs])
    ax1.bar(vs, planted, width=0.7, color=pal.kept, label="Planted")
    ax1.bar(
        vs,
        noise,
        width=0.7,
        bottom=planted,
        color="none",
        edgecolor=pal.muted,
        hatch="///",
        linewidth=0.8,
        label="Noise",
    )
    ax1.set(
        xticks=vs,
        xlabel="Number of backbones that keep the edge",
        ylabel="Edges",
        title="a. Edges by vote count",
    )
    ax1.grid(True, axis="y")
    ax1.grid(False, axis="x")
    ax1.legend(
        handles=[
            Patch(facecolor=pal.kept, label="Planted"),
            Patch(facecolor="none", edgecolor=pal.muted, hatch="///", label="Noise"),
        ],
        loc="upper right",
    )
    for series, values, color, ls, marker in [
        ("Precision", _prec, colors[0], "-", "o"),
        ("Recall", _rec, colors[1], "--", "s"),
    ]:
        ax2.plot(vs, values, color=color, ls=ls, marker=marker, lw=1.6, ms=6)
        ax2.annotate(
            series,
            (vs[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=pal.ink,
            path_effects=halo,
        )
    ax2.scatter(
        [_n + 1.7],
        [_p["Marginal likelihood"]],
        marker="o",
        s=40,
        facecolors="none",
        edgecolors=pal.ink,
        linewidths=1.5,
    )
    ax2.scatter(
        [_n + 1.7],
        [_r["Marginal likelihood"]],
        marker="s",
        s=40,
        facecolors="none",
        edgecolors=pal.ink,
        linewidths=1.5,
    )
    ax2.annotate(
        "marginal\nlikelihood\nalone",
        (_n + 1.7, _r["Marginal likelihood"]),
        xytext=(0, -14),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8,
        color=pal.muted,
    )
    ax2.set(
        xticks=vs,
        xlim=(0.6, _n + 2.3),
        ylim=(0, 1.05),
        xlabel="Kept by at least this many backbones",
        title="b. Precision and recall of the vote",
    )
    ax2.grid(True)
    return fig


figures.show(
    draw_votes,
    "votes",
    width=None,
    alt=(
        "Two charts. a, a stacked bar per vote count from 1 to "
        f"{_n}: solid bars count planted edges, hatched bars noise edges. "
        f"Edges kept by one backbone are mostly noise ({by_votes[1][0] - by_votes[1][1]} "
        f"of {by_votes[1][0]}); edges kept by all {_n} are almost all planted "
        f"({by_votes[_n][1]} of {by_votes[_n][0]}). b, precision (solid line, "
        f"circles) rises from {_prec[0]:.2f} to {_prec[-1]:.2f} as more votes "
        f"are required, and recall (dashed line, squares) falls from "
        f"{_rec[0]:.2f} to {_rec[-1]:.2f}. Hollow markers at the right show the "
        f"marginal likelihood backbone alone: precision {_p['Marginal likelihood']:.2f}, "
        f"recall {_r['Marginal likelihood']:.2f}."
    ),
    caption=(
        "Votes of the six backbones on the planted network. (a) Edges by how "
        "many backbones keep them, planted (solid) and noise (hatched). (b) "
        "Precision and recall of the backbone of edges with at least that many "
        "votes; the hollow markers at the right are the marginal likelihood "
        "backbone on its own. The values are printed above."
    ),
)
# sphinx_gallery_end_ignore

# %%
# Votes sort edges well: an edge that one method alone keeps is usually
# noise, and an edge that all six keep is almost always planted. But
# requiring more votes buys precision with recall, and the consensus of all
# six is only about as precise as the best single method while finding far
# fewer planted ties. A consensus is only as good as the independence of
# its voters: three of our six share a preference for heavy edges, so their
# votes count the same evidence three times.
#
# Use votes to *grade* edges, for example to draw the stable core more
# boldly, or to report how many conclusions rest on edges every method
# keeps. Do not use a consensus as a substitute for choosing a method whose
# assumptions fit the data.
#
# .. _cv-benchmark:
#
# 6. The Les Misérables benchmark, explained
# ------------------------------------------
#
# :doc:`/tutorials/les_miserables_benchmark` lists how many of the 254
# edges of Les Misérables every method keeps. It is computed at build time
# from the same case list that the test suite pins, so its numbers are
# current. It is easy to read it as a ranking. It is not one, for three
# reasons.
#
# **The filter matters as much as the method.** Every row is a
# score-then-filter recipe, and the rows use three kinds of filter:
#
# - a **p-value below 0.05, without correction**, for the statistical
#   methods;
# - the **top 30%** of a score, for the proximity measures, which have no
#   p-value;
# - a **rule of the method**, such as a keep flag or a score cut, for the
#   structural methods.
#
# **The fixed-share rows are equal by construction.** Every proximity
# measure keeps 30% of the edges, so they all show the same count; and the
# spanning tree, like any spanning tree of a connected graph, keeps one edge
# fewer than the number of nodes. Those counts describe the filter, not the
# method.
#
# **The p-value rows are uncorrected.** That keeps the table comparable
# with other studies, but on 254 tests about 13 edges pass at
# :math:`\alpha = 0.05` by chance alone. Here are the statistical rows
# again, computed the same way, with the BH correction beside them:

PVALUE_METHODS = {
    "disparity_filter": "disparity_pvalue",
    "noise_corrected_filter": "nc_pvalue",
    "marginal_likelihood_filter": "ml_pvalue",
    "ecm_filter": "ecm_pvalue",
    "lans_filter": "lans_pvalue",
    "polya_filter": "polya_pvalue",
    "gloss_filter": "gloss_pvalue",
    "glab_filter": "glab_pvalue",
}
chance = 0.05 * L.number_of_edges()
print(f"expected to pass by chance at alpha = 0.05: about {chance:.0f}")
print(f"{'method':<28}{'no correction':>14}{'BH':>5}")
benchmark_pvalues = {}
for method, attr in PVALUE_METHODS.items():
    scored = getattr(nb, method)(L)
    counts = [
        nb.threshold_filter(scored, attr, 0.05, mtc=mtc).number_of_edges()
        for mtc in ("none", "bh")
    ]
    benchmark_pvalues[method] = counts
    print(f"{method:<28}{counts[0]:>14}{counts[1]:>5}")

# sphinx_gallery_start_ignore
# The benchmark itself, run from its single definition: the counts above
# must be the table's.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    BENCH = []
    for _case in lm.CASES:
        _G = lm.graph_for(_case)
        _scored = _case.score(_G)
        _kept = _case.filter_graph(_scored).number_of_edges()
        _pv = _case.filter == "threshold" and _case.attr.endswith("_pvalue")
        _bh = (
            nb.threshold_filter(
                _scored, _case.attr, lm.ALPHA, mtc="bh"
            ).number_of_edges()
            if _pv
            else None
        )
        BENCH.append((_case, _kept, _bh))
_table = {c.name: kept for c, kept, _ in BENCH}
for _method, (_none, _bhc) in benchmark_pvalues.items():
    assert _table[_method] == _none, _method
assert lm.ALPHA == 0.05 and lm.TOP_FRACTION == 0.3
_fraction = {kept for c, kept, _ in BENCH if c.filter == "fraction"}
assert len(_fraction) == 1
assert _table["maximum_spanning_tree_backbone"] == L.number_of_nodes() - 1
_below_chance = [m for m, (none, _) in benchmark_pvalues.items() if none < chance]
assert {"disparity_filter", "gloss_filter", "polya_filter"} <= set(_below_chance)
assert all(benchmark_pvalues[m][1] <= 2 for m in _below_chance)
assert (
    benchmark_pvalues["marginal_likelihood_filter"][1]
    > 0.75 * (benchmark_pvalues["marginal_likelihood_filter"][0])
)
assert benchmark_pvalues["glab_filter"][1] > 0.75 * benchmark_pvalues["glab_filter"][0]
assert benchmark_pvalues["ecm_filter"][1] == 0 < benchmark_pvalues["ecm_filter"][0]

FAMILIES = [
    (
        "p-value below 0.05, no correction",
        lambda c: c.filter == "threshold" and c.attr.endswith("_pvalue"),
    ),
    (
        "A rule of the method",
        lambda c: (
            c.filter == "boolean"
            or (c.filter == "threshold" and not c.attr.endswith("_pvalue"))
        ),
    ),
    ("Top 30% of a score", lambda c: c.filter == "fraction"),
]


def draw_benchmark(mode):
    pal = style.palette(mode)
    rows = []
    for title, member in FAMILIES:
        group = [b for b in BENCH if member(b[0])]
        group.sort(key=lambda b: -b[1])
        rows.append(("header", title, None, None))
        rows += [("row", c.name, kept, bh) for c, kept, bh in group]
    fig, ax = plt.subplots(figsize=(6.6, 0.2 * len(rows) + 1.2), layout="constrained")
    y = np.arange(len(rows))[::-1]
    labels = []
    for yi, (kind, label, kept, bh) in zip(y, rows):
        if kind == "header":
            ax.text(
                -0.01,
                yi,
                label,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=pal.ink,
            )
            labels.append("")
            continue
        labels.append(label)
        if bh is not None:
            ax.plot([bh, kept], [yi, yi], color=pal.muted, lw=1, zorder=2)
            ax.scatter(
                bh,
                yi,
                s=34,
                facecolors=pal.surface,
                edgecolors=pal.kept,
                lw=1.4,
                zorder=3,
            )
        ax.scatter(
            kept, yi, s=34, color=pal.kept, edgecolors=pal.surface, lw=1, zorder=4
        )
    ax.set_yticks(y, labels, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-5, L.number_of_edges() + 5)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.axvline(chance, color=pal.muted, lw=0.8)
    ax.text(
        chance + 3,
        len(rows) - 0.6,
        "chance at α = 0.05",
        fontsize=8,
        color=pal.muted,
        va="top",
    )
    ax.set_xlabel(f"Edges kept, of {L.number_of_edges()}")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                color=pal.kept,
                markeredgecolor=pal.surface,
                label="As in the benchmark table",
            ),
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                markerfacecolor=pal.surface,
                markeredgecolor=pal.kept,
                label="p-value rows with BH",
            ),
        ],
        loc="lower right",
    )
    ax.set_title("The Les Misérables benchmark by kind of filter")
    return fig


_n_rows = len(BENCH)
figures.show(
    draw_benchmark,
    "benchmark",
    width=None,
    alt=(
        f"A dot chart of the {_n_rows} rows of the Les Misérables benchmark, "
        "edges kept on a horizontal axis from 0 to "
        f"{L.number_of_edges()}, grouped by kind of filter. The p-value rows "
        "each have a filled dot at their uncorrected count and a hollow dot "
        "at their count with the BH correction, joined by a line; several "
        "fall from near the chance line to 0. The rule rows spread across the "
        "axis. The top-30% rows all sit at the same count, "
        f"{next(iter(_fraction))}."
    ),
    caption=(
        "Every row of :doc:`the benchmark table </tutorials/les_miserables_benchmark>`, "
        "grouped by kind of filter. Filled dots are the table's counts; for the "
        "p-value rows, hollow dots are the counts after the BH correction. The "
        "vertical line is the number of edges expected to pass "
        ":math:`p < 0.05` by chance."
    ),
)
# sphinx_gallery_end_ignore

# %%
# Read with those three reasons in mind, the table says something useful:
#
# - The disparity filter, Pólya filter and GloSS filter keep fewer edges
#   than chance alone would pass, and almost nothing survives the
#   correction: on this small network their tests have little power.
# - The marginal likelihood filter and the GLAB filter keep most of their
#   edges after correction: their evidence is strong.
# - The ECM filter keeps a sizeable backbone uncorrected but nothing after
#   correction, so its uncorrected count says little about the network.
#
# A count says how much a recipe keeps, not whether what it keeps is right.
# To rank methods you need a criterion: a planted truth as in section 2, or
# a property your analysis depends on.
#
# .. _cv-report:
#
# 7. Reporting
# ------------
#
# A reader should be able to rebuild your backbone and judge it.
# :func:`~nx_backbone.describe_backbone` writes the core of the methods
# paragraph from the same arguments you passed to the filter: the method and
# its null model, :math:`\alpha` and the correction, how much of the network
# was kept, and the citations, including the version of nx-backbone you
# used.

print(
    nb.describe_backbone(
        "marginal_likelihood_filter",
        alpha=0.05,
        mtc="bh",
        original=P,
        backbone=ml_backbone,
    )
)

# %%
# ``style="bibtex"`` returns the same references as BibTeX entries, for a
# reference manager:

print(nb.describe_backbone("marginal_likelihood_filter", mtc="bh", style="bibtex"))

# %%
# The paragraph covers what the function knows. It cannot know your
# sensitivity analysis, your validation or, for an order-dependent method,
# the input order. For example, the paragraph for the modularity backbone
# lists the seed but says nothing about order:

text = nb.describe_backbone(
    "modularity_backbone",
    original=L,
    backbone=original_order["Modularity backbone"],
    seed=42,
)
print(text.split("\n\n")[0])

# sphinx_gallery_start_ignore
assert "seed=42" in text and "order" not in text.split("\n\n")[0]
# sphinx_gallery_end_ignore

# %%
# A checklist for the methods section
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Before you report a backbone, check that the methods section says:
#
# #. **What a weight is**: what was counted or measured, and any transform
#    applied to it (:func:`~nx_backbone.describe_backbone` reports the
#    transforms recorded by :func:`~nx_backbone.transform_weights`).
# #. **The method and why**: its null model or criterion, and why it suits
#    these weights (:doc:`/user_guide/choosing_method`).
# #. **The level and the correction**, chosen in advance, and which error
#    the correction controls (section 3).
# #. **How much was kept**: edges, nodes and weight, from
#    :func:`~nx_backbone.describe_backbone`.
# #. **Sensitivity**: the backbone's size, and ideally your main result, at
#    a few levels of :math:`\alpha` and with another correction.
# #. **Input order and seed** for the spanning tree, the PMFG and the
#    modularity backbone, or the consensus over shuffled rebuilds
#    (section 4).
# #. **A comparison** with at least one method built on a different idea,
#    with its agreement, reported as agreement and not as validation
#    (section 1).
# #. **Validation**: a check against a property you know, or a planted
#    experiment simulated to resemble your data (section 2), and which
#    conclusions rest on edges that every method keeps (section 5).
# #. **Citations** for the method, the correction and the software with its
#    version, from :func:`~nx_backbone.describe_backbone` (or the
#    ``CITATION.cff`` file in the repository).
# #. **Code and data**, so that a reader can run the same recipe.
#
# Where next
# ----------
#
# - **The how-to pages.** :doc:`/tutorials/comparing_methods` and
#   :doc:`/tutorials/validating_backbones` give the calls in brief, the
#   latter with a signed backbone and its own checklist;
#   :doc:`/tutorials/les_miserables_benchmark` has the full table.
# - **Narrative tutorials.** :doc:`Start here
#   </auto_examples/start_here/plot_hairball_to_backbone>` for the whole
#   workflow on one network, and :doc:`Statistical null models side by side
#   </auto_examples/statistical_nulls/plot_statistical_nulls>` for what each
#   null model assumes.
# - **The user guide.** :doc:`/user_guide/evaluation` (every evaluation
#   measure) and :doc:`/user_guide/choosing_method`.
# - **The API.** :func:`~nx_backbone.compare_backbones`,
#   :func:`~nx_backbone.consensus_backbone`,
#   :func:`~nx_backbone.adjust_pvalues`,
#   :func:`~nx_backbone.threshold_filter`,
#   :func:`~nx_backbone.describe_backbone`,
#   :func:`~nx_backbone.maximum_spanning_tree_backbone`,
#   :func:`~nx_backbone.planar_maximally_filtered_graph`,
#   :func:`~nx_backbone.modularity_backbone` and
#   :func:`~nx_backbone.ultrametric_backbone`.
#
# References
# ----------
#
# - Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
#   rate: a practical and powerful approach to multiple testing. *Journal of
#   the Royal Statistical Society: Series B*, 57(1), 289-300.
# - Dianati, N. (2016). Unwinding the hairball graph: Pruning algorithms for
#   weighted complex networks. *Physical Review E*, 93, 012304.
#   https://doi.org/10.1103/PhysRevE.93.012304
# - Foti, N. J., Hughes, J. M., & Rockmore, D. N. (2011). Nonparametric
#   sparsification of complex multiscale networks. *PLoS ONE*, 6(2), e16431.
#   https://doi.org/10.1371/journal.pone.0016431
# - Rajeh, S., Savonnet, M., Leclercq, E., & Cherifi, H. (2022).
#   Modularity-based backbone extraction in weighted complex networks. In
#   *NetSci-X 2022*, pp. 67-79.
# - Serrano, M. A., Boguñá, M., & Vespignani, A. (2009). Extracting the
#   multiscale backbone of complex weighted networks. *Proceedings of the
#   National Academy of Sciences*, 106(16), 6483-6488.
#   https://doi.org/10.1073/pnas.0808904106
# - Simas, T., Correia, R. B., & Rocha, L. M. (2021). The distance backbone
#   of complex networks. *Journal of Complex Networks*, 9(6), cnab021.
#   https://doi.org/10.1093/comnet/cnab021
# - Tumminello, M., Aste, T., Di Matteo, T., & Mantegna, R. N. (2005). A
#   tool for filtering information in complex systems. *Proceedings of the
#   National Academy of Sciences*, 102(30), 10421-10426.
