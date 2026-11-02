"""
.. _tutorial-weight-meaning:

What your weights mean: transforms, units, and nulls
====================================================

Before a backbone method can decide which edges matter, it has to read each
edge's **weight**, the number attached to the edge.  Every weighted method
reads that number in a particular way: as a count of events, as a strength
whose size only matters relative to its neighbors, or as a distance.  Hand a
method weights of the wrong kind and it still returns a backbone, with no
error, but the backbone answers a different question.

These choices often change a backbone more than the choice of method does.
This tutorial takes one network and follows the decision through:

1. how to tell whether your weights are counts, strengths or distances, and
   which methods assume which;
2. how a backbone moves when the same weights are transformed;
3. why the count-based null models need ``unit=`` when the weights are not
   counts of single events;
4. when the disparity filter's uniform null model is wrong, and what the
   generalized null models change;
5. how to check that a test is calibrated, by running it on data with no
   structure, with two fixes from version 0.3.0 as examples of what a
   miscalibrated test looks like.

It is not a tour of the null models: the
:ref:`statistical-nulls tutorial <tutorial-statistical-nulls>` compares them
side by side, and :doc:`/user_guide/weight_transforms` is the reference for
:func:`~nx_backbone.transform_weights` and ``unit=``.  Every number below is
computed when the documentation is built.
"""

# sphinx_gallery_thumbnail_path = "_static/example_figures/weight_meaning/thumbnail.png"

# %%
# The network
# -----------
#
# The Les Misérables co-appearance network that ships with NetworkX links two
# characters when they appear in the same chapter of the novel.  The weight
# counts the chapters they share.  We use it throughout because we know
# exactly what its weights are, which is the point of this tutorial.
#
# Two terms recur.  A node's **degree** is its number of links, and its
# **strength** is the sum of their weights.  A statistical method gives each
# edge a **p-value**: the probability, under the method's **null model** (its
# description of weights with no structure), of a weight at least as extreme
# as the one observed.  We keep edges with a p-value below 0.05.

import math
import warnings
from itertools import pairwise

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import nx_backbone as nb

# sphinx_gallery_start_ignore
# isort: split
from _figures import style
from _figures.tutorial import TutorialFigures
from matplotlib.ticker import NullLocator

figures = TutorialFigures("weight_meaning")
# sphinx_gallery_end_ignore

G = nx.les_miserables_graph()
ALPHA = 0.05


def table(header, rows):
    """Print rows as an aligned plain-text table."""
    rows = [[str(c) for c in row] for row in [header, *rows]]
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    for k, row in enumerate(rows):
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
        if k == 0:
            print("  ".join("-" * w for w in widths))


def edge_set(H):
    """The edges of *H* as a set of unordered pairs."""
    return {frozenset(e) for e in H.edges()}


def jaccard(a, b):
    """Shared edges over edges in either set."""
    return len(a & b) / len(a | b) if a | b else 1.0


print(f"{G.number_of_nodes()} characters, {G.number_of_edges()} edges")

# %%
# Counts, strengths, or distances?
# --------------------------------
#
# Most weights are one of three kinds.
#
# - A **count** is a number of events: messages sent, chapters shared, trips
#   taken.  Counts are whole numbers, and they add up: two observation
#   periods give the sum of their counts.  A larger count is also more
#   *evidence*, since one shared chapter could be chance and thirty could
#   not.
# - A **strength** (or intensity) is a magnitude where larger means closer
#   but the scale itself is arbitrary: a correlation, a similarity score, a
#   rating, money spent.  Only how large a weight is *relative to other
#   weights* carries meaning.
# - A **distance** (or cost) is a magnitude where larger means *farther*:
#   travel time, a dissimilarity, a price of reaching someone.
#
# Three questions sort most data.  Are the weights whole numbers that count
# something?  Would observing for twice as long roughly double them?  Does a
# larger weight mean a closer tie or a weaker one?  A quick profile of the
# weights helps with the first:


def weight_profile(H, weight="weight"):
    """Print what the weights of *H* look like."""
    w = np.array([d for *_, d in H.edges(data=weight)], dtype=float)
    whole = np.mean(w == np.round(w))
    print(f"  {len(w)} weights from {w.min():g} to {w.max():g}")
    print(f"  share of whole numbers: {whole:.0%}")
    print(f"  median {np.median(w):g}, mean {w.mean():.2f}")


weight_profile(G)

# sphinx_gallery_start_ignore
_w = [d for *_, d in G.edges(data="weight")]
assert all(float(x).is_integer() for x in _w) and min(_w) >= 1
# sphinx_gallery_end_ignore

# %%
# Every weight is a whole number of chapters, and more shared chapters means
# a closer tie: these weights are counts, and counts are also strengths.
#
# Each method, in turn, reads a weight in one of four ways.  The full table,
# method by method and checked by the test suite, is in
# :doc:`/user_guide/weight_transforms`; in short:
#
# - **As a count.**  :func:`~nx_backbone.marginal_likelihood_filter`,
#   :func:`~nx_backbone.ecm_filter`,
#   :func:`~nx_backbone.noise_corrected_filter` and the exact
#   :func:`~nx_backbone.polya_filter` treat a weight as a number of events,
#   so its absolute size matters.
# - **As a share.**  :func:`~nx_backbone.disparity_filter` and its relatives
#   use the fraction of a node's strength that an edge carries, so a common
#   scale factor does not matter but the shape of the weights does.
# - **As an order.**  :func:`~nx_backbone.lans_filter`,
#   :func:`~nx_backbone.maximum_spanning_tree_backbone` and
#   :func:`~nx_backbone.ultrametric_backbone` use only which weights are
#   larger than which.
# - **As a proximity, turned into a distance by 1/w.**
#   :func:`~nx_backbone.metric_backbone`,
#   :func:`~nx_backbone.high_salience_skeleton` and the other shortest-path
#   methods take the distance of an edge to be one over its weight, so a heavy
#   edge is a short one.  The ultrametric backbone does too, but uses only the
#   order of those distances.
#
# The last group is where a distance does the most damage.  Suppose the same
# network arrived as distances, say "chapters apart per shared chapter",
# :math:`d = 1/w`, and we passed them to
# :func:`~nx_backbone.metric_backbone` as they are.  The **metric backbone**
# (Simas et al., 2021) keeps an edge when no indirect path between its
# endpoints is shorter than the edge itself.  Reading :math:`d` as a
# proximity, the method takes :math:`1/d = w` as the distance, so the
# closest pairs become the farthest.


def metric_edges(H):
    return edge_set(nb.boolean_filter(nb.metric_backbone(H), "metric_keep"))


def ultrametric_edges(H):
    return edge_set(nb.boolean_filter(nb.ultrametric_backbone(H), "ultrametric_keep"))


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    D = nb.transform_weights(G, lambda w: 1 / w)  # the network as distances
    D_fixed = nb.transform_weights(D, lambda d: 1 / d)  # back to proximities

right = metric_edges(G)
misread = metric_edges(D)
fixed = metric_edges(D_fixed)
print(f"Metric backbone, weights read as proximities: {len(right)} edges")
print(f"Metric backbone, distances passed as weights: {len(misread)} edges")
print(f"Edges in both: {len(right & misread)} (Jaccard {jaccard(right, misread):.2f})")
print(f"After inverting the distances first: same backbone? {fixed == right}")
print(f"\ntransform_weights warned: {str(caught[0].message)[:70]}...")

# sphinx_gallery_start_ignore
assert fixed == right
assert misread != right and jaccard(right, misread) < 0.6
assert any("does not preserve the order" in str(c.message) for c in caught)
pos_les = style.layout(G, "les_miserables")
misread_graph = nx.Graph(list(misread))
right_graph = nx.Graph(list(right))


def draw_misread(mode):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6), layout="constrained")
    panels = [
        ("Weights read as proximities", right_graph),
        ("Distances passed as weights", misread_graph),
    ]
    for ax, (title, H) in zip(axes, panels):
        counts = style.draw_backbone(
            ax, G, H, pos_les, mode=mode, legend=False, node_size=20
        )
        ax.set_title(f"{title}\n{counts.short}", fontsize=10)
    fig.legend(handles=style.legend_handles(mode), loc="outside lower center", ncols=4)
    return fig


figures.show(
    draw_misread,
    "misread_distances",
    fmt="png",
    alt=(
        "Two drawings of the Les Misérables network in the same layout. Left: "
        f"the metric backbone of the co-appearance counts, {len(right)} of "
        f"{G.number_of_edges()} edges in solid blue over the removed edges in "
        "light gray. Right: the metric backbone computed from distances passed "
        f"as weights, {len(misread)} edges, of which only "
        f"{len(right & misread)} are also in the left backbone."
    ),
    caption=(
        "The metric backbone of the same network, read correctly (left) and "
        f"with distances passed as weights (right).  Only {len(right & misread)} "
        f"edges are in both; inverting the distances first with "
        "``transform_weights`` gives back the left backbone exactly."
    ),
    thumbnail=True,
)
# sphinx_gallery_end_ignore

# %%
# Nothing failed: both calls returned a backbone of plausible size.
# :func:`~nx_backbone.transform_weights` did warn that ``1/w`` reverses the
# order of the weights, which is the one signal that something changed
# meaning.  If your data are distances, convert them to proximities before
# any method that reads weights as proximities, and check the direction
# again before an order-based method such as
# :func:`~nx_backbone.maximum_spanning_tree_backbone`.
#
# Transforms: one network, many backbones
# ---------------------------------------
#
# Even with the direction right, the *scale* of a strength is a choice.  Is a
# tie with 16 shared chapters twice as close as one with 8, four times, or
# only a little closer?  A **transform** answers that by replacing each
# weight :math:`w` with :math:`f(w)`.  The power family :math:`w^\lambda`
# covers the usual answers: :math:`\lambda = 1` leaves the weights alone,
# :math:`\lambda = 1/2` (a square root) compresses large weights, small
# :math:`\lambda` behaves like a logarithm, and :math:`\lambda > 1`
# stretches them.  Every positive :math:`\lambda` keeps the order of the
# weights.
#
# We transform the weights with each :math:`\lambda` and extract three
# backbones that read weights in three different ways: the metric backbone
# (distances :math:`1/w`), the disparity filter (shares) and LANS (order
# within each node).

LAMBDAS = np.geomspace(1 / 32, 16, 19)


def three_backbones(H):
    return {
        "Metric": metric_edges(H),
        "Disparity": edge_set(
            nb.threshold_filter(nb.disparity_filter(H), "disparity_pvalue", ALPHA)
        ),
        "LANS": edge_set(nb.threshold_filter(nb.lans_filter(H), "lans_pvalue", ALPHA)),
    }


sweep = [
    three_backbones(nb.transform_weights(G, lambda w, lam=lam: w**lam))
    for lam in LAMBDAS
]
baseline = three_backbones(G)
ultrametric = ultrametric_edges(G)

NAMED = {
    "w": lambda w: w,
    "sqrt(w)": math.sqrt,
    "log(1 + w)": math.log1p,
    "w^2": lambda w: w**2,
}
named = {
    label: three_backbones(nb.transform_weights(G, f)) for label, f in NAMED.items()
}
table(
    ["Transform", *(f"{m} (Jaccard to w)" for m in baseline)],
    [
        [
            label,
            *(f"{len(bb[m])} ({jaccard(bb[m], baseline[m]):.2f})" for m in baseline),
        ]
        for label, bb in named.items()
    ],
)
print(f"\nUltrametric backbone: {len(ultrametric)} edges")

# sphinx_gallery_start_ignore
_metric = [len(s["Metric"]) for s in sweep]
_disp = [len(s["Disparity"]) for s in sweep]
assert all(s["LANS"] == baseline["LANS"] for s in sweep)
assert all(x >= y for x, y in pairwise(_metric))
assert all(x <= y for x, y in pairwise(_disp))
assert _metric[0] == G.number_of_edges()
assert sweep[-1]["Metric"] == ultrametric
assert ultrametric <= baseline["Metric"]
SWEEP_SERIES = {
    name: [len(s[name]) for s in sweep] for name in ("Metric", "Disparity", "LANS")
}
# sphinx_gallery_end_ignore

# %%
# The three behave very differently:

# sphinx_gallery_start_ignore


def draw_power_sweep(mode):
    p = style.palette(mode)
    colors = style.categorical(3, mode)
    marks = {"Metric": "o", "Disparity": "s", "LANS": "^"}
    labels = {
        "Metric": "Metric backbone (distances 1/w)",
        "Disparity": "Disparity filter (shares)",
        "LANS": "LANS (order)",
    }
    fig, ax = plt.subplots(figsize=(8.4, 4.0), layout="constrained")
    for ref, text in [
        (G.number_of_edges(), "every edge"),
        (len(ultrametric), "ultrametric backbone"),
    ]:
        ax.axhline(ref, color=p.muted, lw=0.8)
        ax.text(LAMBDAS[0], ref + 4, text, color=p.muted, fontsize=8, va="bottom")
    ax.axvline(1, color=p.hairline, lw=1.2, zorder=0)
    ax.text(0.94, 75, "original\nweights", color=p.muted, fontsize=8, ha="right")
    for (name, ys), color in zip(SWEEP_SERIES.items(), colors):
        ax.plot(LAMBDAS, ys, color=color, marker=marks[name], markersize=4)
        ax.annotate(
            labels[name],
            (LAMBDAS[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=p.ink,
            fontsize=8.5,
        )
    ax.set_xscale("log", base=2)
    ax.xaxis.set_minor_locator(NullLocator())
    ticks = [1 / 32, 1 / 8, 1 / 2, 1, 2, 4, 16]
    ax.set_xticks(ticks, ["1/32", "1/8", "1/2", "1", "2", "4", "16"])
    ax.set_xlim(LAMBDAS[0] * 0.8, LAMBDAS[-1] * 1.2)
    ax.set_ylim(-5, G.number_of_edges() + 25)
    ax.set_xlabel("Power λ in the transform w → w^λ (log scale)")
    ax.set_ylabel(f"Edges kept (of {G.number_of_edges()})")
    ax.set_title("Backbone size as the weights are compressed or stretched")
    ax.grid(True, axis="y")
    # room on the right for the direct labels
    fig.get_layout_engine().set(rect=(0, 0, 0.76, 1))
    return fig


figures.show(
    draw_power_sweep,
    "power_sweep",
    alt=(
        "A line chart of edges kept (vertical) against the power lambda of the "
        "transform w to the lambda, from 1/32 to 16 on a log scale (horizontal). "
        f"The metric backbone (circles) falls from all {G.number_of_edges()} "
        f"edges at small lambda to the {len(ultrametric)}-edge ultrametric "
        f"backbone at large lambda. The disparity filter (squares) rises from "
        f"{SWEEP_SERIES['Disparity'][0]} to {SWEEP_SERIES['Disparity'][-1]} "
        f"edges. LANS (triangles) stays at {len(baseline['LANS'])} edges."
    ),
    caption=(
        "Backbone size under :math:`w^\\lambda`.  Every positive power keeps the "
        "order of the weights, so LANS never moves; the metric backbone and the "
        "disparity filter move in opposite directions.  Exact values for four "
        "common transforms are in the table above."
    ),
)
# sphinx_gallery_end_ignore

# %%
# - **LANS** compares each weight only with the other weights of the same
#   node, so any order-keeping transform leaves it exactly where it was.
# - The **disparity filter** tests whether an edge carries an unusually
#   large share of a node's strength.  Stretching the weights makes the
#   heaviest edge's share larger, so more edges pass; compressing them evens
#   the shares out, and almost nothing passes.
# - The **metric backbone** moves the other way.  A path's length is the sum
#   of its distances :math:`w^{-\lambda}`.  As :math:`\lambda` shrinks, every
#   distance approaches 1, a direct edge is always shorter than a path of two
#   or more, and every edge is kept.  As :math:`\lambda` grows, the longest
#   step dominates a path's sum, so the test approaches the *ultrametric*
#   one, which compares the edge with the longest step of the best path; at
#   the right of the chart the two backbones are identical.
#
# So the metric backbone is not one backbone but a family indexed by how
# you scale the weights, running from the whole graph to the ultrametric
# backbone.  Where along that range the original weights fall depends on the
# data.  Here is how the edges themselves move:

# sphinx_gallery_start_ignore
SHOWN = [0.5, 1.0, 4.0]
shown_backbones = {
    lam: metric_edges(nb.transform_weights(G, lambda w, lam=lam: w**lam))
    for lam in SHOWN
}
assert shown_backbones[1.0] == right
assert shown_backbones[0.5] >= right >= shown_backbones[4.0]


def draw_metric_family(mode):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.0), layout="constrained")
    for ax, lam in zip(axes, SHOWN):
        H = nx.Graph(list(shown_backbones[lam]))
        counts = style.draw_backbone(
            ax, G, H, pos_les, mode=mode, legend=False, node_size=14
        )
        ax.set_title(f"λ = {lam:g}\n{counts.short}", fontsize=10)
    fig.legend(handles=style.legend_handles(mode), loc="outside lower center", ncols=4)
    return fig


figures.show(
    draw_metric_family,
    "metric_family",
    fmt="png",
    alt=(
        "Three drawings of Les Misérables in the same layout, showing the "
        "metric backbone of the weights raised to the powers "
        f"{', '.join(f'{x:g}' for x in SHOWN)}: "
        + ", ".join(f"{len(shown_backbones[x])} edges" for x in SHOWN)
        + " in solid blue over the removed edges in light gray. Each backbone "
        "contains the next."
    ),
    caption=(
        "The metric backbone of :math:`w^\\lambda` for three powers.  The "
        "backbones are nested: compressing the weights adds edges, stretching "
        "them removes edges, and the strongest ties stay throughout."
    ),
)
# sphinx_gallery_end_ignore

# %%
# A transform is therefore a modeling decision, not preprocessing.  Choose
# it from what a weight means (is 16 chapters twice as close as 8?), not from
# which backbone looks best, and report it:
# :func:`~nx_backbone.transform_weights` records it in
# ``H.graph["weight_transforms"]`` and
# :func:`~nx_backbone.describe_backbone` puts it in the methods text.
#
# Units: what is one event?
# -------------------------
#
# The count-based null models ask whether an edge has more events than its
# endpoints' activity predicts.  The answer depends on how many events there
# are: a share that could be chance over 3 events is convincing over 300.  So
# these methods need to know what *one* event is, and the ``unit=`` argument
# tells them: the model uses ``weight / unit`` events.  The default,
# ``unit=1``, takes the weights themselves as counts.
#
# To see why that matters, imagine Les Misérables' co-appearances logged as
# time on the page instead: each shared chapter as a quarter of an hour, so
# the weights are in hours (0.25, 0.5, ...), or as 15 minutes, so the weights
# are in minutes.  The data are the same; only the unit changed.  We run two
# count models, the noise-corrected filter (Coscia and Neffke, 2017) and the
# ECM filter (Gemmetto et al., 2017), and, for contrast, the disparity filter,
# which reads shares.

RECORDINGS = {
    "chapters, unit=1": (G, 1),
    "hours, unit=1 (default)": (nb.transform_weights(G, lambda w: w / 4), 1),
    "minutes, unit=1 (default)": (nb.transform_weights(G, lambda w: 15 * w), 1),
    "hours, unit=0.25": (nb.transform_weights(G, lambda w: w / 4), 0.25),
    "minutes, unit=15": (nb.transform_weights(G, lambda w: 15 * w), 15),
}
COUNT_METHODS = {
    "Noise-corrected": (nb.noise_corrected_filter, "nc_pvalue"),
    "ECM": (nb.ecm_filter, "ecm_pvalue"),
}

unit_results = {}
unit_warnings = {}
for label, (H, unit) in RECORDINGS.items():
    for name, (score, attr) in COUNT_METHODS.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scored = score(H, unit=unit)
        unit_warnings[label, name] = len(caught)
        unit_results[label, name] = edge_set(nb.threshold_filter(scored, attr, ALPHA))
    unit_results[label, "Disparity"] = edge_set(
        nb.threshold_filter(nb.disparity_filter(H), "disparity_pvalue", ALPHA)
    )

UNIT_METHODS = [*COUNT_METHODS, "Disparity"]
reference = "chapters, unit=1"
table(
    ["Weights recorded as", *UNIT_METHODS],
    [
        [
            label,
            *(
                f"{len(unit_results[label, m])}"
                + ("" if unit_results[label, m] == unit_results[reference, m] else " *")
                for m in UNIT_METHODS
            ),
        ]
        for label in RECORDINGS
    ],
)
print("\n* a different backbone from the one on the chapter counts")
print(
    "Warned that the weights are not whole counts:",
    ", ".join(sorted({label for (label, _), n in unit_warnings.items() if n})),
)

# sphinx_gallery_start_ignore
for _m in COUNT_METHODS:
    _ref = unit_results[reference, _m]
    assert unit_results["hours, unit=0.25", _m] == _ref
    assert unit_results["minutes, unit=15", _m] == _ref
    assert len(unit_results["hours, unit=1 (default)", _m]) < len(_ref)
    assert len(unit_results["minutes, unit=1 (default)", _m]) > len(_ref)
    assert unit_warnings["hours, unit=1 (default)", _m]
    assert not unit_warnings["minutes, unit=15", _m]
for _label in RECORDINGS:
    assert unit_results[_label, "Disparity"] == unit_results[reference, "Disparity"]


def draw_units(mode):
    p = style.palette(mode)
    colors = style.categorical(3, mode)
    marks = ["o", "s", "^"]
    labels = list(RECORDINGS)
    fig, ax = plt.subplots(figsize=(7.4, 3.6), layout="constrained")
    y = np.arange(len(labels))[::-1]
    offsets = [0.18, 0.0, -0.18]
    for m, color, marker, dy in zip(UNIT_METHODS, colors, marks, offsets):
        xs = [len(unit_results[label, m]) for label in labels]
        ax.scatter(
            xs,
            y + dy,
            color=color,
            marker=marker,
            s=42,
            edgecolors=p.surface,
            linewidths=1.0,
            label=m,
            zorder=3,
        )
    for m, color in zip(COUNT_METHODS, colors):
        ref = len(unit_results[reference, m])
        ax.axvline(ref, color=color, lw=0.8, alpha=0.6, zorder=1)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, G.number_of_edges())
    ax.set_xlabel(f"Edges kept at p < {ALPHA} (of {G.number_of_edges()})")
    ax.set_title("Same data, three units: the count models move, shares do not")
    ax.grid(True, axis="x")
    ax.legend(loc="lower right", fontsize=8.5)
    return fig


def _counts(label):
    return ", ".join(f"{m} {len(unit_results[label, m])}" for m in UNIT_METHODS)


figures.show(
    draw_units,
    "units",
    alt=(
        "A dot chart with one row per recording of the same weights and the "
        "number of edges kept on the horizontal axis, for the noise-corrected "
        "filter (circles), the ECM filter (squares) and the disparity filter "
        "(triangles). "
        + "; ".join(f"{label}: {_counts(label)}" for label in RECORDINGS)
        + "."
    ),
    caption=(
        "Edges kept by two count models and the disparity filter when the same "
        "co-appearances are recorded in chapters, hours or minutes.  Thin "
        "vertical lines mark the count models' backbones on the chapter "
        "counts; setting ``unit=`` to one chapter's worth brings every row back "
        "to them.  The values are in the table above."
    ),
)
# sphinx_gallery_end_ignore

# %%
# With the default ``unit=1``, recording in hours divides the evidence by
# four and recording in minutes multiplies it by fifteen: the count models
# keep far fewer edges in hours and far more in minutes, from the same data.
# Setting ``unit=`` to the weight of one event (0.25 hours or 15 minutes)
# gives back exactly the backbone of the chapter counts.  The disparity
# filter never moves, because shares do not depend on the unit.  The scorers
# warn when ``weight / unit`` is not a whole number, as for the hours with
# ``unit=1``; a warning-free run in minutes is wrong all the same, because
# 15 is a whole number of *something*, just not of events.
#
# What if the weights are real-valued with no natural event, such as hours
# of contact with no fixed granularity?  Then ``unit=`` is a modeling
# decision: the amount of weight you are willing to treat as one independent
# observation.  The
# :ref:`statistical-nulls tutorial <tutorial-statistical-nulls>` shows how
# far the backbone moves as that unit changes.  If no unit is defensible, a
# share-based or order-based method avoids the question.
#
# The disparity filter's null: when uniform shares are wrong
# ----------------------------------------------------------
#
# The disparity filter (Serrano et al., 2009) has no unit, but it has an
# assumption about the *shape* of the weights.  Its null model splits a
# node's strength at random points, which is what you would see if the
# node's weights were independent draws from an exponential distribution.
# Real weights may be more uneven than that (a few very heavy ties) or more
# even.  Then the uniform null is wrong.  With uneven weights, edges look
# significant because the weights are heavy-tailed in general, not because
# those edges stand out; with even weights, real standouts are missed.
#
# ``disparity_filter(G, a=...)`` generalizes the null to weights drawn from
# a Gamma distribution with shape :math:`1/a`: ``a=1`` is the original
# (exponential) filter, ``a > 1`` allows more uneven weights and ``a < 1``
# more even ones, and ``a="ml"`` fits *a* to the network by maximum
# likelihood.  :doc:`/user_guide/weight_transforms` gives the formulas.
#
# The cleanest way to see when the uniform null is wrong is to give it data
# with no structure at all, where every edge is noise and a calibrated test
# at 0.05 keeps about 5% of them.  We build stars: a center with between 2 and
# 12 leaves, whose weights are independent Gamma draws, with a random scale
# per star.  Each leaf has only one link, so each edge's p-value is the test
# at its center, and one edge per star gives independent tests.

SHAPES = {
    "uneven (shape 0.3)": 0.3,
    "exponential (shape 1)": 1.0,
    "even (shape 3)": 3.0,
}


def gamma_stars(shape, seed, n_stars=1000):
    """Disjoint stars with i.i.d. Gamma(*shape*) weights; one test edge each."""
    rng = np.random.default_rng(seed)
    S = nx.Graph()
    tests = []
    for i in range(n_stars):
        k = int(rng.integers(2, 13))
        scale = rng.lognormal(0.0, 1.0)
        for j, w in enumerate(rng.gamma(shape, scale, size=k)):
            S.add_edge(("center", i), ("leaf", i, j), weight=float(w))
        tests.append((("center", i), ("leaf", i, 0)))
    return S, tests


REPLICATES = range(1, 4)  # three seeded replicates of 1,000 stars each
star_rates = {}
for label, shape in SHAPES.items():
    uniform_p, fitted_p, fitted_a = [], [], []
    for seed in REPLICATES:
        S, tests = gamma_stars(shape, seed)
        uniform = nb.disparity_filter(S)
        fitted = nb.disparity_filter(S, a="ml")
        uniform_p += [uniform.edges[e]["disparity_pvalue"] for e in tests]
        fitted_p += [fitted.edges[e]["disparity_pvalue"] for e in tests]
        fitted_a.append(fitted.graph["disparity_a"])
    star_rates[label] = {
        "uniform": np.mean(np.array(uniform_p) < ALPHA),
        "fitted": np.mean(np.array(fitted_p) < ALPHA),
        "fitted a": np.mean(fitted_a),
        "true a": 1 / shape,
    }
n_tests = len(uniform_p)
noise_band = 2 * math.sqrt(ALPHA * (1 - ALPHA) / n_tests)
table(
    ["Weights", "true a", "fitted a", "p < 0.05, a=1", 'p < 0.05, a="ml"'],
    [
        [
            label,
            f"{r['true a']:.2f}",
            f"{r['fitted a']:.2f}",
            f"{r['uniform']:.1%}",
            f"{r['fitted']:.1%}",
        ]
        for label, r in star_rates.items()
    ],
)
print(
    f"\n{n_tests} independent tests per row: a calibrated test gives "
    f"{ALPHA:.0%} +/- {noise_band:.1%} (two standard errors)"
)

# sphinx_gallery_start_ignore
_r = star_rates
assert _r["uneven (shape 0.3)"]["uniform"] > 2 * ALPHA
assert _r["even (shape 3)"]["uniform"] < ALPHA / 2
for _x in _r.values():
    assert abs(_x["fitted"] - ALPHA) < 1.5 * noise_band
    assert abs(_x["fitted a"] / _x["true a"] - 1) < 0.15
assert abs(_r["exponential (shape 1)"]["uniform"] - ALPHA) < 1.5 * noise_band


def draw_star_calibration(mode):
    p = style.palette(mode)
    colors = style.categorical(2, mode)
    fig, ax = plt.subplots(figsize=(7.0, 3.6), layout="constrained")
    x = np.arange(len(SHAPES))
    ax.axhline(ALPHA, color=p.muted, lw=0.9)
    ax.text(
        -0.55,
        ALPHA * 1.04,
        "calibrated: 5%",
        color=p.muted,
        fontsize=8.5,
        va="bottom",
        ha="left",
    )
    for key, color, marker, dx, text in [
        ("uniform", colors[0], "o", -0.08, "uniform null, a = 1"),
        ("fitted", colors[1], "s", 0.08, 'fitted null, a = "ml"'),
    ]:
        ys = [r[key] for r in star_rates.values()]
        ax.scatter(
            x + dx,
            ys,
            color=color,
            marker=marker,
            s=55,
            edgecolors=p.surface,
            linewidths=1.0,
            label=text,
            zorder=3,
        )
        for xi, yi in zip(x + dx, ys):
            ax.annotate(
                f"{yi:.1%}",
                (xi, yi),
                xytext=(-6 if dx < 0 else 6, 0),
                textcoords="offset points",
                ha="right" if dx < 0 else "left",
                va="center",
                color=p.ink,
                fontsize=8,
            )
    ax.set_xticks(x, [f"Gamma weights,\n{label}" for label in SHAPES])
    ax.set_xlim(-0.6, len(SHAPES) - 0.4)
    ax.set_ylim(0, max(r["uniform"] for r in star_rates.values()) * 1.2)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel(f"Share of pure-noise edges\nwith p < {ALPHA}")
    ax.set_title("The uniform null is right only for exponential-like weights")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=8.5)
    return fig


figures.show(
    draw_star_calibration,
    "disparity_null",
    width="85%",
    alt=(
        "A dot chart of the share of pure-noise edges with p below 0.05 for "
        "three kinds of Gamma weights (uneven, exponential, even). The uniform "
        "null (circles) gives "
        + ", ".join(f"{r['uniform']:.1%}" for r in star_rates.values())
        + "; the fitted null (squares) gives "
        + ", ".join(f"{r['fitted']:.1%}" for r in star_rates.values())
        + ". A horizontal line marks 5%."
    ),
    caption=(
        "Disparity filter on stars with no structure.  With uneven weights the "
        "uniform null calls too many edges significant, and with even weights "
        "too few; fitting *a* brings each case back to about 5% "
        f"({n_tests} tests per point; sampling error about +/- "
        f"{noise_band:.1%})."
    ),
)
# sphinx_gallery_end_ignore

# %%
# With exponential weights, the uniform null is right and both filters keep
# about 5% of the noise, within sampling error.  With uneven weights the
# uniform null keeps far too many, and with even weights almost none, while
# ``a="ml"`` recovers the true *a* and holds its level in all three.
#
# What does that mean for real data?  The karate club network's weights are
# based on the number of contexts in which two members interacted: small
# whole numbers, far more even than exponential.

K = nx.karate_club_graph()
weight_profile(K)
karate_uniform = nb.disparity_filter(K)
karate_fitted = nb.disparity_filter(K, a="ml")
a_karate = karate_fitted.graph["disparity_a"]
kept_uniform = nb.threshold_filter(karate_uniform, "disparity_pvalue", ALPHA)
kept_fitted = nb.threshold_filter(karate_fitted, "disparity_pvalue", ALPHA)
print(f"Fitted a = {a_karate:.2f}")
print(f"Edges kept, a=1: {kept_uniform.number_of_edges()} of {K.number_of_edges()}")
print(f'Edges kept, a="ml": {kept_fitted.number_of_edges()} of {K.number_of_edges()}')

# sphinx_gallery_start_ignore
assert a_karate < 0.5
assert kept_fitted.number_of_edges() > kept_uniform.number_of_edges()
# sphinx_gallery_end_ignore

# %%
# Under the uniform null, nothing on the karate club is significant: its
# shares are so even that no edge can stand out against a null that expects
# exponential spread.  The fitted null judges each edge against the
# network's own, much smaller, spread.  Which is right depends on the
# question.  The fitted null treats the network's typical unevenness as
# noise, so it finds edges that are heavy *for this network*; the uniform
# null takes exponential spread as the benchmark, whatever the network.
# Either way, report *a*.
#
# One caveat applies to both: the filter tests each edge from both of its
# endpoints and keeps the smaller p-value.  Each endpoint test can be
# calibrated while the edge test is not, because an edge gets two chances.
# On random 8-regular graphs with exponential weights, where the uniform null
# holds exactly at every node:

rng = np.random.default_rng(7)
or_rates = []
for seed in range(3):
    R = nx.random_regular_graph(8, 300, seed=seed)
    for u, v in R.edges():
        R[u][v]["weight"] = float(rng.exponential())
    p = np.array([d for *_, d in nb.disparity_filter(R).edges(data="disparity_pvalue")])
    or_rates.append(np.mean(p < ALPHA))
print(f"Share of edges with p < {ALPHA}: {np.mean(or_rates):.1%}")

# sphinx_gallery_start_ignore
assert 0.065 < np.mean(or_rates) < 0.095
# sphinx_gallery_end_ignore

# %%
# That is above 5% with nothing to find, and a multiple-testing correction
# does not remove it.  The same holds for every filter that tests an edge
# from both ends.
#
# Checking calibration on data with no structure
# ----------------------------------------------
#
# The star experiment is an instance of a general check.  A test is
# **calibrated** when, on data drawn from its own null model, its p-values
# are uniform: a share *t* of them fall below *t*, for every *t*.  The
# recipe:
#
# 1. simulate data from the method's null model, with no structure in it;
# 2. run the method as you would on real data;
# 3. compare the share of p-values below each threshold with the threshold.
#
# If the share is larger, the test is **anti-conservative**: it finds
# structure in noise.  If it is smaller, the test is conservative: valid,
# but with less power.  Look at more than 0.05, because a test can be fine
# there and badly wrong at the small thresholds that a multiple-testing
# correction uses.
#
# Version 0.3.0 fixed two tests that failed this check.  Rerunning the old
# behavior, which is still available as an option, shows what
# miscalibration looks like.
#
# The noise-corrected filter: an exact tail versus the delta rule
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The noise-corrected filter's null model sends each unit of interaction
# from a node chosen in proportion to its activity to a node chosen the same
# way, independently (a node can be chosen twice, giving a self-loop, as in
# the model).  Its default p-value is the exact binomial tail of that model.  The
# paper's own rule, ``nc_score > 1.64 * nc_sdev``, read as a z-test, is
# ``pvalue_method="delta"``.  We simulate directed graphs from the null:
# 800 unit interactions among 40 nodes with lognormal activity, in six
# seeded replicates.


def interaction_null(seed, n=40, interactions=800):
    """Directed counts from the noise-corrected null model."""
    rng = np.random.default_rng(seed)
    activity = rng.lognormal(0.0, 1.0, n)
    activity /= activity.sum()
    pairs = rng.choice(n, size=(interactions, 2), p=activity)
    N = nx.DiGraph()
    for u, v in pairs.tolist():
        w = N[u][v]["weight"] + 1 if N.has_edge(u, v) else 1
        N.add_edge(u, v, weight=w)
    return N


nc_pvalues = {
    method: np.concatenate(
        [
            [
                d
                for *_, d in nb.noise_corrected_filter(
                    interaction_null(seed), pvalue_method=method
                ).edges(data="nc_pvalue")
            ]
            for seed in range(6)
        ]
    )
    for method in ("binomial", "delta")
}
for method, p in nc_pvalues.items():
    print(f"{method:>8}: {len(p)} edges, {np.mean(p < ALPHA):.1%} with p < {ALPHA}")

# sphinx_gallery_start_ignore
assert np.mean(nc_pvalues["binomial"] < ALPHA) < 0.08
assert np.mean(nc_pvalues["delta"] < ALPHA) > 4 * ALPHA
# sphinx_gallery_end_ignore

# %%
# The exact tail rejects close to 5% of the noise; the edges of one graph
# are not independent tests, so expect some scatter around it.  The delta
# rule rejects several times that.  Its standard deviation is evaluated at
# the observed weight, through a transform that flattens as the weight grows,
# so the ratio of score to standard deviation is not bounded and its normal
# tail is far too small on graphs with low counts.
#
# SDSM on sparse data: an exact tail versus a normal approximation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The stochastic degree sequence model (SDSM; Neal, 2014) scores a
# **bipartite projection**: agents linked by the number of artifacts they
# share, a count.  Its p-value is the tail of a sum of independent Bernoulli
# draws.  ``pvalue_method="rna"`` approximates that tail with a refined
# normal approximation, as R ``backbone`` does (Neal, 2022); the default
# ``"auto"`` computes it exactly whenever that is affordable.  Our null data
# are incidence matrices of 300 agents and 250 artifacts in which every cell
# is an independent coin flip with probability 0.01, three seeded replicates.
# That is the SDSM's null model with equal degrees, and it is sparse: two
# agents are expected to share far less than one artifact.

rng = np.random.default_rng(108)
null_matrices = [(rng.random((300, 250)) < 0.01).astype(int) for _ in range(3)]
sdsm_scored = {
    method: [nb.sdsm(M, pvalue_method=method) for M in null_matrices]
    for method in ("auto", "rna")
}
sdsm_pvalues = {
    method: np.concatenate([[d for *_, d in H.edges(data="sdsm_pvalue")] for H in Hs])
    for method, Hs in sdsm_scored.items()
}
THRESHOLDS = [0.05, 1e-3, 1e-5]
table(
    ["pvalue_method", *(f"p < {t:g}" for t in THRESHOLDS), "pairs kept after BH"],
    [
        [
            method,
            *(f"{np.mean(p < t):.2g}" for t in THRESHOLDS),
            sum(
                nb.threshold_filter(H, "sdsm_pvalue", ALPHA, mtc="bh").number_of_edges()
                for H in sdsm_scored[method]
            ),
        ]
        for method, p in sdsm_pvalues.items()
    ],
)
print(f"\n{len(sdsm_pvalues['auto'])} agent pairs in all")

# sphinx_gallery_start_ignore
_auto, _rna = sdsm_pvalues["auto"], sdsm_pvalues["rna"]
assert np.mean(_auto < ALPHA) < ALPHA and np.mean(_rna < ALPHA) < ALPHA
assert np.mean(_rna < 1e-5) > 10 * 1e-5
assert np.mean(_auto < 1e-5) <= 1e-5
_bh = {
    m: sum(
        nb.threshold_filter(H, "sdsm_pvalue", ALPHA, mtc="bh").number_of_edges()
        for H in Hs
    )
    for m, Hs in sdsm_scored.items()
}
assert _bh["auto"] == 0 and _bh["rna"] > 0
# sphinx_gallery_end_ignore

# %%
# At 0.05 both look fine, even conservative: most pairs share no artifact at
# all, so their p-value is 1.  The trouble is in the far tail.  Below
# :math:`10^{-5}` the approximation puts many times more pairs than a
# calibrated test would, and those are exactly the p-values that survive a
# Benjamini-Hochberg (BH) correction for testing every pair.  On data with
# nothing in it, the approximation reports a backbone and the exact tail
# reports none.  The share below 0.05 alone would have missed this.
#
# Both checks on one chart: each curve is the share of p-values below a
# threshold, against the threshold, on log scales.  A calibrated test lies
# on or below the diagonal.

# sphinx_gallery_start_ignore
GRID = np.geomspace(1e-7, 1, 29)


def share_below(p):
    return np.array([np.mean(p < t) for t in GRID])


CURVES = {
    "Noise-corrected filter": {
        "exact binomial (default)": share_below(nc_pvalues["binomial"]),
        'delta rule ("delta")': share_below(nc_pvalues["delta"]),
    },
    "SDSM, sparse data": {
        'exact tail ("auto", default)': share_below(sdsm_pvalues["auto"]),
        'normal approximation ("rna")': share_below(sdsm_pvalues["rna"]),
    },
}
FLOORS = {
    "Noise-corrected filter": 1 / len(nc_pvalues["binomial"]),
    "SDSM, sparse data": 1 / len(sdsm_pvalues["auto"]),
}


def draw_calibration(mode):
    p = style.palette(mode)
    colors = style.categorical(2, mode)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), layout="constrained")
    for ax, (title, curves) in zip(axes, CURVES.items()):
        ax.plot(GRID, GRID, color=p.muted, lw=0.9)
        ax.text(2e-6, 2e-6 * 2.2, "calibrated", color=p.muted, fontsize=8, rotation=33)
        for (label, ys), color, marker, ls in zip(
            curves.items(), colors, ["o", "s"], ["solid", (0, (4, 2))]
        ):
            ok = ys > 0
            ax.plot(
                GRID[ok],
                ys[ok],
                color=color,
                marker=marker,
                markersize=3.5,
                ls=ls,
                label=label,
            )
        ax.axhline(FLOORS[title], color=p.hairline, lw=1.0)
        ax.text(
            1.5e-7,
            FLOORS[title] * 1.3,
            "one p-value",
            color=p.muted,
            fontsize=7.5,
            va="bottom",
        )
        ax.axvline(ALPHA, color=p.hairline, lw=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(GRID[0], 1.5)
        ax.set_ylim(min(FLOORS.values()) / 3, 1.5)
        ax.set_xlabel("Threshold t")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True)
    axes[0].set_ylabel("Share of null p-values below t")
    return fig


_rna_curve = CURVES["SDSM, sparse data"]['normal approximation ("rna")']
rna_cross = max(t for t, y in zip(GRID, _rna_curve) if y > t)


figures.show(
    draw_calibration,
    "calibration",
    alt=(
        "Two log-log charts of the share of null p-values below a threshold t "
        "against t, with a diagonal marking a calibrated test. Left, the "
        "noise-corrected filter: the exact binomial p-values (circles, solid) "
        "follow the diagonal, while the delta rule (squares, dashed) lies far "
        "above it, with "
        f"{np.mean(nc_pvalues['delta'] < 1e-4):.0%} of p-values below 0.0001. "
        "Right, SDSM on sparse data: the exact tail (circles, solid) stays "
        "below the diagonal and ends where no p-value is smaller, while the "
        "normal approximation (squares, dashed) is above the diagonal for "
        f"every threshold up to {rna_cross:.1g}, where "
        f"{_rna_curve[GRID == rna_cross][0]:.2g} of the p-values lie below it, "
        f"and still has {_rna_curve[0]:.1g} of them below {GRID[0]:g}."
    ),
    caption=(
        "Calibration curves on data with no structure.  The default p-values of "
        "both methods stay close to the diagonal or below it; the options they "
        "replaced in 0.3.0 rise above it, the delta rule everywhere and the normal "
        "approximation in the far tail that multiple-testing corrections use.  "
        "Points with no p-value below *t* are not drawn; the horizontal line "
        "marks the share of a single p-value."
    ),
)
# sphinx_gallery_end_ignore

# %%
# Run this check on any method before you trust its p-values on a new kind
# of data, with simulated data that look like yours: the same size, the
# same density, the same range of degrees and counts.  It takes a few lines
# and seconds, and it catches both a wrong null model (the stars) and a
# wrong computation of the right null model (the two fixes).
#
# Deciding what your weights mean
# -------------------------------
#
# - **Name the kind of weight.**  Counts of events, strengths where only
#   relative size matters, or distances where larger means farther.  Check
#   the direction before any method that reads weights as proximities or
#   orders them.
# - **Pick methods that read that kind.**  Count models for counts, with
#   ``unit=`` set to one event; share- or order-based methods for strengths;
#   distances inverted to proximities first.
# - **Treat a transform as part of the model.**  Choose it from what the
#   weights mean, and report it.
# - **Check the null model's shape.**  For the disparity filter, compare
#   ``a=1`` with ``a="ml"``, and decide which benchmark your question needs.
# - **Check calibration on noise.**  Simulate from the null at your data's
#   size and density, and look at the share of p-values below 0.05 and below
#   the thresholds your correction will use.
#
# Where next
# ----------
#
# - :doc:`/user_guide/weight_transforms`: the reference for
#   :func:`~nx_backbone.transform_weights`, ``unit=`` and the generalized
#   disparity null, with the table of which methods each transform changes.
# - :ref:`Statistical null models side by side <tutorial-statistical-nulls>`:
#   the null models compared on one network, including sweeps of ``a`` and
#   ``unit=``.
# - :doc:`/tutorials/validating_backbones`: multiple-testing corrections
#   and robustness.
# - API pages: :doc:`/api/statistical`, :doc:`/api/structural` and
#   :doc:`/api/filters`.
#
# References
# ----------
#
# - Coscia, M., & Neffke, F. M. (2017). Network backboning with noisy data.
#   *Proceedings of the IEEE International Conference on Data Engineering
#   (ICDE)*, 425-436. https://doi.org/10.1109/ICDE.2017.100
# - Gemmetto, V., Cardillo, A., & Garlaschelli, D. (2017). Irreducible
#   network backbones: unbiased graph filtering via maximum entropy.
#   arXiv:1706.00230.
# - Neal, Z. P. (2014). The backbone of bipartite projections: Inferring
#   relationships from co-authorship, co-sponsorship, co-attendance and other
#   co-behaviors. *Social Networks*, 39, 84-97.
#   https://doi.org/10.1016/j.socnet.2014.06.001
# - Neal, Z. P. (2022). backbone: An R package to extract network
#   backbones. *PLoS ONE*, 17(5), e0269137.
#   https://doi.org/10.1371/journal.pone.0269137
# - Serrano, M. A., Boguñá, M., & Vespignani, A. (2009). Extracting the
#   multiscale backbone of complex weighted networks. *Proceedings of the
#   National Academy of Sciences*, 106(16), 6483-6488.
#   https://doi.org/10.1073/pnas.0808904106
# - Simas, T., Correia, R. B., & Rocha, L. M. (2021). The distance backbone
#   of complex networks. *Journal of Complex Networks*, 9(6), cnab021.
#   https://doi.org/10.1093/comnet/cnab021
