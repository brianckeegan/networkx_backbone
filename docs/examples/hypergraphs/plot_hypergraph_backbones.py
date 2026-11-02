"""
Hypergraph backbones: keeping the groups that matter
====================================================

Many social and biological records are lists of *groups*: who attended which
meeting, which authors wrote which paper, which proteins form which complex.
The usual first step turns every group into all of its pairs and analyzes the
resulting graph. This tutorial is about the cases where that step throws away
the thing you care about, and about the backbone methods that work on the
groups directly.

We follow one small, real dataset from start to finish: the 18 women and 14
social events of Davis, Gardner & Gardner's *Deep South* (1941), bundled with
NetworkX as ``nx.davis_southern_women_graph()``. Along the way we

1. compare a group view and a pairwise view of the same events;
2. extract the minimum description length (MDL) backbone of Kirkley, Felippe,
   Malizia & Battiston (2026) and look at the trade-off it optimizes;
3. run its local variant and see how its three aggregation rules differ;
4. switch to a synthetic meeting log to test which groups recur more often
   than chance, with the statistically validated hypergraphs and cores of
   Musciotto, Battiston & Mantegna (2021);
5. use the structural helpers (maximal hyperedges, order filtering,
   s-components); and
6. move hypergraphs in and out of the library with the HIF interchange format.

Every number on this page is computed when the documentation is built. The
plotting code is left out of the page to keep the analysis readable; it is in
the script and notebook you can download at the end. For a
shorter, code-first reference to the same functions, see
:doc:`/tutorials/hypergraph_backbone`.
"""

# sphinx_gallery_thumbnail_path = "_static/example_figures/hypergraphs/thumbnail.png"

import io
import itertools
import json
import math
import random

import matplotlib.pyplot as plt
import networkx as nx
from _figures import style
from _figures.tutorial import TutorialFigures
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import nx_backbone as nb

# sphinx_gallery_start_ignore
# Figure plumbing, hidden on the rendered page but kept in the downloads.
#
# Each figure is drawn twice with the shared docs style (docs/_figures/style.py),
# once per theme, and shown as an HTML pair whose ``only-light``/``only-dark``
# classes let the docs theme display the variant that matches the reader's
# theme.
#
# The figures go through the docs' tutorial figure helper
# (docs/_figures/tutorial.py): in the docs build the files are linked from
# docs/_static/example_figures/hypergraphs/; run anywhere else (as a script,
# or as the downloaded notebook), they are embedded in the output.  Every
# figure here is light enough to stay SVG: the drawings have at most a few
# hundred marks, and PNG would be larger.
figures = TutorialFigures("hypergraphs")


_SQUARE = 64.0  # marker area (pt^2) of a hyperedge (event) square


def _event_color(mode):
    """Events (hyperedges) wear categorical slot 2; people wear the node blue."""
    return style.categorical(2, mode)[1]


def _scatter(ax, xy, **kwargs):
    xy = list(xy)
    if xy:
        ax.scatter([x for x, _ in xy], [y for _, y in xy], **kwargs)


def _legend_marker(label, marker, face, edge=None, size=7.0, width=0.0):
    return Line2D(
        [],
        [],
        linestyle="",
        marker=marker,
        markersize=size,
        markerfacecolor=face,
        markeredgecolor=edge if edge is not None else face,
        markeredgewidth=width,
        label=label,
    )


def _legend_line(label, color, width=style.KEPT_WIDTH):
    return Line2D([], [], color=color, lw=width, label=label)


def _first_name(woman):
    return woman.split()[0]


# sphinx_gallery_end_ignore

# %%
# The data: women and the events they attended
# --------------------------------------------
#
# A **hypergraph** is a set of nodes together with a set of **hyperedges**,
# where a hyperedge is any group of nodes, of any size. The size of a
# hyperedge is its **order**; an ordinary graph is a hypergraph whose
# hyperedges all have order two. NetworkX has no hypergraph class, so
# ``nx_backbone`` uses the simplest representation there is: a list of
# hyperedges, each an iterable of node labels.
#
# In the Davis data each event is a hyperedge (the guest list) over the women.
# NetworkX ships the data as a bipartite graph, so the guest lists are the
# neighborhoods of the event nodes.

davis = nx.davis_southern_women_graph()
women = davis.graph["top"]
events = davis.graph["bottom"]
guest_lists = {event: frozenset(davis[event]) for event in events}
hyperedges = list(guest_lists.values())

# Two events can have identical guest lists.  The backbone functions merge
# such duplicates, so keep track of which events share one.
events_of = {}
for event, guests in guest_lists.items():
    events_of.setdefault(guests, []).append(event)
distinct = list(events_of)
label = {guests: "/".join(names) for guests, names in events_of.items()}

sizes = sorted(len(guests) for guests in hyperedges)
print(f"{len(women)} women, {len(events)} events, {len(distinct)} distinct guest lists")
print(f"event sizes (orders): {sizes}")
for guests, names in events_of.items():
    if len(names) > 1:
        print(f"{' and '.join(names)} had the same {len(guests)} guests")

# %%
# 1. Groups versus pairs
# ----------------------
#
# The **pairwise projection** (or clique expansion) of a hypergraph links two
# nodes whenever they share at least one hyperedge. It is what you get when
# you analyze co-attendance as a graph of women. The projection is not
# reversible. A meeting of three people and three separate meetings of two
# produce exactly the same triangle:

one_meeting = [("A", "B", "C")]
three_meetings = [("A", "B"), ("B", "C"), ("A", "C")]


def projection(hyperedges):
    """The pairwise projection: one edge per pair of nodes sharing a hyperedge."""
    graph = nx.Graph()
    for members in hyperedges:
        graph.add_nodes_from(members)
        graph.add_edges_from(itertools.combinations(sorted(members), 2))
    return graph


same = nx.utils.edges_equal(
    projection(one_meeting).edges(), projection(three_meetings).edges()
)
print(f"Both hypergraphs project to the same triangle: {same}")

# %%
# The **incidence graph** keeps the difference. It is a bipartite graph with
# one node per person, one node per hyperedge, and an edge for every
# membership; :func:`~nx_backbone.hypergraph_to_bipartite` builds it.

incidence_one, people = nb.hypergraph_to_bipartite(one_meeting)
incidence_three, _ = nb.hypergraph_to_bipartite(three_meetings)
for name, graph in [
    ("one meeting", incidence_one),
    ("three meetings", incidence_three),
]:
    print(
        f"{name}: {graph.number_of_nodes()} nodes "
        f"({len(people)} people), {graph.number_of_edges()} memberships"
    )

# sphinx_gallery_start_ignore
assert same
assert max(len(names) for names in events_of.values()) <= 2


def _draw_groups_vs_pairs(mode):
    p = style.palette(mode)
    corners = {"A": (0.0, 1.0), "B": (-0.87, -0.5), "C": (0.87, -0.5)}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    panels = [
        (axes[0], incidence_one, "One meeting of three"),
        (axes[1], incidence_three, "Three meetings of two"),
    ]
    for ax, graph, title in panels:
        pos = dict(corners)
        for node, data in graph.nodes(data=True):
            if data["bipartite"] == 1:
                members = data["members"]
                x = sum(corners[m][0] for m in members) / len(members)
                y = sum(corners[m][1] for m in members) / len(members)
                push = 1.0 if len(members) == 3 else 1.75
                pos[node] = (x * push, y * push)
        for u, v in graph.edges():
            (x0, y0), (x1, y1) = pos[u], pos[v]
            ax.plot([x0, x1], [y0, y1], color=p.kept, lw=style.KEPT_WIDTH, zorder=1)
        groups = [n for n, d in graph.nodes(data=True) if d["bipartite"] == 1]
        _scatter(
            ax,
            (pos[g] for g in groups),
            marker="s",
            s=_SQUARE * 1.6,
            c=_event_color(mode),
            edgecolors=p.surface,
            linewidths=style.RING_WIDTH,
            zorder=3,
        )
        ax.set_title(title)
    axes[2].set_title("Projection of either")
    for u, v in projection(one_meeting).edges():
        (x0, y0), (x1, y1) = corners[u], corners[v]
        axes[2].plot([x0, x1], [y0, y1], color=p.kept, lw=style.KEPT_WIDTH, zorder=1)
    for ax in axes:
        _scatter(
            ax,
            corners.values(),
            s=style.NODE_SIZE * 2.5,
            c=p.node,
            edgecolors=p.surface,
            linewidths=style.RING_WIDTH,
            zorder=4,
        )
        for name, (x, y) in corners.items():
            ax.annotate(
                name,
                (x, y),
                xytext=(0, 9) if y > 0 else (0, -15),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=p.ink,
            )
        ax.set_xlim(-1.75, 1.75)
        ax.set_ylim(-1.35, 1.45)
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.legend(
        handles=[
            _legend_marker("Person", "o", p.node, p.surface, 8, style.RING_WIDTH),
            _legend_marker(
                "Meeting (hyperedge)",
                "s",
                _event_color(mode),
                p.surface,
                8,
                style.RING_WIDTH,
            ),
            _legend_line("Membership, or a pair in the projection", p.kept),
        ],
        loc="lower center",
        ncols=3,
        bbox_to_anchor=(0.5, -0.04),
    )
    return fig


figures.show(
    _draw_groups_vs_pairs,
    "groups-vs-pairs",
    alt="Three panels. Left: three people, A, B and C, joined to one square "
    "meeting node. Middle: the same three people joined in pairs to three "
    "square meeting nodes. Right: a triangle linking A, B and C, the pairwise "
    "projection of both.",
    caption="Figure 1. Two different hypergraphs, drawn as incidence graphs "
    "(people are circles, meetings are squares), have the same pairwise "
    "projection. The projection cannot tell a three-person meeting from three "
    "two-person meetings.",
)
# sphinx_gallery_end_ignore

# %%
# The same loss happens at scale. In the Davis data most pairs of women
# attended at least one event together, so the projection is close to a
# complete graph, and every pair of co-attendees looks alike whether they met
# at a small gathering or among a dozen other guests.

women_projection = projection(hyperedges)
print(
    f"incidence graph: {davis.number_of_nodes()} nodes, "
    f"{davis.number_of_edges()} memberships"
)
print(
    f"projection: {women_projection.number_of_nodes()} women, "
    f"{women_projection.number_of_edges()} pairs, "
    f"density {nx.density(women_projection):.2f}"
)

# sphinx_gallery_start_ignore
assert nx.density(women_projection) > 0.5
davis_pos = style.layout(davis, "davis_southern_women")


def _draw_event_squares(ax, mode, kept=None, labels=True):
    """Event squares (and labels) over a Davis drawing; *kept* marks a backbone."""
    p = style.palette(mode)
    kept = set(events) if kept is None else set(kept)
    _scatter(
        ax,
        (davis_pos[e] for e in events if e in kept),
        marker="s",
        s=_SQUARE,
        c=_event_color(mode),
        edgecolors=p.surface,
        linewidths=style.RING_WIDTH,
        zorder=5,
    )
    _scatter(
        ax,
        (davis_pos[e] for e in events if e not in kept),
        marker="s",
        s=_SQUARE,
        c=p.surface,
        edgecolors=p.node_removed,
        linewidths=1.2,
        zorder=5,
    )
    if labels:
        for e, (x, y) in _event_label_positions().items():
            ax.text(
                x,
                y,
                e,
                ha="center",
                va="center",
                fontsize=7,
                color=p.ink if e in kept else p.muted,
                zorder=6,
            )


def _event_label_positions(gap=0.1):
    """A label spot next to each event square that clears the other labels.

    Labels point away from the center of the drawing; when that spot is too
    close to a label or square already placed, the opposite and the two
    perpendicular directions are tried in turn.
    """
    cx = sum(x for x, _ in davis_pos.values()) / len(davis_pos)
    cy = sum(y for _, y in davis_pos.values()) / len(davis_pos)
    placed = {}
    squares = {e: (float(davis_pos[e][0]), float(davis_pos[e][1])) for e in events}
    for e in events:
        x, y = squares[e]
        dx, dy = x - cx, y - cy
        norm = math.hypot(dx, dy) or 1.0
        dx, dy = dx / norm, dy / norm
        options = [(dx, dy), (-dx, -dy), (-dy, dx), (dy, -dx)]
        spots = [(x + gap * u, y + gap * v) for u, v in options]

        def clearance(spot, e=e):
            others = [*placed.values(), *(s for f, s in squares.items() if f != e)]
            return min(math.dist(spot, other) for other in others)

        fits = [s for s in spots if clearance(s) >= 1.2 * gap]
        placed[e] = fits[0] if fits else max(spots, key=clearance)
    return placed


def _draw_davis_views(mode):
    p = style.palette(mode)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.8))
    style.draw_backbone(
        left, davis, davis, davis_pos, mode=mode, legend=False, kept_width=0.8
    )
    _draw_event_squares(left, mode)
    left.set_title(f"Incidence graph: {len(events)} events")
    women_pos = {w: davis_pos[w] for w in women}
    style.draw_backbone(
        right,
        women_projection,
        women_projection,
        women_pos,
        mode=mode,
        legend=False,
        kept_width=0.5,
    )
    right.set_xlim(left.get_xlim())
    right.set_ylim(left.get_ylim())
    right.set_title(f"Projection: {women_projection.number_of_edges()} pairs")
    fig.legend(
        handles=[
            _legend_marker("Woman", "o", p.node, p.surface, 7, style.RING_WIDTH),
            _legend_marker(
                "Event (hyperedge)",
                "s",
                _event_color(mode),
                p.surface,
                7,
                style.RING_WIDTH,
            ),
            _legend_line("Attended (left), attended together (right)", p.kept),
        ],
        loc="lower center",
        ncols=3,
        bbox_to_anchor=(0.5, -0.03),
    )
    return fig


figures.show(
    _draw_davis_views,
    "davis-views",
    alt=f"Two network drawings with the same positions for the women. Left: the "
    f"incidence graph, {len(women)} women (circles) linked to the "
    f"{len(events)} events (labeled squares) they attended. Right: the pairwise "
    f"projection, a dense web of {women_projection.number_of_edges()} lines "
    f"between the women.",
    caption=f"Figure 2. The Davis data as a hypergraph (left) and as its pairwise "
    f"projection (right, density {nx.density(women_projection):.2f}). Women "
    f"keep their positions across the panels.",
)
# sphinx_gallery_end_ignore

# %%
# Pairwise backbones of the projection, such as the bipartite projection
# methods in :doc:`/tutorials/bipartite_backbone`, answer "which *pairs* are
# tied more than expected?". The methods below answer a different question:
# "which *groups* do we need?". Their output is a **sub-hypergraph**, a subset
# of the original hyperedges.
#
# 2. The MDL backbone
# -------------------
#
# Look at the guest lists and you see redundancy: some small events are
# almost subsets of bigger ones. The MDL backbone of Kirkley, Felippe, Malizia
# & Battiston (2026) turns that observation into a compression problem.
# Imagine sending the hypergraph to someone, bit by bit. You choose a
# **backbone** of **parent** hyperedges, which you send in full. Every other
# hyperedge, a **child**, is sent as an edit of one parent it overlaps: say
# how many nodes they share, which ones, and which outside nodes complete the
# child. A child that nearly matches its parent is cheap to send.
#
# The **description length** :math:`L(G, B)` is the total number of bits
# needed for hypergraph :math:`G` given backbone :math:`B`. It has two parts
# that pull against each other:
#
# - the bits for the parents, which grow with every hyperedge kept, and
# - the bits for the children given their parents, which grow as the backbone
#   shrinks and children must be sent from poorer matches.
#
# The MDL backbone is the :math:`B` with the smallest total. The method has no
# parameters to tune on an unweighted hypergraph: the trade-off is decided by
# the bit counts alone.

result = nb.mdl_hypergraph_backbone(hyperedges)
print(f"kept {len(result.backbone)} of {result.n_input_hyperedges} distinct events")
for parent in result.backbone:
    children = result.assignment.get(parent, [])
    sent_from = ", ".join(label[c] for c in children) or "no children"
    print(f"  {label[parent]:>8} ({len(parent)} guests) encodes: {sent_from}")

# %%
# To see the trade-off, we can write the description length out ourselves.
# With :math:`N` nodes and :math:`\ell` distinct orders, a parent :math:`p`
# costs :math:`\log_2 \ell + \log_2 \binom{N}{|p|}` bits (its size, then which
# nodes), and a child :math:`c` sent from parent :math:`p`, sharing
# :math:`o = |c \cap p|` nodes, costs
#
# .. math::
#
#    \log_2 \ell + \log_2 \min(|c|, |p|) + \log_2 \binom{|p|}{o}
#    + \log_2 \binom{N - |p|}{|c| - o}
#
# bits: its size, the size of the overlap, which parent nodes are shared, and
# which outside nodes complete it. Each child goes to the overlapping parent
# that sends it most cheaply. The function below implements these two
# equations and checks itself against the library.

n_nodes, n_orders = result.n_nodes, result.n_orders


def parent_bits(parent):
    return math.log2(n_orders) + math.log2(math.comb(n_nodes, len(parent)))


def child_bits(child, parent):
    shared = len(child & parent)
    return (
        math.log2(n_orders)
        + math.log2(min(len(child), len(parent)))
        + math.log2(math.comb(len(parent), shared))
        + math.log2(math.comb(n_nodes - len(parent), len(child) - shared))
    )


def description_length(backbone):
    """(bits for the parents, bits for the children) of a candidate backbone."""
    backbone = set(backbone)
    parents = sum(parent_bits(p) for p in backbone)
    children = sum(
        min(child_bits(c, p) for p in backbone if c & p)
        for c in distinct
        if c not in backbone
    )
    return parents, children


print(f"library:   {result.description_length:.2f} bits")
print(f"by hand:   {sum(description_length(result.backbone)):.2f} bits")

# sphinx_gallery_start_ignore
assert math.isclose(sum(description_length(result.backbone)), result.description_length)
assert math.isclose(
    sum(description_length(distinct)), result.baseline_description_length
)
# sphinx_gallery_end_ignore

# %%
# Now compare backbones. Sending every event in full is one extreme: no
# children, but every bit is spent on parents. Keeping only the **maximal**
# events (those not contained in another event) is a purely structural
# reduction. At the other extreme, a greedy cover keeps the largest event and
# then only events that share no guest with what is already kept, which
# leaves almost everything to be sent as children.
#
# Finding the exact optimum is a combinatorial search, so the library uses the
# two greedy optimizers of the paper's Appendix D. ``method="edge"`` adds
# parent-child links in order of how many bits they save; ``method="node"``
# adds parents in order of their total savings. ``method="auto"`` (the
# default, and what the paper does) runs both and keeps whichever gives the
# shorter description.

by_edge = nb.mdl_hypergraph_backbone(hyperedges, method="edge")
by_node = nb.mdl_hypergraph_backbone(hyperedges, method="node")

cover = []
for guests in sorted(distinct, key=len, reverse=True):
    if not any(guests & kept for kept in cover):
        cover.append(guests)

candidates = {
    "Every event": distinct,
    "Maximal events": nb.maximal_hyperedges(hyperedges),
    'MDL, method="edge"': by_edge.backbone,
    'MDL, method="node"': by_node.backbone,
    "Largest-first cover": cover,
}
baseline = result.baseline_description_length
print(
    f"{'backbone':<22}{'kept':>5}{'parents':>10}{'children':>10}{'total':>9}{'ratio':>7}"
)
for name, backbone in candidates.items():
    parents, children = description_length(backbone)
    total = parents + children
    print(
        f"{name:<22}{len(backbone):>5}{parents:>10.1f}{children:>10.1f}"
        f"{total:>9.1f}{total / baseline:>7.2f}"
    )
print(
    f'method="auto" kept {len(result.backbone)} events, {result.description_length:.1f} bits'
)

# sphinx_gallery_start_ignore
assert (
    result.description_length
    <= min(by_edge.description_length, by_node.description_length) + 1e-9
)
assert all(
    result.description_length <= sum(description_length(b)) + 1e-9
    for b in candidates.values()
)
_bars = {name: description_length(b) for name, b in candidates.items()}
assert sum(_bars["Largest-first cover"]) > baseline


def _draw_tradeoff(mode):
    p = style.palette(mode)
    parent_color, child_color = style.categorical(2, mode)
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    names = list(_bars)
    for row, name in enumerate(names):
        parents, children = _bars[name]
        ax.barh(
            row,
            parents,
            height=0.62,
            color=parent_color,
            edgecolor=p.surface,
            linewidth=1.5,
        )
        ax.barh(
            row,
            children,
            left=parents,
            height=0.62,
            color=child_color,
            edgecolor=p.surface,
            linewidth=1.5,
        )
        total = parents + children
        ax.annotate(
            f"{total:.0f} bits, ratio {total / baseline:.2f}",
            (total, row),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=p.ink,
            fontweight="bold"
            if math.isclose(total, result.description_length)
            else None,
        )
    ax.set_yticks(
        range(len(names)),
        [
            f"{name}\n{len(candidates[name])} of {len(distinct)} events"
            for name in names
        ],
    )
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, labelcolor=p.ink)
    ax.set_xlabel("Description length (bits)")
    ax.set_xlim(0, max(sum(v) for v in _bars.values()) * 1.32)
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    ax.legend(
        handles=[
            _legend_marker("Parents, sent in full", "s", parent_color, size=9),
            _legend_marker("Children, sent from a parent", "s", child_color, size=9),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncols=2,
    )
    return fig


_lowest = min(_bars, key=lambda n: sum(_bars[n]))
figures.show(
    _draw_tradeoff,
    "description-length",
    alt="Horizontal stacked bars, one per candidate backbone, split into the "
    "bits spent on parents and the bits spent on children. "
    + " ".join(
        f"{name}: {len(candidates[name])} events kept, {_bars[name][0]:.0f} plus "
        f"{_bars[name][1]:.0f} bits."
        for name in _bars
    ),
    caption=f"Figure 3. Description length of five candidate backbones of the "
    f"Davis events. Keeping every event spends all bits on parents; the "
    f"largest-first cover spends almost all of them on children. The shortest "
    f"total ({_lowest}, {sum(_bars[_lowest]):.0f} bits) is the MDL backbone. "
    f"The ratio is the total divided by the cost of sending every event in "
    f"full; the table above gives the same values.",
)
# sphinx_gallery_end_ignore

# %%
# For the MDL backbone, the ratio in the last column is the paper's **inverse
# compression ratio** :math:`\eta = L(G, B^*) / L(G, G)`: the description
# length of the MDL backbone divided by that of sending everything. It lies
# between 0 and 1. A value near 0 means the hypergraph is highly redundant; a
# value of 1 means no backbone shortens the description. A poor backbone can
# do worse than no backbone at all: the largest-first cover's ratio is above
# 1. The MDL ratio is stored on the result, and
# :func:`~nx_backbone.hypergraph_compression_ratio` returns it directly.

print(f"eta = {result.compression_ratio:.3f}")
print(f"eta = {nb.hypergraph_compression_ratio(hyperedges):.3f}")

# %%
# Figure 4 puts the MDL backbone back on the incidence graph. The kept events
# are the ones the others can be sent from cheaply; the pruned events are
# still in the data, encoded as edits of the kept ones (the printout above
# lists which parent encodes which child). The backbone holds guest lists,
# so map them back to the events:

kept_events = [e for e in events if guest_lists[e] in set(result.backbone)]
pruned_events = [e for e in events if e not in kept_events]
print(f"kept:   {', '.join(kept_events)}")
print(f"pruned: {', '.join(pruned_events)}")

# sphinx_gallery_start_ignore
_kept_events = kept_events
_davis_backbone = davis.edge_subgraph(
    (w, e) for e in _kept_events for w in davis[e]
).copy()


def _draw_mdl_backbone(mode):
    p = style.palette(mode)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    counts = style.draw_backbone(
        ax, davis, _davis_backbone, davis_pos, mode=mode, legend=False, kept_width=1.1
    )
    _draw_event_squares(ax, mode, kept=_kept_events)
    dropped_women = len(women) - len(set(_davis_backbone) & set(women))
    handles = [
        _legend_marker(
            "Event in the backbone",
            "s",
            _event_color(mode),
            p.surface,
            7,
            style.RING_WIDTH,
        ),
        _legend_marker("Event pruned", "s", p.surface, p.node_removed, 7, 1.2),
        _legend_line("Attended a kept event", p.kept),
        _legend_line("Attended a pruned event", p.removed),
        _legend_marker("Woman", "o", p.node, p.surface, 7, style.RING_WIDTH),
    ]
    if dropped_women:
        handles.append(
            _legend_marker("Woman at pruned events only", "o", p.node_removed, size=6)
        )
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncols=2)
    ax.set_title(f"MDL backbone: {len(_kept_events)} of {len(events)} events")
    return fig, counts


_mdl_counts = style.backbone_counts(davis, _davis_backbone)
figures.show(
    lambda mode: _draw_mdl_backbone(mode)[0],
    "mdl-backbone",
    thumbnail=True,
    alt=f"The Davis incidence graph with the {len(_kept_events)} backbone events "
    f"({', '.join(_kept_events)}) drawn as filled squares with solid lines to "
    f"their guests, and the other events as hollow squares with faint lines.",
    caption=f"Figure 4. The MDL backbone of the Davis events. It keeps "
    f"{len(_kept_events)} of {len(events)} events (counting both events of a "
    f"shared guest list) and {_mdl_counts.edges_kept} of "
    f"{_mdl_counts.edges_total} memberships; "
    f"{len(set(_davis_backbone) & set(women))} of {len(women)} women attended "
    f"at least one kept event.",
)
# sphinx_gallery_end_ignore

# %%
# **Weighted hypergraphs.** When hyperedges carry integer weights, such as how
# many times a group met, ``weights=`` and a single knob ``gamma`` in
# :math:`(0, 1]` let the weights pull heavy hyperedges into the backbone;
# ``gamma=1`` ignores them. :doc:`/tutorials/hypergraph_backbone` shows the
# call.
#
# **Reporting.** :func:`~nx_backbone.describe_backbone` writes a methods
# paragraph with the citation, from the same reference list the library uses
# everywhere:

print(
    nb.describe_backbone(
        "mdl_hypergraph_backbone", original=hyperedges, backbone=result
    )
)

# %%
# 3. The local MDL backbone
# -------------------------
#
# The MDL backbone is *global*: an event is pruned whenever any overlapping
# event, anywhere in the data, can send it cheaply. That can be unfair to a
# woman whose only events are pruned. The **local** variant (Appendix E of the
# paper) is the higher-order cousin of the disparity filter. Each woman's
# **neighborhood** is the set of events she attended. The MDL backbone is
# found for every neighborhood on its own, so each woman "votes" for the
# events in her neighborhood backbone. An **aggregation rule** then decides
# how many votes an event needs:
#
# - ``"union"``: at least one of its guests (every woman keeps at least one
#   event, but the backbone is dense);
# - ``"majority"`` (the default, used for Table II of the paper): at least
#   half of its guests;
# - ``"all"``: every guest (the most conservative rule).

local = {
    rule: nb.local_mdl_hypergraph_backbone(hyperedges, aggregation=rule)
    for rule in ("union", "majority", "all")
}
print(f"{'rule':<10}{'kept':>6}{'eta (local)':>13}")
print(f"{'global':<10}{len(result.backbone):>6}{result.compression_ratio:>13.3f}")
for rule, res in local.items():
    print(f"{rule:<10}{len(res.backbone):>6}{res.compression_ratio:>13.3f}")

# %%
# The per-woman results are on the result object. For example, the first
# woman's neighborhood and its backbone:

first = women[0]
neighborhood = [label[g] for g in distinct if first in g]
own_backbone = [label[g] for g in local["majority"].local_backbones[first]]
print(
    f"{first} attended {len(neighborhood)} distinct events: {', '.join(neighborhood)}"
)
print(f"her neighborhood backbone keeps: {', '.join(own_backbone)}")
print(f"in {local['majority'].local_description_lengths[first]:.1f} bits")

# sphinx_gallery_start_ignore
assert len(local["union"]) >= len(local["majority"]) >= len(local["all"])
assert local["union"].local_backbones == local["all"].local_backbones
assert len(local["majority"]) >= len(result) and all(
    r.compression_ratio >= result.compression_ratio for r in local.values()
)
assert any(
    g not in set(result.backbone)
    and g in set(local["majority"].backbone)
    and 2 * sum(g in set(local["union"].local_backbones[w]) for w in g) > len(g)
    for g in distinct
)
_rows = sorted(distinct, key=lambda g: (-len(g), events.index(events_of[g][0])))
_rules = ["union", "majority", "all"]


def _draw_local_votes(mode):
    p = style.palette(mode)
    n_women = len(women)
    decision_x = {"votes": n_women + 1.0}
    for j, rule in enumerate([*_rules, "global"]):
        decision_x[rule] = n_women + 2.6 + 1.3 * j
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    neighborhood_backbone = local["union"].local_backbones
    global_kept = set(result.backbone)
    for row, g in enumerate(_rows):
        votes = 0
        for col, w in enumerate(women):
            if w not in g:
                continue
            in_own = g in set(neighborhood_backbone[w])
            votes += in_own
            ax.scatter(
                col,
                row,
                marker="s",
                s=46,
                c=p.kept if in_own else p.surface,
                edgecolors=p.surface if in_own else p.muted,
                linewidths=0.6 if in_own else 0.9,
                zorder=3,
            )
        ax.annotate(
            f"{votes}/{len(g)}",
            (decision_x["votes"], row),
            ha="center",
            va="center",
            fontsize=8,
            color=p.ink,
        )
        for rule in [*_rules, "global"]:
            backbone = global_kept if rule == "global" else set(local[rule].backbone)
            kept = g in backbone
            ax.scatter(
                decision_x[rule],
                row,
                marker="o",
                s=40,
                c=p.node if kept else p.surface,
                edgecolors=p.node if kept else p.node_removed,
                linewidths=1.2,
                zorder=3,
            )
    for x in [n_women - 0.5, decision_x["votes"] + 0.8]:
        ax.axvline(x, color=p.hairline, lw=0.75, zorder=0)
    ax.set_yticks(range(len(_rows)), [f"{label[g]} ({len(g)})" for g in _rows])
    xticks = [*range(n_women), *decision_x.values()]
    xlabels = [
        *(_first_name(w) for w in women),
        "Votes",
        "Union",
        "Majority",
        "All",
        "Global MDL",
    ]
    ax.set_xticks(xticks, xlabels, rotation=90)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelcolor=p.ink, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.8, max(decision_x.values()) + 0.8)
    ax.set_ylim(len(_rows) - 0.4, -0.6)
    ax.set_ylabel("Event (guests)", color=p.ink)
    ax.legend(
        handles=[
            _legend_marker(
                "In this guest's neighborhood backbone", "s", p.kept, size=7
            ),
            _legend_marker(
                "Attended, not in her neighborhood backbone",
                "s",
                p.surface,
                p.muted,
                7,
                0.9,
            ),
            _legend_marker("Event kept by the rule", "o", p.node, p.node, 7, 1.2),
            _legend_marker(
                "Event dropped by the rule", "o", p.surface, p.node_removed, 7, 1.2
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncols=2,
    )
    return fig


figures.show(
    _draw_local_votes,
    "local-votes",
    alt=f"A grid with one row per distinct guest list ({len(_rows)} rows) and "
    f"one column per woman ({len(women)} columns). A filled square marks a "
    f"woman whose neighborhood backbone keeps that event; a hollow square marks "
    f"an event she attended that her neighborhood backbone drops. Columns on the "
    f"right give the vote count and whether the union "
    f"({len(local['union'].backbone)} kept), majority "
    f"({len(local['majority'].backbone)} kept), all "
    f"({len(local['all'].backbone)} kept) and global MDL "
    f"({len(result.backbone)} kept) backbones keep the event.",
    caption="Figure 5. How the local MDL backbone votes. Each row is a guest "
    "list (events with identical lists are merged), each column a woman; "
    "filled squares are her votes. The union rule needs one vote, the majority "
    "rule needs votes from at least half of the guests, the all rule from every "
    "guest. "
    "Filled circles mark the events each rule keeps, next to the global MDL "
    "backbone for comparison.",
)
# sphinx_gallery_end_ignore

# %%
# Read Figure 5 row by row. An event that is the natural parent in many of
# its guests' neighborhoods collects many votes; an event that sits almost
# inside a bigger one collects few or none, whatever its size. The global
# backbone (last column) can still drop an event that most of its guests
# vote for, when an event elsewhere sends it cheaply; the majority rule keeps
# such events. Choose the local backbone when each person's important groups
# matter more than overall compression, and expect it to be denser and to
# compress less than the global one.
#
# 4. Statistically validated hypergraphs and cores
# ------------------------------------------------
#
# The MDL backbone asks which groups *explain the structure*. A different
# question is which groups *recur more often than chance*. That needs
# repeated observations, and in the Davis data no guest list repeats more
# than twice. So for this section we generate a synthetic **meeting log**:
# many small meetings drawn at random among colleagues, plus a few planted
# groups that meet again and again:
#
# - a pair, a trio and a quartet, each meeting several times on their own;
# - a trio that meets several times too, but every time with a different
#   fourth colleague, so it never appears as a hyperedge by itself.

rng = random.Random(7)
colleagues = list(range(60))
meeting_log = [tuple(rng.sample(colleagues, rng.choice([2, 3, 4]))) for _ in range(600)]
planted = {(0, 1): 8, (2, 3, 4): 6, (5, 6, 7, 8): 5}
for group, times in planted.items():
    meeting_log += [group] * times
hidden_core = (9, 10, 11)
others = [c for c in colleagues if c not in hidden_core]
meeting_log += [(*hidden_core, rng.choice(others)) for _ in range(8)]
rng.shuffle(meeting_log)
print(f"{len(meeting_log)} meetings among {len(colleagues)} colleagues")

# %%
# The **statistically validated hypergraph** (SVH) of Musciotto, Battiston &
# Mantegna (2021) tests every observed hyperedge against a **null model**, a
# random baseline that keeps how active each person is. Within each order
# :math:`k`, with :math:`n_k` meetings of that size and :math:`d_i` of them
# including person :math:`i`, the number of times a group meets is compared
# with a binomial distribution whose mean is
# :math:`n_k \prod_i (d_i / n_k)`. The **p-value** is the chance of seeing the
# group meet at least as often under that baseline. Because thousands of
# groups are tested, the p-values are controlled with the Benjamini-Hochberg
# false discovery rate (FDR) at level ``alpha``, corrected for the number of
# possible groups of each order.

svh = nb.statistically_validated_hypergraph(meeting_log, alpha=0.01)


def expected_meetings(group, log):
    """Mean of the SVH null: n_k * prod(d_i / n_k) over meetings of size k."""
    same_order = [frozenset(m) for m in log if len(set(m)) == len(group)]
    n_k = len(same_order)
    expected = n_k
    for person in group:
        expected *= sum(person in m for m in same_order) / n_k
    return expected


print(f"validated {len(svh.validated)} of {len(svh.pvalues)} tested hyperedges")
print(f"{'group':<16}{'met':>5}{'expected':>10}{'p-value':>11}  validated")
for group in [*planted, hidden_core]:
    key = frozenset(group)
    if key in svh.pvalues:
        print(
            f"{group!s:<16}{svh.counts[key]:>5}{expected_meetings(group, meeting_log):>10.3f}"
            f"{svh.pvalues[key]:>11.1e}  {key in svh.validated}"
        )
    else:
        print(f"{group!s:<16}{0:>5}{'':>10}{'not tested':>11}  {key in svh.validated}")

# sphinx_gallery_start_ignore
assert frozenset((2, 3, 4)) in svh.validated
assert frozenset((5, 6, 7, 8)) in svh.validated
assert frozenset((0, 1)) not in svh.validated
assert frozenset(hidden_core) not in svh.pvalues
_orders = [2, 3, 4]
_order_names = {2: "Pairs", 3: "Trios", 4: "Quartets"}
_jitter = random.Random(0)
_points = {
    k: [
        (
            svh.counts[g] + _jitter.uniform(-0.18, 0.18),
            -math.log10(max(svh.pvalues[g], 1e-300)),
            g in set(svh.validated),
            g,
        )
        for g in svh.pvalues
        if len(g) == k
    ]
    for k in _orders
}


def _draw_svh(mode):
    p = style.palette(mode)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
    for ax, k in zip(axes, _orders):
        points = _points[k]
        _scatter(
            ax,
            ((x, y) for x, y, ok, _ in points if not ok),
            s=16,
            facecolors="none",
            edgecolors=p.muted,
            linewidths=0.7,
            zorder=2,
        )
        _scatter(
            ax,
            ((x, y) for x, y, ok, _ in points if ok),
            s=36,
            c=p.kept,
            edgecolors=p.surface,
            linewidths=style.RING_WIDTH,
            zorder=3,
        )
        for x, y, _, g in points:
            if tuple(sorted(g)) in planted:
                ax.annotate(
                    "{" + ", ".join(map(str, sorted(g))) + "}",
                    (x, y),
                    xytext=(-6, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=8,
                    color=p.ink,
                )
        n_valid = sum(ok for _, _, ok, _ in points)
        ax.set_title(
            f"{_order_names[k]}: {n_valid} of {len(points)} validated", fontsize=9
        )
        ax.set_xlabel("Times the group met")
        ax.grid(axis="y")
        ax.set_xlim(0, max(x for x, *_ in points) + 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("−log₁₀ p-value")
    fig.legend(
        handles=[
            _legend_marker(
                "Validated (FDR, alpha = 0.01)",
                "o",
                p.kept,
                p.surface,
                7,
                style.RING_WIDTH,
            ),
            _legend_marker("Not validated", "o", "none", p.muted, 5.5, 0.7),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        ncols=2,
    )
    return fig


figures.show(
    _draw_svh,
    "svh",
    alt="Three scatter plots, for pairs, trios and quartets, of how many times "
    "each observed group met (horizontal) against minus log ten of its SVH "
    "p-value (vertical). Almost all groups met once and sit near zero. "
    + " ".join(
        f"The planted group {set(g)} met {svh.counts[frozenset(g)]} times and is "
        f"{'validated' if frozenset(g) in svh.validated else 'not validated'}."
        for g in planted
    ),
    caption="Figure 6. Every hyperedge of the meeting log tested by the SVH, one "
    "panel per order. Filled points are validated, hollow points are not; "
    "points are jittered horizontally so that groups with the same count do "
    "not hide each other. The planted groups are labeled.",
)
# sphinx_gallery_end_ignore

# %%
# The printout and Figure 6 show two things. First, repetition is judged
# relative to the order: a quartet that meets a few times is far more
# surprising than a pair that meets a few more times, because random pairs
# overlap often while random quartets almost never do (compare the expected
# counts in the table). Second, the trio that always came with a fourth
# person is invisible to the SVH, since it never appears as a hyperedge of
# its own.
#
# The **statistically validated cores** (SVC) close that gap. They test every
# *sub-group* of every hyperedge, counting how many meetings contain it, from
# the largest order down; once a group is validated its subsets are not
# tested again, so significance goes to the largest validated **core**.

svc = nb.statistically_validated_cores(meeting_log, alpha=0.01)
print(f"tested {len(svc.pvalues)} groups, validated {len(svc.validated)} cores")
for core in svc.validated:
    alone = meeting_log.count(tuple(sorted(core)))
    print(
        f"  {sorted(core)!s:<14} in {svc.counts[core]} meetings "
        f"(alone in {alone}), p = {svc.pvalues[core]:.1e}"
    )

# sphinx_gallery_start_ignore
assert frozenset(hidden_core) in svc.validated
# sphinx_gallery_end_ignore

# %%
# The hidden trio is now found. SVC enumerates sub-groups, up to
# :math:`2^k` for a hyperedge of size :math:`k`, so on data with large
# hyperedges set ``max_order`` (a guard, ``max_groups``, stops runaway
# enumerations). Both functions need SciPy.
#
# 5. Structural helpers
# ---------------------
#
# Three cheap, parameter-light tools complete the picture. They return plain
# lists of hyperedges and need nothing beyond NetworkX.
#
# - :func:`~nx_backbone.maximal_hyperedges` keeps the **toplexes**,
#   hyperedges not contained in any other. This removes exact nesting only;
#   the MDL backbone also removes near-nesting and overlap.
# - :func:`~nx_backbone.order_filter` keeps hyperedges by order, for
#   example only the large gatherings.
# - :func:`~nx_backbone.s_components` groups hyperedges into
#   **s-components**: two hyperedges are *s-adjacent* when they share at
#   least ``s`` nodes, and an s-component is a connected set of them. Raising
#   ``s`` shows at what overlap the hypergraph falls apart.

maximal = nb.maximal_hyperedges(hyperedges)
print(
    f"maximal events: {len(maximal)} of {len(distinct)}: "
    f"{', '.join(label[g] for g in maximal)}"
)
in_both = [label[g] for g in result.backbone if g in set(maximal)]
print(f"MDL backbone events that are also maximal: {', '.join(in_both) or 'none'}")

large = nb.order_filter(hyperedges, min_order=8)
print(f"events with at least 8 guests: {', '.join(label[g] for g in large)}")

print(f"{'s':>3}{'components':>12}{'largest':>9}")
for s in range(1, max(sizes) + 1):
    components = nb.s_components(hyperedges, s=s)
    print(f"{s:>3}{len(components):>12}{len(components[0]):>9}")
    if len(components) == len(distinct):
        break

# %%
# 6. Moving hypergraphs in and out
# --------------------------------
#
# Hypergraph data usually lives in a dedicated library. The **Hypergraph
# Interchange Format** (HIF) is a JSON standard that xgi, HyperNetX,
# HypergraphX and HAT all read and write, and
# :func:`~nx_backbone.write_hif` and :func:`~nx_backbone.read_hif`
# need only the standard library. Here the distinct guest lists go out with a
# weight (how many events share the list) and come back unchanged.

weights = [len(events_of[g]) for g in distinct]
buffer = io.StringIO()
nb.write_hif(distinct, buffer, weights=weights)
text = buffer.getvalue()
hif = json.loads(text)
print(f"{len(text)} characters of HIF JSON with keys {sorted(hif)}")
print(f"first incidences: {hif['incidences'][:2]}")

read_back, read_weights = nb.read_hif(io.StringIO(text), return_weights=True)
print(f"round trip preserved hyperedges: {read_back == distinct}")
print(f"round trip preserved weights: {read_weights == weights}")
again = nb.mdl_hypergraph_backbone(read_back)
print(f"same MDL backbone after the round trip: {again.backbone == result.backbone}")

# sphinx_gallery_start_ignore
assert read_back == distinct and read_weights == weights
assert again.backbone == result.backbone
# sphinx_gallery_end_ignore

# %%
# Pass a path instead of a buffer to write a file. The library-specific
# adapters convert directly. They import their library only when called, and
# the libraries are optional extras (for example,
# ``pip install "nx-backbone[xgi]"``), so the snippets below are shown
# but not run when this page is built.
#
# .. code-block:: python
#
#    import xgi
#
#    H = xgi.read_hif("events.hif.json")        # or any xgi.Hypergraph
#    result = nb.mdl_hypergraph_backbone(nb.from_xgi(H))
#    backbone_xgi = nb.to_xgi(result.backbone)  # back to an xgi.Hypergraph
#
# .. code-block:: python
#
#    import hypernetx as hnx
#
#    H = hnx.Hypergraph({"E1": ["Ann", "Bea"], "E2": ["Bea", "Cy", "Dee"]})
#    result = nb.mdl_hypergraph_backbone(nb.from_hypernetx(H))
#    backbone_hnx = nb.to_hypernetx(result.backbone)
#
# .. code-block:: python
#
#    # HypergraphX keeps multiplicities as weights; SVH treats them as counts.
#    edges, counts = nb.from_hypergraphx(H_hgx, return_weights=True)
#    svh = nb.statistically_validated_hypergraph(edges, weights=counts)
#    validated_hgx = nb.to_hypergraphx(svh.validated)
#
#    # HAT (pip name HypergraphAnalysisToolbox) works the same way:
#    edges = nb.from_hat(H_hat)
#    backbone_hat = nb.to_hat(nb.mdl_hypergraph_backbone(edges).backbone)
#
# Each ``from_*`` function also takes ``return_weights=True``, and each
# ``to_*`` function takes ``weights=``.
#
# Where next
# ----------
#
# - :doc:`/api/hypergraph`: every hypergraph function, with its parameters
#   and the choices the implementation makes where the papers leave details
#   open.
# - :doc:`/tutorials/hypergraph_backbone`: the short, code-first reference
#   for the same functions, including weighted MDL backbones.
# - :doc:`/tutorials/bipartite_backbone`: backbones of the pairwise
#   projection, for when the question really is about pairs.
# - :doc:`/user_guide/choosing_method` and :doc:`/user_guide/evaluation`:
#   choosing among methods and judging the result.
# - :doc:`/tutorials/index`: the other tutorials.
#
# References
# ----------
#
# - Davis, A., Gardner, B. B., & Gardner, M. R. (1941). *Deep South*.
#   University of Chicago Press.
# - Kirkley, A., Felippe, H., Malizia, F., & Battiston, F. (2026). Hypergraph
#   backboning. arXiv:2606.00893.
# - Musciotto, F., Battiston, F., & Mantegna, R. N. (2021). Detecting
#   informative higher-order interactions in statistically validated
#   hypergraphs. *Communications Physics*, 4, 218.
