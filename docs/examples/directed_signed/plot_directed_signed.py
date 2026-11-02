"""
Directed and signed backbones
=============================

Two switches change what a backbone *means* more than any other: the
direction of the edges and the ``signed=True`` option. A directed edge has
two ends that see it differently, so "is this edge significant?" becomes
"significant *to whom*?". A signed backbone keeps edges that are
significantly *weak* as well as significantly strong, and "weak" means
something different under each null model. Both are where results most often
surprise users.

This tutorial builds a small directed network whose structure we know,
because we planted it, and reads its backbones end to end:

1. what a directed null model tests at the source and at the target, and why
   :func:`~nx_backbone.disparity_filter` and :func:`~nx_backbone.lans_filter`
   test both ends;
2. the upper and lower tails of a test, and what "significantly weak" means
   under a local and a joint null model;
3. a signed backbone of the network, read edge by edge;
4. how each scorer assigns a sign, including exact ties, which are signed
   -1, and the lower tail of the signed LANS test;
5. pointers to signed bipartite projections and to matching R.

Every number below is computed when the documentation is built.
"""

# sphinx_gallery_thumbnail_path = "_static/example_figures/directed_signed/thumbnail.png"

# %%
# Setup
# -----
#
# Every figure is drawn twice, once for the light theme and once for the
# dark theme, and the page shows the one matching your theme. The plotting
# code is left out of the page; it is in the script and notebook you can
# download.

import math
from fractions import Fraction

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats

import nx_backbone as nb

# sphinx_gallery_start_ignore
# isort: split
from _figures import style
from _figures.tutorial import TutorialFigures
from matplotlib.lines import Line2D

figures = TutorialFigures("directed_signed")
# sphinx_gallery_end_ignore

ALPHA = 0.05


def table(header, rows):
    """Print rows as an aligned plain-text table."""
    cells = [[str(c) for c in header]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    for i, row in enumerate(cells):
        print(
            "  ".join(
                c.rjust(w) if j else c.ljust(w)
                for j, (c, w) in enumerate(zip(row, widths))
            )
        )
        if i == 0:
            print("  ".join("-" * w for w in widths))


# %%
# A network with known asymmetries
# --------------------------------
#
# Our data is the number of messages sent within a small research group over
# one term. It is a **directed** network: an edge *u* → *v* is a different
# edge from *v* → *u*, and its **weight** counts the messages *u* sent to
# *v*. *u* is the edge's **source** and *v* its **target**.
#
# Most of it is background: each member writes to the group's lead and to a
# few other members, in random amounts. On top of it we plant four
# relationships whose answer we know in advance:
#
# - **Admin → Lead.** The administrator sends almost all of their messages
#   to the lead, who receives many messages from everyone. The edge
#   dominates the *sender's* activity, not the *receiver's*.
# - **List → Visitor.** A mailing list sends the same digest to everybody,
#   and a visitor receives little else. The edge dominates the *receiver's*
#   inbox, not the *sender's* output.
# - **Lead → Admin.** The lead writes a lot, to everyone, but only once to
#   the administrator.
# - **m14 → Lead.** One member writes generously to every other member, but
#   only once to the lead, whom everybody else writes to.

rng = np.random.default_rng(7)
members = [f"m{i:02d}" for i in range(1, 15)]
G = nx.DiGraph()

# Background: members m01-m13 write to three other members and to the lead.
for u in members[:-1]:
    others = [m for m in members if m != u]
    for v in rng.choice(others, size=3, replace=False):
        G.add_edge(u, str(v), weight=int(rng.integers(1, 7)))
    G.add_edge(u, "Lead", weight=int(rng.integers(15, 30)))
# m14 writes to every other member, and only once to the lead.
for v in members[:-1]:
    G.add_edge("m14", v, weight=int(rng.integers(5, 10)))
G.add_edge("m14", "Lead", weight=1)
# The lead writes a lot to every member, and once to the administrator.
for v in members:
    G.add_edge("Lead", v, weight=int(rng.integers(30, 60)))
G.add_edge("Lead", "Admin", weight=1)
# The administrator writes mostly to the lead.
G.add_edge("Admin", "Lead", weight=40)
for v in rng.choice(members, size=3, replace=False):
    G.add_edge("Admin", str(v), weight=1)
# The list sends the same digest to every member and to the visitor, who
# hears from a few members and writes to the lead.
for v in [*members, "Visitor"]:
    G.add_edge("List", v, weight=10)
for v in rng.choice(members, size=4, replace=False):
    G.add_edge(str(v), "Visitor", weight=1)
G.add_edge("Visitor", "Lead", weight=2)

PLANTED = [("Admin", "Lead"), ("List", "Visitor"), ("Lead", "Admin"), ("m14", "Lead")]
reciprocal = sum(G.has_edge(v, u) for u, v in G.edges) // 2
print(
    f"{G.number_of_nodes()} people, {G.number_of_edges()} directed edges, "
    f"{reciprocal} pairs linked both ways, "
    f"{G.size(weight='weight')} messages"
)

# %%
# In a directed network every node has two degrees and two strengths. Its
# **out-degree** is the number of people it writes to and its
# **out-strength** the number of messages it sends; its **in-degree** and
# **in-strength** count the senders and the messages it receives.

table(
    ["node", "out-degree", "out-strength", "in-degree", "in-strength"],
    [
        [
            n,
            G.out_degree(n),
            G.out_degree(n, weight="weight"),
            G.in_degree(n),
            G.in_degree(n, weight="weight"),
        ]
        for n in ["Lead", "Admin", "List", "Visitor", "m14"]
    ],
)

# sphinx_gallery_start_ignore
assert G.in_degree("List") == 0 and G.out_degree("List") == len(members) + 1
assert G.in_degree("Admin") == 1
assert all(G.has_edge(*e) for e in PLANTED)
assert reciprocal > 0
# sphinx_gallery_end_ignore

# %%
# Two ends of a directed edge
# ---------------------------
#
# The disparity filter (Serrano et al., 2009) asks whether an edge carries
# a larger share of a node's strength than chance would give it. Its **null
# model**, the definition of "chance", is that a node splits its strength
# over its *k* links at random, every split equally likely. The share
# :math:`x = w / s` of a link of weight *w* at a node of strength *s* then
# has the **upper-tail p-value**
#
# .. math::
#
#    P(X \geq x) = (1 - x)^{k - 1},
#
# the probability of a share at least that large. A node with a single link
# has nothing to compare it with, so its test gives 1.
#
# In a directed network each edge is seen from two nodes, and each sees a
# different budget. The **source test** asks whether the edge is a large
# share of what the *source sends* (its out-strength, over its out-degree);
# the **target test** asks whether it is a large share of what the *target
# receives* (its in-strength, over its in-degree). Here is that computation
# for the four planted edges, next to the library's p-value:


def disparity_end(w, strength, degree):
    """Upper and lower tails of the disparity test at one end of an edge."""
    if degree <= 1:
        return 1.0, 1.0
    upper = (1.0 - w / strength) ** (degree - 1)
    return upper, 1.0 - upper


def disparity_ends(G, u, v):
    """The source test (u's out-links) and the target test (v's in-links)."""
    w = G[u][v]["weight"]
    source = disparity_end(w, G.out_degree(u, weight="weight"), G.out_degree(u))
    target = disparity_end(w, G.in_degree(v, weight="weight"), G.in_degree(v))
    return source, target


H = nb.disparity_filter(G)
ends = {(u, v): disparity_ends(G, u, v) for u, v in G.edges}
table(
    ["edge", "source test", "target test", "disparity_pvalue"],
    [
        [
            f"{u} -> {v}",
            f"{ends[u, v][0][0]:.2g}",
            f"{ends[u, v][1][0]:.2g}",
            f"{H[u][v]['disparity_pvalue']:.2g}",
        ]
        for u, v in PLANTED
    ],
)

# sphinx_gallery_start_ignore
# The library's directed p-value is the smaller of the two end tests.
for (u, v), (source, target) in ends.items():
    assert math.isclose(
        H[u][v]["disparity_pvalue"],
        min(source[0], target[0]),
        rel_tol=1e-12,
        abs_tol=1e-300,
    )
# Admin -> Lead: significant from the source only; List -> Visitor: from the
# target only.
assert ends["Admin", "Lead"][0][0] < ALPHA <= ends["Admin", "Lead"][1][0]
assert ends["List", "Visitor"][1][0] < ALPHA <= ends["List", "Visitor"][0][0]
# sphinx_gallery_end_ignore

# %%
# The library computes both tests and keeps the *smaller* p-value, so an
# edge is kept when it is significant to *either* end (this page checks
# that for every edge when it is built). The :func:`~nx_backbone.disparity_filter` docstring describes this
# rule, following Serrano et al.: the source is tested on its out-links and
# the target on its in-links.
# :func:`~nx_backbone.lans_filter` (Foti et al., 2011) does the same with
# its nonparametric test: it compares the edge with the source's other
# out-links and with the target's other in-links.
#
# Why both? Because each end alone loses a planted edge. Admin → Lead is
# most of what the administrator sends, but a small part of what the lead
# receives: only the source test sees it. List → Visitor is a sliver of the
# list's output but most of the visitor's inbox: only the target test sees
# it. Here is the backbone that the two tests keep together:

src = {e: ends[e][0][0] for e in ends}  # source test p-values
tgt = {e: ends[e][1][0] for e in ends}  # target test p-values
quadrant = {
    "both": [e for e in G.edges if src[e] < ALPHA and tgt[e] < ALPHA],
    "source only": [e for e in G.edges if src[e] < ALPHA <= tgt[e]],
    "target only": [e for e in G.edges if tgt[e] < ALPHA <= src[e]],
    "neither": [e for e in G.edges if src[e] >= ALPHA and tgt[e] >= ALPHA],
}
table(["significant at alpha", "edges"], [[k, len(v)] for k, v in quadrant.items()])
backbone = nb.threshold_filter(H, "disparity_pvalue", ALPHA)
print(
    f"disparity backbone: {backbone.number_of_edges()} of {G.number_of_edges()} edges"
)

# sphinx_gallery_start_ignore
assert backbone.number_of_edges() == G.number_of_edges() - len(quadrant["neither"])
assert quadrant["source only"] and quadrant["target only"]
P_FLOOR = 1e-4
assert min(min(src.values()), min(tgt.values())) >= P_FLOOR


def radial_layout():
    """The lead in the middle, the members on a ring, the others outside it.

    A fixed layout of this network (rather than ``style.layout``'s spring
    layout, which piles every node around the lead) keeps the arrows apart.
    """
    pos = {"Lead": np.array([0.0, 0.0])}
    step = 2 * math.pi / len(members)
    for i, m in enumerate(members):
        angle = math.pi / 2 + step * (i + 0.5)
        pos[m] = np.array([math.cos(angle), math.sin(angle)])
    for node, angle in (("List", 90), ("Visitor", -180 / len(members)), ("Admin", 270)):
        a = math.radians(angle)
        pos[node] = 1.55 * np.array([math.cos(a), math.sin(a)])
    return pos


pos = radial_layout()
LABEL_NODES = {"Lead", "Admin", "List", "Visitor", "m14"}
DASH = (0, (3.0, 2.0))


def draw_arrows(ax, edges, color, width, style_="solid", alpha=1.0):
    """Curved arrows, so that the two directions of a pair do not overlap."""
    if not edges:
        return
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        ax=ax,
        edge_color=color,
        width=width,
        style=style_,
        alpha=alpha,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=8 if width >= style.KEPT_WIDTH else 5,
        connectionstyle="arc3,rad=0.12",
        node_size=80,
        min_source_margin=0,
        min_target_margin=3,
    )


def draw_nodes(ax, mode, kept_nodes):
    p = style.palette(mode)
    on = [n for n in G if n in kept_nodes]
    off = [n for n in G if n not in kept_nodes]
    for nodes, color, ring in (
        (off, p.node_removed, 0),
        (on, p.node, style.RING_WIDTH),
    ):
        if nodes:
            xy = np.array([pos[n] for n in nodes])
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                s=style.NODE_SIZE * 1.6,
                c=color,
                edgecolors=p.surface,
                linewidths=ring,
                zorder=4,
            )
    for n in G:
        x, y = pos[n]
        r = math.hypot(x, y)
        dx, dy = (0.0, -1.0) if r == 0 else (x / r, y / r)
        ax.annotate(
            n,
            (x, y),
            xytext=(9 * dx, 9 * dy),
            textcoords="offset points",
            ha="center" if abs(dx) < 0.4 else ("left" if dx > 0 else "right"),
            va="center" if abs(dy) < 0.4 else ("bottom" if dy > 0 else "top"),
            fontsize=8.5 if n in LABEL_NODES else 7.5,
            fontweight="bold" if n in LABEL_NODES else "normal",
            color=p.ink if n in LABEL_NODES else p.muted,
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": p.surface,
                "ec": "none",
                "alpha": 0.85,
            }
            if r == 0
            else None,
            zorder=5,
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.06)
    ax.set_axis_off()


def draw_directed_backbone(ax, mode, kept, signs=None, title=None):
    """Kept edges as solid (or, when negative, dashed) arrows over the faint rest."""
    p = style.palette(mode)
    kept = set(kept)
    removed = [e for e in G.edges if e not in kept]
    draw_arrows(ax, removed, p.removed, style.REMOVED_WIDTH)
    if signs is None:
        draw_arrows(ax, sorted(kept), p.kept, style.KEPT_WIDTH)
    else:
        draw_arrows(
            ax, sorted(e for e in kept if signs[e] > 0), p.positive, style.KEPT_WIDTH
        )
        draw_arrows(
            ax,
            sorted(e for e in kept if signs[e] < 0),
            p.negative,
            style.KEPT_WIDTH,
            DASH,
        )
    draw_nodes(ax, mode, {n for e in kept for n in e})
    if title:
        ax.set_title(title)


def legend_handles(mode, *backbones, signed=False):
    """The style's legend, without "Node dropped" when no backbone drops a node."""
    handles = style.legend_handles(mode, signed=signed)
    if all({n for e in bb for n in e} == set(G) for bb in backbones):
        handles = [h for h in handles if h.get_label() != "Node dropped"]
    return handles


def draw_network(mode):
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    draw_directed_backbone(ax, mode, backbone.edges)
    handles = legend_handles(mode, backbone.edges)
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=3,
        handlelength=2.4,
    )
    return fig


figures.show(
    draw_network,
    "network",
    alt=(
        f"A directed network of {G.number_of_nodes()} people drawn as dots with "
        f"arrows. The {backbone.number_of_edges()} edges of the disparity backbone "
        "are thick solid blue arrows; the other "
        f"{G.number_of_edges() - backbone.number_of_edges()} edges are thin gray "
        "arrows. The lead sits in the middle, with arrows to and from almost "
        "everyone; the list sends an arrow to every member and to the visitor."
    ),
    caption=(
        f"The directed disparity backbone at alpha = {ALPHA}: "
        f"{style.backbone_counts(G, backbone).caption} An edge is kept when it "
        "is significant from its source or from its target. The two directions "
        "of a pair are drawn as separate curved arrows."
    ),
    fmt="png",
    width="90%",
    thumbnail=True,
)
# sphinx_gallery_end_ignore

# %%
# The scatter below places every edge by its two tests. The two lines are
# alpha: edges left of the vertical line are significant from the source,
# edges below the horizontal line from the target. Each of the off-diagonal
# corners holds edges that one end alone would miss.

# sphinx_gallery_start_ignore
QUAD_LABELS = {
    "both": (0.03, 0.03, "left", "bottom"),
    "source only": (0.03, 0.97, "left", "top"),
    "target only": (0.97, 0.03, "right", "bottom"),
    "neither": (0.97, 0.97, "right", "top"),
}


def draw_ends(mode):
    p = style.palette(mode)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    x = np.array([max(src[e], P_FLOOR) for e in G.edges])
    y = np.array([max(tgt[e], P_FLOOR) for e in G.edges])
    ax.axvline(ALPHA, color=p.muted, lw=0.75, zorder=1)
    ax.axhline(ALPHA, color=p.muted, lw=0.75, zorder=1)
    ax.scatter(x, y, s=20, c=p.kept, edgecolors=p.surface, linewidths=0.8, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(P_FLOOR / 2, 2.5)
    ax.set_ylim(P_FLOOR / 2, 2.5)
    ax.set_xlabel("source test p-value (out-links of the source)")
    ax.set_ylabel("target test p-value (in-links of the target)")
    for name, (qx, qy, ha, va) in QUAD_LABELS.items():
        ax.text(
            qx,
            qy,
            f"{name}: {len(quadrant[name])}",
            transform=ax.transAxes,
            ha=ha,
            va=va,
            color=p.muted,
            fontsize=8,
        )
    ax.annotate(
        "alpha",
        (ALPHA, P_FLOOR / 2),
        xytext=(3, 3),
        textcoords="offset points",
        color=p.muted,
        fontsize=7.5,
    )
    for e, (dx, dy) in {
        ("Admin", "Lead"): (0, 16),
        ("List", "Visitor"): (-8, -34),
    }.items():
        ax.annotate(
            f"{e[0]} → {e[1]}",
            (max(src[e], P_FLOOR), max(tgt[e], P_FLOOR)),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=p.ink,
            arrowprops={"arrowstyle": "-", "color": p.muted, "lw": 0.6},
        )
    ax.grid(True, which="major")
    return fig


figures.show(
    draw_ends,
    "two_ends",
    alt=(
        f"Scatter plot on log axes of the {G.number_of_edges()} edges: the source "
        "test p-value across, the target test p-value up, with lines at alpha on "
        f"both axes. {len(quadrant['both'])} edges are significant from both ends, "
        f"{len(quadrant['source only'])} from the source only (including Admin to "
        f"Lead), {len(quadrant['target only'])} from the target only (including "
        f"List to Visitor) and {len(quadrant['neither'])} from neither."
    ),
    caption=(
        "The two disparity tests of every directed edge. The library's p-value is "
        "the smaller of the two, so every edge outside the top-right corner is in "
        "the backbone. A test of 1 is an end with a single link."
    ),
    width="75%",
)
# sphinx_gallery_end_ignore

# %%
# Two consequences follow. First, the OR rule rejects more often than alpha
# even when the null model holds; the statistical nulls tutorial measures by
# how much (:doc:`/auto_examples/statistical_nulls/plot_statistical_nulls`).
# Second, the two directions of a pair are separate tests, and a backbone of
# a directed network can keep one direction and drop the other:

both_ways = [(u, v) for u, v in G.edges if G.has_edge(v, u) and u < v]
one_way = [
    (u, v) for u, v in both_ways if backbone.has_edge(u, v) != backbone.has_edge(v, u)
]
print(
    f"{len(both_ways)} pairs linked both ways; the backbone keeps one direction only of {len(one_way)}"
)
print(
    "Admin -> Lead kept:",
    backbone.has_edge("Admin", "Lead"),
    "| Lead -> Admin kept:",
    backbone.has_edge("Lead", "Admin"),
)

# sphinx_gallery_start_ignore
assert backbone.has_edge("Admin", "Lead") and not backbone.has_edge("Lead", "Admin")
assert one_way
# sphinx_gallery_end_ignore

# %%
# Do not throw the direction away before scoring. NetworkX's
# ``G.to_undirected()`` does not add the two directions of a pair; it keeps
# the attributes of whichever direction it meets last, so the undirected
# Admin–Lead tie carries the weight of one direction only:

U = G.to_undirected()
print(
    f"Admin -> Lead: {G['Admin']['Lead']['weight']}, Lead -> Admin: "
    f"{G['Lead']['Admin']['weight']}, undirected: {U['Admin']['Lead']['weight']}"
)

# sphinx_gallery_start_ignore
assert U["Admin"]["Lead"]["weight"] in (
    G["Admin"]["Lead"]["weight"],
    G["Lead"]["Admin"]["weight"],
)
# sphinx_gallery_end_ignore

# %%
# If the question really is undirected, add the two weights yourself
# (``U.add_edge(u, v, weight=w_uv + w_vu)``) and say so in your methods.
#
# **Joint null models test once.** The marginal likelihood filter (Dianati,
# 2016), the noise-corrected filter (Coscia & Neffke, 2017) and the ECM
# (Gemmetto et al., 2017) do
# not test the ends separately. Each gives the ordered pair one null
# distribution built from *both* margins at once, the source's out-strength
# and the target's in-strength (the ECM also uses the degrees), so a
# directed edge gets one test and there is no OR rule. On directed integer
# weights the noise-corrected filter's default p-value is the marginal
# likelihood p-value, as its docstring says, which we check here:

MLF = nb.marginal_likelihood_filter(G)
NC = nb.noise_corrected_filter(G)
print(
    "largest |MLF - noise-corrected| p-value difference:",
    f"{max(abs(MLF.edges[e]['ml_pvalue'] - NC.edges[e]['nc_pvalue']) for e in G.edges):.1e}",
)

# sphinx_gallery_start_ignore
assert all(
    math.isclose(
        MLF.edges[e]["ml_pvalue"], NC.edges[e]["nc_pvalue"], rel_tol=1e-9, abs_tol=1e-12
    )
    for e in G.edges
)
# sphinx_gallery_end_ignore

# %%
# Strong and weak: the two tails
# ------------------------------
#
# So far every test was **one-sided**: it asks only whether an edge is
# heavier than chance, through the upper tail. The **lower tail**,
# :math:`P(X \leq x)`, asks the opposite question: is the edge *lighter*
# than chance? An edge with a small lower tail is **significantly weak**:
# it exists, but carries less weight than the null model expects of it.
#
# With ``signed=True``, five scorers of weighted networks run both tests at
# once. The stored
# p-value becomes **two-sided**, :math:`\min(1, 2\min(\text{upper},
# \text{lower}))`, and a ``"sign"`` attribute records which tail it came
# from: +1 for a significantly strong edge, -1 for a significantly weak one.
#
# What "expects" means depends on the null model, so "significantly weak"
# does too:
#
# - Under the **disparity filter**, a *local* null, an edge is weak when it
#   is an improbably small share of an endpoint's budget. The lower tail of
#   one end is :math:`1 - (1 - x)^{k - 1}`, which is close to
#   :math:`(k - 1)x` for small shares, so a weak edge needs a share below
#   about :math:`\alpha / (2(k - 1))`: only a busy node, with many links and
#   a large strength, can have a significantly weak link.
# - Under the **marginal likelihood filter**, a *joint* null, the weight of
#   *u* → *v* is binomial: each of the *T* messages picks its sender in
#   proportion to out-strength and its receiver in proportion to
#   in-strength. An edge is weak when two active people exchange far fewer
#   messages than their activity predicts.
# - Under **LANS**, an edge is weak when it is among the lightest links of
#   one of its ends; only the order of the weights counts.
#
# Two planted edges show the first two. Lead → Admin is a tiny share of the
# lead's output; m14 → Lead is a tiny count between two of the most active
# people.

SIGNED = {
    "disparity": (nb.disparity_filter(G, signed=True), "disparity_pvalue"),
    "MLF": (nb.marginal_likelihood_filter(G, signed=True), "ml_pvalue"),
    "LANS": (nb.lans_filter(G, signed=True), "lans_pvalue"),
}
T = G.size(weight="weight")
expected = {
    e: G.out_degree(e[0], weight="weight") * G.in_degree(e[1], weight="weight") / T
    for e in G.edges
}
table(
    [
        "edge",
        "weight",
        "share of source",
        "MLF expects",
        "disparity p, sign",
        "MLF p, sign",
    ],
    [
        [
            f"{u} -> {v}",
            G[u][v]["weight"],
            f"{G[u][v]['weight'] / G.out_degree(u, weight='weight'):.4f}",
            f"{expected[u, v]:.1f}",
            f"{SIGNED['disparity'][0][u][v]['disparity_pvalue']:.3f}, {SIGNED['disparity'][0][u][v]['sign']:+d}",
            f"{SIGNED['MLF'][0][u][v]['ml_pvalue']:.1e}, {SIGNED['MLF'][0][u][v]['sign']:+d}",
        ]
        for u, v in [("Lead", "Admin"), ("m14", "Lead")]
    ],
)


# sphinx_gallery_start_ignore
def signed_of(name, e):
    H_, attr = SIGNED[name]
    return H_.edges[e][attr], H_.edges[e]["sign"]


# Lead -> Admin is weak under the disparity filter but not under MLF;
# m14 -> Lead is weak under MLF but not under the disparity filter.
assert (
    signed_of("disparity", ("Lead", "Admin"))[0] < ALPHA
    and signed_of("disparity", ("Lead", "Admin"))[1] == -1
)
assert signed_of("MLF", ("Lead", "Admin"))[0] >= ALPHA
assert (
    signed_of("MLF", ("m14", "Lead"))[0] < ALPHA
    and signed_of("MLF", ("m14", "Lead"))[1] == -1
)
assert signed_of("disparity", ("m14", "Lead"))[0] >= ALPHA
# The disparity weak bound: Lead -> Admin's share is below alpha / (2 (k - 1)).
k_lead = G.out_degree("Lead")
share_admin = G["Lead"]["Admin"]["weight"] / G.out_degree("Lead", weight="weight")
assert share_admin < ALPHA / (2 * (k_lead - 1))
# m14 and the lead are among the most active: top three by out-strength and
# the top in-strength.
_out = sorted(G, key=lambda n: -G.out_degree(n, weight="weight"))
assert "m14" in _out[:3]
assert max(G, key=lambda n: G.in_degree(n, weight="weight")) == "Lead"
# MLF's model: tails of B(T, s_out s_in / T^2), the MLF two-sided p-value.
_e = ("m14", "Lead")
_p = expected[_e] / T
_lo = stats.binom.cdf(G.edges[_e]["weight"], T, _p)
assert math.isclose(signed_of("MLF", _e)[0], min(1, 2 * _lo), rel_tol=1e-9)
# sphinx_gallery_end_ignore

# %%
# Each tail is drawn below for its edge: the solid curve is the upper tail,
# the dashed curve the lower tail, and the observed edge sits on the lower
# tail, below the two-sided threshold :math:`\alpha / 2`.

# sphinx_gallery_start_ignore
s_lead = G.out_degree("Lead", weight="weight")
lead_shares = [G["Lead"][v]["weight"] / s_lead for v in G.successors("Lead")]
mlf_x = np.arange(0, int(3 * expected[_e]) + 1)
mlf_upper = stats.binom.sf(mlf_x - 1, T, _p)
mlf_lower = stats.binom.cdf(mlf_x, T, _p)
w_m14 = G.edges[_e]["weight"]
lo_admin = 1 - (1 - share_admin) ** (k_lead - 1)


def draw_tails(mode):
    p = style.palette(mode)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    xs = np.logspace(-4, 0, 400)[:-1]
    up = (1 - xs) ** (k_lead - 1)
    lo = 1 - up
    for ax in (ax1, ax2):
        ax.axhline(ALPHA / 2, color=p.muted, lw=0.75, zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(1e-11, 3)
        ax.grid(True, which="major", axis="y")
    ax1.text(1.2e-4, ALPHA / 2 / 1.4, "α/2", color=p.muted, fontsize=8, va="top")
    ax1.plot(xs, up, color=p.positive, zorder=2)
    ax1.plot(xs, lo, color=p.negative, linestyle=DASH, zorder=2)
    ax1.set_xscale("log")
    ax1.set_xlim(1e-4, 1)
    ax1.text(1.2e-4, 0.5, "upper tail\nP(X ≥ x)", color=p.ink, fontsize=8, va="top")
    ax1.text(1.2e-4, 4e-4, "lower tail\nP(X ≤ x)", color=p.ink, fontsize=8, va="top")
    ax1.plot(
        lead_shares,
        [3e-11] * len(lead_shares),
        marker="|",
        linestyle="",
        color=p.muted,
        markersize=8,
    )
    ax1.plot(
        [share_admin],
        [lo_admin],
        marker="o",
        color=p.negative,
        markeredgecolor=p.surface,
        zorder=3,
    )
    ax1.annotate(
        f"Lead → Admin\nlower tail {lo_admin:.3f}",
        (share_admin, lo_admin),
        xytext=(12, -14),
        va="top",
        textcoords="offset points",
        fontsize=8,
        color=p.ink,
    )
    ax1.set_xlabel("share x of the lead's out-strength (ticks: the lead's links)")
    ax1.set_ylabel("tail probability")
    ax1.set_title(f"Disparity, the lead's {k_lead} out-links", fontsize=9)

    ax2.step(mlf_x, mlf_upper, where="mid", color=p.positive, zorder=2)
    ax2.step(
        mlf_x,
        np.maximum(mlf_lower, 1e-12),
        where="mid",
        color=p.negative,
        linestyle=DASH,
        zorder=2,
    )
    ax2.axvline(expected[_e], color=p.hairline, lw=0.75, zorder=1)
    ax2.text(expected[_e], 1e-10, " expected", color=p.muted, fontsize=8)
    ax2.text(40, 1e-2, "upper tail", color=p.ink, fontsize=8)
    ax2.text(12, 1e-6, "lower tail", color=p.ink, fontsize=8)
    ax2.plot(
        [w_m14],
        [_lo],
        marker="o",
        color=p.negative,
        markeredgecolor=p.surface,
        zorder=3,
    )
    ax2.annotate(
        f"m14 → Lead, {w_m14} message\nlower tail {_lo:.1e}",
        (w_m14, _lo),
        xytext=(10, -4),
        textcoords="offset points",
        fontsize=8,
        color=p.ink,
    )
    ax2.set_xlim(-1, mlf_x[-1])
    ax2.set_xlabel("messages m14 → Lead under the null")
    ax2.set_title("Marginal likelihood, one ordered pair", fontsize=9)
    fig.tight_layout()
    return fig


figures.show(
    draw_tails,
    "tails",
    alt=(
        "Two panels with tail probabilities on a log scale. Left: for the lead's "
        f"{k_lead} out-links, a solid upper-tail curve falling and a dashed "
        "lower-tail curve rising with the share of the lead's out-strength; Lead "
        f"to Admin, at share {share_admin:.4f}, sits on the lower tail at "
        f"{lo_admin:.3f}, below the alpha/2 line. Right: binomial tails for "
        f"the messages from m14 to the lead, expected {expected[_e]:.1f}; the "
        f"observed {w_m14} has a lower tail of {_lo:.1e}."
    ),
    caption=(
        "What *significantly weak* means under a local and a joint null. Left: "
        "the disparity test at the lead, where an edge is weak if it is a tiny "
        "share of the lead's output. Right: the marginal likelihood test of one "
        "ordered pair, where an edge is weak if it carries far fewer messages "
        "than the two people's activity predicts. Upper tails are solid, lower "
        "tails dashed; the horizontal line is alpha/2."
    ),
)
# sphinx_gallery_end_ignore

# %%
# A signed backbone, read edge by edge
# ------------------------------------
#
# Now the signed backbones of the whole network, at the same alpha, from one
# local parametric null (disparity), one joint null (MLF) and one
# nonparametric local test (LANS).


def sign_counts(H, attr):
    kept = [d["sign"] for *_, d in H.edges(data=True) if d[attr] < ALPHA]
    return kept.count(1), kept.count(-1)


signed_backbones = {
    name: nb.threshold_filter(H, attr, ALPHA) for name, (H, attr) in SIGNED.items()
}
table(
    ["scorer", "kept", "+1 (strong)", "-1 (weak)"],
    [
        [name, bb.number_of_edges(), *sign_counts(H, attr)]
        for name, (H, attr), bb in zip(
            SIGNED, SIGNED.values(), signed_backbones.values()
        )
    ],
)

# sphinx_gallery_start_ignore
signs = {
    name: {e: H.edges[e]["sign"] for e in H.edges} for name, (H, _) in SIGNED.items()
}
counts = {name: sign_counts(H, attr) for name, (H, attr) in SIGNED.items()}
assert counts["LANS"][1] > counts["MLF"][1] + counts["disparity"][1]
_weak = {
    name: {e for e in bb.edges if bb.edges[e]["sign"] == -1}
    for name, bb in signed_backbones.items()
}
assert _weak["disparity"] == {("Lead", "Admin")}
assert _weak["MLF"] == {("m14", "Lead")}


def draw_signed(mode):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 9.0))
    axes = axes.ravel()
    for ax, name in zip(axes, SIGNED):
        plus, minus = counts[name]
        draw_directed_backbone(
            ax,
            mode,
            signed_backbones[name].edges,
            signs=signs[name],
            title=f"{name[0].upper()}{name[1:]}: {plus} strong, {minus} weak",
        )
    axes[3].set_axis_off()
    axes[3].legend(
        handles=legend_handles(
            mode, *(bb.edges for bb in signed_backbones.values()), signed=True
        ),
        loc="center",
        handlelength=2.4,
        fontsize=9.5,
    )
    fig.tight_layout()
    return fig


figures.show(
    draw_signed,
    "signed",
    alt=(
        "Three drawings of the same directed network, one per signed backbone: "
        f"disparity ({counts['disparity'][0]} strong, {counts['disparity'][1]} weak), "
        f"marginal likelihood ({counts['MLF'][0]} strong, {counts['MLF'][1]} weak) "
        f"and LANS ({counts['LANS'][0]} strong, {counts['LANS'][1]} weak). Strong "
        "edges are solid blue arrows, weak edges dashed red arrows, over thin gray "
        "arrows for the edges not kept."
    ),
    caption=(
        f"Signed backbones at alpha = {ALPHA} (two-sided). Significantly strong "
        "edges are solid, significantly weak edges dashed. Many of LANS's weak "
        "edges are ties, explained in the next section."
    ),
    fmt="png",
)
# sphinx_gallery_end_ignore

# %%
# The disparity and MLF backbones each call one planted edge weak, and a
# different one: the disparity filter flags Lead → Admin (a small share of
# a busy sender), MLF flags m14 → Lead (a small count between two active
# people). Neither is "the" weak tie of the network; each answers its own
# question. LANS calls far more edges weak. To see why, we need the sign
# rule itself.
#
# How each scorer assigns a sign
# ------------------------------
#
# Every signed scorer computes an upper tail and a lower tail for each edge
# and passes both to one rule (``_combine_tails`` in
# ``nx_backbone/statistical.py`` and ``_signed_pvalues`` in
# ``nx_backbone/bipartite.py``; the one exception, a normal approximation in
# the noise-corrected filter, is in the table below):
#
# .. math::
#
#    p = \min(1,\, 2\min(\text{upper}, \text{lower})), \qquad
#    \text{sign} = \begin{cases} +1 & \text{upper} < \text{lower} \\
#                                -1 & \text{otherwise.} \end{cases}
#
# So an edge whose two tails are **exactly equal** is signed -1. This is the
# rule of R ``backbone`` 3.0.4 (Neal, 2022), whose ``.retain`` takes
# ``upper < lower`` as positive; ``nx-backbone`` adopted it in version 0.3.0
# (`#84 <https://github.com/brianckeegan/nx-backbone/pull/84>`__). The tails
# themselves differ by scorer:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 24 38 38
#
#    * - Scorer
#      - Upper and lower tail
#      - When can the tails tie?
#    * - :func:`~nx_backbone.disparity_filter`
#      - Per end :math:`(1 - x)^{k - 1}` and its complement; the edge takes the
#        smaller upper tail and the smaller lower tail of its two ends.
#      - An end of degree 1 gives 1 to both tails, so an edge whose two ends
#        both have one link ties at 1.
#    * - :func:`~nx_backbone.marginal_likelihood_filter`,
#        :func:`~nx_backbone.noise_corrected_filter`,
#        :func:`~nx_backbone.ecm_filter`
#      - :math:`P(X \geq w)` and :math:`P(X \leq w)` of a count model (binomial
#        or ECM). The noise-corrected filter with ``pvalue_method="delta"``
#        (or ``method="zscore"``) signs by its standardized score instead: +1
#        when it is positive.
#      - The two tails overlap at :math:`X = w`, so they add up to more than 1:
#        a tie needs both at least 1/2, a p-value of 1. Untestable edges,
#        such as an ECM self-loop, tie too, and so does a standardized
#        noise-corrected score of exactly 0.
#    * - :func:`~nx_backbone.lans_filter`
#      - Per end :math:`\#\{\text{weights} > w\}/k` and
#        :math:`\#\{\text{weights} < w\}/k`, neither counting the edge itself;
#        the smaller of each over the two ends.
#      - Often, and at p = 0: see below.
#    * - :func:`~nx_backbone.sdsm`, :func:`~nx_backbone.fdsm`,
#        :func:`~nx_backbone.fixedfill`, :func:`~nx_backbone.fixedrow`,
#        :func:`~nx_backbone.fixedcol`
#      - Upper and lower tails of the co-occurrence under the null model.
#      - As for the count models; see the bipartite tutorial.
#
# :func:`~nx_backbone.polya_filter` and :func:`~nx_backbone.gloss_filter`
# define only the upper-tail test and have no ``signed`` option.
#
# **Ties in LANS.** The LANS tails are counts over the same *k* links, so
# equal counts give exactly equal tails. Both tails are 0, a tie at 0, when
# one end of the edge has no link heavier and no link lighter than it:
#
# - the only out-link of a source with out-degree 1, or the only in-link of
#   a target with in-degree 1;
# - any link of an end whose links all have the same weight;
# - an edge that is the heaviest at one end and the lightest at the other.
#
# The signed backbone keeps every such edge, with p-value 0 and sign -1.
# Directed networks produce many of them, because a node's out-degree and
# in-degree are each smaller than its total degree, and because a sender
# such as a mailing list often sends the same amount to everyone. We
# recompute the
# LANS tails from their definition, with exact fractions, and check them
# against the library:


def lans_end(w, weights):
    """LANS upper and lower tails at one end: #{weights > w}/k and #{weights < w}/k."""
    k = len(weights)
    return Fraction(sum(x > w for x in weights), k), Fraction(
        sum(x < w for x in weights), k
    )


def lans_tails(G, u, v):
    """The smaller upper and the smaller lower tail over the edge's two ends."""
    w = G[u][v]["weight"]
    up_s, lo_s = lans_end(w, [d for *_, d in G.out_edges(u, data="weight")])
    up_t, lo_t = lans_end(w, [d for *_, d in G.in_edges(v, data="weight")])
    return min(up_s, up_t), min(lo_s, lo_t)


tails = {e: lans_tails(G, *e) for e in G.edges}
L, _ = SIGNED["LANS"]
for e, (up, lo) in tails.items():
    assert L.edges[e]["sign"] == (1 if up < lo else -1)
    assert math.isclose(
        L.edges[e]["lans_pvalue"], min(1, 2 * min(up, lo)), abs_tol=1e-12
    )


def why_tie(G, u, v):
    """Why an edge's two LANS tails are both 0."""
    w = G[u][v]["weight"]
    for end, weights in (
        ("source", [d for *_, d in G.out_edges(u, data="weight")]),
        ("target", [d for *_, d in G.in_edges(v, data="weight")]),
    ):
        if len(weights) == 1:
            return "an end has a single link"
        if lans_end(w, weights) == (0, 0):
            return f"all of the {end}'s links weigh the same"
    return "heaviest at one end, lightest at the other"


lans_kept = [e for e in G.edges if L.edges[e]["lans_pvalue"] < ALPHA]
weak = [e for e in lans_kept if L.edges[e]["sign"] == -1]
ties = [e for e in weak if tails[e][0] == tails[e][1]]
causes = {}
for e in ties:
    causes.setdefault(why_tie(G, *e), []).append(e)
print(
    f"signed LANS backbone: {len(lans_kept)} edges, {len(weak)} signed -1, {len(ties)} of them exact ties"
)
table(["why the tails tie at 0", "edges"], [[k, len(v)] for k, v in causes.items()])

# sphinx_gallery_start_ignore
assert all(tails[e] == (0, 0) for e in ties)
assert all(L.edges[e]["lans_pvalue"] == 0 for e in ties)
assert 0 < len(ties) < len(weak)
assert set(causes) <= {
    "an end has a single link",
    "all of the source's links weigh the same",
    "all of the target's links weigh the same",
    "heaviest at one end, lightest at the other",
}
# The list's links all weigh the same, so all of them are ties at 0.
assert all(e in ties for e in G.out_edges("List"))
# Lead -> Admin is a LANS tie: the administrator receives only from the lead.
assert ("Lead", "Admin") in ties and G.in_degree("Admin") == 1
# List -> Visitor too: the same digest weight is both the heaviest and the
# lightest of the list's out-links.
assert tails["List", "Visitor"] == (0, 0)
# sphinx_gallery_end_ignore

# %%
# The figure places every edge by its two LANS tails. Above the diagonal the
# upper tail is smaller and the edge is signed +1; below it, -1; *on* the
# diagonal the tails tie, and the edge is signed -1 too. The shaded band is
# where the smaller tail is below :math:`\alpha / 2`, so the edge is kept.

# sphinx_gallery_start_ignore
lans_points = {}
for e, (up, lo) in tails.items():
    kind = "tie" if up == lo else ("+1" if up < lo else "-1")
    lans_points.setdefault((kind, float(up), float(lo)), []).append(e)
n_kind = {
    k: sum(len(v) for (kind, *_), v in lans_points.items() if kind == k)
    for k in ("+1", "-1", "tie")
}
n_origin = len(lans_points.get(("tie", 0.0, 0.0), []))
assert n_origin == sum(1 for e in ties)
assert n_kind["+1"] + n_kind["-1"] + n_kind["tie"] == G.number_of_edges()


def draw_lans(mode):
    p = style.palette(mode)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    fwd = (np.sqrt, np.square)
    ax.set_xscale("function", functions=fwd)
    ax.set_yscale("function", functions=fwd)
    band = ALPHA / 2
    ax.axvspan(0, band, color=p.hairline, zorder=0, lw=0)
    ax.axhspan(0, band, color=p.hairline, zorder=0, lw=0)
    ax.plot([0, 1], [0, 1], color=p.muted, lw=0.75, zorder=1)
    marker = {
        "+1": ("P", p.positive, p.positive),
        "-1": ("_", p.negative, p.negative),
        "tie": ("D", "none", p.negative),
    }
    for (kind, up, lo), edges in lans_points.items():
        m, face, edge = marker[kind]
        size = 18 + 14 * (len(edges) - 1) ** 0.8
        if kind == "-1":
            ax.scatter(
                up,
                lo,
                s=size * 1.6,
                marker=m,
                c=edge,
                linewidths=1.8,
                zorder=3,
                clip_on=False,
            )
        else:
            ax.scatter(
                up,
                lo,
                s=size,
                marker=m,
                facecolors=face,
                edgecolors=edge,
                linewidths=1.2,
                zorder=3,
                clip_on=False,
            )
    ax.annotate(
        f"{n_origin} ties at 0",
        (0, 0),
        xytext=(14, 6),
        textcoords="offset points",
        fontsize=8,
        color=p.ink,
    )
    ax.text(
        0.04,
        0.96,
        "+1: upper tail smaller",
        transform=ax.transAxes,
        color=p.ink,
        fontsize=8,
        va="top",
    )
    ax.text(
        0.96,
        0.30,
        "−1: lower tail smaller",
        transform=ax.transAxes,
        color=p.ink,
        fontsize=8,
        ha="right",
    )
    ax.text(
        0.62,
        0.69,
        "tie: −1",
        rotation=45,
        color=p.muted,
        fontsize=8,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.025, 0.1, 0.25, 0.5, 1])
    ax.set_yticks([0, 0.025, 0.1, 0.25, 0.5, 1])
    ax.set_xticklabels(["0", ".025", ".1", ".25", ".5", "1"])
    ax.set_yticklabels(["0", ".025", ".1", ".25", ".5", "1"])
    ax.set_xlabel("upper tail  #{weights > w}/k")
    ax.set_ylabel("lower tail  #{weights < w}/k")
    ax.set_aspect("equal")
    handles = [
        Line2D(
            [],
            [],
            marker="P",
            color=p.positive,
            linestyle="",
            markersize=7,
            label=f"+1 ({n_kind['+1']})",
        ),
        Line2D(
            [],
            [],
            marker="_",
            color=p.negative,
            linestyle="",
            markersize=9,
            markeredgewidth=1.8,
            label=f"−1 ({n_kind['-1']})",
        ),
        Line2D(
            [],
            [],
            marker="D",
            markerfacecolor="none",
            markeredgecolor=p.negative,
            linestyle="",
            markersize=6,
            label=f"tie, −1 ({n_kind['tie']})",
        ),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=3)
    return fig


figures.show(
    draw_lans,
    "lans_ties",
    alt=(
        f"Scatter of the {G.number_of_edges()} edges by their LANS upper tail "
        "(across) and lower tail (up), on square-root axes, with the diagonal "
        f"drawn. {n_kind['+1']} edges signed +1 are plus signs above the "
        f"diagonal, {n_kind['-1']} edges signed -1 are minus signs below it, and "
        f"{n_kind['tie']} tied edges are hollow diamonds on it, {n_origin} of "
        "them at the origin. Shaded bands along both axes mark tails below alpha/2."
    ),
    caption=(
        "The sign rule of the signed LANS test. Marker size grows with the number "
        f"of edges at a point, and the legend counts all {G.number_of_edges()} "
        "edges, kept or not. Ties on the diagonal are signed -1; those at the "
        f"origin have p-value 0 and are all kept, which is why {len(ties)} of the "
        f"{len(weak)} weak edges in the LANS backbone are ties."
    ),
    width="70%",
)
# sphinx_gallery_end_ignore

# %%
# **The signed LANS lower tail.** The lower tail
# :math:`\#\{\text{weights} < w\}/k` is the mirror image of the upper tail
# and does not count the edge itself. Before version 0.3.0 the signed LANS
# test used the empirical CDF, :math:`\#\{\text{weights} \leq w\}/k`, which
# counts the edge and so is never below :math:`1/k`; at alpha = 0.05 an
# edge could then be significantly weak only at a node with more than 40
# links (`#63 <https://github.com/brianckeegan/nx-backbone/pull/63>`__). The
# old rule, recomputed here, finds no weak edge in this network:


def old_lower(G, u, v):
    w = G[u][v]["weight"]
    out_w = [d for *_, d in G.out_edges(u, data="weight")]
    in_w = [d for *_, d in G.in_edges(v, data="weight")]
    return min(Fraction(sum(x <= w for x in ws), len(ws)) for ws in (out_w, in_w))


old_weak = [
    e
    for e in G.edges
    if old_lower(G, *e) < tails[e][0] and 2 * old_lower(G, *e) < ALPHA
]
print(
    f"weak edges with the old lower tail: {len(old_weak)}; with the current one: {len(weak)}"
)
print(
    f"largest out- or in-degree here: {max(max(d for _, d in G.out_degree()), max(d for _, d in G.in_degree()))}"
)

# sphinx_gallery_start_ignore
assert not old_weak
assert max(max(d for _, d in G.out_degree()), max(d for _, d in G.in_degree())) <= 40
# sphinx_gallery_end_ignore

# %%
# **Reading the signed backbone.** A -1 is a statement about the null model,
# not about the relationship. Before reading the weak edges as avoidance:
#
# 1. **Name the null.** Lead → Admin is weak *as a share of the lead's
#    output*; m14 → Lead is weak *given how active both are*.
# 2. **Separate the ties.** A LANS tie at 0 says that one end had a single
#    link or links of equal weight, or that the edge is extreme at its two
#    ends in opposite directions;
#    it does not say the edge is light. Many of the LANS backbone's -1
#    edges in this network are such ties. The helper above
#    (``lans_tails``) separates them, and for any scorer a -1 edge whose
#    one-sided p-value (``signed=False``) is half its two-sided p-value has
#    equal tails.
# 3. **Check the direction.** In a directed network, ask from which end the
#    edge is weak: the disparity and LANS tests take the smaller lower tail
#    of the two ends, so a -1 may come from the source or from the target.
#
# A small check of the second point with the public API alone:

H1 = nb.lans_filter(G)  # one-sided: the upper tail
tie_by_api = [
    e
    for e in lans_kept
    if L.edges[e]["sign"] == -1
    and math.isclose(
        H1.edges[e]["lans_pvalue"] * 2, L.edges[e]["lans_pvalue"], abs_tol=1e-12
    )
]
print(
    f"ties found from the two p-values: {len(tie_by_api)} (from the tails: {len(ties)})"
)

# sphinx_gallery_start_ignore
assert sorted(tie_by_api) == sorted(ties)
assert len(ties) > len(weak) / 4
# sphinx_gallery_end_ignore

# %%
# Absent pairs: ``missing_as_zero``
# ---------------------------------
#
# A signed test can also ask about pairs that *never* interact. With
# ``missing_as_zero=True``, every ordered pair without an edge is tested as
# an edge of weight 0 (on a directed graph that is every ordered pair
# :math:`(u, v)`, :math:`u \neq v`). A pair whose source sends nothing or
# whose target receives nothing cannot be tested by the local nulls. Under
# LANS a weight of 0 is below every link, so its lower tail is 0 at either
# end; under the disparity filter it is 0 at any end with more than one
# link. So LANS keeps every absent pair it tests as -1, and the disparity
# filter every one except pairs whose two ends both have a single link (the
# docstrings say so, and R does the same). Only MLF gives a graded answer:

for name, func, attr in [
    ("disparity", nb.disparity_filter, "disparity_pvalue"),
    ("MLF", nb.marginal_likelihood_filter, "ml_pvalue"),
    ("LANS", nb.lans_filter, "lans_pvalue"),
]:
    Z = func(G, signed=True, missing_as_zero=True)
    added = [e for e in Z.edges if not G.has_edge(*e)]
    kept_absent = [e for e in added if Z.edges[e][attr] < ALPHA]
    print(
        f"{name:<10} tests {len(added):>3} absent pairs, keeps {len(kept_absent):>3} (all -1: {all(Z.edges[e]['sign'] == -1 for e in kept_absent)})"
    )
    # sphinx_gallery_start_ignore
    assert all(Z.edges[e]["sign"] == -1 for e in kept_absent)
    if name == "LANS":
        assert len(kept_absent) == len(added)
    if name == "disparity":
        single = [
            e for e in added if G.out_degree(e[0]) == 1 and G.in_degree(e[1]) == 1
        ]
        assert single and len(kept_absent) == len(added) - len(single)
    if name == "MLF":
        assert 0 < len(kept_absent) < len(added)
    # sphinx_gallery_end_ignore

# %%
# So ``missing_as_zero=True`` with the disparity or LANS null mostly
# restates which pairs are absent. Use it with a joint null, and count the
# extra tests in any multiple-testing correction.
#
# Signed bipartite projections
# ----------------------------
#
# Projections of two-mode data, such as who attends which events, are
# undirected, but they are often signed: a pair of people can attend
# together significantly more, or significantly less, than their activity
# predicts. :func:`~nx_backbone.sdsm`, :func:`~nx_backbone.fdsm` and the
# fixed models take ``signed=True`` and sign each pair with the same rule
# as above. The bipartite tutorial builds a signed FDSM backbone of the
# Davis Southern Women data, reads its negative ties, and extends the SDSM
# to fixed cells (SDSM-EC): see
# :doc:`/auto_examples/bipartite/plot_bipartite_projections`. The count
# models' ties sit at p = 1, so on a projection the tie rule changes the
# ``sign`` of untested or non-significant pairs, never which pairs are kept.
#
# Matching R
# ----------
#
# Two conventions decide whether a signed backbone matches R ``backbone``:
#
# - **The tie rule** is R's, by default, since 0.3.0: there is nothing to
#   set. Older versions signed ties +1.
# - **The signed disparity lower tail** is kept different on purpose. R
#   takes the lower tail as one minus the combined upper tail; here, as the
#   :func:`~nx_backbone.disparity_filter` Notes explain, the OR rule is
#   applied to each tail, so the lower tail is never larger than R's. The
#   difference matters only when both ends have more than one link:

R_style = {}
for e, (source, target) in ends.items():
    up = min(source[0], target[0])
    ours_lo, r_lo = min(source[1], target[1]), 1 - up
    R_style[e] = (min(1, 2 * min(up, r_lo)), 1 if up < r_lo else -1)
D, _ = SIGNED["disparity"]
ours_kept = {
    (e, D.edges[e]["sign"]) for e in G.edges if D.edges[e]["disparity_pvalue"] < ALPHA
}
r_kept = {(e, s) for e, (pv, s) in R_style.items() if pv < ALPHA}
print(
    f"signed disparity backbone: {len(ours_kept)} edges; with R's lower tail: {len(r_kept)}; same edges and signs: {ours_kept == r_kept}"
)

# sphinx_gallery_start_ignore
for e, (source, target) in ends.items():
    assert min(source[1], target[1]) <= 1 - min(source[0], target[0]) + 1e-15
    if G.out_degree(e[0]) == 1 or G.in_degree(e[1]) == 1:
        # An end with a single link: both rules give the same lower tail.
        assert math.isclose(
            min(source[1], target[1]), 1 - min(source[0], target[0]), abs_tol=1e-15
        )
    assert math.isclose(
        D.edges[e]["disparity_pvalue"],
        min(1, 2 * min(min(source[0], target[0]), min(source[1], target[1]))),
        abs_tol=1e-15,
    )
# sphinx_gallery_end_ignore

# %%
# The other R conventions (which pairs a projection tests, ``lspar``, and
# the global threshold) and a worked comparison against R's own output are
# in :doc:`/auto_examples/r_and_scale/plot_coming_from_r`.
#
# A checklist for directed and signed backbones
# ---------------------------------------------
#
# 1. Keep the direction. Score the :class:`~networkx.DiGraph`; if you must
#    symmetrize, add the two weights yourself.
# 2. Know which ends your scorer tests: both ends with an OR rule
#    (disparity, LANS) or one joint test of the ordered pair (MLF,
#    noise-corrected, ECM).
# 3. With ``signed=True``, say what "weak" means under your null model.
# 4. Separate ties from weak edges before interpreting -1, above all with
#    LANS on a directed graph.
# 5. With ``missing_as_zero=True``, prefer a joint null, and count the extra
#    tests in any correction.
# 6. Report the scorer, alpha, the correction and the sign convention;
#    :func:`~nx_backbone.describe_backbone` writes the methods text.
#
# Where next
# ----------
#
# - :doc:`/user_guide/directed_graphs`: which methods accept directed graphs.
# - :doc:`/auto_examples/statistical_nulls/plot_statistical_nulls`: the seven
#   null models side by side, with signed backbones of an undirected network.
# - :doc:`/auto_examples/bipartite/plot_bipartite_projections`: signed SDSM
#   and FDSM projections and SDSM-EC.
# - :doc:`/auto_examples/r_and_scale/plot_coming_from_r`: the R conventions
#   and their switches.
# - :doc:`/tutorials/validating_backbones`: multiple-testing corrections and
#   robustness checks.
# - API: :func:`~nx_backbone.disparity_filter`,
#   :func:`~nx_backbone.lans_filter`,
#   :func:`~nx_backbone.marginal_likelihood_filter` and
#   :func:`~nx_backbone.backbone_from_weighted` (``signed=``,
#   ``missing_as_zero=``).
#
# References
# ----------
#
# - Coscia, M., & Neffke, F. M. (2017). Network backboning with noisy data.
#   *Proceedings of the IEEE International Conference on Data Engineering
#   (ICDE)*, 425-436. https://doi.org/10.1109/ICDE.2017.100
# - Dianati, N. (2016). Unwinding the hairball graph: Pruning algorithms for
#   weighted complex networks. *Physical Review E*, 93, 012304.
#   https://doi.org/10.1103/PhysRevE.93.012304
# - Foti, N. J., Hughes, J. M., & Rockmore, D. N. (2011). Nonparametric
#   sparsification of complex multiscale networks. *PLoS ONE*, 6(2), e16431.
#   https://doi.org/10.1371/journal.pone.0016431
# - Gemmetto, V., Cardillo, A., & Garlaschelli, D. (2017). Irreducible
#   network backbones: unbiased graph filtering via maximum entropy.
#   arXiv:1706.00230.
# - Neal, Z. P. (2022). backbone: An R package to extract network
#   backbones. *PLoS ONE*, 17(5), e0269137.
#   https://doi.org/10.1371/journal.pone.0269137
# - Serrano, M. A., Boguñá, M., & Vespignani, A. (2009). Extracting the
#   multiscale backbone of complex weighted networks. *Proceedings of the
#   National Academy of Sciences*, 106(16), 6483-6488.
#   https://doi.org/10.1073/pnas.0808904106
