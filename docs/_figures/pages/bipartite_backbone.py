"""Figures of ``docs/tutorials/bipartite_backbone.rst``."""

from __future__ import annotations

import functools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import BoundaryNorm

import nx_backbone as nb
from _figures import style
from _figures.pages import figure


@functools.cache
def _davis():
    """Davis incidence (women x events) and the simple projection's weights."""
    B = nx.davis_southern_women_graph()
    women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    events = [n for n, d in B.nodes(data=True) if d["bipartite"] == 1]
    M = nx.to_numpy_array(B, nodelist=women + events)[: len(women), len(women) :]
    G = nb.simple_projection(B, women)
    W = nx.to_numpy_array(G, nodelist=women, weight="weight")
    # The simple projection counts shared events: W = M M^T off the diagonal.
    shared = M @ M.T
    np.fill_diagonal(shared, 0)
    assert np.array_equal(W, shared)
    # The page: the four weighted projections share these edges, only the
    # weights differ.
    for project in (nb.hyper_projection, nb.probs_projection, nb.ycn_projection):
        assert {frozenset(e) for e in project(B, women).edges()} == {
            frozenset(e) for e in G.edges()
        }
    return women, events, M, G, W


def _stats() -> dict:
    women, events, M, G, W = _davis()
    n = len(women)
    pairs = n * (n - 1) // 2
    return {
        "women": n,
        "events": len(events),
        "attendances": int(M.sum()),
        "edges": G.number_of_edges(),
        "pairs": pairs,
        "density": G.number_of_edges() / pairs,
        "max_shared": int(W.max()),
    }


def _alt() -> str:
    s = _stats()
    return (
        "Two grids side by side, with the same rows: the Davis Southern Women. "
        f"Left, the incidence matrix of {s['women']} women by {s['events']} "
        f"events, a filled square for each of the {s['attendances']} "
        "attendances. Right, the simple projection onto the women, a "
        f"{s['women']} by {s['women']} grid shaded by the number of events two "
        f"women share, from 0 to {s['max_shared']}. Almost every off-diagonal "
        f"cell is shaded: {s['edges']} of the {s['pairs']} pairs share at "
        "least one event."
    )


def _caption() -> str:
    s = _stats()
    return (
        "From incidence to projection. Left, who attended what: "
        f"{s['women']} women by {s['events']} events. Right, "
        ":func:`~nx_backbone.simple_projection` counts the events each pair "
        "of women shares (the incidence times its transpose). "
        f"{s['edges']} of the {s['pairs']} pairs ({s['density']:.0%}) share at "
        "least one event, so the projection is nearly complete: the null "
        "models below decide which of those counts are more than the women's "
        "and the events' degrees would produce."
    )


@figure("incidence-projection", alt=_alt, caption=_caption)
def draw_incidence_projection(mode):
    women, events, M, _, W = _davis()
    p = style.palette(mode)
    n_w, n_e = M.shape
    fig, (ax_m, ax_w) = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.1),
        gridspec_kw={"width_ratios": [n_e, n_w], "wspace": 0.04},
    )

    # Incidence: one filled square per attendance, faint squares elsewhere.
    rows, cols = np.nonzero(M)
    empty_r, empty_c = np.nonzero(M == 0)
    ax_m.scatter(empty_c, empty_r, marker="s", s=44, c=p.hairline, linewidths=0)
    ax_m.scatter(cols, rows, marker="s", s=44, c=p.kept, linewidths=0)
    ax_m.set_xticks(range(n_e), events, rotation=90)
    ax_m.set_yticks(range(n_w), women)
    ax_m.set_title("Incidence: women × events")

    # Projection: shared-event counts; pairs sharing none in the empty-cell
    # gray of the incidence, the diagonal left blank.
    top = int(W.max())
    cmap = (
        style.sequential_cmap(mode)
        .resampled(top)
        .with_extremes(under=p.hairline, bad="none")
    )
    shown = np.ma.masked_where(np.eye(n_w, dtype=bool), W)
    image = ax_w.imshow(
        shown,
        cmap=cmap,
        norm=BoundaryNorm(np.arange(0.5, top + 1), top),
        interpolation="nearest",
    )
    ax_w.set_xticks(range(n_w), [w.split()[0] for w in women], rotation=90)
    ax_w.set_yticks([])
    ax_w.set_title("Projection: events shared")
    # The color bar in its own inset, so both grids keep the same row height.
    cax = ax_w.inset_axes((1.04, 0.15, 0.04, 0.7))
    bar = fig.colorbar(image, cax=cax, ticks=range(1, top + 1))
    bar.set_label("Events shared (gray: none)")
    bar.outline.set_visible(False)
    cax.tick_params(length=0)

    for ax in (ax_m, ax_w):
        ax.set_ylim(n_w - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.tick_params(length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
    return fig
