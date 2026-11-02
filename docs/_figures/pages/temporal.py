"""Figures of ``docs/user_guide/temporal.rst``.

The page's own small examples, built exactly as its doctests build them:
the snapshots ``S``, the events ``E`` and the spans ``P`` on one time axis,
and the events binned into snapshots with ``freq=``.  (The temporal
tutorial's figure draws a different, larger badge log.)
"""

from __future__ import annotations

import functools

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle

import nx_backbone as nb
from _figures import style
from _figures.pages import figure

PAIRS = [("ann", "bob"), ("bob", "cat")]
FREQS = (1, 2)
T_MAX = 5


@functools.cache
def _forms():
    """``S``, ``E`` and ``P`` of the page's "The three forms"."""
    S = {
        0: nx.Graph([("ann", "bob")]),
        1: nx.Graph([("ann", "bob"), ("bob", "cat")]),
        2: nx.Graph(),
        3: nx.Graph([("bob", "cat")]),
    }
    S[2].add_node("dan")
    E = nx.MultiGraph()
    E.add_edges_from(
        [
            ("ann", "bob", {"time": 0.5}),
            ("ann", "bob", {"time": 1.2, "weight": 3}),
            ("bob", "cat", {"time": 3.7}),
        ]
    )
    P = nx.MultiGraph()
    P.add_edges_from(
        [
            ("ann", "bob", {"start": 0, "end": 2}),
            ("bob", "cat", {"start": 1, "end": 4}),
        ]
    )
    forms = (nb.temporal_form(S), nb.temporal_form(E), nb.temporal_form(P))
    assert forms == ("snapshots", "events", "spans"), forms
    return S, E, P


@functools.cache
def _binned():
    """``{freq: snapshots_from_events(E, freq=freq, origin=0)}``."""
    _, E, _ = _forms()
    return {f: nb.snapshots_from_events(E, freq=f, origin=0) for f in FREQS}


def _only_isolated():
    """``(time, nodes)`` of the snapshots that hold isolated nodes only."""
    S, _, _ = _forms()
    return [(t, sorted(G)) for t, G in S.items() if G and not G.number_of_edges()]


def _row(u, v):
    return PAIRS.index((u, v)) if (u, v) in PAIRS else PAIRS.index((v, u))


def _pair_axes(ax, title, p):
    labels = [f"{u}–{v}" for u, v in PAIRS]
    ax.set_yticks(range(len(PAIRS)), labels)
    ax.set_ylim(len(PAIRS) - 0.4, -0.6)
    ax.set_xlim(-0.2, T_MAX)
    ax.tick_params(axis="y", length=0, labelcolor=p.ink)
    ax.spines["left"].set_visible(False)
    ax.set_title(title, loc="left")


def _event_ticks(ax, E, p):
    for u, v, d in E.edges(data=True):
        y, t = _row(u, v), d["time"]
        ax.plot([t, t], [y - 0.3, y + 0.3], color=p.kept, lw=2.0)
        if "weight" in d:
            ax.annotate(
                f"weight {d['weight']}",
                (t, y - 0.3),
                xytext=(3, 1),
                textcoords="offset points",
                fontsize=7.5,
                color=p.ink,
                va="bottom",
            )


# -- figure 1: the three forms on one time axis -------------------------------


def _forms_alt() -> str:
    S, E, P = _forms()
    return (
        f"Three timelines on a shared time axis from 0 to {T_MAX}, one row per "
        "pair (ann-bob, bob-cat). Snapshots: a filled square at each snapshot "
        f"time, {', '.join(map(str, S))}, for each pair present; "
        + "; ".join(
            f"the snapshot at time {t} holds only the isolated node {', '.join(nodes)}"
            for t, nodes in _only_isolated()
        )
        + ". Events: "
        f"a tick at each of the {E.number_of_edges()} event times "
        f"({', '.join(f'{t:g}' for *_, t in E.edges(data='time'))}). Spans: "
        "a bar from each start to its end, "
        + ", ".join(f"{u}-{v} from {s} to {e}" for u, v, s, e in _spans(P))
        + "."
    )


def _spans(P):
    return [(u, v, d["start"], d["end"]) for u, v, d in P.edges(data=True)]


def _forms_caption() -> str:
    (t, nodes), *_ = _only_isolated()
    return (
        "The page's three examples on one time axis. Snapshots ``S`` are "
        f"graphs at given times (time {t} holds only the isolated "
        f"``{', '.join(nodes)}``); "
        "events ``E`` are instants; spans ``P`` last from ``start`` up to, "
        "but not including, ``end`` (the open circle)."
    )


@figure("forms", alt=_forms_alt, caption=_forms_caption)
def draw_forms(mode):
    S, E, P = _forms()
    p = style.palette(mode)
    fig, axes = plt.subplots(
        3, 1, figsize=(6.4, 4.2), sharex=True, layout="constrained"
    )
    ax = axes[0]
    for t, G in S.items():
        ax.axvline(t, color=p.hairline, lw=0.75, zorder=0)
        for u, v in G.edges():
            ax.scatter(t, _row(u, v), marker="s", s=60, c=p.kept, zorder=3)
        isolated = [n for n in G if G.degree(n) == 0]
        if isolated and G.number_of_edges() == 0:
            ax.annotate(
                f"only {', '.join(isolated)}",
                (t, (len(PAIRS) - 1) / 2),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=7.5,
                color=p.muted,
                bbox={"boxstyle": "round,pad=0.2", "fc": p.surface, "ec": p.hairline},
            )
    _pair_axes(ax, "Snapshots S: a graph at each time", p)

    ax = axes[1]
    _event_ticks(ax, E, p)
    _pair_axes(ax, "Events E: one time per event", p)

    ax = axes[2]
    for u, v, start, end in _spans(P):
        y = _row(u, v)
        ax.add_patch(
            Rectangle(
                (start, y - 0.22), end - start, 0.44, facecolor=p.kept, linewidth=0
            )
        )
        ax.scatter(start, y, s=36, c=p.ink, zorder=3)
        ax.scatter(
            end, y, s=36, facecolors=p.surface, edgecolors=p.ink, lw=1.2, zorder=3
        )
    _pair_axes(ax, "Spans P: from start up to (not including) end", p)
    ax.set_xlabel("Time")
    ax.set_xticks(range(T_MAX + 1))
    return fig


# -- figure 2: binning the events with freq= -----------------------------------


def _bins_alt() -> str:
    parts = []
    for f, B in _binned().items():
        cells = "; ".join(
            f"bin {t}: "
            + (
                ", ".join(f"{u}-{v} weight {w}" for u, v, w in G.edges(data="weight"))
                or "empty"
            )
            for t, G in B.items()
        )
        parts.append(f"With freq={f}, {cells}")
    return (
        "Three timelines of the events E on a shared time axis: the events "
        "as ticks, then the snapshots binned with freq=1 and freq=2, each "
        "bin a cell labeled with its summed weight. " + ". ".join(parts) + "."
    )


def _bins_caption() -> str:
    B2 = _binned()[2]
    first = min(B2)
    (u, v, d), *_ = B2[first].edges(data=True)
    empty = [t for t, G in _binned()[1].items() if G.number_of_edges() == 0]
    return (
        ":func:`~nx_backbone.snapshots_from_events` with ``origin=0``. Each "
        "bin is a snapshot keyed by its start; a pair's events in one bin "
        f"become one edge with their summed ``weight`` and ``edge_count``. "
        f"With ``freq=2`` the {d['edge_count']} {u}–{v} events fall in bin "
        f"{first} and add up to weight {d['weight']}. With ``freq=1`` the "
        f"empty bin at time {', '.join(map(str, empty))} is kept."
    )


@figure("binning", alt=_bins_alt, caption=_bins_caption)
def draw_binning(mode):
    _, E, _ = _forms()
    p = style.palette(mode)
    fill = p.sequential[0]
    fig, axes = plt.subplots(
        1 + len(FREQS), 1, figsize=(6.4, 4.2), sharex=True, layout="constrained"
    )
    _event_ticks(axes[0], E, p)
    _pair_axes(axes[0], "Events E", p)
    for ax, (f, B) in zip(axes[1:], _binned().items()):
        for t, G in B.items():
            ax.axvline(t, color=p.hairline, lw=0.75, zorder=0)
            ax.axvline(t + f, color=p.hairline, lw=0.75, zorder=0)
            for u, v, w in G.edges(data="weight"):
                y = _row(u, v)
                ax.add_patch(
                    Rectangle(
                        (t + 0.05, y - 0.3),
                        f - 0.1,
                        0.6,
                        facecolor=fill,
                        linewidth=0,
                    )
                )
                ax.text(
                    t + f / 2,
                    y,
                    f"weight {w}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=p.ink,
                )
            if G.number_of_edges() == 0:
                ax.text(
                    t + f / 2,
                    (len(PAIRS) - 1) / 2,
                    "empty",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=p.muted,
                )
        _pair_axes(ax, f"snapshots_from_events(E, freq={f}, origin=0)", p)
    axes[-1].set_xlabel("Time (bins start at their key)")
    axes[-1].set_xticks(range(T_MAX + 1))
    return fig
