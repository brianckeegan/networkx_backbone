"""Figures of ``docs/tutorials/temporal_backbones.rst``."""

from __future__ import annotations

import functools
import math
import random
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx

import nx_backbone as nb
from _figures import style
from _figures.pages import figure

BLOCK = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
FLICKER = [(10, 11), (12, 13), (14, 15), (16, 17), (18, 19)]
N_NOISE = 3  # noise rows drawn: the most often observed noise edges
STEPS = 20


def _snapshots():
    """The page's synthetic snapshots, built exactly as its doctest builds them."""
    rng = random.Random(42)
    snaps = {t: nx.Graph() for t in range(STEPS)}
    for G in snaps.values():
        G.add_nodes_from(range(30))
    for u, v in BLOCK:
        first = rng.randrange(11)
        for t in range(first, first + 10):
            snaps[t].add_edge(u, v, weight=10)
    for u, v in FLICKER:
        for t in rng.sample(range(STEPS), 10):
            snaps[t].add_edge(u, v, weight=10)
    for G in snaps.values():
        for u, v in combinations(range(30), 2):
            if (u, v) not in BLOCK + FLICKER and rng.random() < 0.1:
                G.add_edge(u, v, weight=1)
    return snaps


def _mean(values) -> float:
    values = [x for x in values if not math.isnan(x)]
    return sum(values) / len(values)


@functools.cache
def _scored():
    snaps = _snapshots()
    H = nb.observation_fraction(snaps)
    P = nb.persistence(snaps)
    noise = [e for e in H.edges if tuple(sorted(e)) not in BLOCK + FLICKER]
    loud = sorted(noise, key=lambda e: (-H.edges[e]["obs_fraction"], sorted(e)))
    groups = {"block": BLOCK, "flicker": FLICKER, "noise": loud[:N_NOISE]}
    means = {
        name: {
            score: _mean(G.edges[e][score] for e in edges)
            for score, G in (("obs_fraction", H), ("persistence", P))
        }
        for name, edges in (("block", BLOCK), ("flicker", FLICKER), ("noise", noise))
    }
    # The page's doctest values (rounded to two decimals there).
    assert H.number_of_edges() == 382
    assert round(means["block"]["obs_fraction"], 2) == 0.5
    assert round(means["flicker"]["obs_fraction"], 2) == 0.5
    assert round(means["block"]["persistence"], 2) == 0.92
    assert round(means["flicker"]["persistence"], 2) == 0.49
    return snaps, H, P, groups, means, len(noise)


def _fmt(x: float) -> str:
    return "–" if math.isnan(x) else f"{x:.2f}"


def _alt() -> str:
    _, H, _, groups, means, n_noise = _scored()
    m = means
    top_noise = H.edges[groups["noise"][0]]["obs_fraction"]
    return (
        f"A grid of edges (rows) by the {STEPS} time steps (columns), with a "
        "filled square in each step where the edge is observed, and two "
        "columns of numbers on the right: observation fraction and "
        "persistence. The five block edges each fill one unbroken run of 10 "
        "steps; the five flicker edges are on in 10 scattered steps each; the "
        f"{N_NOISE} noise edges shown, the most often observed of {n_noise}, "
        f"are on in at most {top_noise:.0%} of the steps. Mean observation "
        f"fraction: block {m['block']['obs_fraction']:.2f}, flicker "
        f"{m['flicker']['obs_fraction']:.2f}, noise "
        f"{m['noise']['obs_fraction']:.2f}. Mean persistence: block "
        f"{m['block']['persistence']:.2f}, flicker "
        f"{m['flicker']['persistence']:.2f}."
    )


def _caption() -> str:
    _, _, _, _, means, n_noise = _scored()
    m = means
    return (
        "When each edge is observed. The block and the flicker edges are both "
        f"observed in half of the steps (mean ``obs_fraction`` "
        f"{m['block']['obs_fraction']:.2f} and "
        f"{m['flicker']['obs_fraction']:.2f}), which sets them apart from the "
        f"noise ({m['noise']['obs_fraction']:.2f} over all {n_noise} noise "
        f"edges; the {N_NOISE} shown are the most often observed) but not from "
        "each other. Only the order of the observations differs, and "
        ":func:`~nx_backbone.persistence` reads it: "
        f"{m['block']['persistence']:.2f} for the block against "
        f"{m['flicker']['persistence']:.2f} for the flicker."
    )


@figure("edge-activity", alt=_alt, caption=_caption)
def draw_edge_activity(mode):
    snaps, H, P, groups, _, _ = _scored()
    p = style.palette(mode)
    rows, labels, y = [], [], 0.0
    group_y = {}
    for name, edges in groups.items():
        group_y[name] = y
        for e in edges:
            rows.append((y, e))
            labels.append(f"{e[0]}–{e[1]}")
            y += 1
        y += 0.8  # gap between groups
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.11, top=0.9)
    for yy, (u, v) in rows:
        on = [t for t, G in snaps.items() if G.has_edge(u, v)]
        off = [t for t in snaps if t not in on]
        ax.scatter(off, [yy] * len(off), marker="s", s=6, c=p.hairline, linewidths=0)
        ax.scatter(on, [yy] * len(on), marker="s", s=48, c=p.kept, linewidths=0)
        for x, score, G in (
            (STEPS + 1.6, "obs_fraction", H),
            (STEPS + 5.1, "persistence", P),
        ):
            ax.text(x, yy, _fmt(G.edges[u, v][score]), ha="center", va="center")
    for x, label in ((STEPS + 1.6, "obs_fraction"), (STEPS + 5.1, "persistence")):
        ax.text(x, -0.9, label, ha="center", va="bottom", color=p.muted, fontsize=8)
    for name, gy in group_y.items():
        ax.text(
            -5.6,
            gy - 0.75,
            name if name != "noise" else f"noise (top {N_NOISE})",
            ha="left",
            va="center",
            fontweight="bold",
            color=p.ink,
            fontsize=8.5,
        )
    ax.set_yticks([yy for yy, _ in rows], labels)
    ax.set_xticks(range(0, STEPS, 2))
    ax.set_xlim(-0.6, STEPS + 6.6)
    ax.set_ylim(y - 0.5, -1.4)
    ax.set_xlabel("Time step")
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"Edge activity over the {STEPS} steps", pad=14)
    return fig
