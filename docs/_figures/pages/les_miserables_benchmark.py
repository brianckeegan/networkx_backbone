"""Figures of ``docs/tutorials/les_miserables_benchmark.rst``.

The chart plots the page's table: the same cases
(``benchmarks/benchmarks/les_miserables.py``), each run as the table's
``.. les-miserables-table::`` directive (``docs/conf.py``) runs it.
"""

from __future__ import annotations

import functools
import warnings

import matplotlib.pyplot as plt

import nx_backbone as nb
from _figures import style
from _figures.pages import figure

# The API module of each scorer, in the table's order, and its heading.
FAMILIES = {
    "statistical": "Statistical",
    "structural": "Structural",
    "proximity": "Proximity",
    "hybrid": "Hybrid",
    "unweighted": "Unweighted",
}


@functools.cache
def _counts():
    """``(family, case name, edges kept)`` per case, and the original count."""
    from benchmarks.benchmarks import les_miserables as lm

    rows = []
    for case in lm.CASES:
        with warnings.catch_warnings():
            # As in the table: convergence warnings are the tests' concern.
            warnings.simplefilter("ignore")
            kept = case.run(lm.graph_for(case)).number_of_edges()
        family = getattr(nb, case.method).__module__.rpartition(".")[2]
        rows.append((family, case.name, kept))
    unknown = {f for f, *_ in rows} - set(FAMILIES)
    assert not unknown, f"add {unknown} to FAMILIES"
    return lm.les_miserables_weighted().number_of_edges(), rows


def _grouped():
    """Rows grouped by family, in FAMILIES order, largest first within each."""
    total, rows = _counts()
    groups = []
    for family, heading in FAMILIES.items():
        members = sorted((r for r in rows if r[0] == family), key=lambda r: -r[2])
        if members:
            groups.append((heading, [(name, kept) for _, name, kept in members]))
    return total, groups


def _alt() -> str:
    total, groups = _grouped()
    parts = []
    for heading, members in groups:
        listed = ", ".join(f"{name} {kept}" for name, kept in members)
        parts.append(f"{heading}: {listed}.")
    n = sum(len(m) for _, m in groups)
    return (
        f"Horizontal bar chart of the edges kept by each of the {n} methods "
        f"of the table, out of {total}, grouped by family and sorted within "
        "each family, largest first. " + " ".join(parts)
    )


def _caption() -> str:
    total, groups = _grouped()
    every = [(kept, name) for _, members in groups for name, kept in members]
    (lo, lo_name), (hi, hi_name) = min(every), max(every)
    proximity = dict(groups).get("Proximity", [])
    same = {kept for _, kept in proximity}
    tail = (
        f" Every proximity score keeps {same.pop()}: the table cuts them all "
        "at the same top fraction, so only *which* edges differ."
        if len(same) == 1
        else ""
    )
    return (
        f"The table above as a chart: edges kept of {total}, from {lo} "
        f"(``{lo_name}``) to {hi} (``{hi_name}``)." + tail
    )


@figure("edges-kept", alt=_alt, caption=_caption)
def draw_edges_kept(mode):
    total, groups = _grouped()
    pal = style.palette(mode)
    labels, values, headings = [], [], []
    y = 0.0
    ys = []
    for heading, members in groups:
        headings.append((y, heading))
        y += 1.0
        for name, kept in members:
            labels.append(name)
            values.append(kept)
            ys.append(y)
            y += 1.0
        y += 0.4
    fig, ax = plt.subplots(figsize=(4.4, 0.2 * y + 0.6))
    ax.barh(ys, values, height=0.7, color=pal.kept)
    for yi, v in zip(ys, values, strict=True):
        ax.text(v + total * 0.01, yi, str(v), va="center", fontsize=8.5, color=pal.ink)
    for yh, heading in headings:
        ax.text(
            0,
            yh,
            heading,
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color=pal.ink,
        )
    ax.set_yticks(ys, labels, fontsize=8.5)
    ax.tick_params(axis="y", length=0, labelcolor=pal.ink)
    ax.set_ylim(y - 0.4, -0.8)
    ax.set_xlim(0, total)
    ax.set_xlabel(f"edges kept (of {total})")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(True)
    return fig
