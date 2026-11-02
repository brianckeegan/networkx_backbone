"""Figures of ``docs/user_guide/backends.rst``.

The committed nx-parallel timings, read from the page's own table (and the
date, CPU count and versions from the sentence above it), so the chart and
the table cannot disagree.  Nothing is timed during the build.
"""

from __future__ import annotations

import functools
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt

from _figures import style
from _figures.pages import figure

PAGE = Path(__file__).resolve().parents[2] / "user_guide" / "backends.rst"


@functools.cache
def _measured():
    """Rows ``(method, networkx_s, parallel_s, ratio)`` and the run's details."""
    text = PAGE.read_text(encoding="utf-8")
    rows = []
    for name, nx_s, par_s, ratio in re.findall(
        r"^``(\w+)``\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\S+\s*$", text, flags=re.M
    ):
        nx_s, par_s, ratio = float(nx_s), float(par_s), float(ratio)
        # The Ratio column is NetworkX time / nx-parallel time, computed from
        # the unrounded times: equal to the rounded times' ratio within 1%.
        assert math.isclose(nx_s / par_s, ratio, rel_tol=0.01), name
        rows.append((name, nx_s, par_s, ratio))
    prose = " ".join(text.split())
    details = {
        "date": r"Measured on (\d{4}-\d{2}-\d{2})",
        "cpus": r"(\d+) CPUs",
        "parallel": r"with and without nx-parallel ([\d.]+)",
        "nodes": r"graph with ([\d,]+) nodes",
        "edges": r"and ([\d,]+) integer-weighted edges",
    }
    found = {}
    for key, pattern in details.items():
        match = re.search(pattern, prose)
        assert match, f"backends.rst no longer states {key!r}: update this module"
        found[key] = match.group(1)
    assert len(rows) >= 3, "the timing table moved: update this module"
    return rows, found


def _title() -> str:
    _, d = _measured()
    return (
        f"Speed-up with nx-parallel {d['parallel']}, measured {d['date']} "
        f"on {d['cpus']} CPUs"
    )


def _alt() -> str:
    rows, d = _measured()
    each = "; ".join(
        f"{name} {ratio:.2f} times ({nx_s:.2f} s to {par_s:.2f} s)"
        for name, nx_s, par_s, ratio in sorted(rows, key=lambda r: -r[3])
    )
    return (
        "Horizontal bar chart of the speed-up from nx-parallel, the NetworkX "
        "time divided by the nx-parallel time, with a vertical line at 1 for "
        f"no change, measured on {d['date']} on {d['cpus']} CPUs: {each}."
    )


def _caption() -> str:
    rows, d = _measured()
    faster = [r[0] for r in rows if r[3] > 1.5]
    # The page: only the edge-betweenness methods gain.
    assert set(faster) == {"edge_betweenness_filter", "h_backbone"}, faster
    return (
        f"The timings in the table above, measured once on {d['date']} on a "
        f"{d['cpus']}-CPU machine with nx-parallel {d['parallel']}, on a "
        f"{d['nodes']}-node graph; the chart reads them from the table and "
        "times nothing. A bar past the line at 1 ran faster with nx-parallel. "
        f"Only the {len(faster)} methods built on edge betweenness gain; the "
        "others call NetworkX functions that nx-parallel does not speed up "
        "here. Other machines and graphs will give other numbers."
    )


@figure("speedups", alt=_alt, caption=_caption)
def draw_speedups(mode):
    rows, _ = _measured()
    p = style.palette(mode)
    rows = sorted(rows, key=lambda r: r[3])
    fig, ax = plt.subplots(figsize=(6.6, 2.9), layout="constrained")
    top = max(r[3] for r in rows)
    for y, (_name, nx_s, par_s, ratio) in enumerate(rows):
        ax.barh(
            y,
            ratio,
            height=0.55,
            color=p.kept if ratio > 1 else p.removed,
            linewidth=0,
        )
        ax.annotate(
            f"{ratio:.2f}×  ({nx_s:.2f} s → {par_s:.2f} s)",
            (max(ratio, 1.0), y),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=p.ink,
        )
    ax.axvline(1.0, color=p.ink, lw=0.9)
    ax.annotate(
        "1 = no change",
        (1.0, len(rows) - 0.45),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=p.muted,
    )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0, labelcolor=p.ink)
    ax.set_xlim(0, top * 1.55)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("Speed-up: NetworkX time / nx-parallel time")
    ax.set_title(_title())
    ax.grid(True, axis="x")
    ax.spines["left"].set_visible(False)
    return fig
