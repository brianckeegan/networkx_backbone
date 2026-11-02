"""Figures of ``docs/user_guide/choosing_method.rst``.

The page's "Decision guide" as a diagram: the questions in the page's order
(bipartite? unweighted? weighted: which goal?) and the method family each
answer leads to.  Every function, module and goal the diagram names is
checked against the text of that section, so the two cannot drift apart.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _figures import style
from _figures.pages import figure

PAGE = Path(__file__).resolve().parents[2] / "user_guide" / "choosing_method.rst"

QUESTIONS = [
    "Is the graph bipartite?",
    "Is it unweighted?",
    "It is weighted:\nwhat is your goal?",
]
#: (question index, answer on the arrow, family, lines, names in the text)
OUTCOMES = [
    (
        0,
        "yes",
        "bipartite",
        "sdsm (analytical), fdsm (Monte Carlo), or fixedfill,\n"
        "fixedrow, fixedcol; then filter by p-value",
        ["bipartite projection", "sdsm", "fdsm", "fixedfill", "fixedrow", "fixedcol"],
    ),
    (
        1,
        "yes",
        "unweighted",
        "sparsify, lspar, local_degree;\nthen boolean_filter on sparsify_keep",
        [
            "unweighted",
            "sparsify",
            "lspar",
            "local_degree",
            "boolean_filter",
            "sparsify_keep",
        ],
    ),
    (
        2,
        "Statistical significance",
        "statistical",
        "p-values under a null model;\nfilter at a significance level",
        ["statistical", "null model", "p-values", "significance level"],
    ),
    (
        2,
        "Topological structure",
        "structural",
        "shortest paths, spanning trees, planarity",
        ["structural", "shortest paths", "spanning trees", "planarity"],
    ),
    (
        2,
        "Neighborhood similarity",
        "proximity",
        "how similar the endpoints' neighborhoods are",
        ["proximity", "neighborhoods", "similar"],
    ),
    (
        2,
        "Combined approach",
        "hybrid: glab_filter",
        "betweenness and degree information",
        ["hybrid", "glab_filter", "betweenness and degree"],
    ),
]


@functools.cache
def _guide() -> str:
    """The text of the page's "Decision guide" section, whitespace collapsed."""
    text = PAGE.read_text(encoding="utf-8")
    match = re.search(r"Decision guide\n-+\n(.*?)\n\S[^\n]*\n-+\n", text, re.S)
    assert match, "the Decision guide section moved: update this module"
    guide = " ".join(match.group(1).split())
    for q in ("bipartite?", "unweighted?", "weighted?"):
        assert f"Is your graph {q}" in guide, q
    for _, answer, _, _, names in OUTCOMES:
        missing = [n for n in names if n not in guide]
        assert not missing, f"diagram names {missing} the guide does not"
        if answer != "yes":
            assert f"**{answer}**" in guide, answer
    return guide


def _alt() -> str:
    _guide()
    goals = "; ".join(
        f"{answer}: {family}, {' '.join(lines.split())}"
        for q, answer, family, lines, _ in OUTCOMES
        if q == 2
    )
    first, second = OUTCOMES[0], OUTCOMES[1]
    return (
        "Decision diagram read top to bottom. First question, is the graph "
        f"bipartite? Yes leads to {first[2]}: {' '.join(first[3].split())}. "
        f"No leads to the second question, is it unweighted? Yes leads to the "
        f"{second[2]} module: {' '.join(second[3].split())}. No means "
        f"weighted, and the goal picks the module: {goals}."
    )


def _caption() -> str:
    _guide()
    return (
        "The decision guide below as a diagram. The question order is the "
        "guide's; for a weighted graph the goal of the analysis picks the "
        "module. The comparison table and the section on choosing a "
        "statistical method narrow the choice within a module."
    )


def _box(ax, x, y, w, h, text, p, *, filled, title=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=p.hairline if filled else p.surface,
            edgecolor=p.muted if not filled else p.hairline,
            linewidth=0.9,
            zorder=2,
        )
    )
    if title:
        ax.text(
            x + 0.12,
            y + 0.02,
            title,
            fontsize=8.5,
            fontweight="bold",
            color=p.ink,
            va="bottom",
            zorder=3,
        )
        ax.text(x + 0.12, y - 0.02, text, fontsize=8, color=p.ink, va="top", zorder=3)
    else:
        ax.text(
            x + w / 2,
            y,
            text,
            fontsize=8.5,
            color=p.ink,
            ha="center",
            va="center",
            zorder=3,
        )


def _arrow(ax, start, end, p):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            color=p.muted,
            linewidth=0.9,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


@figure("decision", alt=_alt, caption=_caption)
def draw_decision(mode):
    _guide()
    p = style.palette(mode)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.set_axis_off()

    q_x, q_w, q_h = 0.2, 2.5, 0.62
    fam_x, fam_w = 5.6, 4.2
    rows = {0: 5.65, 1: 4.55}  # y of the first two questions and their outcomes
    goal_ys = [3.35, 2.4, 1.45, 0.5]
    q_ys = [rows[0], rows[1], (goal_ys[0] + goal_ys[-1]) / 2]
    trunk = q_x + q_w + 0.35  # x of the line the four goals branch off

    for i, (question, y) in enumerate(zip(QUESTIONS, q_ys)):
        _box(ax, q_x, y, q_w, q_h, question, p, filled=False)
        if i + 1 < len(QUESTIONS):
            x = q_x + q_w / 2
            _arrow(ax, (x, y - q_h / 2), (x, q_ys[i + 1] + q_h / 2), p)
            ax.text(
                x + 0.08,
                (y - q_h / 2 + q_ys[i + 1] + q_h / 2) / 2,
                "no",
                fontsize=8,
                color=p.muted,
                va="center",
            )
    # The goals: one line out of the last question, branching to each family.
    ax.plot([q_x + q_w, trunk], [q_ys[2], q_ys[2]], color=p.muted, lw=0.9, zorder=1)
    ax.plot([trunk, trunk], [goal_ys[0], goal_ys[-1]], color=p.muted, lw=0.9, zorder=1)

    goals = iter(goal_ys)
    for q, answer, family, lines, _ in OUTCOMES:
        fam_h = 0.8 if "\n" in lines else 0.66
        y = rows[q] if q in rows else next(goals)
        _box(ax, fam_x, y, fam_w, fam_h, lines, p, filled=True, title=family)
        start = (q_x + q_w, y) if q in rows else (trunk, y)
        _arrow(ax, start, (fam_x, y), p)
        ax.text(
            (start[0] + fam_x) / 2,
            y + 0.06,
            answer,
            fontsize=8,
            color=p.muted if q in rows else p.ink,
            ha="center",
            va="bottom",
            zorder=3,
        )
    return fig
