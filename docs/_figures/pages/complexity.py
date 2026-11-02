"""Figures of ``docs/user_guide/complexity.rst``.

The time classes in the page's own tables, read from the page, and what each
one predicts for a graph ten times larger.  Nothing is timed: the measured
run times are in the scaling guide, which the page links.
"""

from __future__ import annotations

import functools
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

from _figures import style
from _figures.pages import figure

PAGE = Path(__file__).resolve().parents[2] / "user_guide" / "complexity.rst"
#: A sparse graph growing tenfold: n nodes and m = EDGES_PER_NODE * n edges.
N0 = 1_000
EDGES_PER_NODE = 3
GROWTH = 10

_ROW = re.compile(
    r"^   \* - ``(?P<name>\w+)``\n"
    r"     - ``(?P<time>[^`]*)``\n"
    r"     - ``[^`]*``\n"
    r"     - ?(?P<notes>.*)$",
    re.M,
)
_TOKEN = re.compile(r"log [nm]|[nm]|\d+|[\^+*()]")


def _cost(expr: str):
    """``O(...)`` in n and m only -> a function ``cost(n, m)``; else ``None``.

    Classes with other sizes (iterations, trials, snapshots, ...) or prose
    have no single growth factor and are counted separately.
    """
    match = re.fullmatch(r"O\((.*)\)", expr.strip())
    if not match:
        return None
    inner = match.group(1)
    tokens = _TOKEN.findall(inner)
    # Only n, m, log, integers and + * ^ ( ): anything else has other sizes.
    if "".join(tokens).replace(" ", "") != inner.replace(" ", ""):
        return None
    out = []
    ends_factor = False
    for tok in tokens:
        starts_factor = tok in ("n", "m", "(") or tok.startswith("log") or tok.isdigit()
        if ends_factor and starts_factor and out[-1] != "**":
            out.append("*")
        out.append(
            "**" if tok == "^" else f"log({tok[-1]})" if tok.startswith("log") else tok
        )
        ends_factor = tok in ("n", "m", ")") or tok.startswith("log") or tok.isdigit()
    code = "".join(out)

    def cost(n, m):
        return eval(code, {"__builtins__": {}}, {"n": n, "m": m, "log": math.log})

    cost(2.0, 3.0)  # fails here, at build time, on anything unexpected
    return cost


@functools.cache
def _classes():
    """``{class: [functions]}`` for the n-and-m classes, and the rest."""
    text = PAGE.read_text(encoding="utf-8")
    by_class: dict[str, list[str]] = defaultdict(list)
    other: list[str] = []
    rows = _ROW.findall(text)
    assert len(rows) >= 60, "the complexity tables moved: update this module"
    for name, time, notes in rows:
        if notes.strip().startswith("Alias for"):
            continue
        if _cost(time) is None:
            other.append(name)
        else:
            by_class[time].append(name)
    return dict(by_class), other


def _factor(expr: str) -> float:
    cost = _cost(expr)
    n0, n1 = N0, N0 * GROWTH
    return cost(n1, EDGES_PER_NODE * n1) / cost(n0, EDGES_PER_NODE * n0)


#: (power of ten, extra log factor) -> name of the growth family.
FAMILIES = {
    (0, False): "constant",
    (1, False): "linear",
    (1, True): "n log n",
    (2, False): "quadratic",
    (2, True): "n² log n",
    (3, False): "cubic",
}


def _family(factor: float) -> tuple[int, bool]:
    power = round(math.log10(factor))
    return power, factor / 10**power > 1.15


@functools.cache
def _rows():
    """``[(family, [factors], [classes], [functions])]``, slowest-growing first."""
    by_class, _ = _classes()
    groups: dict[tuple[int, bool], tuple[list, list, list]] = {}
    for expr, names in by_class.items():
        factor = _factor(expr)
        key = _family(factor)
        assert key in FAMILIES, (expr, factor)
        factors, classes, funcs = groups.setdefault(key, ([], [], []))
        factors.append(factor)
        classes.append(expr)
        funcs.extend(names)
    return [
        (FAMILIES[key], sorted(fs), sorted(cs, key=len), sorted(ns))
        for key, (fs, cs, ns) in sorted(groups.items())
    ]


def _num(x: float) -> str:
    return f"{x:,.0f}" if x >= 100 else f"{x:.3g}"


def _span(factors) -> str:
    lo, hi = _num(min(factors)), _num(max(factors))
    return f"{lo}×" if lo == hi else f"{lo}–{hi}×"


def _plural(k: int, word: str) -> str:
    return f"{k} {word}{'' if k == 1 else 's'}"


def _alt() -> str:
    rows = _rows()
    _, other = _classes()
    each = "; ".join(
        f"{fam} ({', '.join(cs)}): {_span(fs)}, {_plural(len(ns), 'function')}"
        for fam, fs, cs, ns in rows
    )
    return (
        "Dot chart, one row per family of time classes from the tables on "
        "this page, placed on a logarithmic axis at how many times longer "
        f"the classes predict for a sparse graph {GROWTH} times larger: "
        f"{each}. {_plural(len(other), 'function')} whose classes depend on "
        "other sizes are not shown."
    )


def _caption() -> str:
    rows = _rows()
    _, other = _classes()
    total = sum(len(ns) for *_, ns in rows)
    return (
        f"What the time classes below predict when a sparse graph grows "
        f"{GROWTH} times, from {N0:,} nodes and {EDGES_PER_NODE * N0:,} edges "
        f"to {GROWTH * N0:,} and {GROWTH * EDGES_PER_NODE * N0:,}: the ratio "
        "of each class's expression at the two sizes, with the classes "
        "grouped by how fast they grow. Constants and lower-order costs are "
        "left out, so this is a prediction for large graphs, not a "
        "measurement; the scaling guide measures run times. The chart reads "
        f"the classes of {total} functions from the tables (aliases left "
        f"out); {len(other)} more depend on other sizes, such as iterations, "
        "trials or snapshots."
    )


def _wrap(classes, width=34) -> str:
    lines, line = [], ""
    for c in classes:
        piece = c if not line else f", {c}"
        if line and len(line) + len(piece) > width:
            lines.append(line + ",")
            line = c
        else:
            line += piece
    return "\n".join([*lines, line])


@figure("growth", alt=_alt, caption=_caption)
def draw_growth(mode):
    rows = _rows()
    p = style.palette(mode)
    height = sum(0.3 + 0.16 * _wrap(cs).count("\n") for _, _, cs, _ in rows)
    fig, ax = plt.subplots(figsize=(7.0, height + 1.2), layout="constrained")
    top = max(max(fs) for _, fs, _, _ in rows)
    for y, (fam, factors, _, funcs) in enumerate(rows):
        mid = math.sqrt(min(factors) * max(factors))
        ax.plot([1, mid], [y, y], color=p.hairline, lw=2.0, zorder=1)
        ax.scatter(mid, y, s=46, c=p.kept, edgecolors=p.surface, zorder=3)
        ax.annotate(
            f"{fam}: {_span(factors)}, {_plural(len(funcs), 'function')}",
            (mid, y),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=p.ink,
        )
    ax.set_yticks(range(len(rows)), [_wrap(cs) for _, _, cs, _ in rows])
    ax.tick_params(axis="y", length=0, labelcolor=p.ink, labelsize=7.5)
    ax.set_xscale("log")
    ax.set_xlim(0.7, top * 40)
    ticks = [t for t in (1, 10, 100, 1000, 10000) if t <= top * 2]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}×"))
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel(
        f"Predicted time for a graph {GROWTH}× larger (m = {EDGES_PER_NODE}n), "
        "relative to now"
    )
    ax.set_title("What the time classes predict (not measured)", loc="left")
    ax.grid(True, axis="x")
    ax.spines["left"].set_visible(False)
    return fig
