"""Figures of ``docs/user_guide/weight_transforms.rst``.

Les Misérables (co-appearance counts), its weights under three increasing
transforms, and the edge counts of backbones under those transforms and
under two count units.  The build also checks the page's "Count units"
numbers and its rescaling column against these runs.
"""

from __future__ import annotations

import functools
import math
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.ticker import MultipleLocator

import nx_backbone as nb
from _figures import style
from _figures.pages import figure

PAGE = Path(__file__).resolve().parents[2] / "user_guide" / "weight_transforms.rst"
ALPHA = 0.05
NODE = "Valjean"
#: name -> (label, function); identity first.  At most three series.
TRANSFORMS = {
    "w": ("w", None),
    "sqrt": ("√w", math.sqrt),
    "log1p": ("log(1 + w)", math.log1p),
}
UNITS = (1, 0.1)


@functools.cache
def _graph():
    return nx.les_miserables_graph()


@functools.cache
def _transformed(name):
    G = _graph()
    f = TRANSFORMS[name][1]
    return G if f is None else nb.transform_weights(G, f)


def _shares(name):
    """NODE's weight shares w/s under a transform, largest first."""
    H = _transformed(name)
    ws = sorted((w for *_, w in H.edges(NODE, data="weight")), reverse=True)
    s = math.fsum(ws)
    return [w / s for w in ws]


def _kept_from(name):
    """NODE's links in the disparity backbone at ALPHA under a transform."""
    H = nb.disparity_filter(_transformed(name))
    return sum(1 for *_, p in H.edges(NODE, data="disparity_pvalue") if p < ALPHA)


# Share-based and rank-based methods, as (label, scorer -> backbone).
SHARE_METHODS = {
    "disparity_filter": lambda H: nb.threshold_filter(
        nb.disparity_filter(H), "disparity_pvalue", ALPHA
    ),
    "multiple_linkage_analysis": lambda H: nb.boolean_filter(
        nb.multiple_linkage_analysis(H), "mla_keep"
    ),
    "lans_filter": lambda H: nb.threshold_filter(
        nb.lans_filter(H), "lans_pvalue", ALPHA
    ),
}
COUNT_METHODS = {
    "marginal_likelihood_filter": (nb.marginal_likelihood_filter, "ml_pvalue"),
    "ecm_filter": (nb.ecm_filter, "ecm_pvalue"),
    "noise_corrected_filter": (nb.noise_corrected_filter, "nc_pvalue"),
}


@functools.cache
def _transform_counts():
    """{method: {transform: edges kept}}, checked against rescaling by 4."""
    G = _graph()
    counts = {
        m: {t: run(_transformed(t)).number_of_edges() for t in TRANSFORMS}
        for m, run in SHARE_METHODS.items()
    }
    # The page's table: all three are unchanged by rescaling.
    G4 = nb.transform_weights(G, lambda w: 4 * w)
    for m, run in SHARE_METHODS.items():
        assert run(G4).number_of_edges() == counts[m]["w"], m
    return counts


@functools.cache
def _unit_counts():
    """{method: {unit: edges kept}}, checked against the page's prose."""
    G = _graph()
    counts = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # non-integer counts at unit=0.1
        for m, (scorer, attr) in COUNT_METHODS.items():
            counts[m] = {
                u: nb.threshold_filter(scorer(G, unit=u), attr, ALPHA).number_of_edges()
                for u in UNITS
            }
    # "keeps 215 instead of 126 edges for MLF at alpha = 0.05, 89 instead of
    # 72 for the ECM, and 215 instead of 126 for the noise-corrected filter."
    prose = " ".join(PAGE.read_text(encoding="utf-8").split())
    found = re.search(
        r"keeps (\d+) instead of (\d+) edges for MLF.*?(\d+) instead of (\d+) "
        r"for the ECM, and (\d+) instead of (\d+) for the noise-corrected",
        prose,
    )
    assert found, "the 'Count units' numbers moved: update this check"
    claimed = [int(x) for x in found.groups()]
    actual = [counts[m][u] for m in COUNT_METHODS for u in reversed(UNITS)]
    assert claimed == actual, f"page says {claimed}, the library gives {actual}"
    assert all(row[0.1] > row[1] for row in counts.values())  # "keep more edges"
    return counts


# -- figure 1: one node's weight shares under each transform -------------------


def _shares_alt() -> str:
    k = len(_shares("w"))
    tops = ", ".join(f"{lab} {_shares(t)[0]:.0%}" for t, (lab, _) in TRANSFORMS.items())
    return (
        f"Line chart of the weight shares of {NODE}'s {k} links in Les "
        "Misérables, sorted from largest to smallest, under three transforms "
        f"of the weights, with a gray line at the even share 1/{k}. The "
        f"largest share is {tops}; the transformed shares lie closer to the "
        "even share."
    )


def _shares_caption() -> str:
    k = len(_shares("w"))
    # "pull the shares toward that line, so fewer of them stand out"
    tops = [_shares(t)[0] for t in TRANSFORMS]
    kept = [_kept_from(t) for t in TRANSFORMS]
    assert tops == sorted(tops, reverse=True) and kept == sorted(kept, reverse=True)
    kept = ", ".join(
        f"{_kept_from(t)} under ``{lab}``" for t, (lab, _) in TRANSFORMS.items()
    )
    return (
        f"A transform changes what the disparity filter tests. {NODE} has {k} "
        "links; the filter compares each link's share of his strength with "
        f"an even split (gray line, 1/{k}). Square roots and logs pull the "
        "shares toward that line, so fewer of them stand out: of his links, "
        f"the backbone at p < {ALPHA} keeps {kept}."
    )


@figure("shares", alt=_shares_alt, caption=_shares_caption)
def draw_shares(mode):
    p = style.palette(mode)
    colors = style.categorical(len(TRANSFORMS), mode)
    markers = ["o", "s", "^"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
    k = len(_shares("w"))
    ax.axhline(1 / k, color=p.muted, lw=0.75, zorder=1)
    ax.annotate(
        f"even share, 1/{k}",
        (k, 1 / k),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color=p.muted,
    )
    for (name, (label, _)), color, marker in zip(TRANSFORMS.items(), colors, markers):
        shares = _shares(name)
        ranks = range(1, len(shares) + 1)
        ax.plot(
            ranks,
            shares,
            color=color,
            lw=1.5,
            marker=marker,
            markersize=5,
            markeredgecolor=p.surface,
            markeredgewidth=0.6,
            label=label,
            zorder=3,
        )
    ax.set_xlim(0, k + 1)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel(f"{NODE}'s links, largest share first")
    ax.set_ylabel("Share of his strength, w / s")
    ax.set_title(f"{NODE}'s weight shares under three transforms")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", ncols=3)
    return fig


# -- figure 2: edge counts under transforms and under unit= -----------------------


def _counts_alt() -> str:
    tc, uc = _transform_counts(), _unit_counts()
    left = "; ".join(
        f"{m}: " + ", ".join(f"{TRANSFORMS[t][0]} {n}" for t, n in row.items())
        for m, row in tc.items()
    )
    right = "; ".join(
        f"{m}: " + ", ".join(f"unit={u:g} {n}" for u, n in row.items())
        for m, row in uc.items()
    )
    return (
        "Two panels of horizontal bars, one above the other, edges kept out of "
        f"{_graph().number_of_edges()}. Top, by transform: {left}. Bottom, by "
        f"count unit: {right}."
    )


def _counts_caption() -> str:
    uc = _unit_counts()
    m = _graph().number_of_edges()
    return (
        f"Edges kept out of Les Misérables' {m} (p < {ALPHA}, or the method's "
        "keep flag). Top: an increasing transform changes the share-based "
        "methods but not :func:`~nx_backbone.lans_filter`, which uses only the "
        "order of each node's weights; multiplying every weight by 4 changes "
        "none of the three. Bottom: the count models read ``weight / unit`` as "
        "a number of events, so ``unit=0.1`` (ten events per co-appearance) "
        "gives them more evidence and they keep more edges, "
        f"{uc['marginal_likelihood_filter'][0.1]} instead of "
        f"{uc['marginal_likelihood_filter'][1]} for the marginal likelihood "
        "filter."
    )


def _bars(ax, rows, series, labels, mode, total):
    """Grouped horizontal bars, each labeled at its end with series and value."""
    p = style.palette(mode)
    colors = style.categorical(len(series), mode)
    height = 0.8 / len(series)
    for y, row in enumerate(rows.values()):
        for j, (key, color) in enumerate(zip(series, colors)):
            yy = y - 0.4 + height * (j + 0.5)
            n = row[key]
            ax.barh(yy, n, height=height * 0.82, color=color, linewidth=0)
            ax.annotate(
                f"{labels[j]}: {n}",
                (n, yy),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=p.ink,
            )
    ax.set_yticks(range(len(rows)), list(rows))
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, labelcolor=p.ink, labelsize=8)
    ax.set_xlim(0, total * 1.12)
    ax.set_xlabel(f"Edges kept (of {total})")
    ax.grid(True, axis="x")
    ax.spines["left"].set_visible(False)


@figure("edge-counts", alt=_counts_alt, caption=_counts_caption)
def draw_edge_counts(mode):
    tc, uc = _transform_counts(), _unit_counts()
    total = _graph().number_of_edges()
    fig, (ax_t, ax_u) = plt.subplots(
        2, 1, figsize=(6.6, 5.4), sharex=True, layout="constrained"
    )
    _bars(
        ax_t,
        tc,
        list(TRANSFORMS),
        [lab for lab, _ in TRANSFORMS.values()],
        mode,
        total,
    )
    ax_t.set_title("Transform the weights")
    ax_t.set_xlabel("")
    _bars(ax_u, uc, list(UNITS), [f"unit={u:g}" for u in UNITS], mode, total)
    ax_u.set_title("Change the count unit")
    return fig
