"""Figures of ``docs/tutorials/graph_gallery.rst``.

One overview grid: a backbone from each API module, built with the method
gallery's own specs and datasets (``scripts/generate_visualizations.py``), so
every panel is a small copy of a figure on that module's API page.  The page's
redraw fingerprint does not cover that script; ``rm -rf docs/_page_figures``
after changing it locally (a clean build always redraws).
"""

from __future__ import annotations

import functools
import importlib.util
import sys
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _figures import style
from _figures.pages import ROOT, figure

# One representative method per API module, in the page's order.
PANELS = [
    ("statistical", "disparity_filter"),
    ("structural", "metric_backbone"),
    ("proximity", "jaccard_backbone"),
    ("hybrid", "glab_filter"),
    ("unweighted", "lspar"),
    ("bipartite", "fixedrow"),
    ("hypergraph", "mdl_hypergraph_backbone"),
    ("temporal", "persistence"),
]


def _gallery():
    """The method gallery's module, loaded as ``docs/conf.py`` loads it."""
    name = "generate_visualizations"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "scripts" / "generate_visualizations.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses look the module up here
        spec.loader.exec_module(module)
    return module


def _short(c: style.BackboneCounts) -> str:
    return (
        f"{c.edges_kept} of {c.edges_total} edges, "
        f"{c.nodes_kept} of {c.nodes_total} nodes"
    )


@functools.cache
def _panels():
    """Per panel: module, method, original, backbone, positions, subtitle."""
    gv = _gallery()
    ctx = gv.context()
    specs = {(s.module, s.file): s for s in gv.method_specs()}
    panels = []
    for module, name in PANELS:
        if module == "hypergraph":
            hyperedges = ctx["davis_hyperedges"]
            kept = gv._hypergraph_backbone(name, ctx)
            original = gv._incidence(hyperedges)
            backbone = gv._incidence({e: hyperedges[e] for e in kept})
            pos = style.layout(original, "davis_hypergraph")
            covered = {v for e in kept for v in hyperedges[e]}
            panels.append(
                {
                    "module": module,
                    "name": name,
                    "dataset": "Davis Southern Women as a hypergraph",
                    "original": original,
                    "backbone": backbone,
                    "pos": pos,
                    "events": (list(hyperedges), kept),
                    "short": f"{len(kept)} of {len(hyperedges)} hyperedges",
                    "more": f", covering {len(covered)} of {len(ctx['women'])} women",
                }
            )
            continue
        spec = specs[module, name]
        dataset = gv.DATASETS[spec.dataset]
        original = dataset.graph(ctx)
        key = gv._LAYOUT_KEY.get(spec.dataset, spec.dataset)
        pos = style.layout(original, key, weight=dataset.layout_weight)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            backbone = spec.build(ctx)
        panels.append(
            {
                "module": module,
                "name": name,
                "dataset": dataset.title,
                "original": original,
                "backbone": backbone,
                "pos": pos,
                "events": None,
                "short": _short(style.backbone_counts(original, backbone)),
            }
        )
    return panels


def _alt() -> str:
    parts = "; ".join(
        f"{p['module']}, {p['name']} on {p['dataset']}: {p['short']}{p.get('more', '')}"
        for p in _panels()
    )
    return (
        f"A grid of {len(PANELS)} small network drawings, one backbone from each "
        "group of methods, each drawn over its original graph with kept edges "
        f"solid and removed edges faint. {parts}."
    )


def _caption() -> str:
    links = ", ".join(f":doc:`{p['module']} </api/{p['module']}>`" for p in _panels())
    return (
        "One backbone from each group of methods, drawn as in the gallery: "
        "kept edges solid, removed edges faint gray, and the edges and nodes "
        "kept under each title. The Davis hypergraph panel draws the events as "
        "squares joined to the women who attended them. The full-size figure "
        f"of every method is on its API page: {links}."
    )


@figure("overview", alt=_alt, caption=_caption, fmt="png")
def draw_overview(mode):
    p = style.palette(mode)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 5.4))
    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.85, bottom=0.12, wspace=0.06, hspace=0.5
    )
    for ax, panel in zip(axes.flat, _panels(), strict=True):
        style.draw_backbone(
            ax,
            panel["original"],
            panel["backbone"],
            panel["pos"],
            mode=mode,
            legend=False,
            node_size=style.NODE_SIZE * 0.55,
            kept_width=1.2,
            removed_width=0.45,
        )
        if panel["events"] is not None:
            events, kept = panel["events"]
            for group, color in (
                ([e for e in events if e not in kept], p.node_removed),
                (kept, p.node),
            ):
                xy = [panel["pos"]["e", e] for e in group]
                ax.scatter(
                    [x for x, _ in xy],
                    [y for _, y in xy],
                    marker="s",
                    s=style.NODE_SIZE * 0.8,
                    c=color,
                    edgecolors=p.surface,
                    linewidths=1.0,
                    zorder=5,
                )
        ax.set_title(panel["module"], pad=34)
        ax.text(
            0.0,
            1.0,
            "\n".join([panel["name"], *panel["short"].split(", ")]),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=p.muted,
            fontsize=7.5,
            linespacing=1.15,
        )
    handles = style.legend_handles(mode)
    handles.append(
        Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            markersize=6.5,
            markerfacecolor=p.node,
            markeredgecolor=p.surface,
            label="Hyperedge (event) kept",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncols=3,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=2.0,
        columnspacing=1.4,
    )
    return fig
