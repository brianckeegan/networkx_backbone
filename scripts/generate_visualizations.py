#!/usr/bin/env python3
"""Generate the per-method gallery shown on the API pages.

Every public backbone method has a :class:`MethodSpec`: the dataset it runs
on, how its scores are cut into a backbone, and a ``build`` function.
:func:`build` runs each one and draws the backbone over its original graph
with the shared docs style (``docs/_figures/style.py``): kept edges
emphasized, removed edges faint, one fixed layout per dataset so the methods
can be compared at a glance, in a light and a dark variant.  It also writes,
per API module, a reStructuredText include (``<module>.inc``) with each
figure, its caption and alt text, both carrying the counts.

``docs/conf.py`` calls :func:`build` on every Sphinx build (like
``scripts/roadmap_figures.py``), writing into the gitignored
``docs/_gallery/``, so the figures always show what the code does now.
Nothing generated is committed.  Run ``python scripts/generate_visualizations.py``
to render them by hand.

:func:`validate_coverage` fails the build when a public method has no spec.
"""

from __future__ import annotations

import argparse
import random
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
for _path in (ROOT, DOCS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _figures import style  # noqa: E402

import nx_backbone as nb  # noqa: E402

# The API modules with a gallery include, in the order the includes are written.
MODULES = (
    "statistical",
    "structural",
    "proximity",
    "hybrid",
    "unweighted",
    "bipartite",
    "hypergraph",
    "temporal",
)


@dataclass(frozen=True)
class Dataset:
    """A dataset the gallery draws on, with its original graph."""

    key: str
    title: str  # used in captions: "on Les Misérables"
    alt: str  # what the original graph is, for alt text
    graph: Callable[[dict], nx.Graph]
    layout_weight: str | None = None


@dataclass(frozen=True)
class MethodSpec:
    """One gallery figure: a public method, its dataset, and its cut."""

    name: str  # the public function, shown as the figure title
    module: str  # which API page include the figure goes to
    dataset: str
    rule: str  # how scores become a backbone, shown under the title
    build: Callable[[dict], Any]
    sign: str | None = None  # edge attribute with the sign, for signed backbones
    slug: str | None = None  # file name, when one method has several figures
    note: str = ""  # extra sentence for the include text

    @property
    def file(self) -> str:
        return self.slug or self.name


# -- data -------------------------------------------------------------------------


def _les_unweighted(G):
    H = nx.Graph()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges())
    return H


def _temporal_snapshots():
    """The synthetic snapshots of docs/tutorials/temporal_backbones.rst."""
    rng = random.Random(42)
    block = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    flicker = [(10, 11), (12, 13), (14, 15), (16, 17), (18, 19)]
    snaps = {t: nx.Graph() for t in range(20)}
    for G in snaps.values():
        G.add_nodes_from(range(30))
    for u, v in block:
        first = rng.randrange(11)
        for t in range(first, first + 10):
            snaps[t].add_edge(u, v, weight=10)
    for u, v in flicker:
        for t in rng.sample(range(20), 10):
            snaps[t].add_edge(u, v, weight=10)
    for G in snaps.values():
        for u, v in combinations(range(30), 2):
            if (u, v) not in block + flicker and rng.random() < 0.1:
                G.add_edge(u, v, weight=1)
    return snaps


def context():
    """Load every dataset once."""
    les = nx.les_miserables_graph()
    davis = nx.davis_southern_women_graph()
    women = [n for n, d in davis.nodes(data=True) if d["bipartite"] == 0]
    events = [n for n, d in davis.nodes(data=True) if d["bipartite"] == 1]
    snaps = _temporal_snapshots()
    return {
        "les_weighted": les,
        "les_unweighted": _les_unweighted(les),
        "davis": davis,
        "women": women,
        "events": events,
        # Project from the bare edge list: NetworkX's projections copy node and
        # graph attributes without declaring that to its dispatcher, which the
        # nx_loopback test harness reports as a mismatch (see tests/test_bipartite.py).
        "women_projection": nx.bipartite.weighted_projected_graph(
            nx.Graph(davis.edges()), women
        ),
        # Davis as a hypergraph: each event is a hyperedge on its attendees.
        "davis_hyperedges": {e: frozenset(davis[e]) for e in events},
        "snapshots": snaps,
        "temporal_union": nb.flatten_temporal(snaps),
    }


DATASETS = {
    d.key: d
    for d in (
        Dataset(
            "les_miserables",
            "Les Misérables",
            "the Les Misérables co-appearance network (77 characters, 254 "
            "weighted edges)",
            lambda c: c["les_weighted"],
        ),
        Dataset(
            "les_miserables_unweighted",
            "Les Misérables (unweighted)",
            "the unweighted Les Misérables co-appearance network (77 "
            "characters, 254 edges)",
            lambda c: c["les_unweighted"],
        ),
        Dataset(
            "davis",
            "Davis Southern Women",
            "the Davis Southern Women co-attendance projection (18 women; an "
            "edge joins two women who attended at least one event together)",
            lambda c: c["women_projection"],
            layout_weight="weight",
        ),
        Dataset(
            "temporal",
            "20 synthetic snapshots",
            "the union of 20 synthetic snapshots on 30 nodes (the temporal "
            "backbones tutorial's data: 5 persistent heavy edges, 5 "
            "flickering heavy edges, and light noise)",
            lambda c: c["temporal_union"],
        ),
    )
}
# Same nodes, so the same positions: every Les Misérables figure is comparable.
_LAYOUT_KEY = {"les_miserables_unweighted": "les_miserables"}


# -- specs ------------------------------------------------------------------------


def _pvalue(module, name, score, *, dataset="les_miserables", alpha=0.05, **kw):
    fn = getattr(nb, name)
    graph = DATASETS[dataset].graph
    return MethodSpec(
        name,
        module,
        dataset,
        f"{score} < {alpha}",
        lambda c: nb.threshold_filter(
            fn(graph(c), **kw), score, alpha, mode="below", include_all_nodes=False
        ),
    )


def _keep(module, name, flag, *, dataset="les_miserables", **kw):
    fn = getattr(nb, name)
    graph = DATASETS[dataset].graph
    args = ", ".join(f"{k}={v}" for k, v in kw.items())
    return MethodSpec(
        name,
        module,
        dataset,
        f"{flag}" + (f" ({args})" if args else ""),
        lambda c: nb.boolean_filter(fn(graph(c), **kw), flag),
    )


def _above(module, name, score, threshold, **kw):
    fn = getattr(nb, name)
    return MethodSpec(
        name,
        module,
        "les_miserables",
        f"{score} ≥ {threshold}",
        lambda c: nb.threshold_filter(
            fn(c["les_weighted"], **kw),
            score,
            threshold,
            mode="above",
            include_all_nodes=False,
        ),
    )


def _top(name, score, fraction=0.3, **kw):
    fn = getattr(nb, name)
    return MethodSpec(
        name,
        "proximity",
        "les_miserables",
        f"top {fraction:.0%} of edges by {score}",
        lambda c: nb.fraction_filter(
            fn(c["les_weighted"], **kw), score, fraction, ascending=False
        ),
    )


def _bipartite(name, *, signed=False, note="", **kw):
    fn = getattr(nb, name)
    if name in ("sdsm", "fdsm"):
        score = f"{name}_pvalue"
        rule = f"{score} < 0.05"

        def build(c):
            return nb.threshold_filter(
                fn(c["davis"], c["women"], signed=signed, **kw),
                score,
                0.05,
                mode="below",
                include_all_nodes=False,
            )

    else:
        rule = "alpha = 0.05" + (", signed=True" if signed else "")

        def build(c):
            return fn(c["davis"], c["women"], alpha=0.05, signed=signed, **kw)

    return MethodSpec(
        name,
        "bipartite",
        "davis",
        rule,
        build,
        sign="sign" if signed else None,
        slug=f"{name}_signed" if signed else None,
        note=note,
    )


def _temporal(name, score, threshold, **kw):
    fn = getattr(nb, name)
    return MethodSpec(
        name,
        "temporal",
        "temporal",
        f"{score} ≥ {threshold}",
        lambda c: nb.threshold_filter(
            fn(c["snapshots"], **kw),
            score,
            threshold,
            mode="above",
            include_all_nodes=False,
        ),
    )


def method_specs():
    """Every gallery figure, in page order within each module."""
    return [
        # Statistical
        _pvalue("statistical", "disparity_filter", "disparity_pvalue"),
        _pvalue("statistical", "noise_corrected_filter", "nc_pvalue"),
        _pvalue("statistical", "marginal_likelihood_filter", "ml_pvalue"),
        _pvalue("statistical", "ecm_filter", "ecm_pvalue"),
        _pvalue("statistical", "lans_filter", "lans_pvalue"),
        _keep("statistical", "multiple_linkage_analysis", "mla_keep"),
        _pvalue("statistical", "polya_filter", "polya_pvalue"),
        _pvalue("statistical", "gloss_filter", "gloss_pvalue"),
        # Structural
        _keep(
            "structural",
            "global_threshold_filter",
            "global_threshold_keep",
            threshold=2.0,
        ),
        _keep("structural", "strongest_n_ties", "strongest_n_ties_keep", n=2),
        _keep(
            "structural", "global_sparsification", "global_sparsification_keep", s=0.4
        ),
        _keep("structural", "primary_linkage_analysis", "primary_linkage_keep"),
        _keep("structural", "edge_betweenness_filter", "edge_betweenness_keep", s=0.3),
        _keep("structural", "node_degree_filter", "node_degree_keep", min_degree=2),
        _above("structural", "high_salience_skeleton", "salience", 0.5),
        _keep("structural", "metric_backbone", "metric_keep"),
        _keep("structural", "ultrametric_backbone", "ultrametric_keep"),
        _above("structural", "doubly_stochastic_filter", "ds_weight", 0.1),
        _keep("structural", "h_backbone", "h_backbone_keep"),
        _keep("structural", "modularity_backbone", "modularity_keep", seed=42),
        _keep("structural", "planar_maximally_filtered_graph", "pmfg_keep"),
        _keep("structural", "maximum_spanning_tree_backbone", "mst_keep"),
        # Proximity
        _top("neighborhood_overlap", "overlap"),
        _top("jaccard_backbone", "jaccard"),
        _top("dice_backbone", "dice"),
        _top("cosine_backbone", "cosine"),
        _top("hub_promoted_index", "hpi"),
        _top("hub_depressed_index", "hdi"),
        _top("lhn_local_index", "lhn_local"),
        _top("preferential_attachment_score", "pa"),
        _top("adamic_adar_index", "adamic_adar"),
        _top("resource_allocation_index", "resource_allocation"),
        _top("graph_distance_proximity", "dist", all_pairs=False),
        _top("local_path_index", "lp"),
        # Hybrid
        _pvalue("hybrid", "glab_filter", "glab_pvalue"),
        # Unweighted
        _keep(
            "unweighted",
            "sparsify",
            "sparsify_keep",
            dataset="les_miserables_unweighted",
            s=0.5,
        ),
        _keep(
            "unweighted",
            "lspar",
            "sparsify_keep",
            dataset="les_miserables_unweighted",
            s=0.5,
        ),
        _keep(
            "unweighted",
            "local_degree",
            "sparsify_keep",
            dataset="les_miserables_unweighted",
            s=0.3,
        ),
        # Bipartite (Davis Southern Women, projected onto the women)
        _bipartite(
            "sdsm",
            note="No pair is significant at this level; R ``backbone`` agrees "
            "(see the changelog's reference tests).",
        ),
        _bipartite("fdsm", trials=250, seed=42),
        _bipartite("fixedfill"),
        _bipartite("fixedrow"),
        _bipartite("fixedcol"),
        _bipartite("fixedrow", signed=True),
        # Temporal (the temporal backbones tutorial's snapshots)
        _temporal("observation_fraction", "obs_fraction", 0.4),
        _temporal("persistence", "persistence", 0.8),
        _temporal("backbone_stability", "stability", 0.3, method=nb.disparity_filter),
        _temporal("weight_trend", "weight_mean", 3),
        MethodSpec(
            "edge_persistence_hmm",
            "temporal",
            "temporal",
            "hmm_keep (min_steps=8)",
            lambda c: nb.boolean_filter(
                nb.edge_persistence_hmm(c["snapshots"], min_steps=8), "hmm_keep"
            ),
        ),
        MethodSpec(
            "kalman_backbone",
            "temporal",
            "temporal",
            "kalman_pvalue < 0.05 (disparity_filter per step)",
            lambda c: nb.threshold_filter(
                nb.kalman_backbone(c["snapshots"], nb.disparity_filter, seed=0),
                "kalman_pvalue",
                0.05,
                mode="below",
            ),
        ),
    ]


# Hypergraph figures draw hyperedges, not edges: see _draw_hypergraph.
HYPERGRAPH_METHODS = {
    "mdl_hypergraph_backbone": "global MDL, parameter-free",
    "local_mdl_hypergraph_backbone": "local MDL, defaults",
}

# Public names that are not scorers, or are drawn under another name.
NOT_DRAWN = {
    # expands kalman_backbone's per-step series; drawn as kalman_backbone
    "kalman_snapshots",
    # short aliases of drawn methods
    "disparity",
    "mlf",
    "lans",
    "svh",
    "svc",
    # bipartite projections, samplers and convenience wrappers
    "simple_projection",
    "hyper_projection",
    "probs_projection",
    "ycn_projection",
    "bipartite_projection",
    "bicm",
    "fastball",
    "backbone_from_projection",
    "backbone_from_weighted",
    "backbone_from_unweighted",
    "backbone",
    # hypergraph helpers and result classes
    "intersection_graph",
    "hypergraph_compression_ratio",
    "HypergraphBackbone",
    "ValidatedHypergraph",
    "maximal_hyperedges",
    "order_filter",
    "s_components",
    # validates no hyperedge of the 14 Davis events at any usual alpha, so
    # its figure would show nothing; the hypergraph tutorial covers it
    "statistically_validated_hypergraph",
    "statistically_validated_cores",
    # temporal conversions
    "temporal_form",
    "convert_temporal",
    "flatten_temporal",
    "snapshots_from_events",
    "snapshots_from_spans",
    "events_from_snapshots",
    "spans_from_snapshots",
    "events_from_spans",
    "spans_from_events",
    "from_event_table",
    "to_event_table",
}


def validate_coverage(specs):
    """Raise ``RuntimeError`` unless every public method has a gallery figure."""
    import importlib

    expected = set()
    for module in MODULES:
        expected |= set(importlib.import_module(f"nx_backbone.{module}").__all__)
    expected -= NOT_DRAWN
    drawn = {s.name for s in specs} | set(HYPERGRAPH_METHODS)
    missing, extra = sorted(expected - drawn), sorted(drawn - expected)
    if missing or extra:
        raise RuntimeError(
            "gallery coverage: add a MethodSpec in scripts/generate_visualizations.py"
            f" (or list the name in NOT_DRAWN); missing: {missing}, unexpected:"
            f" {extra}"
        )


# -- drawing ----------------------------------------------------------------------


def _figure(mode, title, subtitle):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.6), layout="constrained")
    p = style.palette(mode)
    ax.set_title(title, loc="left", pad=18)
    ax.text(
        0.0,
        1.0,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=p.muted,
        fontsize=8.5,
    )
    return fig, ax


def _draw_method(spec, original, backbone, pos):
    counts = {}

    def draw(mode):
        fig, ax = _figure(mode, spec.name, spec.rule)
        counts[mode] = style.draw_backbone(
            ax, original, backbone, pos, mode=mode, sign=spec.sign
        )
        return fig

    return draw, counts


def _incidence(hyperedges):
    """Bipartite incidence graph: hyperedge nodes ``("e", name)`` -- members."""
    B = nx.Graph()
    for name, members in hyperedges.items():
        B.add_node(("e", name))
        B.add_edges_from((("e", name), v) for v in members)
    return B


def _hypergraph_backbone(name, ctx):
    hyperedges = ctx["davis_hyperedges"]
    H = list(hyperedges.values())
    kept_sets = {frozenset(h) for h in getattr(nb, name)(H).backbone}
    return [e for e, members in hyperedges.items() if members in kept_sets]


def _draw_hypergraph(name, ctx, pos):
    from matplotlib.lines import Line2D

    hyperedges = ctx["davis_hyperedges"]
    original = _incidence(hyperedges)
    kept = _hypergraph_backbone(name, ctx)
    backbone = _incidence({e: hyperedges[e] for e in kept})
    covered = {v for e in kept for v in hyperedges[e]}
    info = {
        "kept": len(kept),
        "total": len(hyperedges),
        "covered": len(covered),
        "women": len(ctx["women"]),
    }

    def draw(mode):
        p = style.palette(mode)
        fig, ax = _figure(mode, name, HYPERGRAPH_METHODS[name])
        style.draw_backbone(ax, original, backbone, pos, mode=mode, legend=False)
        # Hyperedges (events) as squares on top of the round women.
        for group, color in (
            ([e for e in hyperedges if e not in kept], p.node_removed),
            (kept, p.node),
        ):
            if group:
                xy = [pos[("e", e)] for e in group]
                ax.scatter(
                    [x for x, _ in xy],
                    [y for _, y in xy],
                    marker="s",
                    s=style.NODE_SIZE * 1.4,
                    c=color,
                    edgecolors=p.surface,
                    linewidths=style.RING_WIDTH,
                    zorder=5,
                )
        handles = style.legend_handles(mode)
        handles[0].set_label("Membership in a kept hyperedge")
        handles[1].set_label("Membership in a removed hyperedge")
        handles[2].set_label("Woman in a kept hyperedge")
        handles[3].set_label("Woman in no kept hyperedge")
        square = {"marker": "s", "linestyle": "", "markersize": 7.5}
        handles += [
            Line2D(
                [],
                [],
                markerfacecolor=p.node,
                markeredgecolor=p.surface,
                markeredgewidth=style.RING_WIDTH,
                label="Hyperedge (event) kept",
                **square,
            ),
            Line2D(
                [],
                [],
                markerfacecolor=p.node_removed,
                markeredgewidth=0,
                label="Hyperedge (event) removed",
                **square,
            ),
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncols=2,
            handlelength=2.4,
            columnspacing=1.6,
        )
        return fig

    return draw, info


# -- reST -------------------------------------------------------------------------


def _rst_figure(lines, stem, alt, caption):
    lines += style.figure_rst(stem, alt, caption)


def _method_rst(spec, counts, dataset, stem):
    lines = [
        f":func:`~nx_backbone.{spec.name}` on {dataset.title}, keeping "
        f"``{spec.rule}``.{(' ' + spec.note) if spec.note else ''}",
        "",
    ]
    if spec.sign:
        caption = (
            "Backbone over the original graph (gray): positive edges solid blue, "
            f"negative edges dashed red. {counts.caption}"
        )
    else:
        caption = f"Backbone (blue) over the original graph (gray). {counts.caption}"
    alt = (
        f"Network drawing of {dataset.alt}, with the {spec.name} backbone "
        f"({spec.rule}) highlighted and the removed edges faint. {counts.caption}"
    )
    if spec.sign:
        alt += " Positive backbone edges are solid lines, negative ones dashed."
    _rst_figure(lines, stem, alt, caption)
    return lines


def _hypergraph_rst(name, info, stem):
    summary = (
        f"Kept {info['kept']} of {info['total']} hyperedges, covering "
        f"{info['covered']} of {info['women']} women."
    )
    lines = [
        f":func:`~nx_backbone.{name}` on the Davis Southern Women "
        f"hypergraph (each of the 14 events is a hyperedge on the women who "
        f"attended it), {HYPERGRAPH_METHODS[name]}.",
        "",
    ]
    alt = (
        "Bipartite drawing of the Davis Southern Women hypergraph: 14 events "
        "(squares) linked to the 18 women who attended them (circles), with the "
        f"hyperedges kept by {name} highlighted. {summary}"
    )
    _rst_figure(lines, stem, alt, f"Kept hyperedges in blue. {summary}")
    return lines


# Every figure is saved at its full, fixed size (no tight crop), so the text is
# the same size on every API page whatever the shape of the drawing.
_SAVE = {"bbox_inches": None}


# -- build ------------------------------------------------------------------------


def build(out_dir, *, static_prefix="/_gallery"):
    """Render every gallery figure and the per-module includes into *out_dir*.

    Parameters
    ----------
    out_dir : path-like
        Directory for ``<dataset>/<method>-{light,dark}.svg`` and
        ``<module>.inc``; created if needed.
    static_prefix : str, optional
        How the includes refer to *out_dir* (absolute docs path).

    Returns
    -------
    rows : list of dict
        One row per figure: name, module, dataset, edges kept and total.
    """
    import matplotlib

    matplotlib.use("Agg")

    out = Path(out_dir)
    ctx = context()
    specs = method_specs()
    validate_coverage(specs)

    includes = {m: [] for m in MODULES}
    rows = []
    for spec in specs:
        dataset = DATASETS[spec.dataset]
        original = dataset.graph(ctx)
        layout_key = _LAYOUT_KEY.get(spec.dataset, spec.dataset)
        pos = style.layout(original, layout_key, weight=dataset.layout_weight)
        with warnings.catch_warnings():
            # Convergence and same-edge-count warnings belong to the tests.
            warnings.simplefilter("ignore")
            backbone = spec.build(ctx)
        stray = [
            e for e in backbone.edges() if not original.has_edge(*e) and e[0] != e[1]
        ]
        if stray:
            raise RuntimeError(
                f"{spec.file}: {len(stray)} backbone edges are not in the "
                f"{spec.dataset} graph the figure draws, e.g. {stray[0]}"
            )
        draw, counts = _draw_method(spec, original, backbone, pos)
        style.save_variants(draw, out / spec.dataset / spec.file, **_SAVE)
        c = counts["light"]
        includes[spec.module] += _method_rst(
            spec, c, dataset, f"{static_prefix}/{spec.dataset}/{spec.file}"
        )
        rows.append(
            {
                "name": spec.file,
                "module": spec.module,
                "dataset": spec.dataset,
                "edges_kept": c.edges_kept,
                "edges_total": c.edges_total,
                "nodes_kept": c.nodes_kept,
            }
        )

    incidence = _incidence(ctx["davis_hyperedges"])
    pos = style.layout(incidence, "davis_hypergraph")
    for name in HYPERGRAPH_METHODS:
        draw, info = _draw_hypergraph(name, ctx, pos)
        style.save_variants(draw, out / "davis_hypergraph" / name, **_SAVE)
        includes["hypergraph"] += _hypergraph_rst(
            name, info, f"{static_prefix}/davis_hypergraph/{name}"
        )
        rows.append(
            {
                "name": name,
                "module": "hypergraph",
                "dataset": "davis_hypergraph",
                "edges_kept": info["kept"],
                "edges_total": info["total"],
                "nodes_kept": info["covered"],
            }
        )

    out.mkdir(parents=True, exist_ok=True)
    for module, lines in includes.items():
        (out / f"{module}.inc").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        default=str(DOCS / "_gallery"),
        help="Directory for the figures and includes (default: docs/_gallery).",
    )
    args = parser.parse_args()
    rows = build(args.output_dir)
    for row in rows:
        print(
            f"{row['module']:<12} {row['name']:<36} "
            f"{row['edges_kept']:>4}/{row['edges_total']:<4}"
        )
    print(f"Rendered {len(rows)} gallery figures (light and dark) in {args.output_dir}")


if __name__ == "__main__":
    main()
