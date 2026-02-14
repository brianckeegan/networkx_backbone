#!/usr/bin/env python3
"""Generate backbone comparison visualizations and docs gallery content."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import networkx as nx
import networkx_backbone as nb
import networkx_backbone.hybrid as nb_hybrid
import networkx_backbone.proximity as nb_proximity
import networkx_backbone.statistical as nb_statistical
import networkx_backbone.structural as nb_structural
import networkx_backbone.unweighted as nb_unweighted


@dataclass
class MethodSpec:
    """Specification for one backbone method visualization."""

    name: str
    module: str
    dataset: str
    build: Callable


def _to_unweighted(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from(G.edges())
    return H


def _drop_isolates(G):
    H = G.copy()
    isolates = list(nx.isolates(H))
    if isolates:
        H.remove_nodes_from(isolates)
    return H


def _to_undirected_simple(G):
    if not G.is_directed():
        return G
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if not H.has_edge(u, v):
            H.add_edge(u, v, **data)
    return H


def _title_from_name(name: str) -> str:
    return name.replace("_", " ").title()


def _method_specs():
    return [
        # Statistical
        MethodSpec(
            "disparity_filter",
            "statistical",
            "les_miserables",
            lambda c: nb.threshold_filter(
                nb.disparity_filter(c["les_weighted"]),
                "disparity_pvalue",
                0.05,
                mode="below",
                include_all_nodes=False,
            ),
        ),
        MethodSpec(
            "noise_corrected_filter",
            "statistical",
            "les_miserables",
            lambda c: nb.threshold_filter(
                nb.noise_corrected_filter(c["les_weighted"]),
                "nc_score",
                2.0,
                mode="above",
                include_all_nodes=False,
            ),
        ),
        MethodSpec(
            "marginal_likelihood_filter",
            "statistical",
            "les_miserables",
            lambda c: nb.threshold_filter(
                nb.marginal_likelihood_filter(c["les_weighted"]),
                "ml_pvalue",
                0.05,
                mode="below",
                include_all_nodes=False,
            ),
        ),
        MethodSpec(
            "ecm_filter",
            "statistical",
            "les_miserables",
            lambda c: nb.threshold_filter(
                nb.ecm_filter(c["les_weighted"]),
                "ecm_pvalue",
                0.05,
                mode="below",
                include_all_nodes=False,
            ),
        ),
        MethodSpec(
            "lans_filter",
            "statistical",
            "les_miserables",
            lambda c: nb.threshold_filter(
                nb.lans_filter(c["les_weighted"]),
                "lans_pvalue",
                0.05,
                mode="below",
                include_all_nodes=False,
            ),
        ),
        MethodSpec(
            "multiple_linkage_analysis",
            "statistical",
            "les_miserables",
            lambda c: nb.multiple_linkage_analysis(c["les_weighted"], alpha=0.05),
        ),
        # Structural
        MethodSpec(
            "global_threshold_filter",
            "structural",
            "les_miserables",
            lambda c: nb.global_threshold_filter(c["les_weighted"], threshold=2.0),
        ),
        MethodSpec(
            "strongest_n_ties",
            "structural",
            "les_miserables",
            lambda c: nb.strongest_n_ties(c["les_weighted"], n=2),
        ),
        MethodSpec(
            "global_sparsification",
            "structural",
            "les_miserables",
            lambda c: nb.global_sparsification(c["les_weighted"], s=0.4),
        ),
        MethodSpec(
            "primary_linkage_analysis",
            "structural",
            "les_miserables",
            lambda c: nb.primary_linkage_analysis(c["les_weighted"]),
        ),
        MethodSpec(
            "edge_betweenness_filter",
            "structural",
            "les_miserables",
            lambda c: nb.edge_betweenness_filter(c["les_weighted"], s=0.3),
        ),
        MethodSpec(
            "node_degree_filter",
            "structural",
            "les_miserables",
            lambda c: nb.node_degree_filter(c["les_weighted"], min_degree=2),
        ),
        MethodSpec(
            "high_salience_skeleton",
            "structural",
            "les_miserables",
            lambda c: nb.threshold_filter(
                nb.high_salience_skeleton(c["les_weighted"]),
                "salience",
                0.5,
                mode="above",
                include_all_nodes=False,
            ),
        ),
        MethodSpec(
            "metric_backbone",
            "structural",
            "les_miserables",
            lambda c: nb.metric_backbone(c["les_weighted"]),
        ),
        MethodSpec(
            "ultrametric_backbone",
            "structural",
            "les_miserables",
            lambda c: nb.ultrametric_backbone(c["les_weighted"]),
        ),
        MethodSpec(
            "doubly_stochastic_filter",
            "structural",
            "les_miserables",
            lambda c: nb.threshold_filter(
                nb.doubly_stochastic_filter(c["les_weighted"]),
                "ds_weight",
                0.1,
                mode="above",
                include_all_nodes=False,
            ),
        ),
        MethodSpec(
            "h_backbone",
            "structural",
            "les_miserables",
            lambda c: nb.h_backbone(c["les_weighted"]),
        ),
        MethodSpec(
            "modularity_backbone",
            "structural",
            "les_miserables",
            lambda c: nb.threshold_filter(
                nb.modularity_backbone(c["les_weighted"]),
                "vitality",
                0.0,
                mode="above",
                filter_on="nodes",
                include_all_nodes=False,
            ),
        ),
        MethodSpec(
            "planar_maximally_filtered_graph",
            "structural",
            "les_miserables",
            lambda c: nb.planar_maximally_filtered_graph(c["les_weighted"]),
        ),
        MethodSpec(
            "maximum_spanning_tree_backbone",
            "structural",
            "les_miserables",
            lambda c: nb.maximum_spanning_tree_backbone(c["les_weighted"]),
        ),
        # Proximity
        MethodSpec(
            "neighborhood_overlap",
            "proximity",
            "les_miserables",
            lambda c: nb.fraction_filter(
                nb.neighborhood_overlap(c["les_weighted"]),
                "overlap",
                0.3,
                ascending=False,
            ),
        ),
        MethodSpec(
            "jaccard_backbone",
            "proximity",
            "les_miserables",
            lambda c: nb.fraction_filter(
                nb.jaccard_backbone(c["les_weighted"]),
                "jaccard",
                0.3,
                ascending=False,
            ),
        ),
        MethodSpec(
            "dice_backbone",
            "proximity",
            "les_miserables",
            lambda c: nb.fraction_filter(
                nb.dice_backbone(c["les_weighted"]),
                "dice",
                0.3,
                ascending=False,
            ),
        ),
        MethodSpec(
            "cosine_backbone",
            "proximity",
            "les_miserables",
            lambda c: nb.fraction_filter(
                nb.cosine_backbone(c["les_weighted"]),
                "cosine",
                0.3,
                ascending=False,
            ),
        ),
        MethodSpec(
            "hub_promoted_index",
            "proximity",
            "les_miserables",
            lambda c: nb.fraction_filter(
                nb.hub_promoted_index(c["les_weighted"]),
                "hpi",
                0.3,
                ascending=False,
            ),
        ),
        MethodSpec(
            "hub_depressed_index",
            "proximity",
            "les_miserables",
            lambda c: nb.fraction_filter(
                nb.hub_depressed_index(c["les_weighted"]),
                "hdi",
                0.3,
                ascending=False,
            ),
        ),
        MethodSpec(
            "lhn_local_index",
            "proximity",
            "les_miserables",
            lambda c: nb.fraction_filter(
                nb.lhn_local_index(c["les_weighted"]),
                "lhn_local",
                0.3,
                ascending=False,
            ),
        ),
        MethodSpec(
            "preferential_attachment_score",
            "proximity",
            "les_miserables",
            lambda c: nb.fraction_filter(
                nb.preferential_attachment_score(c["les_weighted"]),
                "pa",
                0.3,
                ascending=False,
            ),
        ),
        MethodSpec(
            "adamic_adar_index",
            "proximity",
            "les_miserables",
            lambda c: nb.fraction_filter(
                nb.adamic_adar_index(c["les_weighted"]),
                "adamic_adar",
                0.3,
                ascending=False,
            ),
        ),
        MethodSpec(
            "resource_allocation_index",
            "proximity",
            "les_miserables",
            lambda c: nb.fraction_filter(
                nb.resource_allocation_index(c["les_weighted"]),
                "resource_allocation",
                0.3,
                ascending=False,
            ),
        ),
        MethodSpec(
            "graph_distance_proximity",
            "proximity",
            "les_miserables",
            lambda c: nb.fraction_filter(
                nb.graph_distance_proximity(c["les_weighted"], all_pairs=False),
                "dist",
                0.3,
                ascending=False,
            ),
        ),
        MethodSpec(
            "local_path_index",
            "proximity",
            "les_miserables",
            lambda c: nb.fraction_filter(
                nb.local_path_index(c["les_weighted"]),
                "lp",
                0.3,
                ascending=False,
            ),
        ),
        # Hybrid
        MethodSpec(
            "glab_filter",
            "hybrid",
            "les_miserables",
            lambda c: nb.threshold_filter(
                nb.glab_filter(c["les_weighted"]),
                "glab_pvalue",
                0.05,
                mode="below",
                include_all_nodes=False,
            ),
        ),
        # Unweighted
        MethodSpec(
            "sparsify",
            "unweighted",
            "les_miserables",
            lambda c: nb.sparsify(c["les_unweighted"], s=0.5),
        ),
        MethodSpec(
            "lspar",
            "unweighted",
            "les_miserables",
            lambda c: nb.lspar(c["les_unweighted"], s=0.5),
        ),
        MethodSpec(
            "local_degree",
            "unweighted",
            "les_miserables",
            lambda c: nb.local_degree(c["les_unweighted"], s=0.3),
        ),
        # Bipartite
        MethodSpec(
            "sdsm",
            "bipartite",
            "southern_women",
            lambda c: nb.threshold_filter(
                nb.sdsm(c["davis_bipartite"], c["women_nodes"]),
                "sdsm_pvalue",
                0.05,
                mode="below",
                include_all_nodes=False,
            ),
        ),
        MethodSpec(
            "fdsm",
            "bipartite",
            "southern_women",
            lambda c: nb.threshold_filter(
                nb.fdsm(c["davis_bipartite"], c["women_nodes"], trials=250, seed=42),
                "fdsm_pvalue",
                0.05,
                mode="below",
                include_all_nodes=False,
            ),
        ),
        MethodSpec(
            "fixedfill",
            "bipartite",
            "southern_women",
            lambda c: nb.fixedfill(c["davis_bipartite"], c["women_nodes"], alpha=0.05),
        ),
        MethodSpec(
            "fixedrow",
            "bipartite",
            "southern_women",
            lambda c: nb.fixedrow(c["davis_bipartite"], c["women_nodes"], alpha=0.05),
        ),
        MethodSpec(
            "fixedcol",
            "bipartite",
            "southern_women",
            lambda c: nb.fixedcol(c["davis_bipartite"], c["women_nodes"], alpha=0.05),
        ),
    ]


def _validate_coverage(specs):
    spec_names = {spec.name for spec in specs}

    expected = set(nb_statistical.__all__) - {"disparity", "mlf", "lans"}
    expected |= set(nb_structural.__all__)
    expected |= set(nb_proximity.__all__)
    expected |= set(nb_hybrid.__all__)
    expected |= set(nb_unweighted.__all__)
    expected |= {"sdsm", "fdsm", "fixedfill", "fixedrow", "fixedcol"}

    missing = sorted(expected - spec_names)
    extra = sorted(spec_names - expected)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing methods: {missing}")
        if extra:
            parts.append(f"unexpected methods: {extra}")
        raise RuntimeError("Invalid gallery method coverage: " + "; ".join(parts))


def _context():
    les_weighted = nx.les_miserables_graph()
    les_unweighted = _to_unweighted(les_weighted)

    davis = nx.davis_southern_women_graph()
    women_nodes = [
        n for n, data in davis.nodes(data=True) if data.get("bipartite") == 0
    ]

    women_projection = nx.Graph()
    women_projection.add_nodes_from(women_nodes)
    for i, u in enumerate(women_nodes):
        for v in women_nodes[i + 1 :]:
            women_projection.add_edge(u, v)

    return {
        "les_weighted": les_weighted,
        "les_unweighted": les_unweighted,
        "davis_bipartite": davis,
        "women_nodes": women_nodes,
        "women_projection": women_projection,
    }


def _base_graphs(context):
    return {
        "les_miserables": context["les_weighted"],
        "southern_women": context["women_projection"],
    }


def _layout_map(base_graphs):
    return {
        "les_miserables": nx.spring_layout(
            base_graphs["les_miserables"], seed=42, weight="weight"
        ),
        "southern_women": nx.spring_layout(base_graphs["southern_women"], seed=42),
    }


def _render_all(
    output_dir: Path,
    drop_isolates: bool,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from networkx_backbone.visualization import compare_graphs

    context = _context()
    base_graphs = _base_graphs(context)
    layout_map = _layout_map(base_graphs)
    specs = _method_specs()
    _validate_coverage(specs)
    rows = []

    for spec in specs:
        base_graph = base_graphs[spec.dataset]
        out_dir = output_dir / spec.dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{spec.name}.png"

        backbone = spec.build(context)
        if drop_isolates:
            backbone = _drop_isolates(backbone)
        if not base_graph.is_directed() and backbone.is_directed():
            backbone = _to_undirected_simple(backbone)

        title = f"{_title_from_name(spec.name)} ({spec.dataset})"
        fig, ax, diff = compare_graphs(
            base_graph,
            backbone,
            pos=layout_map[spec.dataset],
            title=title,
            return_diff=True,
        )
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)

        rows.append(
            {
                "name": spec.name,
                "title": _title_from_name(spec.name),
                "module": spec.module,
                "dataset": spec.dataset,
                "image": f"/_static/graph_gallery/{spec.dataset}/{spec.name}.png",
                "original_nodes": base_graph.number_of_nodes(),
                "original_edges": base_graph.number_of_edges(),
                "backbone_nodes": len(diff["kept_nodes"]),
                "removed_nodes": len(diff["removed_nodes"]),
                "backbone_edges": len(diff["kept_edges"]),
            }
        )

    return rows


def _write_gallery_rst(rows, rst_path: Path):
    rows = sorted(rows, key=lambda r: (r["dataset"], r["module"], r["name"]))
    by_dataset = {"les_miserables": [], "southern_women": []}
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    lines = [
        "Graph Comparison Gallery",
        "========================",
        "",
        "This gallery compares each backbone method against a reference graph.",
        "Removed nodes are colored red and retained edges are drawn as thicker black lines.",
        "",
        "For readability, isolates are removed from each backbone before plotting.",
        "",
    ]

    sections = [
        ("les_miserables", "Les Miserables (Non-Bipartite Methods)"),
        ("southern_women", "Davis Southern Women (Bipartite Methods)"),
    ]
    for dataset_key, title in sections:
        lines.extend([title, "-" * len(title), ""])
        for row in by_dataset[dataset_key]:
            method_title = row["title"]
            lines.extend(
                [
                    method_title,
                    "^" * len(method_title),
                    "",
                    f"- Module: ``{row['module']}``",
                    f"- Original graph: ``{row['original_nodes']}`` nodes, ``{row['original_edges']}`` edges",
                    f"- Backbone graph: ``{row['backbone_nodes']}`` nodes, ``{row['backbone_edges']}`` edges",
                    f"- Nodes removed: ``{row['removed_nodes']}``",
                    "",
                    f".. image:: {row['image']}",
                    "   :width: 800px",
                    "   :alt: Backbone comparison visualization",
                    "",
                ]
            )

    rst_path.parent.mkdir(parents=True, exist_ok=True)
    rst_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="docs/_static/graph_gallery",
        help="Directory to write generated PNG files.",
    )
    parser.add_argument(
        "--gallery-rst",
        default="docs/tutorials/graph_gallery.rst",
        help="Path to write generated gallery RST.",
    )
    parser.add_argument(
        "--keep-isolates",
        action="store_true",
        help="Keep isolates in generated backbone visualizations.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    gallery_rst = Path(args.gallery_rst)

    rows = _render_all(output_dir=output_dir, drop_isolates=not args.keep_isolates)
    _write_gallery_rst(rows, gallery_rst)

    print(f"Generated {len(rows)} visualization images in {output_dir}")
    print(f"Wrote gallery page: {gallery_rst}")


if __name__ == "__main__":
    main()
