"""Les Miserables benchmark tests across non-bipartite backbone methods."""

import networkx as nx
import pytest

import networkx_backbone as nb

DEFAULT_PVALUE = 0.05


def _global_threshold_filter(G):
    return nb.global_threshold_filter(G, threshold=2)


def _strongest_n_ties(G):
    return nb.strongest_n_ties(G, n=2)


def _filter_by_default_pvalue(H):
    """Apply default p-value filtering when a method exposes a p-value attribute."""
    pvalue_attrs = sorted(
        {
            key
            for _, _, data in H.edges(data=True)
            for key in data
            if key.endswith("_pvalue")
        }
    )

    if not pvalue_attrs:
        return H, None
    if len(pvalue_attrs) > 1:
        raise AssertionError(f"Expected at most one p-value attribute, got: {pvalue_attrs}")

    pvalue_attr = pvalue_attrs[0]
    filtered = nb.threshold_filter(H, pvalue_attr, DEFAULT_PVALUE, mode="below")
    return filtered, pvalue_attr


def _flag_full_graph_for_revision(method_name, H, original_edges, pvalue_attr):
    """Flag methods that still return the full original edge set."""
    if H.number_of_edges() == original_edges:
        if pvalue_attr is None:
            pytest.xfail(
                f"REVISION FLAG: {method_name} returned full graph "
                f"({original_edges} edges) without p-value filtering."
            )
        else:
            pytest.xfail(
                f"REVISION FLAG: {method_name} returned full graph "
                f"({original_edges} edges) after {pvalue_attr} <= {DEFAULT_PVALUE} filtering."
            )


METHOD_EDGE_COUNTS = [
    ("disparity_filter", nb.disparity_filter, 247),
    ("noise_corrected_filter", nb.noise_corrected_filter, 254),
    ("marginal_likelihood_filter", nb.marginal_likelihood_filter, 70),
    ("ecm_filter", nb.ecm_filter, 254),
    ("lans_filter", nb.lans_filter, 109),
    ("multiple_linkage_analysis(alpha=0.05)", lambda G: nb.multiple_linkage_analysis(G, alpha=0.05), 109),
    ("global_threshold_filter(threshold=2)", _global_threshold_filter, 157),
    ("strongest_n_ties(n=2)", _strongest_n_ties, 113),
    ("global_sparsification(s=0.5)", lambda G: nb.global_sparsification(G, s=0.5), 127),
    ("primary_linkage_analysis", nb.primary_linkage_analysis, 77),
    ("edge_betweenness_filter(s=0.5)", lambda G: nb.edge_betweenness_filter(G, s=0.5), 127),
    ("node_degree_filter(min_degree=2)", lambda G: nb.node_degree_filter(G, min_degree=2), 237),
    ("high_salience_skeleton", nb.high_salience_skeleton, 254),
    ("metric_backbone", nb.metric_backbone, 163),
    ("ultrametric_backbone", nb.ultrametric_backbone, 118),
    ("doubly_stochastic_filter", nb.doubly_stochastic_filter, 254),
    ("h_backbone", nb.h_backbone, 22),
    ("modularity_backbone", nb.modularity_backbone, 254),
    ("planar_maximally_filtered_graph", nb.planar_maximally_filtered_graph, 162),
    ("maximum_spanning_tree_backbone", nb.maximum_spanning_tree_backbone, 76),
    ("neighborhood_overlap", nb.neighborhood_overlap, 254),
    ("jaccard_backbone", nb.jaccard_backbone, 254),
    ("dice_backbone", nb.dice_backbone, 254),
    ("cosine_backbone", nb.cosine_backbone, 254),
    ("hub_promoted_index", nb.hub_promoted_index, 254),
    ("hub_depressed_index", nb.hub_depressed_index, 254),
    ("lhn_local_index", nb.lhn_local_index, 254),
    ("preferential_attachment_score", nb.preferential_attachment_score, 254),
    ("adamic_adar_index", nb.adamic_adar_index, 254),
    ("resource_allocation_index", nb.resource_allocation_index, 254),
    ("graph_distance_proximity", nb.graph_distance_proximity, 254),
    ("local_path_index", nb.local_path_index, 254),
    ("glab_filter", nb.glab_filter, 5),
]

UNWEIGHTED_METHOD_EDGE_COUNTS = [
    ("sparsify", nb.sparsify, 136),
    ("lspar", nb.lspar, 136),
    ("local_degree", nb.local_degree, 135),
]


def test_les_miserables_baseline(les_miserables_weighted, les_miserables_unweighted):
    """NetworkX Les Miserables generator should remain stable."""
    assert les_miserables_weighted.number_of_nodes() == 77
    assert les_miserables_weighted.number_of_edges() == 254
    assert les_miserables_unweighted.number_of_edges() == 254


@pytest.mark.parametrize(
    ("method_name", "method", "expected_edges"),
    METHOD_EDGE_COUNTS,
    ids=[m[0] for m in METHOD_EDGE_COUNTS],
)
def test_weighted_methods_edge_counts(
    method_name, method, expected_edges, les_miserables_weighted
):
    """Each non-bipartite weighted method should reproduce documented counts."""
    H = method(les_miserables_weighted)
    H, pvalue_attr = _filter_by_default_pvalue(H)
    assert isinstance(H, nx.Graph), method_name
    assert H.number_of_nodes() == les_miserables_weighted.number_of_nodes(), method_name
    assert H.number_of_edges() == expected_edges, method_name
    _flag_full_graph_for_revision(
        method_name,
        H,
        les_miserables_weighted.number_of_edges(),
        pvalue_attr,
    )


@pytest.mark.parametrize(
    ("method_name", "method", "expected_edges"),
    UNWEIGHTED_METHOD_EDGE_COUNTS,
    ids=[m[0] for m in UNWEIGHTED_METHOD_EDGE_COUNTS],
)
def test_unweighted_methods_edge_counts(
    method_name, method, expected_edges, les_miserables_unweighted
):
    """Unweighted methods should reproduce documented Les Miserables counts."""
    H = method(les_miserables_unweighted)
    H, pvalue_attr = _filter_by_default_pvalue(H)
    assert isinstance(H, nx.Graph), method_name
    assert H.number_of_nodes() == les_miserables_unweighted.number_of_nodes(), method_name
    assert H.number_of_edges() == expected_edges, method_name
    _flag_full_graph_for_revision(
        method_name,
        H,
        les_miserables_unweighted.number_of_edges(),
        pvalue_attr,
    )
