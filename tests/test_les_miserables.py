"""Les Miserables benchmark tests across non-bipartite backbone methods."""

import warnings

import networkx as nx
import pytest

from benchmarks.benchmarks.les_miserables import UNWEIGHTED_CASES, WEIGHTED_CASES


def _warn_if_no_edge_reduction(method_name, original_edges, filtered_edges):
    if filtered_edges == original_edges:
        warnings.warn(
            (
                f"{method_name}: filtered graph has the same number of edges "
                f"as the original ({original_edges}). Re-test and validate."
            ),
            UserWarning,
            stacklevel=2,
        )


# Edge count each method keeps on Les Miserables after score-then-filter.  The
# scorer and filter behind every label are defined once, in
# benchmarks/benchmarks/les_miserables.py, which also drives the table in
# docs/tutorials/les_miserables_benchmark.rst and the asv timings.
EXPECTED_EDGES = {
    "disparity_filter": 9,
    "noise_corrected_filter": 195,
    "noise_corrected_filter(method='zscore')": 98,
    "marginal_likelihood_filter": 126,
    "ecm_filter": 72,
    "lans_filter": 109,
    "multiple_linkage_analysis": 223,
    "multiple_linkage_analysis(method='lans', alpha=0.05)": 109,
    "global_threshold_filter(threshold=2)": 157,
    "strongest_n_ties(n=2)": 113,
    "global_sparsification(s=0.5)": 127,
    "primary_linkage_analysis": 69,
    "edge_betweenness_filter(s=0.5)": 127,
    "node_degree_filter(min_degree=2)": 237,
    "high_salience_skeleton": 76,
    "metric_backbone": 163,
    "ultrametric_backbone": 118,
    "doubly_stochastic_filter": 109,
    "h_backbone": 18,
    "modularity_backbone": 72,
    "planar_maximally_filtered_graph": 162,
    "maximum_spanning_tree_backbone": 76,
    "neighborhood_overlap": 76,
    "jaccard_backbone": 76,
    "dice_backbone": 76,
    "cosine_backbone": 76,
    "hub_promoted_index": 76,
    "hub_depressed_index": 76,
    "lhn_local_index": 76,
    "preferential_attachment_score": 76,
    "adamic_adar_index": 76,
    "resource_allocation_index": 76,
    "graph_distance_proximity": 76,
    "local_path_index": 76,
    "glab_filter": 56,
    # Unweighted sparsifiers
    "sparsify": 174,
    "lspar": 174,
    "local_degree": 91,
}

WEIGHTED_METHOD_EDGE_COUNTS = [
    (case.name, case.score, case.filter_graph, EXPECTED_EDGES.get(case.name))
    for case in WEIGHTED_CASES
]

UNWEIGHTED_METHOD_EDGE_COUNTS = [
    (case.name, case.score, case.filter_graph, EXPECTED_EDGES.get(case.name))
    for case in UNWEIGHTED_CASES
]


def test_every_benchmark_case_is_pinned():
    """Each case in the shared definition has exactly one pinned count."""
    names = [case.name for case in WEIGHTED_CASES + UNWEIGHTED_CASES]
    assert len(names) == len(set(names))
    assert set(names) == set(EXPECTED_EDGES)
    assert all(not case.weighted for case in UNWEIGHTED_CASES)
    assert all(case.weighted for case in WEIGHTED_CASES)


def test_les_miserables_baseline(les_miserables_weighted, les_miserables_unweighted):
    """NetworkX Les Miserables generator should remain stable."""
    assert les_miserables_weighted.number_of_nodes() == 77
    assert les_miserables_weighted.number_of_edges() == 254
    assert les_miserables_unweighted.number_of_edges() == 254


def test_warning_trigger_for_no_edge_reduction():
    with pytest.warns(UserWarning, match="same number of edges"):
        _warn_if_no_edge_reduction("dummy_method", 10, 10)


@pytest.mark.parametrize(
    ("method_name", "score_method", "filter_method", "expected_edges"),
    WEIGHTED_METHOD_EDGE_COUNTS,
    ids=[m[0] for m in WEIGHTED_METHOD_EDGE_COUNTS],
)
def test_weighted_methods_edge_counts(
    method_name,
    score_method,
    filter_method,
    expected_edges,
    les_miserables_weighted,
):
    """Weighted methods must use score-then-filter before benchmarking."""
    scored = score_method(les_miserables_weighted)
    filtered = filter_method(scored)
    assert isinstance(filtered, nx.Graph), method_name
    assert filtered.number_of_nodes() == les_miserables_weighted.number_of_nodes(), (
        method_name
    )
    assert filtered.number_of_edges() == expected_edges, method_name
    _warn_if_no_edge_reduction(
        method_name,
        les_miserables_weighted.number_of_edges(),
        filtered.number_of_edges(),
    )


@pytest.mark.parametrize(
    ("method_name", "score_method", "filter_method", "expected_edges"),
    UNWEIGHTED_METHOD_EDGE_COUNTS,
    ids=[m[0] for m in UNWEIGHTED_METHOD_EDGE_COUNTS],
)
def test_unweighted_methods_edge_counts(
    method_name,
    score_method,
    filter_method,
    expected_edges,
    les_miserables_unweighted,
):
    """Unweighted methods must use score-then-filter before benchmarking."""
    scored = score_method(les_miserables_unweighted)
    filtered = filter_method(scored)
    assert isinstance(filtered, nx.Graph), method_name
    assert filtered.number_of_nodes() == les_miserables_unweighted.number_of_nodes(), (
        method_name
    )
    assert filtered.number_of_edges() == expected_edges, method_name
    _warn_if_no_edge_reduction(
        method_name,
        les_miserables_unweighted.number_of_edges(),
        filtered.number_of_edges(),
    )
