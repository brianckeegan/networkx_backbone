"""Les Miserables benchmark tests across non-bipartite backbone methods."""

import networkx as nx
import pytest

from benchmarks.benchmarks.les_miserables import (
    CASES,
    UNWEIGHTED_CASES,
    WEIGHTED_CASES,
    graph_for,
    les_miserables_unweighted,
    les_miserables_weighted,
)

# Edge count each method keeps on Les Miserables after score-then-filter.  The
# scorer and filter behind every label are defined once, in
# benchmarks/benchmarks/les_miserables.py, which also drives the table in
# docs/tutorials/les_miserables_benchmark.rst and the asv timings.
EXPECTED_EDGES = {
    "disparity_filter": 9,
    "noise_corrected_filter": 126,
    "noise_corrected_filter(pvalue_method='delta')": 195,
    "noise_corrected_filter(method='zscore')": 98,
    "marginal_likelihood_filter": 126,
    "ecm_filter": 72,
    "lans_filter": 109,
    "multiple_linkage_analysis": 205,
    "multiple_linkage_analysis(method='lans', alpha=0.05)": 109,
    "polya_filter": 12,
    "gloss_filter": 3,
    "global_threshold_filter(threshold=2)": 157,
    "strongest_n_ties(n=2)": 146,
    "global_sparsification(s=0.5)": 127,
    "primary_linkage_analysis": 69,
    "edge_betweenness_filter(s=0.5)": 127,
    "node_degree_filter(min_degree=2)": 237,
    "high_salience_skeleton": 76,
    "metric_backbone": 163,
    "ultrametric_backbone": 118,
    "doubly_stochastic_filter": 109,
    # Sinkhorn-Knopp does not converge here; see the Notes of the docstring.
    "doubly_stochastic_filter (ds_keep)": 187,
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


def test_every_benchmark_case_is_pinned():
    """Each case in the shared definition has exactly one pinned count."""
    names = [case.name for case in CASES]
    assert len(names) == len(set(names))
    assert set(names) == set(EXPECTED_EDGES)
    assert all(not case.weighted for case in UNWEIGHTED_CASES)
    assert all(case.weighted for case in WEIGHTED_CASES)
    # Every pinned backbone removes edges (Les Miserables has 254).
    assert max(EXPECTED_EDGES.values()) < 254


# Sinkhorn-Knopp does not converge on Les Miserables (see EXPECTED_EDGES).
@pytest.mark.filterwarnings("ignore:doubly_stochastic_filter:RuntimeWarning")
def test_edge_counts():
    """Score-then-filter keeps the pinned number of edges for every case.

    One test for all cases, so a change lists every count that moved.
    """
    # The NetworkX generator itself must not change under the pins.
    G = les_miserables_weighted()
    assert (G.number_of_nodes(), G.number_of_edges()) == (77, 254)
    assert les_miserables_unweighted().number_of_edges() == 254

    failures = []
    for case in CASES:
        G = graph_for(case)
        filtered = case.run(G)
        if not isinstance(filtered, nx.Graph) or set(filtered) != set(G):
            failures.append(f"{case.name}: wrong graph type or node set")
        if filtered.number_of_edges() != EXPECTED_EDGES[case.name]:
            failures.append(
                f"{case.name}: {filtered.number_of_edges()} edges "
                f"(expected {EXPECTED_EDGES[case.name]})"
            )
    assert not failures, "\n".join(failures)
