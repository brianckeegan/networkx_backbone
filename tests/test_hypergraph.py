"""Tests for hypergraph backbone extraction (MDL method, Kirkley et al. 2026)."""

import itertools

import pytest

from networkx_backbone import (
    HypergraphBackbone,
    hypergraph_compression_ratio,
    intersection_graph,
    maximal_hyperedges,
    mdl_hypergraph_backbone,
    order_filter,
    s_components,
)
from networkx_backbone.hypergraph import (
    _child_codelength,
    _parent_codelength,
    _reduced_mutual_information,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def simplex_with_subfaces(nodes, min_order=4):
    """A top face plus all of its nested sub-hyperedges down to ``min_order``."""
    nodes = list(nodes)
    edges = [tuple(nodes)]
    for k in range(min_order, len(nodes)):
        edges.extend(itertools.combinations(nodes, k))
    return edges


# ---------------------------------------------------------------------------
# Information-theoretic primitives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size_p,size_c,overlap",
    [(5, 4, 4), (5, 3, 3), (5, 2, 2), (4, 4, 3), (6, 3, 2)],
)
def test_rmi_symmetric_and_identity(size_p, size_c, overlap):
    n_nodes = 12
    r_pc = _reduced_mutual_information(size_p, size_c, overlap, n_nodes)
    r_cp = _reduced_mutual_information(size_c, size_p, overlap, n_nodes)
    assert r_pc == pytest.approx(r_cp)  # R is symmetric (Eqs. 9-10)

    # R(c, p) == H(c) - H(c|p)  (Eq. 9), using L=1 so the log L term cancels.
    h_c = _parent_codelength(size_c, 1, n_nodes)
    h_c_given_p = _child_codelength(size_c, size_p, overlap, 1, n_nodes)
    assert r_pc == pytest.approx(h_c - h_c_given_p)


def test_rmi_positive_for_nesting_when_universe_large():
    # Nested c ⊂ p gives positive RMI once N is sufficiently larger than |p|.
    assert _reduced_mutual_information(5, 4, 4, 20) > 0
    # ...and can be non-positive when the universe is tiny.
    assert _reduced_mutual_information(5, 4, 4, 5) <= 0


# ---------------------------------------------------------------------------
# intersection_graph
# ---------------------------------------------------------------------------


def test_intersection_graph_basic():
    I = intersection_graph([(1, 2, 3), (2, 3, 4), (5, 6)])
    assert I.number_of_nodes() == 3
    assert I.number_of_edges() == 1  # only the first two overlap
    assert I[0][1]["overlap"] == 2
    assert I.nodes[0]["members"] == frozenset({1, 2, 3})


def test_intersection_graph_disjoint_has_no_edges():
    I = intersection_graph([(0, 1), (2, 3), (4, 5)])
    assert I.number_of_nodes() == 3
    assert I.number_of_edges() == 0


def test_intersection_graph_s_threshold():
    G = [(1, 2, 3), (2, 3, 4), (5, 6)]
    assert intersection_graph(G, s=1).number_of_edges() == 1
    assert intersection_graph(G, s=2).number_of_edges() == 1  # overlap is exactly 2
    assert intersection_graph(G, s=3).number_of_edges() == 0


def test_intersection_graph_invalid_s_raises():
    with pytest.raises(ValueError):
        intersection_graph([(1, 2, 3)], s=0)


# ---------------------------------------------------------------------------
# Structural methods: inclusion reduction, order filter, s-components
# ---------------------------------------------------------------------------


def test_maximal_hyperedges_removes_subsets():
    result = maximal_hyperedges([(1, 2, 3), (1, 2), (2, 3), (4, 5), (4, 5, 6)])
    assert set(result) == {frozenset({1, 2, 3}), frozenset({4, 5, 6})}
    # Ordered by decreasing size.
    assert [len(e) for e in result] == sorted((len(e) for e in result), reverse=True)


def test_maximal_hyperedges_nested_chain():
    assert maximal_hyperedges([(1,), (1, 2), (1, 2, 3)]) == [frozenset({1, 2, 3})]


def test_maximal_hyperedges_all_maximal():
    result = maximal_hyperedges([(1, 2), (3, 4), (5, 6)])
    assert set(result) == {frozenset({1, 2}), frozenset({3, 4}), frozenset({5, 6})}


def test_maximal_hyperedges_merges_duplicates():
    assert maximal_hyperedges([(1, 2, 3), (3, 2, 1)]) == [frozenset({1, 2, 3})]


def test_order_filter_min_max_and_orders():
    G = [(1, 2), (1, 2, 3), (1, 2, 3, 4)]
    assert set(order_filter(G, min_order=3)) == {
        frozenset({1, 2, 3}),
        frozenset({1, 2, 3, 4}),
    }
    assert order_filter(G, max_order=2) == [frozenset({1, 2})]
    assert order_filter(G, min_order=3, max_order=3) == [frozenset({1, 2, 3})]
    assert set(order_filter(G, orders=[2, 4])) == {
        frozenset({1, 2}),
        frozenset({1, 2, 3, 4}),
    }


def test_order_filter_invalid_range_raises():
    with pytest.raises(ValueError):
        order_filter([(1, 2, 3)], min_order=4, max_order=2)


def test_s_components_threshold():
    G = [(1, 2, 3), (2, 3, 4), (5, 6, 7)]
    # s=1 and s=2: first two hyperedges connect; the third is isolated.
    for s in (1, 2):
        comps = s_components(G, s=s)
        assert [len(c) for c in comps] == [2, 1]
    # s=3: no pair shares 3 nodes, so every hyperedge is its own component.
    assert [len(c) for c in s_components(G, s=3)] == [1, 1, 1]


def test_s_components_chain():
    comps = s_components([(1, 2), (2, 3), (3, 4), (10, 11)], s=1)
    assert [len(c) for c in comps] == [3, 1]
    # The big component holds the chain; the disjoint pair stands alone.
    assert frozenset({10, 11}) in comps[1]


# ---------------------------------------------------------------------------
# Core MDL backbone behaviour
# ---------------------------------------------------------------------------


def test_recovers_top_faces_of_nested_simplices():
    # Two disjoint 5-simplices, each with all size-4 sub-faces. The MDL backbone
    # should keep exactly the two top faces and prune every redundant sub-face.
    a, b = range(0, 5), range(5, 10)
    G = simplex_with_subfaces(a) + simplex_with_subfaces(b)

    result = mdl_hypergraph_backbone(G)

    assert isinstance(result, HypergraphBackbone)
    assert len(result.backbone) == 2
    assert frozenset(a) in result.backbone
    assert frozenset(b) in result.backbone
    assert result.compression_ratio < 1.0
    assert result.n_nodes == 10
    # Every pruned sub-face is recorded as a child of a retained parent.
    assert result.n_input_hyperedges == len(G)


def test_downward_closure_compresses_and_keeps_tops():
    a, b = range(0, 5), range(5, 10)
    G = simplex_with_subfaces(a, min_order=2) + simplex_with_subfaces(b, min_order=2)

    result = mdl_hypergraph_backbone(G)

    assert frozenset(a) in result.backbone
    assert frozenset(b) in result.backbone
    assert result.fraction_kept < 0.5  # substantial sparsification
    assert 0.0 <= result.compression_ratio < 1.0


def test_disjoint_hyperedges_are_incompressible():
    G = [(0, 1), (2, 3), (4, 5)]
    result = mdl_hypergraph_backbone(G)
    assert len(result.backbone) == 3
    assert result.assignment == {}
    assert result.compression_ratio == pytest.approx(1.0)


def test_description_length_never_exceeds_baseline():
    G = simplex_with_subfaces(range(0, 6), min_order=2)
    result = mdl_hypergraph_backbone(G)
    assert result.description_length <= result.baseline_description_length
    assert 0.0 <= result.compression_ratio <= 1.0


def test_compression_ratio_helper_matches_backbone():
    G = simplex_with_subfaces(range(0, 5)) + simplex_with_subfaces(range(5, 10))
    eta = hypergraph_compression_ratio(G)
    assert eta == pytest.approx(mdl_hypergraph_backbone(G).compression_ratio)
    assert 0.0 <= eta <= 1.0


def test_deterministic():
    G = simplex_with_subfaces(range(0, 5), min_order=2) + simplex_with_subfaces(
        range(5, 10), min_order=2
    )
    r1 = mdl_hypergraph_backbone(G)
    r2 = mdl_hypergraph_backbone(G)
    assert set(r1.backbone) == set(r2.backbone)


# ---------------------------------------------------------------------------
# Weighted model
# ---------------------------------------------------------------------------


def _family_a_with_filler():
    """Family A (top + one size-4 child) plus a disjoint filler simplex.

    The filler raises the node count so the size-4 child of A is topologically
    redundant (pruned) in the unweighted backbone.
    """
    A = [(0, 1, 2, 3, 4), (0, 1, 2, 3)]
    filler = simplex_with_subfaces(range(5, 10))
    return A, filler


def test_gamma_one_equivalent_to_unweighted():
    A, filler = _family_a_with_filler()
    G = A + filler
    weights = [5.0, 5.0] + [5.0] * len(filler)  # uniform weights

    unweighted = mdl_hypergraph_backbone(G)
    g1 = mdl_hypergraph_backbone(G, weights=weights, gamma=1.0)

    assert g1.weighted is True
    assert set(g1.backbone) == set(unweighted.backbone)


def test_low_gamma_forces_high_weight_hyperedge_into_backbone():
    A, filler = _family_a_with_filler()
    G = A + filler
    weights = [1.0, 100.0] + [1.0] * len(filler)  # the size-4 child is heavy

    unweighted = mdl_hypergraph_backbone(G)
    weighted = mdl_hypergraph_backbone(G, weights=weights, gamma=0.01)

    child = frozenset({0, 1, 2, 3})
    assert child not in unweighted.backbone  # topologically redundant
    assert child in weighted.backbone  # weight keeps it in the backbone


def test_geometric_prior_runs():
    A, filler = _family_a_with_filler()
    G = A + filler
    weights = [1.0, 100.0] + [1.0] * len(filler)
    result = mdl_hypergraph_backbone(G, weights=weights, gamma=0.05, prior="geometric")
    assert frozenset({0, 1, 2, 3}) in result.backbone


# ---------------------------------------------------------------------------
# Edge cases and input handling
# ---------------------------------------------------------------------------


def test_empty_hypergraph():
    result = mdl_hypergraph_backbone([])
    assert len(result.backbone) == 0
    assert result.compression_ratio == pytest.approx(1.0)


def test_single_hyperedge():
    result = mdl_hypergraph_backbone([(1, 2, 3)])
    assert result.backbone == [frozenset({1, 2, 3})]
    assert result.assignment == {}
    assert result.compression_ratio == pytest.approx(1.0)


def test_duplicate_hyperedges_are_merged():
    result = mdl_hypergraph_backbone([(1, 2, 3), (1, 2, 3), (3, 2, 1)])
    assert result.backbone == [frozenset({1, 2, 3})]
    assert result.n_input_hyperedges == 1


def test_repeated_nodes_within_hyperedge_ignored():
    result = mdl_hypergraph_backbone([(1, 1, 2, 2, 3)])
    assert result.backbone == [frozenset({1, 2, 3})]


def test_duplicate_weighted_hyperedges_sum_weights():
    # Two copies of the same hyperedge merge; their weights add (3 + 4 = 7 > 1),
    # so the merged hyperedge is treated as weighted.
    result = mdl_hypergraph_backbone(
        [(1, 2, 3), (1, 2, 3)], weights=[3.0, 4.0], gamma=0.5
    )
    assert result.backbone == [frozenset({1, 2, 3})]
    assert result.weighted is True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gamma", [0.0, -0.1, 1.5])
def test_invalid_gamma_raises(gamma):
    with pytest.raises(ValueError):
        mdl_hypergraph_backbone([(1, 2, 3)], weights=[2.0], gamma=gamma)


def test_invalid_prior_raises():
    with pytest.raises(ValueError):
        mdl_hypergraph_backbone([(1, 2, 3)], weights=[2.0], prior="bogus")


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        mdl_hypergraph_backbone([(1, 2, 3)], method="node")


def test_weights_length_mismatch_raises():
    with pytest.raises(ValueError):
        mdl_hypergraph_backbone([(1, 2, 3), (2, 3, 4)], weights=[1.0])


def test_weight_below_one_raises():
    with pytest.raises(ValueError):
        mdl_hypergraph_backbone([(1, 2, 3)], weights=[0.5])
