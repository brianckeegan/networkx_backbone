"""Tests for filtering utilities."""

import networkx as nx
import pytest

from networkx_backbone import (
    adjust_pvalues,
    boolean_filter,
    consensus_backbone,
    disparity_filter,
    fraction_filter,
    multigraph_to_weighted,
    threshold_filter,
)


class TestAdjustPvalues:
    PS = [0.001, 0.008, 0.012, 0.03, 0.04, 0.2, 0.5, 0.9]

    def test_bonferroni(self):
        assert adjust_pvalues([0.01, 0.02, 0.5], "bonferroni") == [0.03, 0.06, 1.0]

    def test_holm_step_down(self):
        assert adjust_pvalues([0.01, 0.02, 0.5], "holm") == pytest.approx(
            [0.03, 0.04, 0.5]
        )

    def test_none_clips_to_unit_interval(self):
        assert adjust_pvalues([0.2, 1.5, -0.1], "none") == [0.2, 1.0, 0.0]

    def test_order_preserved_and_bounded(self):
        adj = adjust_pvalues(self.PS, "bh")
        assert len(adj) == len(self.PS)
        assert all(0.0 <= a <= 1.0 for a in adj)

    @pytest.mark.parametrize("method", ["bh", "fdr", "by"])
    def test_matches_scipy(self, method):
        sp = pytest.importorskip("scipy.stats")
        import numpy as np

        expected = sp.false_discovery_control(
            self.PS, method="by" if method == "by" else "bh"
        )
        assert np.allclose(adjust_pvalues(self.PS, method), expected)

    def test_correction_is_conservative(self):
        raw = self.PS
        for method in ("bonferroni", "holm", "hochberg", "bh", "by"):
            adj = adjust_pvalues(raw, method)
            assert all(a >= r - 1e-12 for a, r in zip(adj, raw))

    def test_empty(self):
        assert adjust_pvalues([], "bh") == []

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            adjust_pvalues([0.1], "hommel")


class TestThresholdFilterMTC:
    def test_mtc_reduces_or_equal_edges(self):
        G = nx.les_miserables_graph()
        H = disparity_filter(G)
        raw = threshold_filter(H, "disparity_pvalue", 0.05).number_of_edges()
        for method in ("bonferroni", "bh"):
            corrected = threshold_filter(
                H, "disparity_pvalue", 0.05, mtc=method
            ).number_of_edges()
            assert corrected <= raw

    def test_mtc_requires_below_mode(self):
        G = nx.les_miserables_graph()
        H = disparity_filter(G)
        with pytest.raises(ValueError):
            threshold_filter(H, "disparity_pvalue", 0.05, mode="above", mtc="bh")

    def test_mtc_none_matches_plain(self):
        G = nx.les_miserables_graph()
        H = disparity_filter(G)
        a = threshold_filter(H, "disparity_pvalue", 0.3).number_of_edges()
        b = threshold_filter(H, "disparity_pvalue", 0.3, mtc="none").number_of_edges()
        assert a == b


class TestMultigraphToWeighted:
    def test_multigraph_parallel_edge_count(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 1)
        G.add_edge(0, 1)

        H = multigraph_to_weighted(G)
        assert isinstance(H, nx.Graph)
        assert H.number_of_edges() == 1
        assert H[0][1]["weight"] == 3
        assert H[0][1]["edge_count"] == 3

    def test_multigraph_edge_type_count(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, edge_type="a")
        G.add_edge(0, 1, edge_type="b")
        G.add_edge(0, 1, edge_type="b")

        H = multigraph_to_weighted(G, edge_type_attr="edge_type")
        assert H[0][1]["weight"] == 2
        assert H[0][1]["edge_count"] == 3
        assert H[0][1]["edge_type_count"] == 2

    def test_multidigraph_direction_preserved(self):
        G = nx.MultiDiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 1)
        G.add_edge(1, 0)

        H = multigraph_to_weighted(G)
        assert isinstance(H, nx.DiGraph)
        assert H.number_of_edges() == 2
        assert H[0][1]["weight"] == 2
        assert H[1][0]["weight"] == 1

    def test_simple_graph_fills_missing_weight(self):
        G = nx.Graph()
        G.add_edge(0, 1)

        H = multigraph_to_weighted(G)
        assert H[0][1]["weight"] == 1


class TestThresholdFilter:
    def test_below_mode(self, weighted_triangle):
        H = disparity_filter(weighted_triangle)
        filtered = threshold_filter(H, "disparity_pvalue", 0.5, mode="below")
        assert filtered.number_of_edges() <= H.number_of_edges()

    def test_above_mode(self, weighted_triangle):
        H = disparity_filter(weighted_triangle)
        filtered = threshold_filter(H, "disparity_pvalue", 0.5, mode="above")
        assert filtered.number_of_edges() <= H.number_of_edges()

    def test_invalid_mode_raises(self, weighted_triangle):
        with pytest.raises(ValueError):
            threshold_filter(weighted_triangle, "weight", 1.0, mode="invalid")

    def test_preserves_nodes_on_edge_filter(self, weighted_triangle):
        H = disparity_filter(weighted_triangle)
        filtered = threshold_filter(H, "disparity_pvalue", 0.5)
        assert set(filtered.nodes()) == set(H.nodes())

    def test_removes_isolates_on_edge_filter(self, weighted_triangle):
        H = disparity_filter(weighted_triangle)
        filtered = threshold_filter(
            H,
            "disparity_pvalue",
            0.0,  # keep no edges in below mode
            mode="below",
            include_all_nodes=False,
        )
        assert filtered.number_of_edges() == 0
        assert filtered.number_of_nodes() == 0

    def test_node_filter(self):
        G = nx.Graph()
        G.add_node(0, score=0.1)
        G.add_node(1, score=0.9)
        G.add_node(2, score=0.5)
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        filtered = threshold_filter(G, "score", 0.5, mode="below", filter_on="nodes")
        assert 0 in filtered.nodes()
        assert 1 not in filtered.nodes()

    def test_node_filter_remove_isolates_option(self):
        G = nx.Graph()
        G.add_node(0, score=0.1)
        G.add_node(1, score=0.1)
        G.add_node(2, score=0.9)
        G.add_edge(0, 2)

        kept_with_isolates = threshold_filter(
            G, "score", 0.5, mode="below", filter_on="nodes", include_all_nodes=True
        )
        assert set(kept_with_isolates.nodes()) == {0, 1}
        assert 1 in kept_with_isolates.nodes()

        kept_without_isolates = threshold_filter(
            G, "score", 0.5, mode="below", filter_on="nodes", include_all_nodes=False
        )
        assert set(kept_without_isolates.nodes()) == set()


class TestFractionFilter:
    def test_keeps_fraction_of_edges(self, weighted_triangle):
        H = disparity_filter(weighted_triangle)
        filtered = fraction_filter(H, "disparity_pvalue", 0.5)
        assert filtered.number_of_edges() <= H.number_of_edges()

    def test_invalid_fraction_raises(self, weighted_triangle):
        with pytest.raises(ValueError):
            fraction_filter(weighted_triangle, "weight", 0.0)

    def test_fraction_one_keeps_all(self, weighted_triangle):
        H = disparity_filter(weighted_triangle)
        filtered = fraction_filter(H, "disparity_pvalue", 1.0)
        assert filtered.number_of_edges() == H.number_of_edges()


class TestBooleanFilter:
    def test_filters_by_boolean(self):
        G = nx.Graph()
        G.add_edge(0, 1, keep=True)
        G.add_edge(1, 2, keep=False)
        G.add_edge(2, 3, keep=True)
        H = boolean_filter(G, "keep")
        assert sorted(H.edges()) == [(0, 1), (2, 3)]

    def test_preserves_nodes(self):
        G = nx.Graph()
        G.add_node(5)
        G.add_edge(0, 1, keep=True)
        G.add_edge(1, 2, keep=False)
        H = boolean_filter(G, "keep")
        assert 5 in H.nodes()


class TestConsensusBackbone:
    def test_intersection_of_edges(self):
        G1 = nx.Graph([(0, 1), (1, 2)])
        G2 = nx.Graph([(0, 1), (2, 3)])
        H = consensus_backbone(G1, G2)
        assert sorted(H.edges()) == [(0, 1)]

    def test_requires_at_least_two(self):
        G = nx.Graph([(0, 1)])
        with pytest.raises(ValueError):
            consensus_backbone(G)

    def test_empty_intersection(self):
        G1 = nx.Graph([(0, 1)])
        G2 = nx.Graph([(2, 3)])
        H = consensus_backbone(G1, G2)
        assert H.number_of_edges() == 0
