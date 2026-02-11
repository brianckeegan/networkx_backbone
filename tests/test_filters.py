"""Tests for filtering utilities."""

import networkx as nx
import pytest

from networkx_backbone import (
    boolean_filter,
    consensus_backbone,
    disparity_filter,
    fraction_filter,
    threshold_filter,
)


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
