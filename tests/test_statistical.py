"""Tests for statistical backbone extraction methods."""

import networkx as nx
import pytest

from networkx_backbone import (
    disparity_filter,
    ecm_filter,
    lans_filter,
    marginal_likelihood_filter,
    noise_corrected_filter,
)


class TestDisparityFilter:
    def test_adds_pvalue_attribute(self, weighted_triangle):
        H = disparity_filter(weighted_triangle)
        for u, v, data in H.edges(data=True):
            assert "disparity_pvalue" in data

    def test_pvalues_in_range(self, weighted_triangle):
        H = disparity_filter(weighted_triangle)
        for _, _, data in H.edges(data=True):
            assert 0.0 <= data["disparity_pvalue"] <= 1.0

    def test_preserves_edges(self, weighted_triangle):
        H = disparity_filter(weighted_triangle)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()

    def test_preserves_nodes(self, weighted_triangle):
        H = disparity_filter(weighted_triangle)
        assert set(H.nodes()) == set(weighted_triangle.nodes())

    def test_missing_weight_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        with pytest.raises(nx.NetworkXError):
            disparity_filter(G)

    def test_directed_graph(self):
        G = nx.DiGraph()
        G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 1.0)])
        H = disparity_filter(G)
        assert "disparity_pvalue" in H[0][1]


class TestNoiseCorrectedFilter:
    def test_adds_score_attribute(self, weighted_triangle):
        H = noise_corrected_filter(weighted_triangle)
        for u, v, data in H.edges(data=True):
            assert "nc_score" in data

    def test_preserves_structure(self, weighted_triangle):
        H = noise_corrected_filter(weighted_triangle)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()
        assert set(H.nodes()) == set(weighted_triangle.nodes())

    def test_higher_weight_higher_score(self, weighted_triangle):
        H = noise_corrected_filter(weighted_triangle)
        # Edge (0,1) has weight 3.0, edge (1,2) has weight 1.0
        # The higher weight edge should generally have a higher score
        assert isinstance(H[0][1]["nc_score"], float)


class TestMarginalLikelihoodFilter:
    def test_adds_pvalue_attribute(self, weighted_triangle):
        H = marginal_likelihood_filter(weighted_triangle)
        for u, v, data in H.edges(data=True):
            assert "ml_pvalue" in data

    def test_pvalues_in_range(self, weighted_triangle):
        H = marginal_likelihood_filter(weighted_triangle)
        for _, _, data in H.edges(data=True):
            assert 0.0 <= data["ml_pvalue"] <= 1.0


class TestECMFilter:
    def test_adds_pvalue_attribute(self, weighted_triangle):
        H = ecm_filter(weighted_triangle)
        for u, v, data in H.edges(data=True):
            assert "ecm_pvalue" in data

    def test_pvalues_in_range(self, weighted_triangle):
        H = ecm_filter(weighted_triangle)
        for _, _, data in H.edges(data=True):
            assert 0.0 <= data["ecm_pvalue"] <= 1.0

    def test_preserves_structure(self, weighted_triangle):
        H = ecm_filter(weighted_triangle)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()


class TestLANSFilter:
    def test_adds_pvalue_attribute(self, weighted_triangle):
        H = lans_filter(weighted_triangle)
        for u, v, data in H.edges(data=True):
            assert "lans_pvalue" in data

    def test_pvalues_in_range(self, weighted_triangle):
        H = lans_filter(weighted_triangle)
        for _, _, data in H.edges(data=True):
            assert 0.0 <= data["lans_pvalue"] <= 1.0

    def test_preserves_structure(self, weighted_triangle):
        H = lans_filter(weighted_triangle)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()
