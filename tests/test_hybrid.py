"""Tests for hybrid backbone extraction methods."""

import networkx as nx
import pytest

from networkx_backbone import glab_filter


class TestGLABFilter:
    def test_adds_pvalue_attribute(self, weighted_triangle):
        H = glab_filter(weighted_triangle)
        for u, v, data in H.edges(data=True):
            assert "glab_pvalue" in data

    def test_pvalues_in_range(self, weighted_triangle):
        H = glab_filter(weighted_triangle)
        for _, _, data in H.edges(data=True):
            assert 0.0 <= data["glab_pvalue"] <= 1.0

    def test_preserves_structure(self, weighted_triangle):
        H = glab_filter(weighted_triangle)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()
        assert set(H.nodes()) == set(weighted_triangle.nodes())

    def test_rejects_directed(self):
        G = nx.DiGraph()
        G.add_weighted_edges_from([(0, 1, 1.0)])
        with pytest.raises(nx.NetworkXNotImplemented):
            glab_filter(G)

    def test_single_node(self):
        G = nx.Graph()
        G.add_node(0)
        H = glab_filter(G)
        assert H.number_of_nodes() == 1
        assert H.number_of_edges() == 0
