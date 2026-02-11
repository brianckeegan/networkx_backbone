"""Tests for unweighted network backbone methods."""

import networkx as nx
import pytest

from networkx_backbone import local_degree, lspar, sparsify


class TestSparsify:
    def test_reduces_edges(self, karate):
        H = sparsify(karate, s=0.5)
        assert H.number_of_edges() < karate.number_of_edges()

    def test_preserves_nodes(self, karate):
        H = sparsify(karate, s=0.5)
        assert set(H.nodes()) == set(karate.nodes())

    def test_unknown_filter_raises(self, karate):
        with pytest.raises(ValueError):
            sparsify(karate, filter="invalid")

    def test_threshold_filter(self, karate):
        H = sparsify(karate, filter="threshold", s=0.5)
        assert H.number_of_edges() <= karate.number_of_edges()

    def test_different_scoring_methods(self, karate):
        for method in ["jaccard", "degree", "triangles"]:
            H = sparsify(karate, escore=method, s=0.5)
            assert H.number_of_edges() < karate.number_of_edges()

    def test_rejects_directed(self):
        G = nx.DiGraph([(0, 1), (1, 2)])
        with pytest.raises(nx.NetworkXNotImplemented):
            sparsify(G)


class TestLSpar:
    def test_reduces_edges(self, karate):
        H = lspar(karate, s=0.5)
        assert H.number_of_edges() < karate.number_of_edges()


class TestLocalDegree:
    def test_reduces_edges(self, karate):
        H = local_degree(karate, s=0.3)
        assert H.number_of_edges() < karate.number_of_edges()
