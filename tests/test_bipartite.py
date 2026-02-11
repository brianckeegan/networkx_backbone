"""Tests for bipartite backbone extraction methods."""

import networkx as nx
import pytest

from networkx_backbone import fdsm, sdsm


class TestSDSM:
    def test_returns_graph(self, bipartite_graph):
        backbone = sdsm(bipartite_graph, agent_nodes=[1, 2, 3])
        assert isinstance(backbone, nx.Graph)

    def test_adds_pvalue_attribute(self, bipartite_graph):
        backbone = sdsm(bipartite_graph, agent_nodes=[1, 2, 3])
        for u, v, data in backbone.edges(data=True):
            assert "sdsm_pvalue" in data

    def test_agent_nodes_only(self, bipartite_graph):
        backbone = sdsm(bipartite_graph, agent_nodes=[1, 2, 3])
        for node in backbone.nodes():
            assert node in [1, 2, 3]

    def test_non_bipartite_raises(self):
        G = nx.complete_graph(3)  # Not bipartite
        with pytest.raises(nx.NetworkXError):
            sdsm(G, agent_nodes=[0, 1])

    def test_invalid_agent_nodes_raises(self, bipartite_graph):
        with pytest.raises(nx.NetworkXError):
            sdsm(bipartite_graph, agent_nodes=[99])


class TestFDSM:
    def test_returns_graph(self, bipartite_graph):
        backbone = fdsm(bipartite_graph, agent_nodes=[1, 2, 3], trials=50, seed=42)
        assert isinstance(backbone, nx.Graph)

    def test_adds_pvalue_attribute(self, bipartite_graph):
        backbone = fdsm(bipartite_graph, agent_nodes=[1, 2, 3], trials=50, seed=42)
        for u, v, data in backbone.edges(data=True):
            assert "fdsm_pvalue" in data

    def test_pvalues_in_range(self, bipartite_graph):
        backbone = fdsm(bipartite_graph, agent_nodes=[1, 2, 3], trials=50, seed=42)
        for _, _, data in backbone.edges(data=True):
            assert 0.0 <= data["fdsm_pvalue"] <= 1.0
