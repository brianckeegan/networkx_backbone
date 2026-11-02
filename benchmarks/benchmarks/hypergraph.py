"""Hypergraph backbones: MDL pruning and statistical validation.

Guards the greedy MDL optimizers (``_greedy_node`` once scaled roughly
cubically, 8.3 s at m = 2000) and the SVC sub-group enumeration.
"""

import nx_backbone as nb

from ._data import group_hypergraph, nested_hypergraph


class MDLHypergraphBackbone:
    """``mdl_hypergraph_backbone`` on nested hypergraphs of m hyperedges."""

    params = ([500, 2000], ["edge", "node", "auto"])
    param_names = ["m", "method"]
    timeout = 300

    def setup(self, m, method):
        if not hasattr(nb, "mdl_hypergraph_backbone"):
            raise NotImplementedError
        self.hyperedges = nested_hypergraph(m)

    def time_mdl(self, m, method):
        nb.mdl_hypergraph_backbone(self.hyperedges, method=method)


class StatisticallyValidated:
    """SVH and SVC on 2000 hyperedges of size 2-8 over 300 nodes."""

    params = ["statistically_validated_hypergraph", "statistically_validated_cores"]
    param_names = ["method"]
    timeout = 300

    def setup(self, method):
        if not hasattr(nb, method):
            raise NotImplementedError
        self.hyperedges = group_hypergraph(2000, 300)

    def time_validate(self, method):
        getattr(nb, method)(self.hyperedges)
