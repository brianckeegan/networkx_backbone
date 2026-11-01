"""Shortest-path backbones: salience, metric backbone and GLAB.

All three run all-pairs shortest paths, so they are timed on a mid-size
synthetic weighted graph.
"""

import networkx_backbone as nb

from ._data import weighted_graph

METHODS = ["high_salience_skeleton", "metric_backbone", "glab_filter"]


class ShortestPathMethods:
    """Each method on a Barabási-Albert graph (500 nodes, 1491 edges)."""

    params = METHODS
    param_names = ["method"]
    timeout = 300

    def setup(self, method):
        if not hasattr(nb, method):
            raise NotImplementedError
        self.G = weighted_graph(500)

    def time_score(self, method):
        getattr(nb, method)(self.G)
