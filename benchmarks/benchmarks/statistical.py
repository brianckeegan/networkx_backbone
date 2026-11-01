"""Statistical edge scorers on synthetic weighted graphs.

The same scorers on Les Misérables are timed by ``les_miserables.py``.
"""

import networkx_backbone as nb

from ._data import weighted_graph

SCORERS = [
    "disparity_filter",
    "noise_corrected_filter",
    "marginal_likelihood_filter",
    "lans_filter",
    "multiple_linkage_analysis",
]


class StatisticalScorers:
    """Each scorer on a Barabási-Albert graph (2000 nodes, 5991 edges)."""

    params = SCORERS
    param_names = ["method"]
    timeout = 300

    def setup(self, method):
        if not hasattr(nb, method):
            raise NotImplementedError
        self.G = weighted_graph(2000)

    def time_score(self, method):
        getattr(nb, method)(self.G)


class ECMFilter:
    """``ecm_filter`` on a 500-node graph (1491 edges).

    The maximum-entropy fit is far costlier than the closed-form scorers: about
    6 s at 500 nodes and over 100 s at 2000, too slow to repeat in a suite.
    """

    timeout = 300

    def setup(self):
        if not hasattr(nb, "ecm_filter"):
            raise NotImplementedError
        self.G = weighted_graph(500)

    def time_ecm_filter(self):
        nb.ecm_filter(self.G)
