"""Statistical edge scorers on synthetic weighted graphs.

The same scorers on Les Misérables are timed by ``les_miserables.py``.
"""

import networkx_backbone as nb

from ._data import weighted_graph

SCORERS = [
    "disparity_filter",
    "noise_corrected_filter",
    "marginal_likelihood_filter",
    "ecm_filter",
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
    """``ecm_filter`` on a 20,000-node graph (59,991 edges).

    The fit runs over the 1,695 distinct (degree, strength) classes rather
    than the nodes (#21).  Before that, the fit took over 100 s at 2,000
    nodes; one sample is enough at this size.
    """

    timeout = 300
    number = 1
    repeat = 1
    warmup_time = 0

    def setup(self):
        if not hasattr(nb, "ecm_filter"):
            raise NotImplementedError
        self.G = weighted_graph(20000)

    def time_ecm_filter(self):
        nb.ecm_filter(self.G)
