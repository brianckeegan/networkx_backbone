"""Bipartite projection null models on a 400 x 300 bipartite graph.

Guards the vectorized pair loops: ``sdsm`` once took 6.4 s on this size.
"""

import networkx_backbone as nb

from ._data import bipartite_graph

MODELS = ["sdsm", "fdsm", "fixedfill", "fixedrow", "fixedcol"]

# FDSM's cost is linear in the number of Monte Carlo samples; 100 keeps the
# benchmark short while still exercising the curveball sampler.
FDSM_TRIALS = 100


class BipartiteNullModels:
    """Score every agent pair of a 400 x 300 graph (density 0.05)."""

    params = MODELS
    param_names = ["model"]
    timeout = 300

    def setup(self, model):
        if not hasattr(nb, model):
            raise NotImplementedError
        self.B, self.agents = bipartite_graph(400, 300)

    def time_score(self, model):
        if model == "fdsm":
            nb.fdsm(self.B, self.agents, trials=FDSM_TRIALS, seed=0)
        else:
            getattr(nb, model)(self.B, self.agents)
