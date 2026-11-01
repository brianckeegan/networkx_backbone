"""Les Misérables score-then-filter benchmark.

This module is the single definition of the benchmark: which scorer each
method runs, with which arguments, and which filter turns the scores into a
backbone.  Three consumers read it:

- ``tests/test_les_miserables.py`` pins the edge count each case keeps;
- ``docs/conf.py`` runs every case at build time to render the table in
  ``docs/tutorials/les_miserables_benchmark.rst``;
- asv times every scorer (``LesMiserables`` below).

To add a method, append a ``Case`` here and pin its count in the test.
"""

from dataclasses import dataclass, field

import networkx as nx

import networkx_backbone as nb

ALPHA = 0.05  # significance level of the p-value filters
TOP_FRACTION = 0.3  # fraction of edges the proximity scores keep


@dataclass(frozen=True)
class Case:
    """One scorer plus the filter that extracts its backbone.

    Parameters
    ----------
    method : str
        Name of the scoring function in ``networkx_backbone``.
    filter : {"threshold", "fraction", "boolean"}
        Which filter extracts the backbone from the scored graph.
    attr : str
        Edge attribute the filter reads.
    value : float or None, optional
        Threshold for ``"threshold"``, fraction for ``"fraction"``.
    mode : {"below", "above"}, optional
        ``threshold_filter`` mode.
    kwargs : dict, optional
        Extra keyword arguments for the scorer.
    label : str or None, optional
        Test id and table label; defaults to *method*.
    weighted : bool, optional
        Whether the case runs on the weighted or the unweighted graph.
    """

    method: str
    filter: str
    attr: str
    value: float | None = None
    mode: str = "below"
    kwargs: dict = field(default_factory=dict)
    label: str | None = None
    weighted: bool = True

    @property
    def name(self):
        """The label used as test id and in the docs table."""
        return self.label or self.method

    def score(self, G):
        """Run the scorer and return the scored graph."""
        return getattr(nb, self.method)(G, **self.kwargs)

    def filter_graph(self, H):
        """Extract the backbone from a scored graph."""
        if self.filter == "threshold":
            return nb.threshold_filter(H, self.attr, self.value, mode=self.mode)
        if self.filter == "fraction":
            return nb.fraction_filter(H, self.attr, self.value, ascending=False)
        if self.filter == "boolean":
            return nb.boolean_filter(H, self.attr)
        raise ValueError(f"unknown filter {self.filter!r}")

    def run(self, G):
        """Score *G*, then filter it."""
        return self.filter_graph(self.score(G))

    def describe_filter(self):
        """Describe the filter in reStructuredText (for the docs table)."""
        if self.filter == "threshold":
            op = "<" if self.mode == "below" else ">="
            return f"``{self.attr} {op} {self.value!r}``"
        if self.filter == "fraction":
            return f"top {self.value:.0%} by ``{self.attr}``"
        return f"``{self.attr}`` (boolean)"


# -- cases --------------------------------------------------------------------
def _pvalue(method, attr):
    return Case(method, "threshold", attr, ALPHA, "below")


def _top(method, attr):
    return Case(method, "fraction", attr, TOP_FRACTION)


def _keep(method, attr, label=None, **kwargs):
    return Case(method, "boolean", attr, kwargs=kwargs, label=label)


WEIGHTED_CASES = [
    # Statistical
    _pvalue("disparity_filter", "disparity_pvalue"),
    _pvalue("noise_corrected_filter", "nc_pvalue"),
    Case(
        "noise_corrected_filter",
        "threshold",
        "nc_score",
        2.0,
        "above",
        kwargs={"method": "zscore"},
        label="noise_corrected_filter(method='zscore')",
    ),
    _pvalue("marginal_likelihood_filter", "ml_pvalue"),
    _pvalue("ecm_filter", "ecm_pvalue"),
    _pvalue("lans_filter", "lans_pvalue"),
    _keep("multiple_linkage_analysis", "mla_keep"),
    Case(
        "multiple_linkage_analysis",
        "boolean",
        "mla_keep",
        kwargs={"method": "lans", "alpha": ALPHA},
        label="multiple_linkage_analysis(method='lans', alpha=0.05)",
    ),
    _pvalue("polya_filter", "polya_pvalue"),
    # Structural
    _keep(
        "global_threshold_filter",
        "global_threshold_keep",
        "global_threshold_filter(threshold=2)",
        threshold=2,
    ),
    _keep("strongest_n_ties", "strongest_n_ties_keep", "strongest_n_ties(n=2)", n=2),
    _keep(
        "global_sparsification",
        "global_sparsification_keep",
        "global_sparsification(s=0.5)",
        s=0.5,
    ),
    _keep("primary_linkage_analysis", "primary_linkage_keep"),
    _keep(
        "edge_betweenness_filter",
        "edge_betweenness_keep",
        "edge_betweenness_filter(s=0.5)",
        s=0.5,
    ),
    _keep(
        "node_degree_filter",
        "node_degree_keep",
        "node_degree_filter(min_degree=2)",
        min_degree=2,
    ),
    Case("high_salience_skeleton", "threshold", "salience", 0.5, "above"),
    _keep("metric_backbone", "metric_keep"),
    _keep("ultrametric_backbone", "ultrametric_keep"),
    Case("doubly_stochastic_filter", "threshold", "ds_weight", 0.1, "above"),
    _keep("doubly_stochastic_filter", "ds_keep", "doubly_stochastic_filter (ds_keep)"),
    _keep("h_backbone", "h_backbone_keep"),
    _keep("modularity_backbone", "modularity_keep", seed=42),
    _keep("planar_maximally_filtered_graph", "pmfg_keep"),
    _keep("maximum_spanning_tree_backbone", "mst_keep"),
    # Proximity
    _top("neighborhood_overlap", "overlap"),
    _top("jaccard_backbone", "jaccard"),
    _top("dice_backbone", "dice"),
    _top("cosine_backbone", "cosine"),
    _top("hub_promoted_index", "hpi"),
    _top("hub_depressed_index", "hdi"),
    _top("lhn_local_index", "lhn_local"),
    _top("preferential_attachment_score", "pa"),
    _top("adamic_adar_index", "adamic_adar"),
    _top("resource_allocation_index", "resource_allocation"),
    _top("graph_distance_proximity", "dist"),
    _top("local_path_index", "lp"),
    # Hybrid
    _pvalue("glab_filter", "glab_pvalue"),
]

UNWEIGHTED_CASES = [
    Case("sparsify", "boolean", "sparsify_keep", weighted=False),
    Case("lspar", "boolean", "sparsify_keep", weighted=False),
    Case("local_degree", "boolean", "sparsify_keep", weighted=False),
]
# -- end cases ----------------------------------------------------------------

CASES = WEIGHTED_CASES + UNWEIGHTED_CASES


def les_miserables_weighted():
    """Weighted co-appearance graph of *Les Misérables* (77 nodes, 254 edges)."""
    return nx.les_miserables_graph()


def les_miserables_unweighted():
    """The same graph with every edge attribute dropped."""
    G = les_miserables_weighted()
    U = nx.Graph()
    U.add_nodes_from(G.nodes(data=True))
    U.add_edges_from(G.edges())
    return U


def graph_for(case):
    """The Les Misérables graph a case runs on."""
    return les_miserables_weighted() if case.weighted else les_miserables_unweighted()


class LesMiserables:
    """asv: time every scorer of the benchmark table on Les Misérables."""

    params = [case.name for case in CASES]
    param_names = ["method"]

    def setup(self, name):
        self.case = next(case for case in CASES if case.name == name)
        if not hasattr(nb, self.case.method):
            raise NotImplementedError  # asv skips methods an old commit lacks
        self.G = graph_for(self.case)

    def time_score(self, name):
        self.case.score(self.G)
