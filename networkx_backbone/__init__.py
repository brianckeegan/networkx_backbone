"""
Backbone extraction algorithms for complex networks.

This package provides algorithms for extracting backbone structures from
networks, organized into nine submodules:

- **statistical**: Hypothesis-testing methods (disparity, noise-corrected, etc.)
- **structural**: Topology-based methods (threshold, spanning tree, salience, etc.)
- **proximity**: Neighborhood-similarity edge scoring (Jaccard, Dice, cosine, etc.)
- **hybrid**: Combined statistical/structural methods (GLAB)
- **bipartite**: Bipartite projection backbones (SDSM, FDSM, fixed models, wrappers)
- **unweighted**: Sparsification for unweighted graphs (LSpar, local degree)
- **filters**: Post-hoc filtering utilities (threshold, fraction, boolean, consensus)
- **measures**: Evaluation measures for comparing backbones
- **visualization**: Graph-comparison plotting helpers
"""

from networkx_backbone.statistical import *  # noqa: F401,F403
from networkx_backbone.structural import *  # noqa: F401,F403
from networkx_backbone.proximity import *  # noqa: F401,F403
from networkx_backbone.hybrid import *  # noqa: F401,F403
from networkx_backbone.bipartite import *  # noqa: F401,F403
from networkx_backbone.unweighted import *  # noqa: F401,F403
from networkx_backbone.filters import *  # noqa: F401,F403
from networkx_backbone.measures import *  # noqa: F401,F403
from networkx_backbone.visualization import *  # noqa: F401,F403

__all__ = [
    # Statistical
    "disparity_filter",
    "noise_corrected_filter",
    "marginal_likelihood_filter",
    "ecm_filter",
    "lans_filter",
    "multiple_linkage_analysis",
    # Statistical aliases
    "disparity",
    "mlf",
    "lans",
    # Structural
    "global_threshold_filter",
    "strongest_n_ties",
    "global_sparsification",
    "primary_linkage_analysis",
    "edge_betweenness_filter",
    "node_degree_filter",
    "high_salience_skeleton",
    "metric_backbone",
    "ultrametric_backbone",
    "doubly_stochastic_filter",
    "h_backbone",
    "modularity_backbone",
    "planar_maximally_filtered_graph",
    "maximum_spanning_tree_backbone",
    # Proximity
    "neighborhood_overlap",
    "jaccard_backbone",
    "dice_backbone",
    "cosine_backbone",
    "hub_promoted_index",
    "hub_depressed_index",
    "lhn_local_index",
    "preferential_attachment_score",
    "adamic_adar_index",
    "resource_allocation_index",
    "graph_distance_proximity",
    "local_path_index",
    # Hybrid
    "glab_filter",
    # Bipartite
    "simple_projection",
    "hyper_projection",
    "probs_projection",
    "ycn_projection",
    "bipartite_projection",
    "sdsm",
    "fdsm",
    "fixedfill",
    "fixedrow",
    "fixedcol",
    "bicm",
    "fastball",
    # High-level wrappers
    "backbone_from_projection",
    "backbone_from_weighted",
    "backbone_from_unweighted",
    "backbone",
    # Unweighted
    "sparsify",
    "lspar",
    "local_degree",
    # Filters
    "multigraph_to_weighted",
    "threshold_filter",
    "fraction_filter",
    "boolean_filter",
    "consensus_backbone",
    # Measures
    "node_fraction",
    "edge_fraction",
    "weight_fraction",
    "reachability",
    "ks_degree",
    "ks_weight",
    "compare_backbones",
    # Visualization
    "graph_difference",
    "compare_graphs",
    "save_graph_comparison",
]
