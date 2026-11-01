"""
Backbone extraction algorithms for complex networks.

This package provides algorithms for extracting backbone structures from
networks, organized into the following submodules:

- **statistical**: Hypothesis-testing methods (disparity, noise-corrected, etc.)
- **structural**: Topology-based methods (threshold, spanning tree, salience, etc.)
- **proximity**: Neighborhood-similarity edge scoring (Jaccard, Dice, cosine, etc.)
- **hybrid**: Combined statistical/structural methods (GLAB)
- **bipartite**: Bipartite projection backbones (SDSM, FDSM, fixed models, wrappers)
- **hypergraph**: Hypergraph backbones (MDL pruning, statistical validation,
  structural reduction); ingestion/interoperability helpers live in
  ``hypergraph_io`` (HIF, xgi, HyperNetX, HypergraphX, HAT)
- **unweighted**: Sparsification for unweighted graphs (LSpar, local degree)
- **filters**: Post-hoc filtering utilities (threshold, fraction, boolean, consensus)
- **measures**: Evaluation measures for comparing backbones
- **visualization**: Graph-comparison plotting helpers
- **describe**: Methods-section text and citations for a backbone
- **convert**: NumPy/SciPy adjacency matrices and pandas edge lists to graphs
- **temporal**: Temporal networks as snapshots, events or spans, conversions
  between the three forms, and pandas event tables
"""

from networkx_backbone.bipartite import *
from networkx_backbone.convert import *
from networkx_backbone.describe import *
from networkx_backbone.filters import *
from networkx_backbone.hybrid import *
from networkx_backbone.hypergraph import *
from networkx_backbone.hypergraph_io import *
from networkx_backbone.measures import *
from networkx_backbone.proximity import *
from networkx_backbone.statistical import *
from networkx_backbone.structural import *
from networkx_backbone.temporal import *
from networkx_backbone.unweighted import *
from networkx_backbone.visualization import *

__all__ = [
    # Statistical
    "disparity_filter",
    "noise_corrected_filter",
    "marginal_likelihood_filter",
    "ecm_filter",
    "lans_filter",
    "multiple_linkage_analysis",
    "polya_filter",
    "gloss_filter",
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
    # Hypergraph
    "intersection_graph",
    "mdl_hypergraph_backbone",
    "local_mdl_hypergraph_backbone",
    "hypergraph_compression_ratio",
    "HypergraphBackbone",
    "maximal_hyperedges",
    "order_filter",
    "s_components",
    "statistically_validated_hypergraph",
    "statistically_validated_cores",
    "ValidatedHypergraph",
    "svh",
    "svc",
    # Hypergraph interoperability
    "hypergraph_to_bipartite",
    "read_hif",
    "write_hif",
    "from_xgi",
    "to_xgi",
    "from_hypernetx",
    "to_hypernetx",
    "from_hypergraphx",
    "to_hypergraphx",
    "from_hat",
    "to_hat",
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
    "adjust_pvalues",
    "transform_weights",
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
    # Reporting
    "describe_backbone",
    # Conversion
    "to_graph",
    # Temporal networks
    "temporal_form",
    "convert_temporal",
    "snapshots_from_events",
    "snapshots_from_spans",
    "events_from_snapshots",
    "spans_from_snapshots",
    "events_from_spans",
    "spans_from_events",
    "from_event_table",
    "to_event_table",
]
