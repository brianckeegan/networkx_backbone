Workflow
========

This guide explains the canonical workflow for extracting a backbone from
a network.

The score-then-filter pattern
-----------------------------

Most backbone methods follow a two-step process:

1. **Apply a backbone method** to the original graph. This returns a copy of
   the graph with a new edge attribute containing a score (p-value, similarity
   coefficient, salience, etc.).

2. **Apply a filter** from the :mod:`~networkx_backbone.filters` module to
   select the edges that form the backbone.

::

    import networkx as nx
    import networkx_backbone as nb

    G = nx.les_miserables_graph()

    # Step 1: Score
    H = nb.disparity_filter(G)

    # Step 2: Filter
    backbone = nb.threshold_filter(H, "disparity_pvalue", 0.05)

Score attributes by method
--------------------------

Each method adds a specific edge attribute to the returned graph. Use this
attribute name when calling a filter function.

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Method
     - Edge Attribute
     - Interpretation
   * - :func:`~networkx_backbone.disparity_filter`
     - ``disparity_pvalue``
     - p-value (lower = more significant)
   * - :func:`~networkx_backbone.noise_corrected_filter`
     - ``nc_score``
     - z-score (higher = more significant)
   * - :func:`~networkx_backbone.marginal_likelihood_filter`
     - ``ml_pvalue``
     - p-value (lower = more significant)
   * - :func:`~networkx_backbone.ecm_filter`
     - ``ecm_pvalue``
     - p-value (lower = more significant)
   * - :func:`~networkx_backbone.lans_filter`
     - ``lans_pvalue``
     - p-value (lower = more significant)
   * - :func:`~networkx_backbone.high_salience_skeleton`
     - ``salience``
     - Fraction in [0, 1] (higher = more important)
   * - :func:`~networkx_backbone.doubly_stochastic_filter`
     - ``ds_weight``
     - Normalized weight (higher = more important)
   * - :func:`~networkx_backbone.glab_filter`
     - ``glab_pvalue``
     - p-value (lower = more significant)
   * - :func:`~networkx_backbone.neighborhood_overlap`
     - ``overlap``
     - Count (higher = more embedded)
   * - :func:`~networkx_backbone.jaccard_backbone`
     - ``jaccard``
     - Coefficient in [0, 1] (higher = more embedded)
   * - :func:`~networkx_backbone.dice_backbone`
     - ``dice``
     - Coefficient in [0, 1] (higher = more embedded)
   * - :func:`~networkx_backbone.cosine_backbone`
     - ``cosine``
     - Coefficient in [0, 1] (higher = more embedded)
   * - :func:`~networkx_backbone.hub_promoted_index`
     - ``hpi``
     - Index (higher = more embedded)
   * - :func:`~networkx_backbone.hub_depressed_index`
     - ``hdi``
     - Index (higher = more embedded)
   * - :func:`~networkx_backbone.lhn_local_index`
     - ``lhn_local``
     - Index (higher = more embedded)
   * - :func:`~networkx_backbone.preferential_attachment_score`
     - ``pa``
     - Score (higher = more expected)
   * - :func:`~networkx_backbone.adamic_adar_index`
     - ``adamic_adar``
     - Score (higher = more embedded)
   * - :func:`~networkx_backbone.resource_allocation_index`
     - ``resource_allocation``
     - Score (higher = more embedded)
   * - :func:`~networkx_backbone.graph_distance_proximity`
     - ``dist``
     - Reciprocal distance (higher = closer)
   * - :func:`~networkx_backbone.local_path_index`
     - ``lp``
     - Score (higher = more embedded)
   * - :func:`~networkx_backbone.sdsm`
     - ``sdsm_pvalue``
     - p-value (lower = more significant)
   * - :func:`~networkx_backbone.fdsm`
     - ``fdsm_pvalue``
     - p-value (lower = more significant)
   * - :func:`~networkx_backbone.modularity_backbone`
     - ``vitality`` (node)
     - Vitality score (node attribute)

Filtering functions
-------------------

The :mod:`~networkx_backbone.filters` module provides filtering functions and
graph-preparation support utilities.

:func:`~networkx_backbone.multigraph_to_weighted`
    Convert ``MultiGraph`` / ``MultiDiGraph`` inputs into weighted simple
    graphs by collapsing parallel edges. Use ``edge_type_attr`` to count
    distinct edge types per node pair::

        weighted_simple = nb.multigraph_to_weighted(MG, edge_type_attr="edge_type")

    The high-level :func:`~networkx_backbone.backbone_from_weighted` wrapper
    applies this conversion automatically for multigraph inputs by default.

:func:`~networkx_backbone.threshold_filter`
    Keep edges where the score is below (for p-values) or above (for importance
    scores) a given threshold. Use ``mode="below"`` for p-values and
    ``mode="above"`` for importance scores. By default, all input nodes are
    kept; set ``include_all_nodes=False`` to drop isolates::

        # For p-values (keep low values)
        backbone = nb.threshold_filter(H, "disparity_pvalue", 0.05)

        # For scores (keep high values)
        backbone = nb.threshold_filter(H, "salience", 0.5, mode="above")

        # Optionally remove isolate nodes after edge filtering
        backbone = nb.threshold_filter(H, "disparity_pvalue", 0.05, include_all_nodes=False)

:func:`~networkx_backbone.fraction_filter`
    Keep a fixed fraction of edges, sorted by score. Use ``ascending=True``
    to keep the smallest scores (p-values) or ``ascending=False`` to keep
    the largest scores::

        # Keep the 20% most significant edges
        backbone = nb.fraction_filter(H, "disparity_pvalue", 0.2, ascending=True)

:func:`~networkx_backbone.boolean_filter`
    Keep edges where a boolean attribute is truthy::

        backbone = nb.boolean_filter(H, "is_significant")

:func:`~networkx_backbone.consensus_backbone`
    Take the intersection of multiple backbones -- only edges present in
    all input backbones are kept::

        b1 = nb.threshold_filter(nb.disparity_filter(G), "disparity_pvalue", 0.05)
        b2 = nb.metric_backbone(G)
        consensus = nb.consensus_backbone(b1, b2)

Visual comparison
-----------------

Use :mod:`~networkx_backbone.visualization` helpers to compare an original
graph with a backbone and highlight dropped structure::

    backbone = nb.threshold_filter(
        nb.disparity_filter(G), "disparity_pvalue", 0.05, include_all_nodes=False
    )
    fig, ax, diff = nb.compare_graphs(G, backbone, return_diff=True)
    print(f"Removed nodes: {len(diff['removed_nodes'])}")

Methods that return subgraphs directly
--------------------------------------

Some methods return the backbone directly without needing a separate filter
step:

- :func:`~networkx_backbone.metric_backbone` -- edges on shortest paths
- :func:`~networkx_backbone.ultrametric_backbone` -- edges on minimax paths
- :func:`~networkx_backbone.maximum_spanning_tree_backbone` -- maximum spanning tree
- :func:`~networkx_backbone.planar_maximally_filtered_graph` -- planar subgraph
- :func:`~networkx_backbone.global_threshold_filter` -- simple weight cutoff
- :func:`~networkx_backbone.strongest_n_ties` -- top-N edges per node
- :func:`~networkx_backbone.global_sparsification` -- global edge ranking by weight
- :func:`~networkx_backbone.primary_linkage_analysis` -- strongest outgoing edge per node
- :func:`~networkx_backbone.edge_betweenness_filter` -- top edges by betweenness
- :func:`~networkx_backbone.node_degree_filter` -- induced subgraph by node degree
- :func:`~networkx_backbone.h_backbone` -- h-index inspired subgraph
- :func:`~networkx_backbone.sparsify` -- unweighted sparsification
- :func:`~networkx_backbone.lspar` -- local sparsification
- :func:`~networkx_backbone.local_degree` -- degree-based sparsification
- :func:`~networkx_backbone.multiple_linkage_analysis` -- statistical linkage subgraph

For bipartite projections, :func:`~networkx_backbone.sdsm` and
:func:`~networkx_backbone.fdsm` return full projected graphs with p-values
(``sdsm_pvalue`` / ``fdsm_pvalue``), which you can then filter with
:func:`~networkx_backbone.threshold_filter`. Use ``projection=`` to attach
``simple``, ``hyper``, ``probs``, or ``ycn`` projection weights.
