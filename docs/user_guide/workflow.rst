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

2. **Apply a filter** from the :mod:`~nx_backbone.filters` module to
   select the edges that form the backbone.

::

    import networkx as nx
    import nx_backbone as nb

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
   * - :func:`~nx_backbone.disparity_filter`
     - ``disparity_pvalue``
     - p-value (lower = more significant)
   * - :func:`~nx_backbone.noise_corrected_filter`
     - ``nc_score``, ``nc_sdev``, ``nc_pvalue``
     - Lift score in (-1, 1) and its standard deviation (keep
       ``nc_score > 1.64 * nc_sdev``); p-value (lower = more significant)
   * - :func:`~nx_backbone.marginal_likelihood_filter`
     - ``ml_pvalue``
     - p-value (lower = more significant)
   * - :func:`~nx_backbone.ecm_filter`
     - ``ecm_pvalue``
     - p-value (lower = more significant)
   * - :func:`~nx_backbone.lans_filter`
     - ``lans_pvalue``
     - p-value (lower = more significant)
   * - :func:`~nx_backbone.multiple_linkage_analysis`
     - ``mla_keep`` (node attribute ``mla_rank``)
     - Boolean keep flag; ``method="lans"`` gives ``mla_pvalue`` + ``mla_keep``
   * - :func:`~nx_backbone.global_threshold_filter`
     - ``global_threshold_keep``
     - Boolean keep flag
   * - :func:`~nx_backbone.strongest_n_ties`
     - ``strongest_n_ties_keep``
     - Boolean keep flag
   * - :func:`~nx_backbone.global_sparsification`
     - ``global_sparsification_keep``
     - Boolean keep flag
   * - :func:`~nx_backbone.primary_linkage_analysis`
     - ``primary_linkage_keep``
     - Boolean keep flag
   * - :func:`~nx_backbone.edge_betweenness_filter`
     - ``edge_betweenness`` + ``edge_betweenness_keep``
     - centrality score + boolean keep flag
   * - :func:`~nx_backbone.node_degree_filter`
     - ``node_degree_keep`` (node + edge)
     - Boolean keep flag
   * - :func:`~nx_backbone.high_salience_skeleton`
     - ``salience``
     - Fraction in [0, 1] (higher = more important)
   * - :func:`~nx_backbone.metric_backbone`
     - ``metric_keep``
     - Boolean keep flag
   * - :func:`~nx_backbone.ultrametric_backbone`
     - ``ultrametric_keep``
     - Boolean keep flag
   * - :func:`~nx_backbone.doubly_stochastic_filter`
     - ``ds_weight``, ``ds_keep``
     - Normalized weight (higher = more important); boolean keep flag from
       the connectivity stage
   * - :func:`~nx_backbone.h_backbone`
     - ``h_backbone_keep``
     - Boolean keep flag
   * - :func:`~nx_backbone.glab_filter`
     - ``glab_pvalue``
     - p-value (lower = more significant)
   * - :func:`~nx_backbone.neighborhood_overlap`
     - ``overlap``
     - Count (higher = more embedded)
   * - :func:`~nx_backbone.jaccard_backbone`
     - ``jaccard``
     - Coefficient in [0, 1] (higher = more embedded)
   * - :func:`~nx_backbone.dice_backbone`
     - ``dice``
     - Coefficient in [0, 1] (higher = more embedded)
   * - :func:`~nx_backbone.cosine_backbone`
     - ``cosine``
     - Coefficient in [0, 1] (higher = more embedded)
   * - :func:`~nx_backbone.hub_promoted_index`
     - ``hpi``
     - Index (higher = more embedded)
   * - :func:`~nx_backbone.hub_depressed_index`
     - ``hdi``
     - Index (higher = more embedded)
   * - :func:`~nx_backbone.lhn_local_index`
     - ``lhn_local``
     - Index (higher = more embedded)
   * - :func:`~nx_backbone.preferential_attachment_score`
     - ``pa``
     - Score (higher = more expected)
   * - :func:`~nx_backbone.adamic_adar_index`
     - ``adamic_adar``
     - Score (higher = more embedded)
   * - :func:`~nx_backbone.resource_allocation_index`
     - ``resource_allocation``
     - Score (higher = more embedded)
   * - :func:`~nx_backbone.graph_distance_proximity`
     - ``dist``
     - Reciprocal distance (higher = closer)
   * - :func:`~nx_backbone.local_path_index`
     - ``lp``
     - Score (higher = more embedded)
   * - :func:`~nx_backbone.sparsify` / :func:`~nx_backbone.lspar` / :func:`~nx_backbone.local_degree`
     - ``sparsify_score`` + ``sparsify_keep``
     - score + boolean keep flag
   * - :func:`~nx_backbone.sdsm`
     - ``sdsm_pvalue``
     - p-value (lower = more significant)
   * - :func:`~nx_backbone.fdsm`
     - ``fdsm_pvalue``
     - p-value (lower = more significant)
   * - :func:`~nx_backbone.modularity_backbone`
     - ``vitality`` (node) + ``modularity_keep`` (edge)
     - node score + boolean keep flag

Filtering functions
-------------------

The :mod:`~nx_backbone.filters` module provides filtering functions and
graph-preparation support utilities.

:func:`~nx_backbone.multigraph_to_weighted`
    Convert ``MultiGraph`` / ``MultiDiGraph`` inputs into weighted simple
    graphs by collapsing parallel edges. Use ``edge_type_attr`` to count
    distinct edge types per node pair::

        weighted_simple = nb.multigraph_to_weighted(MG, edge_type_attr="edge_type")

    The high-level :func:`~nx_backbone.backbone_from_weighted` wrapper
    applies this conversion automatically for multigraph inputs by default.

:func:`~nx_backbone.threshold_filter`
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

:func:`~nx_backbone.fraction_filter`
    Keep a fixed fraction of edges, sorted by score. Use ``ascending=True``
    to keep the smallest scores (p-values) or ``ascending=False`` to keep
    the largest scores::

        # Keep the 20% most significant edges
        backbone = nb.fraction_filter(H, "disparity_pvalue", 0.2, ascending=True)

:func:`~nx_backbone.boolean_filter`
    Keep edges where a boolean attribute is truthy::

        H = nb.global_sparsification(G, s=0.4)
        backbone = nb.boolean_filter(H, "global_sparsification_keep")

:func:`~nx_backbone.consensus_backbone`
    Take the intersection of multiple backbones -- only edges present in
    all input backbones are kept::

        b1 = nb.threshold_filter(nb.disparity_filter(G), "disparity_pvalue", 0.05)
        b2 = nb.boolean_filter(nb.metric_backbone(G), "metric_keep")
        consensus = nb.consensus_backbone(b1, b2)

Matrices and edge lists
-----------------------

Networks held as a NumPy or SciPy sparse adjacency matrix, or as a pandas
edge-list ``DataFrame``, can be passed straight to
:func:`~nx_backbone.backbone`,
:func:`~nx_backbone.backbone_from_weighted` and
:func:`~nx_backbone.backbone_from_unweighted`. They are converted with
:func:`~nx_backbone.to_graph`, which uses NetworkX's own converters and
takes ``nodelist`` (matrix row and column labels), ``directed``, and
``source`` / ``target`` / ``weight`` (edge-list column names). An undirected
graph must come from a symmetric matrix, and an edge list must not repeat a
node pair. :func:`~networkx.convert_matrix.to_pandas_edgelist` turns the
backbone back into a tidy ``DataFrame`` with one column per weight and
score::

    import networkx as nx
    import pandas as pd
    import nx_backbone as nb

    edges = pd.DataFrame(
        {"from": ["a", "a", "b", "c"], "to": ["b", "c", "c", "d"],
         "trips": [12, 1, 7, 3]}
    )
    backbone = nb.backbone_from_weighted(
        edges, source="from", target="to", weight="trips", alpha=0.3
    )
    nx.to_pandas_edgelist(backbone, source="from", target="to")

The individual scorers such as :func:`~nx_backbone.disparity_filter`
keep graph-only signatures, as NetworkX functions do: convert once with
:func:`~nx_backbone.to_graph` and pass the graph to as many scorers and
filters as you like. Bipartite functions are the exception; they take an
agents x artifacts incidence matrix directly (see :doc:`../api/bipartite`).

When no ``method`` is given, :func:`~nx_backbone.backbone` treats a
``DataFrame`` with the ``source`` and ``target`` columns as an edge list and
any other matrix as a bipartite incidence matrix. Name a weighted or
unweighted method to read a matrix as an adjacency matrix.

Visual comparison
-----------------

Use :mod:`~nx_backbone.visualization` helpers to compare an original
graph with a backbone and highlight dropped structure::

    backbone = nb.threshold_filter(
        nb.disparity_filter(G), "disparity_pvalue", 0.05, include_all_nodes=False
    )
    fig, ax, diff = nb.compare_graphs(G, backbone, return_diff=True)
    print(f"Removed nodes: {len(diff['removed_nodes'])}")

Methods with boolean keep flags
-------------------------------

Several methods provide built-in boolean edge attributes specifically for
the filter step. Apply :func:`~nx_backbone.boolean_filter` to extract
the final backbone:

- :func:`~nx_backbone.global_threshold_filter` -> ``global_threshold_keep``
- :func:`~nx_backbone.strongest_n_ties` -> ``strongest_n_ties_keep``
- :func:`~nx_backbone.global_sparsification` -> ``global_sparsification_keep``
- :func:`~nx_backbone.primary_linkage_analysis` -> ``primary_linkage_keep``
- :func:`~nx_backbone.edge_betweenness_filter` -> ``edge_betweenness_keep``
- :func:`~nx_backbone.node_degree_filter` -> ``node_degree_keep``
- :func:`~nx_backbone.metric_backbone` -> ``metric_keep``
- :func:`~nx_backbone.ultrametric_backbone` -> ``ultrametric_keep``
- :func:`~nx_backbone.h_backbone` -> ``h_backbone_keep``
- :func:`~nx_backbone.modularity_backbone` -> ``modularity_keep``
- :func:`~nx_backbone.planar_maximally_filtered_graph` -> ``pmfg_keep``
- :func:`~nx_backbone.maximum_spanning_tree_backbone` -> ``mst_keep``
- :func:`~nx_backbone.multiple_linkage_analysis` -> ``mla_keep``
- :func:`~nx_backbone.sparsify` / :func:`~nx_backbone.lspar` /
  :func:`~nx_backbone.local_degree` -> ``sparsify_keep``

For bipartite projections, :func:`~nx_backbone.sdsm` and
:func:`~nx_backbone.fdsm` return full projected graphs with p-values
(``sdsm_pvalue`` / ``fdsm_pvalue``), which you can then filter with
:func:`~nx_backbone.threshold_filter`. Use ``projection=`` to attach
``simple``, ``hyper``, ``probs``, or ``ycn`` projection weights.

Method gallery
--------------

Each backbone module's API page has a method gallery: every method's
backbone drawn over its original graph, regenerated whenever the
documentation is built. :doc:`../tutorials/graph_gallery` links to them all
and compares the methods side by side.
