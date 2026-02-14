Les Miserables Benchmark
========================

This benchmark uses NetworkX's built-in ``nx.les_miserables_graph()`` as a
common example dataset across non-bipartite backbone methods.

Setup
-----

::

    import networkx as nx
    import networkx_backbone as nb

    # Weighted graph from NetworkX social generators
    G = nx.les_miserables_graph()

    # Unweighted version for unweighted sparsification methods
    G_unweighted = nx.Graph()
    G_unweighted.add_nodes_from(G.nodes(data=True))
    G_unweighted.add_edges_from(G.edges())

    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

For this graph, the original edge count is **254**.

Edge Counts by Method
---------------------

The table below reports the number of edges after each non-bipartite backbone
function call.

.. list-table::
   :header-rows: 1
   :widths: 45 20 20 15

   * - Function
     - Graph Used
     - Parameters
     - Edges After
   * - :func:`~networkx_backbone.disparity_filter`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.noise_corrected_filter`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.marginal_likelihood_filter`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.ecm_filter`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.lans_filter`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.multiple_linkage_analysis`
     - ``G``
     - ``alpha=0.05``
     - 109
   * - :func:`~networkx_backbone.global_threshold_filter`
     - ``G``
     - ``threshold=2``
     - 157
   * - :func:`~networkx_backbone.strongest_n_ties`
     - ``G``
     - ``n=2``
     - 113
   * - :func:`~networkx_backbone.global_sparsification`
     - ``G``
     - ``s=0.5``
     - 127
   * - :func:`~networkx_backbone.primary_linkage_analysis`
     - ``G``
     - defaults
     - 77
   * - :func:`~networkx_backbone.edge_betweenness_filter`
     - ``G``
     - ``s=0.5``
     - 127
   * - :func:`~networkx_backbone.node_degree_filter`
     - ``G``
     - ``min_degree=2``
     - 237
   * - :func:`~networkx_backbone.high_salience_skeleton`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.metric_backbone`
     - ``G``
     - defaults
     - 163
   * - :func:`~networkx_backbone.ultrametric_backbone`
     - ``G``
     - defaults
     - 118
   * - :func:`~networkx_backbone.doubly_stochastic_filter`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.h_backbone`
     - ``G``
     - defaults
     - 22
   * - :func:`~networkx_backbone.modularity_backbone`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.planar_maximally_filtered_graph`
     - ``G``
     - defaults
     - 162
   * - :func:`~networkx_backbone.maximum_spanning_tree_backbone`
     - ``G``
     - defaults
     - 76
   * - :func:`~networkx_backbone.neighborhood_overlap`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.jaccard_backbone`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.dice_backbone`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.cosine_backbone`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.hub_promoted_index`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.hub_depressed_index`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.lhn_local_index`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.preferential_attachment_score`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.adamic_adar_index`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.resource_allocation_index`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.graph_distance_proximity`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.local_path_index`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.glab_filter`
     - ``G``
     - defaults
     - 254
   * - :func:`~networkx_backbone.sparsify`
     - ``G_unweighted``
     - defaults
     - 136
   * - :func:`~networkx_backbone.lspar`
     - ``G_unweighted``
     - defaults
     - 136
   * - :func:`~networkx_backbone.local_degree`
     - ``G_unweighted``
     - defaults
     - 135

Reproducibility Script
----------------------

::

    import networkx as nx
    import networkx_backbone as nb

    G = nx.les_miserables_graph()
    G_unweighted = nx.Graph()
    G_unweighted.add_nodes_from(G.nodes(data=True))
    G_unweighted.add_edges_from(G.edges())

    methods = [
        ("disparity_filter", lambda: nb.disparity_filter(G)),
        ("noise_corrected_filter", lambda: nb.noise_corrected_filter(G)),
        ("marginal_likelihood_filter", lambda: nb.marginal_likelihood_filter(G)),
        ("ecm_filter", lambda: nb.ecm_filter(G)),
        ("lans_filter", lambda: nb.lans_filter(G)),
        ("multiple_linkage_analysis(alpha=0.05)", lambda: nb.multiple_linkage_analysis(G, alpha=0.05)),
        ("global_threshold_filter(threshold=2)", lambda: nb.global_threshold_filter(G, threshold=2)),
        ("strongest_n_ties(n=2)", lambda: nb.strongest_n_ties(G, n=2)),
        ("global_sparsification(s=0.5)", lambda: nb.global_sparsification(G, s=0.5)),
        ("primary_linkage_analysis", lambda: nb.primary_linkage_analysis(G)),
        ("edge_betweenness_filter(s=0.5)", lambda: nb.edge_betweenness_filter(G, s=0.5)),
        ("node_degree_filter(min_degree=2)", lambda: nb.node_degree_filter(G, min_degree=2)),
        ("high_salience_skeleton", lambda: nb.high_salience_skeleton(G)),
        ("metric_backbone", lambda: nb.metric_backbone(G)),
        ("ultrametric_backbone", lambda: nb.ultrametric_backbone(G)),
        ("doubly_stochastic_filter", lambda: nb.doubly_stochastic_filter(G)),
        ("h_backbone", lambda: nb.h_backbone(G)),
        ("modularity_backbone", lambda: nb.modularity_backbone(G)),
        ("planar_maximally_filtered_graph", lambda: nb.planar_maximally_filtered_graph(G)),
        ("maximum_spanning_tree_backbone", lambda: nb.maximum_spanning_tree_backbone(G)),
        ("neighborhood_overlap", lambda: nb.neighborhood_overlap(G)),
        ("jaccard_backbone", lambda: nb.jaccard_backbone(G)),
        ("dice_backbone", lambda: nb.dice_backbone(G)),
        ("cosine_backbone", lambda: nb.cosine_backbone(G)),
        ("hub_promoted_index", lambda: nb.hub_promoted_index(G)),
        ("hub_depressed_index", lambda: nb.hub_depressed_index(G)),
        ("lhn_local_index", lambda: nb.lhn_local_index(G)),
        ("preferential_attachment_score", lambda: nb.preferential_attachment_score(G)),
        ("adamic_adar_index", lambda: nb.adamic_adar_index(G)),
        ("resource_allocation_index", lambda: nb.resource_allocation_index(G)),
        ("graph_distance_proximity", lambda: nb.graph_distance_proximity(G)),
        ("local_path_index", lambda: nb.local_path_index(G)),
        ("glab_filter", lambda: nb.glab_filter(G)),
        ("sparsify", lambda: nb.sparsify(G_unweighted)),
        ("lspar", lambda: nb.lspar(G_unweighted)),
        ("local_degree", lambda: nb.local_degree(G_unweighted)),
    ]

    print("original_edges", G.number_of_edges())
    for name, fn in methods:
        H = fn()
        print(name, H.number_of_edges())
