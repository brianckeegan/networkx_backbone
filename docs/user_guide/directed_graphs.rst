Directed Graphs
===============

Not all backbone methods support directed graphs. This guide explains which
methods work with directed graphs and how their behavior differs.

Supported methods
-----------------

The following methods support both directed and undirected graphs:

**Statistical methods** (all 5):
    :func:`~networkx_backbone.disparity_filter`,
    :func:`~networkx_backbone.noise_corrected_filter`,
    :func:`~networkx_backbone.marginal_likelihood_filter`,
    :func:`~networkx_backbone.ecm_filter`,
    :func:`~networkx_backbone.lans_filter`

**Structural (2 of 10)**:
    :func:`~networkx_backbone.global_threshold_filter`,
    :func:`~networkx_backbone.strongest_n_ties`

**Proximity methods** (all 12):
    All proximity scoring functions work with directed graphs.

**Filters** (all 4):
    :func:`~networkx_backbone.threshold_filter`,
    :func:`~networkx_backbone.fraction_filter`,
    :func:`~networkx_backbone.boolean_filter`,
    :func:`~networkx_backbone.consensus_backbone`

**Measures** (all 7):
    All evaluation functions work with directed graphs.

Undirected-only methods
-----------------------

The following methods raise ``NetworkXNotImplemented`` when given a directed
graph:

- :func:`~networkx_backbone.high_salience_skeleton`
- :func:`~networkx_backbone.metric_backbone`
- :func:`~networkx_backbone.ultrametric_backbone`
- :func:`~networkx_backbone.doubly_stochastic_filter`
- :func:`~networkx_backbone.h_backbone`
- :func:`~networkx_backbone.modularity_backbone`
- :func:`~networkx_backbone.planar_maximally_filtered_graph`
- :func:`~networkx_backbone.maximum_spanning_tree_backbone`
- :func:`~networkx_backbone.glab_filter`
- :func:`~networkx_backbone.sparsify`
- :func:`~networkx_backbone.lspar`
- :func:`~networkx_backbone.local_degree`

Behavior differences
--------------------

For statistical methods on directed graphs, the p-value computation uses the
out-degree and out-strength of the source node. For proximity methods on
directed graphs, neighborhoods are defined using successors.

Example
-------

::

    import networkx as nx
    import networkx_backbone as nb

    # Create a weighted directed graph
    G = nx.DiGraph()
    G.add_weighted_edges_from([
        (0, 1, 5.0), (0, 2, 3.0), (0, 3, 1.0),
        (1, 0, 2.0), (1, 2, 4.0),
        (2, 3, 6.0),
    ])

    # Statistical methods work on directed graphs
    H = nb.disparity_filter(G)
    backbone = nb.threshold_filter(H, "disparity_pvalue", 0.05)
