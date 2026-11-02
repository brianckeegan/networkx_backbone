Directed Graphs
===============

Not all backbone methods support directed graphs. This guide explains which
methods work with directed graphs and how their behavior differs.

.. seealso:: :doc:`/auto_examples/directed_signed/plot_directed_signed`, a narrative tutorial on source and target tests, signed tails and ties.

Supported methods
-----------------

The following methods support both directed and undirected graphs:

**Statistical methods** (all 8):
    :func:`~nx_backbone.disparity_filter`,
    :func:`~nx_backbone.noise_corrected_filter`,
    :func:`~nx_backbone.marginal_likelihood_filter`,
    :func:`~nx_backbone.ecm_filter`,
    :func:`~nx_backbone.lans_filter`,
    :func:`~nx_backbone.multiple_linkage_analysis`,
    :func:`~nx_backbone.polya_filter`,
    :func:`~nx_backbone.gloss_filter`

**Structural (6 of 14)**:
    :func:`~nx_backbone.global_threshold_filter`,
    :func:`~nx_backbone.strongest_n_ties`,
    :func:`~nx_backbone.global_sparsification`,
    :func:`~nx_backbone.primary_linkage_analysis`,
    :func:`~nx_backbone.edge_betweenness_filter`,
    :func:`~nx_backbone.node_degree_filter`

**Hybrid methods**:
    :func:`~nx_backbone.glab_filter`

**Proximity methods** (all 12):
    All proximity scoring functions work with directed graphs.

**Filters** (all 4):
    :func:`~nx_backbone.threshold_filter`,
    :func:`~nx_backbone.fraction_filter`,
    :func:`~nx_backbone.boolean_filter`,
    :func:`~nx_backbone.consensus_backbone`

**Measures** (all 7):
    All evaluation functions work with directed graphs.

Undirected-only methods
-----------------------

The following methods raise ``NetworkXNotImplemented`` when given a directed
graph:

- :func:`~nx_backbone.high_salience_skeleton`
- :func:`~nx_backbone.metric_backbone`
- :func:`~nx_backbone.ultrametric_backbone`
- :func:`~nx_backbone.doubly_stochastic_filter`
- :func:`~nx_backbone.h_backbone`
- :func:`~nx_backbone.modularity_backbone`
- :func:`~nx_backbone.planar_maximally_filtered_graph`
- :func:`~nx_backbone.maximum_spanning_tree_backbone`
- :func:`~nx_backbone.sparsify`
- :func:`~nx_backbone.lspar`
- :func:`~nx_backbone.local_degree`

Behavior differences
--------------------

The statistical methods test a directed edge *u* → *v* against the
source's out-links and the target's in-links. The local filters
(:func:`~nx_backbone.disparity_filter`, :func:`~nx_backbone.lans_filter`,
:func:`~nx_backbone.polya_filter`) run one test from each side and keep the
smaller p-value: an edge can be significant from either end.
:func:`~nx_backbone.noise_corrected_filter`,
:func:`~nx_backbone.marginal_likelihood_filter`,
:func:`~nx_backbone.ecm_filter` and :func:`~nx_backbone.gloss_filter`
build one null model from the source's out-strength (and out-degree) and the
target's in-strength (and in-degree).
:func:`~nx_backbone.multiple_linkage_analysis` fits each node's out-links
only. For proximity methods on directed graphs, neighborhoods are defined
using successors.

.. page-figure:: directed_graphs/out-in-tests

:func:`~nx_backbone.glab_filter` follows Section 2.3 of Zhang et al.
(2014): each link *(i, j)* gets an *out-importance* from the shortest paths
that leave *i* through it (null model on the out-degree of *i*) and an
*in-importance* from the shortest paths that reach *j* through it (null
model on the in-degree of *j*); ``glab_pvalue`` is the smaller of the two.
On a symmetric directed graph this equals the undirected p-value.

Example
-------

::

    import networkx as nx
    import nx_backbone as nb

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
