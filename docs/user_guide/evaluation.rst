Evaluating Backbones
====================

The :mod:`~networkx_backbone.measures` module provides metrics for evaluating
how well a backbone preserves the properties of the original network.

Basic fraction metrics
----------------------

These functions measure what fraction of the original graph is retained in
the backbone.

:func:`~networkx_backbone.node_fraction`
    Fraction of nodes that have at least one edge in the backbone. Isolated
    nodes are excluded from the count in both the original and backbone::

        nb.node_fraction(G, backbone)  # e.g. 0.85

:func:`~networkx_backbone.edge_fraction`
    Fraction of original edges retained in the backbone::

        nb.edge_fraction(G, backbone)  # e.g. 0.30

:func:`~networkx_backbone.weight_fraction`
    Fraction of total edge weight retained::

        nb.weight_fraction(G, backbone)  # e.g. 0.65

Connectivity
------------

:func:`~networkx_backbone.reachability`
    Fraction of node pairs that can communicate through the backbone.
    Returns 1.0 for a connected backbone, 0.0 for a completely disconnected
    one::

        nb.reachability(backbone)  # e.g. 1.0

Distribution preservation
-------------------------

These functions use the Kolmogorov-Smirnov statistic to compare distributions
between the original graph and the backbone. Lower values indicate better
preservation.

:func:`~networkx_backbone.ks_degree`
    KS statistic between the degree distributions::

        nb.ks_degree(G, backbone)  # e.g. 0.12

:func:`~networkx_backbone.ks_weight`
    KS statistic between the edge weight distributions::

        nb.ks_weight(G, backbone)  # e.g. 0.08

Comparing multiple backbones
----------------------------

:func:`~networkx_backbone.compare_backbones` provides a systematic way to
evaluate multiple backbone methods on the same graph::

    import networkx as nx
    import networkx_backbone as nb

    G = nx.les_miserables_graph()

    backbones = {
        "disparity": nb.threshold_filter(
            nb.disparity_filter(G), "disparity_pvalue", 0.05
        ),
        "metric": nb.metric_backbone(G),
        "mst": nb.maximum_spanning_tree_backbone(G),
    }

    results = nb.compare_backbones(G, backbones)

The ``results`` dictionary maps each backbone name to a dictionary of measure
names and their values.

You can also pass custom measures as callables::

    results = nb.compare_backbones(
        G,
        backbones,
        measures=[nb.edge_fraction, nb.reachability, nb.ks_degree],
    )
