Structural Backbone Methods
===========================

This tutorial demonstrates the structural methods, which use topological
properties of the network to identify important edges.

Setup
-----

::

    import networkx as nx
    import networkx_backbone as nb

    G = nx.karate_club_graph()
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0

    print(f"Original: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

Simple filters
--------------

**Global threshold**: Keep edges whose weight meets a minimum::

    backbone = nb.global_threshold_filter(G, threshold=1.0)
    print(f"Global threshold: {backbone.number_of_edges()} edges")

**Strongest N ties**: Keep each node's N strongest edges::

    backbone = nb.strongest_n_ties(G, n=2)
    print(f"Strongest 2 ties: {backbone.number_of_edges()} edges")

Shortest-path methods
---------------------

**High salience skeleton** (Grady et al., 2012): Each edge receives a
salience score -- the fraction of shortest-path trees that include it.
Edges with high salience are consistently important for communication::

    H = nb.high_salience_skeleton(G)

    # Filter to keep edges with salience above 0.5
    backbone = nb.threshold_filter(H, "salience", 0.5, mode="above")
    print(f"High salience (>0.5): {backbone.number_of_edges()} edges")

**Metric backbone** (Simas et al., 2021): Keeps only edges that lie on
shortest paths (using the inverse of weight as distance). Always preserves
connectivity::

    backbone = nb.metric_backbone(G)
    print(f"Metric backbone: {backbone.number_of_edges()} edges")
    print(f"Connected: {nx.is_connected(backbone)}")

**Ultrametric backbone** (Simas et al., 2021): Similar to metric backbone
but uses minimax paths instead of shortest paths::

    backbone = nb.ultrametric_backbone(G)
    print(f"Ultrametric backbone: {backbone.number_of_edges()} edges")

Normalization
-------------

**Doubly stochastic filter** (Slater, 2009): Normalizes the weight matrix
so that each row and column sums to 1 using Sinkhorn-Knopp iteration. Edges
with high normalized weight are important::

    H = nb.doubly_stochastic_filter(G)

    # Filter: keep edges with high doubly-stochastic weight
    backbone = nb.threshold_filter(H, "ds_weight", 0.1, mode="above")
    print(f"Doubly stochastic (>0.1): {backbone.number_of_edges()} edges")

Index-based
-----------

**H-backbone** (Zhang et al., 2018): Uses an h-index inspired criterion
to identify important edges. Returns the backbone directly::

    backbone = nb.h_backbone(G)
    print(f"H-backbone: {backbone.number_of_edges()} edges")

Community-based
---------------

**Modularity backbone** (Rajeh et al., 2022): Computes a vitality score for
each node based on how much removing it changes the network's modularity.
Use :func:`~networkx_backbone.threshold_filter` with ``filter_on="nodes"``::

    H = nb.modularity_backbone(G)

    # Filter nodes by vitality (keep nodes with high vitality)
    backbone = nb.threshold_filter(H, "vitality", 0.0, mode="above", filter_on="nodes")
    print(f"Modularity backbone: {backbone.number_of_nodes()} nodes")

Constrained methods
-------------------

**Planar maximally filtered graph** (Tumminello et al., 2005): Greedily adds
edges from heaviest to lightest while maintaining planarity::

    backbone = nb.planar_maximally_filtered_graph(G)
    print(f"PMFG: {backbone.number_of_edges()} edges")

**Maximum spanning tree**: Wrapper around NetworkX's MST algorithm.
Always produces a connected tree with exactly N-1 edges::

    backbone = nb.maximum_spanning_tree_backbone(G)
    print(f"MST: {backbone.number_of_edges()} edges")
    print(f"Connected: {nx.is_connected(backbone)}")

Comparing structural methods
-----------------------------

::

    backbones = {
        "metric": nb.metric_backbone(G),
        "ultrametric": nb.ultrametric_backbone(G),
        "h_backbone": nb.h_backbone(G),
        "pmfg": nb.planar_maximally_filtered_graph(G),
        "mst": nb.maximum_spanning_tree_backbone(G),
    }

    results = nb.compare_backbones(G, backbones)
    for name, metrics in results.items():
        ef = metrics["edge_fraction"]
        nf = metrics["node_fraction"]
        print(f"{name:15s}: edges={ef:.1%}, nodes={nf:.1%}")
