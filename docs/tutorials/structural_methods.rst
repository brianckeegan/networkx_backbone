Structural Backbone Methods
===========================

This tutorial demonstrates the structural methods, which use topological
properties of the network to identify important edges.

Setup
-----

::

    import networkx as nx
    import networkx_backbone as nb

    G = nx.les_miserables_graph()

    print(f"Original: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

Simple filters
--------------

**Global threshold**: Keep edges whose weight meets a minimum::

    scored = nb.global_threshold_filter(G, threshold=1.0)
    backbone = nb.boolean_filter(scored, "global_threshold_keep")
    print(f"Global threshold: {backbone.number_of_edges()} edges")

**Strongest N ties**: Keep each node's N strongest edges::

    scored = nb.strongest_n_ties(G, n=2)
    backbone = nb.boolean_filter(scored, "strongest_n_ties_keep")
    print(f"Strongest 2 ties: {backbone.number_of_edges()} edges")

**Global sparsification** (Satuluri et al., 2011): Keep the globally
strongest fraction of edges::

    scored = nb.global_sparsification(G, s=0.4)
    backbone = nb.boolean_filter(scored, "global_sparsification_keep")
    print(f"Global sparsification (40%): {backbone.number_of_edges()} edges")

Linkage and centrality filters
------------------------------

**Primary linkage analysis** (Nystuen & Dacey, 1961): Keep each node's
strongest outgoing edge::

    scored = nb.primary_linkage_analysis(G)
    backbone = nb.boolean_filter(scored, "primary_linkage_keep")
    print(f"Primary linkage: {backbone.number_of_edges()} edges")

**Edge betweenness filter** (Girvan & Newman, 2002): Keep edges with the
highest edge-betweenness centrality::

    scored = nb.edge_betweenness_filter(G, s=0.3)
    backbone = nb.boolean_filter(scored, "edge_betweenness_keep")
    print(f"Edge betweenness (30%): {backbone.number_of_edges()} edges")

**Node degree filter**: Keep only nodes whose degree meets a threshold::

    scored = nb.node_degree_filter(G, min_degree=2)
    backbone = nb.boolean_filter(scored, "node_degree_keep")
    print(f"Node degree (>=2): {backbone.number_of_nodes()} nodes")

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

    scored = nb.metric_backbone(G)
    backbone = nb.boolean_filter(scored, "metric_keep")
    print(f"Metric backbone: {backbone.number_of_edges()} edges")
    print(f"Connected: {nx.is_connected(backbone)}")

**Ultrametric backbone** (Simas et al., 2021): Similar to metric backbone
but uses minimax paths instead of shortest paths::

    scored = nb.ultrametric_backbone(G)
    backbone = nb.boolean_filter(scored, "ultrametric_keep")
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
to identify important edges::

    scored = nb.h_backbone(G)
    backbone = nb.boolean_filter(scored, "h_backbone_keep")
    print(f"H-backbone: {backbone.number_of_edges()} edges")

Community-based
---------------

**Modularity backbone** (Rajeh et al., 2022): Computes a vitality score for
each node based on how much removing it changes the network's modularity.
Use the boolean keep flag to extract retained edges::

    scored = nb.modularity_backbone(G)
    backbone = nb.boolean_filter(scored, "modularity_keep")
    print(f"Modularity backbone: {backbone.number_of_nodes()} nodes")

Constrained methods
-------------------

**Planar maximally filtered graph** (Tumminello et al., 2005): Greedily adds
edges from heaviest to lightest while maintaining planarity::

    scored = nb.planar_maximally_filtered_graph(G)
    backbone = nb.boolean_filter(scored, "pmfg_keep")
    print(f"PMFG: {backbone.number_of_edges()} edges")

**Maximum spanning tree**: Wrapper around NetworkX's MST algorithm.
Always produces a connected tree with exactly N-1 edges::

    scored = nb.maximum_spanning_tree_backbone(G)
    backbone = nb.boolean_filter(scored, "mst_keep")
    print(f"MST: {backbone.number_of_edges()} edges")
    print(f"Connected: {nx.is_connected(backbone)}")

Comparing structural methods
-----------------------------

::

    backbones = {
        "metric": nb.boolean_filter(nb.metric_backbone(G), "metric_keep"),
        "ultrametric": nb.boolean_filter(nb.ultrametric_backbone(G), "ultrametric_keep"),
        "h_backbone": nb.boolean_filter(nb.h_backbone(G), "h_backbone_keep"),
        "global_sparsification": nb.boolean_filter(
            nb.global_sparsification(G, s=0.4), "global_sparsification_keep"
        ),
        "edge_betweenness": nb.boolean_filter(
            nb.edge_betweenness_filter(G, s=0.3), "edge_betweenness_keep"
        ),
        "pmfg": nb.boolean_filter(nb.planar_maximally_filtered_graph(G), "pmfg_keep"),
        "mst": nb.boolean_filter(nb.maximum_spanning_tree_backbone(G), "mst_keep"),
    }

    results = nb.compare_backbones(G, backbones)
    for name, metrics in results.items():
        ef = metrics["edge_fraction"]
        nf = metrics["node_fraction"]
        print(f"{name:15s}: edges={ef:.1%}, nodes={nf:.1%}")
