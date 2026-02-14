Quick Start
===========

This guide walks through the basic workflow of extracting a backbone from
a network using ``networkx-backbone``.

Import
------

The recommended import convention is::

    import networkx as nx
    import networkx_backbone as nb

Step 1: Create or load a weighted graph
---------------------------------------

Most backbone methods operate on weighted graphs. Here we use the built-in
Les Miserables co-appearance network::

    G = nx.les_miserables_graph()

Step 2: Apply a backbone method
-------------------------------

Apply the disparity filter to compute a p-value for each edge. This returns
a copy of the graph with an added ``"disparity_pvalue"`` edge attribute::

    H = nb.disparity_filter(G)

Step 3: Filter to extract the backbone
---------------------------------------

Use :func:`~networkx_backbone.threshold_filter` to keep only edges whose
p-value is below a significance threshold::

    backbone = nb.threshold_filter(H, "disparity_pvalue", 0.05)

Step 4: Evaluate the backbone
------------------------------

Use the :mod:`~networkx_backbone.measures` module to compare the backbone
to the original graph::

    print(f"Edges kept: {nb.edge_fraction(G, backbone):.1%}")
    print(f"Nodes kept: {nb.node_fraction(G, backbone):.1%}")

Other approaches
----------------

**Proximity-based scoring**: Score edges by how structurally embedded they are
using neighborhood similarity, then keep the top fraction::

    H = nb.jaccard_backbone(G)
    backbone = nb.fraction_filter(H, "jaccard", 0.2, ascending=False)

**Structural backbone**: Extract the metric backbone, which keeps only edges
that lie on shortest paths::

    backbone = nb.metric_backbone(G)

**Bipartite backbone**: Score a bipartite projection, then filter by p-value::

    B = nx.davis_southern_women_graph()
    women_nodes = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    H = nb.sdsm(B, agent_nodes=women_nodes, projection="hyper")
    backbone = nb.threshold_filter(H, "sdsm_pvalue", 0.05, mode="below")

**Comparing methods**: Systematically compare multiple backbones::

    backbones = {
        "disparity": nb.threshold_filter(nb.disparity_filter(G), "disparity_pvalue", 0.05),
        "mst": nb.maximum_spanning_tree_backbone(G),
    }
    results = nb.compare_backbones(G, backbones)

Next steps
----------

- :doc:`concepts` -- Learn about the different categories of backbone methods
- :doc:`user_guide/index` -- Detailed guides on workflows, method selection, and evaluation
- :doc:`tutorials/index` -- Step-by-step tutorials for each method category
- :doc:`api/index` -- Complete API reference for all 65 functions
