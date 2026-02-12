Proximity-Based Edge Scoring
============================

This tutorial demonstrates the proximity methods, which score edges based on
how similar the neighborhoods of their endpoints are. High-scoring edges are
structurally embedded within communities, while low-scoring edges tend to be
bridges.

Setup
-----

::

    import networkx as nx
    import networkx_backbone as nb

    G = nx.karate_club_graph()
    print(f"Original: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

Local proximity methods
-----------------------

These methods use only the immediate neighborhoods of the edge endpoints.

**Jaccard coefficient**: Ratio of common neighbors to total neighbors::

    H = nb.jaccard_backbone(G)
    # Edges now have a "jaccard" attribute in [0, 1]

**Dice coefficient**: Similar to Jaccard but uses average degree::

    H = nb.dice_backbone(G)

**Cosine similarity**: Geometric mean normalization::

    H = nb.cosine_backbone(G)

**Hub promoted / depressed indices**: Normalize by the minimum or maximum
degree of the endpoints::

    H_hpi = nb.hub_promoted_index(G)
    H_hdi = nb.hub_depressed_index(G)

**Leicht-Holme-Newman index**: Normalize by the product of degrees::

    H = nb.lhn_local_index(G)

**Preferential attachment score**: Product of endpoint degrees -- measures
how expected an edge is under a random attachment model::

    H = nb.preferential_attachment_score(G)

**Adamic-Adar index**: Weights common neighbors by the inverse log of
their degree, giving more weight to rare shared neighbors::

    H = nb.adamic_adar_index(G)

**Resource allocation index**: Similar to Adamic-Adar but uses inverse degree
instead of inverse log-degree::

    H = nb.resource_allocation_index(G)

Quasi-local methods
-------------------

These methods consider paths beyond immediate neighborhoods.

**Graph distance proximity**: Reciprocal of shortest-path distance::

    H = nb.graph_distance_proximity(G)

**Local path index**: Considers paths of length 2 and 3 using adjacency
matrix powers::

    H = nb.local_path_index(G, epsilon=0.01)

Filtering by proximity
-----------------------

Use :func:`~networkx_backbone.fraction_filter` to keep the most structurally
embedded edges. Since higher proximity scores indicate more embedded edges,
use ``ascending=False``::

    H = nb.jaccard_backbone(G)

    # Keep the top 30% most embedded edges
    backbone = nb.fraction_filter(H, "jaccard", 0.3, ascending=False)
    print(f"Backbone: {backbone.number_of_edges()} edges")
    print(f"Edge fraction: {nb.edge_fraction(G, backbone):.1%}")

Comparing proximity methods
----------------------------

::

    methods = {
        "jaccard": ("jaccard", nb.jaccard_backbone(G)),
        "dice": ("dice", nb.dice_backbone(G)),
        "cosine": ("cosine", nb.cosine_backbone(G)),
        "adamic_adar": ("adamic_adar", nb.adamic_adar_index(G)),
        "resource_alloc": ("resource_allocation", nb.resource_allocation_index(G)),
    }

    for name, (attr, H) in methods.items():
        backbone = nb.fraction_filter(H, attr, 0.3, ascending=False)
        ef = nb.edge_fraction(G, backbone)
        print(f"{name:20s}: {backbone.number_of_edges()} edges ({ef:.1%})")
