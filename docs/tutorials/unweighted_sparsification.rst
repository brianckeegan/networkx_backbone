Unweighted Graph Sparsification
================================

This tutorial demonstrates methods for sparsifying unweighted graphs using the
:mod:`~networkx_backbone.unweighted` module.

Setup
-----

::

    import networkx as nx
    import networkx_backbone as nb

    G_full = nx.les_miserables_graph()
    G = nx.Graph()
    G.add_nodes_from(G_full.nodes(data=True))
    G.add_edges_from(G_full.edges())
    print(f"Original: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

The sparsify pipeline
---------------------

The :func:`~networkx_backbone.sparsify` function implements a four-step pipeline:

1. **Score**: Assign a score to each edge based on a chosen metric
2. **Normalize**: Optionally rank-transform scores within each node's neighborhood
3. **Filter**: Select edges based on the normalized scores
4. **Connect** (optional): Add a union of maximum spanning trees to ensure connectivity

Edge scoring methods
^^^^^^^^^^^^^^^^^^^^

The ``escore`` parameter controls how edges are scored:

- ``"jaccard"``: Jaccard similarity of endpoint neighborhoods (default)
- ``"degree"``: Sum of endpoint degrees
- ``"triangles"``: Number of triangles containing the edge
- ``"quadrangles"``: Number of quadrangles containing the edge
- ``"random"``: Random scores (baseline)

::

    # Jaccard scoring (default)
    backbone_jaccard = nb.sparsify(G, escore="jaccard", s=0.5)
    print(f"Jaccard: {backbone_jaccard.number_of_edges()} edges")

    # Degree scoring
    backbone_degree = nb.sparsify(G, escore="degree", s=0.5)
    print(f"Degree: {backbone_degree.number_of_edges()} edges")

    # Triangle scoring
    backbone_tri = nb.sparsify(G, escore="triangles", s=0.5)
    print(f"Triangles: {backbone_tri.number_of_edges()} edges")

Normalization
^^^^^^^^^^^^^

The ``normalize`` parameter controls normalization:

- ``"rank"``: Rank-transform scores within each node's neighborhood (default)
- ``None``: Skip normalization

::

    backbone_ranked = nb.sparsify(G, normalize="rank", s=0.5)
    backbone_raw = nb.sparsify(G, normalize=None, filter="threshold", s=0.5)

Filtering
^^^^^^^^^

The ``filter`` parameter controls how edges are selected:

- ``"degree"``: Keep the top ``ceil(d^s)`` edges per node, where ``d`` is the
  node's degree and ``s`` controls sparsity (default)
- ``"threshold"``: Keep edges with score >= ``s``

The ``s`` parameter controls the level of sparsification. For degree filtering,
lower values produce sparser backbones::

    # More aggressive (sparser)
    sparse = nb.sparsify(G, s=0.3)
    print(f"s=0.3: {sparse.number_of_edges()} edges")

    # Less aggressive (denser)
    dense = nb.sparsify(G, s=0.7)
    print(f"s=0.7: {dense.number_of_edges()} edges")

UMST connectivity guarantee
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``umst=True`` to add a union of maximum spanning trees, guaranteeing
that the backbone is connected (if the original graph is connected)::

    backbone = nb.sparsify(G, s=0.3, umst=True)
    print(f"Connected: {nx.is_connected(backbone)}")

Convenience wrappers
--------------------

Two pre-configured wrappers are available:

**LSpar** (Satuluri et al., 2011): Uses Jaccard scoring, rank normalization,
and degree filtering::

    backbone = nb.lspar(G, s=0.5)
    print(f"LSpar: {backbone.number_of_edges()} edges")

**Local degree** (Hamann et al., 2016): Uses degree scoring, rank
normalization, and degree filtering::

    backbone = nb.local_degree(G, s=0.3)
    print(f"Local degree: {backbone.number_of_edges()} edges")

Comparing unweighted methods
-----------------------------

::

    backbones = {
        "lspar_0.3": nb.lspar(G, s=0.3),
        "lspar_0.5": nb.lspar(G, s=0.5),
        "local_degree_0.3": nb.local_degree(G, s=0.3),
        "local_degree_0.5": nb.local_degree(G, s=0.5),
    }

    for name, bb in backbones.items():
        ef = nb.edge_fraction(G, bb)
        reach = nb.reachability(bb)
        print(f"{name:20s}: {bb.number_of_edges()} edges ({ef:.1%}), reachability={reach:.1%}")
