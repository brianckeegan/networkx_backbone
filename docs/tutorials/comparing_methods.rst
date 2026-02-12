Comparing Multiple Methods
==========================

This tutorial demonstrates how to systematically compare multiple backbone
extraction methods on the same network.

Setup
-----

::

    import networkx as nx
    import networkx_backbone as nb

    G = nx.karate_club_graph()
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0

    print(f"Original: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

Creating multiple backbones
----------------------------

Apply several different backbone methods to the same graph::

    backbones = {
        "disparity": nb.threshold_filter(
            nb.disparity_filter(G), "disparity_pvalue", 0.05
        ),
        "noise_corrected": nb.threshold_filter(
            nb.noise_corrected_filter(G), "nc_score", 2.0, mode="above"
        ),
        "metric": nb.metric_backbone(G),
        "mst": nb.maximum_spanning_tree_backbone(G),
        "strongest_3": nb.strongest_n_ties(G, n=3),
    }

Using compare_backbones
-----------------------

The :func:`~networkx_backbone.compare_backbones` function evaluates each
backbone using a set of measures. By default, it computes
:func:`~networkx_backbone.node_fraction`,
:func:`~networkx_backbone.edge_fraction`, and
:func:`~networkx_backbone.weight_fraction`::

    results = nb.compare_backbones(G, backbones)

    for name, metrics in results.items():
        print(f"{name}:")
        for measure, value in metrics.items():
            print(f"  {measure}: {value:.3f}")

Custom measures
---------------

Pass a list of callables to evaluate custom measures. Each callable should
accept the original graph and the backbone as arguments::

    results = nb.compare_backbones(
        G,
        backbones,
        measures=[nb.edge_fraction, nb.node_fraction, nb.reachability],
    )

    # Print as a table
    print(f"{'Method':25s} {'Edges':>8s} {'Nodes':>8s} {'Reach':>8s}")
    print("-" * 51)
    for name, metrics in results.items():
        ef = metrics["edge_fraction"]
        nf = metrics["node_fraction"]
        r = metrics["reachability"]
        print(f"{name:25s} {ef:8.1%} {nf:8.1%} {r:8.1%}")

Consensus backbone
------------------

The :func:`~networkx_backbone.consensus_backbone` function finds edges that
appear in all provided backbones. This identifies the most robust edges that
multiple methods agree are important::

    b1 = nb.threshold_filter(nb.disparity_filter(G), "disparity_pvalue", 0.05)
    b2 = nb.metric_backbone(G)
    b3 = nb.maximum_spanning_tree_backbone(G)

    consensus = nb.consensus_backbone(b1, b2, b3)
    print(f"Consensus backbone: {consensus.number_of_edges()} edges")
    print(f"Edge fraction: {nb.edge_fraction(G, consensus):.1%}")

Interpreting results
--------------------

When comparing backbones, consider:

- **Edge fraction**: How aggressively the method sparsifies. Lower values
  indicate more aggressive filtering.
- **Node fraction**: Whether nodes become isolated. Ideally close to 1.0.
- **Weight fraction**: How much of the total weight is preserved. Important
  for weighted analysis.
- **Reachability**: Whether the backbone preserves connectivity. Critical
  for path-based analysis.
- **KS statistics**: How well degree and weight distributions are preserved.
  Lower is better.

Different applications may prioritize different metrics. For visualization,
aggressive sparsification (low edge fraction) may be desirable. For
communication network analysis, high reachability is essential.
