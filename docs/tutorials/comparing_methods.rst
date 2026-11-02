Comparing Multiple Methods
==========================

This tutorial demonstrates how to systematically compare multiple backbone
extraction methods on the same network.

.. seealso::

   See the narrative tutorial: :ref:`Start here, section 4: comparing
   methods <hb-compare>`, a worked case study of why backbones from
   different methods differ and how to read the comparison.

Setup
-----

::

    import networkx as nx
    import nx_backbone as nb

    G = nx.les_miserables_graph()

    print(f"Original: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

Creating multiple backbones
----------------------------

Apply several different backbone methods to the same graph::

    backbones = {
        "disparity": nb.threshold_filter(
            nb.disparity_filter(G), "disparity_pvalue", 0.05
        ),
        "noise_corrected": nb.threshold_filter(
            nb.noise_corrected_filter(G), "nc_pvalue", 0.05
        ),
        "metric": nb.boolean_filter(nb.metric_backbone(G), "metric_keep"),
        "mst": nb.boolean_filter(nb.maximum_spanning_tree_backbone(G), "mst_keep"),
        "strongest_3": nb.boolean_filter(nb.strongest_n_ties(G, n=3), "strongest_n_ties_keep"),
    }

Using compare_backbones
-----------------------

The :func:`~nx_backbone.compare_backbones` function evaluates each
backbone using a set of measures. By default, it computes
:func:`~nx_backbone.node_fraction`,
:func:`~nx_backbone.edge_fraction`, and
:func:`~nx_backbone.weight_fraction`::

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

Agreement between backbones
---------------------------

The measures above summarize each backbone on its own. To see whether two
methods keep the *same* edges, compare their edge sets, for example with the
Jaccard index: the edges both keep over the edges either keeps::

    edges = {name: {frozenset(e) for e in B.edges()} for name, B in backbones.items()}
    a, b = edges["metric"], edges["mst"]
    print(f"Jaccard(metric, mst) = {len(a & b) / len(a | b):.2f}")

.. page-figure:: comparing_methods/agreement

Consensus backbone
------------------

The :func:`~nx_backbone.consensus_backbone` function finds edges that
appear in all provided backbones. This identifies the most robust edges that
multiple methods agree are important::

    b1 = nb.threshold_filter(nb.disparity_filter(G), "disparity_pvalue", 0.05)
    b2 = nb.boolean_filter(nb.metric_backbone(G), "metric_keep")
    b3 = nb.boolean_filter(nb.maximum_spanning_tree_backbone(G), "mst_keep")

    consensus = nb.consensus_backbone(b1, b2, b3)
    print(f"Consensus backbone: {consensus.number_of_edges()} edges")
    print(f"Edge fraction: {nb.edge_fraction(G, consensus):.1%}")

.. page-figure:: comparing_methods/consensus

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
