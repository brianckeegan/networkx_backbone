Les Miserables Benchmark
========================

This benchmark uses NetworkX's built-in ``nx.les_miserables_graph()`` and
applies a strict **score-then-filter** workflow for every method:

1. Run a scoring method that returns the full graph with edge attributes.
2. Apply a filter (`threshold_filter`, `fraction_filter`, or `boolean_filter`).
3. Report edge counts only from the filtered graph.

If the filtered graph has the same edge count as the original graph, issue a
warning and re-test the method parameters.

Setup
-----

::

    import warnings
    import networkx as nx
    import networkx_backbone as nb

    G = nx.les_miserables_graph()
    G_unweighted = nx.Graph()
    G_unweighted.add_nodes_from(G.nodes(data=True))
    G_unweighted.add_edges_from(G.edges())

    ORIGINAL_EDGES = G.number_of_edges()  # 254

    def warn_if_no_reduction(name, filtered):
        if filtered.number_of_edges() == ORIGINAL_EDGES:
            warnings.warn(
                f"{name}: filtered graph has the same number of edges as "
                f"the original ({ORIGINAL_EDGES}). Re-test and validate.",
                UserWarning,
            )

Filtered Edge Counts
--------------------

All counts below are after explicit filtering. They are computed when the
documentation is built, from the same method definitions whose counts the test
suite pins (``tests/test_les_miserables.py``), so the table always matches the
installed version of the library. Methods marked *unweighted* run on
``G_unweighted``; every other method runs on the weighted graph ``G``.

.. les-miserables-table::

Reproducibility Script
----------------------

The table is generated from the case list below, kept in
``benchmarks/benchmarks/les_miserables.py`` in the source repository. Each
case names a scorer, its keyword arguments, and the filter applied to its
scores:

.. literalinclude:: ../../benchmarks/benchmarks/les_miserables.py
   :language: python
   :start-after: # -- cases
   :end-before: # -- end cases

Each case expands to one score-then-filter call, for example::

    H = nb.threshold_filter(nb.disparity_filter(G), "disparity_pvalue", 0.05, mode="below")
    H = nb.threshold_filter(nb.noise_corrected_filter(G), "nc_score", 2.0, mode="above")
    H = nb.boolean_filter(nb.multiple_linkage_analysis(G, alpha=0.05), "mla_keep")

From a checkout of the repository, the whole table can be reproduced with::

    import warnings

    from benchmarks.benchmarks.les_miserables import CASES, graph_for

    for case in CASES:
        G = graph_for(case)
        filtered = case.run(G)
        if filtered.number_of_edges() == G.number_of_edges():
            warnings.warn(f"{case.name}: no edge was removed", UserWarning)
        print(case.name, filtered.number_of_edges())
