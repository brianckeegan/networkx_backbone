Backends
========

NetworkX can hand an algorithm to a *backend*, a separately installed package
such as `nx-parallel <https://github.com/networkx/nx-parallel>`_ (multicore
CPU) or `nx-cugraph <https://github.com/rapidsai/nx-cugraph>`_ (NVIDIA GPU),
through its dispatch mechanism (``nx._dispatchable``). See NetworkX's
`backends guide <https://networkx.org/documentation/stable/reference/backends.html>`_
for how backends are installed and configured.

What is dispatchable
--------------------

Every function that takes a graph and returns a scored copy of it is
dispatchable, as NetworkX's own algorithms are:

- statistical scorers (``disparity_filter``, ``noise_corrected_filter``,
  ``marginal_likelihood_filter``, ``ecm_filter``, ``lans_filter``,
  ``multiple_linkage_analysis``, ``polya_filter``, ``gloss_filter``);
- every structural method (``global_threshold_filter``, ``metric_backbone``,
  ``h_backbone``, ``maximum_spanning_tree_backbone``, ...);
- proximity scores, ``glab_filter``, and the unweighted sparsifiers
  (``sparsify``, ``lspar``, ``local_degree``);
- ``multigraph_to_weighted``, ``threshold_filter``, ``fraction_filter``,
  ``boolean_filter``, and ``transform_weights``.

These functions accept the ``backend=`` keyword and follow
``nx.config.backend_priority``. They are registered under the dispatch name
``"backbone_" + name`` (``disparity_filter`` is ``"backbone_disparity_filter"``),
because NetworkX keeps one registry of names for all packages:
``adamic_adar_index`` and ``resource_allocation_index`` are already NetworkX
link-prediction functions, which nx-parallel implements.

The other functions are not dispatchable. The bipartite functions accept
incidence matrices as well as graphs, the hypergraph functions take lists of
hyperedges, the evaluation measures return numbers,
``consensus_backbone`` takes any number of graphs, and the temporal functions
take snapshots (several graphs) or a multigraph of events or spans.

Using a backend
---------------

No backend implements a ``backbone_*`` function yet, so
``backend="parallel"`` raises ``NotImplementedError``. ``backend="networkx"``
(the default) always works:

.. code-block:: python

    import networkx as nx
    import networkx_backbone as nb

    G = nx.les_miserables_graph()
    H = nb.metric_backbone(G, backend="networkx")

A backend can still speed up the NetworkX algorithms that the backbone
methods call, such as ``all_pairs_dijkstra_path_length`` (metric backbone)
and ``edge_betweenness_centrality`` (``edge_betweenness_filter``,
``h_backbone``). List the backend in ``nx.config.backend_priority``; the
backbone function runs its NetworkX code, and each NetworkX algorithm that
the backend implements runs on the backend:

.. code-block:: python

    # pip install nx-parallel
    with nx.config.backend_priority(algos=["parallel"]):
        H = nb.edge_betweenness_filter(G)

    # or for the whole session
    nx.config.backend_priority.algos = ["parallel"]

Setting the environment variable ``NETWORKX_BACKEND_PRIORITY=parallel``
before importing NetworkX does the same. nx-parallel's own settings, such as
the number of jobs, are under ``nx.config.backends.parallel``.

NetworkX caches the converted graph on the input graph
(``G.__networkx_cache__``), so a second call on the same graph skips the
conversion and warns that the cache is being used. The cache is only correct
if the graph has not been modified in place since; see
``nx.config.cache_converted_graphs`` and
``nx.config.warnings_to_ignore``.

Measured timings
----------------

``scripts/benchmark_backends.py`` times the shortest-path methods on a
Barabási-Albert graph with 1,000 nodes and 2,991 integer-weighted edges (the
seeded graph in ``benchmarks/benchmarks/_data.py``), with and without
nx-parallel 0.4 in ``backend_priority``. Best of 3 runs, Python 3.13, NetworkX
3.7, 4 CPUs, nx-parallel's default ``n_jobs=-1``:

===========================  ============  ============  =====  ========
Method                       NetworkX (s)  parallel (s)  Ratio  Max diff
===========================  ============  ============  =====  ========
``metric_backbone``          2.23          2.13          1.05   0.0e+00
``edge_betweenness_filter``  4.29          1.13          3.79   3.5e-18
``h_backbone``               1.82          0.54          3.38   1.4e-14
``high_salience_skeleton``   3.16          3.40          0.93   0.0e+00
``glab_filter``              3.50          3.84          0.91   0.0e+00
===========================  ============  ============  =====  ========

*Ratio* is the NetworkX time divided by the nx-parallel time. *Max diff* is
the largest difference between any score the two runs return. nx-parallel
adds up edge betweenness in a different order, so those scores can differ in
the last bits; on this graph every ``*_keep`` flag was the same. These
timings are from one machine. Measure on your own graphs and hardware before
relying on them.

``high_salience_skeleton`` and ``glab_filter`` run one single-source search
per node (``dijkstra_predecessor_and_distance``), which nx-parallel 0.4 does
not implement, so they gain nothing. ``metric_backbone`` calls
``all_pairs_dijkstra_path_length``, which nx-parallel implements, but it was
not faster here.

Testing with a backend
----------------------

The test suite can run under NetworkX's dispatch test harness, as NetworkX's
own tests do. With ``nx_loopback``, the test backend that ships with
NetworkX, every dispatchable call converts its input graphs, runs, and
checks that the result equals the plain NetworkX result::

    NETWORKX_TEST_BACKEND=nx_loopback pytest --doctest-modules networkx_backbone tests

Add ``NETWORKX_FALLBACK_TO_NX=1`` to run with NetworkX the functions a
backend does not implement, instead of marking their tests as expected
failures.

The narrative guide :doc:`/auto_examples/r_and_scale/plot_scaling_up` puts
these timings next to run times measured at build time and the other ways to
handle large networks.
