API Reference
=============

Complete API documentation for backbone functions across 14 modules
(the hypergraph page covers both ``hypergraph`` and ``hypergraph_io``).
API examples are standardized on ``nx.les_miserables_graph()`` for
non-bipartite methods and ``nx.davis_southern_women_graph()`` for
bipartite methods.
Each module page includes Sphinx Gallery links showing score-then-filter
visualizations for the listed functions.
Each function entry includes complexity information (time/space), with
an aggregate summary in :doc:`../user_guide/complexity`.

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Module
     - Functions
     - Description
   * - :doc:`statistical`
     - 9
     - Hypothesis-testing methods plus short alias names
   * - :doc:`structural`
     - 14
     - Topology-based methods (threshold, sparsification, linkage, centrality, spanning tree, salience, metric, planarity)
   * - :doc:`proximity`
     - 12
     - Neighborhood-similarity edge scoring (Jaccard, Dice, cosine, Adamic-Adar)
   * - :doc:`hybrid`
     - 1
     - Combined statistical/structural methods (GLAB)
   * - :doc:`bipartite`
     - 16
     - Projection backbones, fixed null models, and high-level wrappers
   * - :doc:`hypergraph`
     - 23
     - MDL backbone, compression ratio, intersection / s-line graph, inclusion (toplex) reduction, order filter, s-components, and statistically validated hypergraphs / cores
   * - :doc:`unweighted`
     - 3
     - Sparsification for unweighted graphs (LSpar, local degree)
   * - :doc:`filters`
     - 6
     - Post-hoc filtering utilities, multiple-testing correction, and graph-conversion support
   * - :doc:`measures`
     - 7
     - Evaluation measures for comparing backbones
   * - :doc:`visualization`
     - 3
     - Graph comparison and plotting helpers for backbone differences
   * - :doc:`describe`
     - 1
     - Methods-section text and citations for a backbone (``describe_backbone``)
   * - :doc:`convert`
     - 1
     - NumPy/SciPy adjacency matrices and pandas edge lists to NetworkX graphs (``to_graph``)
   * - :doc:`temporal`
     - 10
     - Temporal networks as snapshots, events, or spans: form detection, conversions, binning, and pandas event tables

.. toctree::
   :maxdepth: 1
   :hidden:

   statistical
   structural
   proximity
   hybrid
   bipartite
   hypergraph
   unweighted
   filters
   measures
   visualization
   describe
   convert
   temporal
