API Reference
=============

Complete API documentation for backbone functions across 9 modules.
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
     - 11
     - Projection backbones, fixed null models, and high-level wrappers
   * - :doc:`unweighted`
     - 3
     - Sparsification for unweighted graphs (LSpar, local degree)
   * - :doc:`filters`
     - 5
     - Post-hoc filtering utilities and graph-conversion support
   * - :doc:`measures`
     - 7
     - Evaluation measures for comparing backbones
   * - :doc:`visualization`
     - 3
     - Graph comparison and plotting helpers for backbone differences

.. toctree::
   :maxdepth: 1
   :hidden:

   statistical
   structural
   proximity
   hybrid
   bipartite
   unweighted
   filters
   measures
   visualization
