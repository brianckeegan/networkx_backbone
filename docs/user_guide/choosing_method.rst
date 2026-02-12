Choosing a Method
=================

With 47 functions available, selecting the right backbone method depends on
the properties of your network and the goals of your analysis.

Decision guide
--------------

**Is your graph bipartite?**
    Use :func:`~networkx_backbone.sdsm` (analytical) or
    :func:`~networkx_backbone.fdsm` (Monte Carlo) to extract significant
    edges from a bipartite projection.

**Is your graph unweighted?**
    Use the :mod:`~networkx_backbone.unweighted` module:
    :func:`~networkx_backbone.sparsify` (generic framework),
    :func:`~networkx_backbone.lspar` (local sparsification), or
    :func:`~networkx_backbone.local_degree` (degree-based).

**Is your graph weighted?**
    Choose based on your analysis goals:

    - **Statistical significance**: Use methods from
      :mod:`~networkx_backbone.statistical` to test whether edge weights are
      significant under a null model. These produce p-values that can be
      filtered at a chosen significance level.

    - **Topological structure**: Use methods from
      :mod:`~networkx_backbone.structural` to extract edges that are important
      for the network's topology (shortest paths, spanning trees, planarity).

    - **Neighborhood similarity**: Use methods from
      :mod:`~networkx_backbone.proximity` to score edges based on how
      similar the neighborhoods of their endpoints are.

    - **Combined approach**: Use :func:`~networkx_backbone.glab_filter` from
      :mod:`~networkx_backbone.hybrid` to combine betweenness and degree
      information.

Method comparison table
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 15 15 20

   * - Method
     - Weighted
     - Directed
     - Output Type
     - Connectivity
     - Approach
   * - :func:`~networkx_backbone.disparity_filter`
     - Yes
     - Yes
     - p-value
     - No guarantee
     - Statistical
   * - :func:`~networkx_backbone.noise_corrected_filter`
     - Yes
     - Yes
     - z-score
     - No guarantee
     - Statistical
   * - :func:`~networkx_backbone.marginal_likelihood_filter`
     - Yes
     - Yes
     - p-value
     - No guarantee
     - Statistical
   * - :func:`~networkx_backbone.ecm_filter`
     - Yes
     - Yes
     - p-value
     - No guarantee
     - Statistical
   * - :func:`~networkx_backbone.lans_filter`
     - Yes
     - Yes
     - p-value
     - No guarantee
     - Statistical
   * - :func:`~networkx_backbone.global_threshold_filter`
     - Yes
     - Yes
     - Subgraph
     - No guarantee
     - Structural
   * - :func:`~networkx_backbone.strongest_n_ties`
     - Yes
     - Yes
     - Subgraph
     - No guarantee
     - Structural
   * - :func:`~networkx_backbone.high_salience_skeleton`
     - Yes
     - No
     - Score
     - No guarantee
     - Structural
   * - :func:`~networkx_backbone.metric_backbone`
     - Yes
     - No
     - Subgraph
     - Preserved
     - Structural
   * - :func:`~networkx_backbone.ultrametric_backbone`
     - Yes
     - No
     - Subgraph
     - Preserved
     - Structural
   * - :func:`~networkx_backbone.doubly_stochastic_filter`
     - Yes
     - No
     - Score
     - No guarantee
     - Structural
   * - :func:`~networkx_backbone.h_backbone`
     - Yes
     - No
     - Subgraph
     - No guarantee
     - Structural
   * - :func:`~networkx_backbone.modularity_backbone`
     - Yes
     - No
     - Node score
     - No guarantee
     - Structural
   * - :func:`~networkx_backbone.planar_maximally_filtered_graph`
     - Yes
     - No
     - Subgraph
     - Preserved
     - Structural
   * - :func:`~networkx_backbone.maximum_spanning_tree_backbone`
     - Yes
     - No
     - Subgraph
     - Preserved
     - Structural
   * - :func:`~networkx_backbone.glab_filter`
     - Yes
     - No
     - p-value
     - No guarantee
     - Hybrid
   * - Proximity methods (12)
     - No
     - Yes
     - Score
     - No guarantee
     - Proximity
   * - :func:`~networkx_backbone.sdsm`
     - Optional
     - No
     - Subgraph
     - No guarantee
     - Bipartite
   * - :func:`~networkx_backbone.fdsm`
     - Optional
     - No
     - Subgraph
     - No guarantee
     - Bipartite
   * - :func:`~networkx_backbone.sparsify`
     - No
     - No
     - Subgraph
     - Optional (UMST)
     - Unweighted
   * - :func:`~networkx_backbone.lspar`
     - No
     - No
     - Subgraph
     - No guarantee
     - Unweighted
   * - :func:`~networkx_backbone.local_degree`
     - No
     - No
     - Subgraph
     - No guarantee
     - Unweighted

Choosing a statistical method
-----------------------------

All five statistical methods test edge significance, but they differ in their
null models:

- :func:`~networkx_backbone.disparity_filter`: Assumes each node's total
  strength is uniformly distributed across its edges. Works well for
  heterogeneous networks with broad degree distributions.

- :func:`~networkx_backbone.noise_corrected_filter`: Uses a binomial framework
  to model edge weights as outcomes of random processes. Produces z-scores
  rather than p-values.

- :func:`~networkx_backbone.marginal_likelihood_filter`: Considers both
  endpoints of each edge in a binomial null model. Treats weights as integer
  counts.

- :func:`~networkx_backbone.ecm_filter`: Uses a maximum-entropy null model that
  preserves both degree and strength sequences. Most computationally expensive
  but most principled.

- :func:`~networkx_backbone.lans_filter`: Nonparametric method using empirical
  CDFs. Makes no distributional assumptions but relies on sufficient data at
  each node.
