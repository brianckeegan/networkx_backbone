Choosing a Method
=================

With 65 functions available, selecting the right backbone method depends on
the properties of your network and the goals of your analysis.

Decision guide
--------------

.. page-figure:: choosing_method/decision

**Is your graph bipartite?**
    Use :func:`~nx_backbone.sdsm` (analytical) or
    :func:`~nx_backbone.fdsm` (Monte Carlo) to score edges in a
    bipartite projection and then filter by p-value. Fixed null-model variants are also
    available via :func:`~nx_backbone.fixedfill`,
    :func:`~nx_backbone.fixedrow`, and
    :func:`~nx_backbone.fixedcol`. Projection weighting can be selected
    with ``projection="simple"|"hyper"|"probs"|"ycn"``.

**Is your graph unweighted?**
    Use the :mod:`~nx_backbone.unweighted` module:
    :func:`~nx_backbone.sparsify` (generic framework),
    :func:`~nx_backbone.lspar` (local sparsification), or
    :func:`~nx_backbone.local_degree` (degree-based), then apply
    :func:`~nx_backbone.boolean_filter` with ``"sparsify_keep"``.

**Is your graph weighted?**
    Choose based on your analysis goals:

    - **Statistical significance**: Use methods from
      :mod:`~nx_backbone.statistical` to test whether edge weights are
      significant under a null model. These produce p-values that can be
      filtered at a chosen significance level.

    - **Topological structure**: Use methods from
      :mod:`~nx_backbone.structural` to extract edges that are important
      for the network's topology (shortest paths, spanning trees, planarity).

    - **Neighborhood similarity**: Use methods from
      :mod:`~nx_backbone.proximity` to score edges based on how
      similar the neighborhoods of their endpoints are.

    - **Combined approach**: Use :func:`~nx_backbone.glab_filter` from
      :mod:`~nx_backbone.hybrid` to combine betweenness and degree
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
   * - :func:`~nx_backbone.disparity_filter`
     - Yes
     - Yes
     - p-value
     - No guarantee
     - Statistical
   * - :func:`~nx_backbone.noise_corrected_filter`
     - Yes
     - Yes
     - Score + p-value
     - No guarantee
     - Statistical
   * - :func:`~nx_backbone.marginal_likelihood_filter`
     - Yes
     - Yes
     - p-value
     - No guarantee
     - Statistical
   * - :func:`~nx_backbone.ecm_filter`
     - Yes
     - Yes
     - p-value
     - No guarantee
     - Statistical
   * - :func:`~nx_backbone.lans_filter`
     - Yes
     - Yes
     - p-value
     - No guarantee
     - Statistical
   * - :func:`~nx_backbone.polya_filter`
     - Yes
     - Yes
     - p-value
     - No guarantee
     - Statistical
   * - :func:`~nx_backbone.gloss_filter`
     - Yes
     - Yes
     - p-value
     - No guarantee
     - Statistical
   * - :func:`~nx_backbone.multiple_linkage_analysis`
     - Yes
     - Yes
     - Keep flag
     - No guarantee
     - Statistical
   * - :func:`~nx_backbone.global_threshold_filter`
     - Yes
     - Yes
     - Scored graph
     - No guarantee
     - Structural
   * - :func:`~nx_backbone.strongest_n_ties`
     - Yes
     - Yes
     - Scored graph
     - No guarantee
     - Structural
   * - :func:`~nx_backbone.global_sparsification`
     - Yes
     - Yes
     - Scored graph
     - No guarantee
     - Structural
   * - :func:`~nx_backbone.primary_linkage_analysis`
     - Yes
     - Yes
     - Scored graph
     - No guarantee
     - Structural
   * - :func:`~nx_backbone.edge_betweenness_filter`
     - Yes
     - Yes
     - Scored graph
     - No guarantee
     - Structural
   * - :func:`~nx_backbone.node_degree_filter`
     - No
     - Yes
     - Scored graph
     - No guarantee
     - Structural
   * - :func:`~nx_backbone.high_salience_skeleton`
     - Yes
     - No
     - Score
     - No guarantee
     - Structural
   * - :func:`~nx_backbone.metric_backbone`
     - Yes
     - No
     - Scored graph
     - Depends on filter
     - Structural
   * - :func:`~nx_backbone.ultrametric_backbone`
     - Yes
     - No
     - Scored graph
     - Depends on filter
     - Structural
   * - :func:`~nx_backbone.doubly_stochastic_filter`
     - Yes
     - No
     - Score + keep flag
     - Spans each component (``ds_keep``)
     - Structural
   * - :func:`~nx_backbone.h_backbone`
     - Yes
     - No
     - Scored graph
     - No guarantee
     - Structural
   * - :func:`~nx_backbone.modularity_backbone`
     - Yes
     - No
     - Node score
     - No guarantee
     - Structural
   * - :func:`~nx_backbone.planar_maximally_filtered_graph`
     - Yes
     - No
     - Scored graph
     - Depends on filter
     - Structural
   * - :func:`~nx_backbone.maximum_spanning_tree_backbone`
     - Yes
     - No
     - Scored graph
     - Depends on filter
     - Structural
   * - :func:`~nx_backbone.glab_filter`
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
   * - :func:`~nx_backbone.sdsm`
     - Optional
     - No
     - Scored graph
     - No guarantee
     - Bipartite
   * - :func:`~nx_backbone.fdsm`
     - Optional
     - No
     - Scored graph
     - No guarantee
     - Bipartite
   * - :func:`~nx_backbone.sparsify`
     - No
     - No
     - Scored graph
     - Optional (UMST)
     - Unweighted
   * - :func:`~nx_backbone.lspar`
     - No
     - No
     - Scored graph
     - No guarantee
     - Unweighted
   * - :func:`~nx_backbone.local_degree`
     - No
     - No
     - Scored graph
     - No guarantee
     - Unweighted

Choosing a statistical method
-----------------------------

All eight statistical methods test edge significance, but they differ in their
null models (the tutorial :ref:`tutorial-statistical-nulls` runs seven of them
on one network and shows what each choice does):

- :func:`~nx_backbone.disparity_filter`: Assumes each node's total
  strength is uniformly distributed across its edges. Works well for
  heterogeneous networks with broad degree distributions. ``a=...`` tolerates
  more (``a > 1``) or less (``a < 1``) heterogeneity, and ``a="ml"`` fits it;
  see :doc:`weight_transforms`.

- :func:`~nx_backbone.noise_corrected_filter`: Scores each edge by its
  lift over the weight expected from its endpoints' strengths, and estimates
  the score's standard deviation with a Bayesian model of noisy counts.
  ``nc_pvalue`` is the exact binomial p-value of its null model, which is the
  marginal likelihood filter's null counted on a symmetric weight matrix. The
  paper's rule ``nc_score > 1.64 * nc_sdev`` (``pvalue_method="delta"``) is
  not calibrated: it keeps most edges of low-count graphs, even pure noise.

- :func:`~nx_backbone.marginal_likelihood_filter`: Considers both
  endpoints of each edge in a binomial null model. Treats weights as integer
  counts; ``unit=`` sets the weight of one count (also on ``ecm_filter`` and
  ``noise_corrected_filter``).

- :func:`~nx_backbone.ecm_filter`: Uses a maximum-entropy null model that
  preserves both degree and strength sequences. Most computationally expensive
  but most principled.

- :func:`~nx_backbone.lans_filter`: Nonparametric method using empirical
  CDFs. Makes no distributional assumptions but relies on sufficient data at
  each node.

- :func:`~nx_backbone.polya_filter`: Lets each node hand out its
  strength through a self-reinforcing Pólya urn. The reinforcement parameter
  ``a`` sets how much heterogeneity the null tolerates; ``a=None`` fits it to
  the data by maximum likelihood. Treats weights as integer counts. Yassin et
  al. (2025) found it best preserves the weight distribution.

- :func:`~nx_backbone.gloss_filter`: A global null that keeps the
  topology and draws every edge weight from the observed weight
  distribution, conditioned on the endpoints' degrees and strengths. Keeps
  the fewest edges of these methods in the Yassin et al. (2025) comparison.

- :func:`~nx_backbone.multiple_linkage_analysis`: For each node, compares
  the sorted shares of its link weights with ideal "k equally dominant links"
  distributions and keeps the links of the best fit (``mla_keep``, with the
  number per node in ``mla_rank``). There is no significance level, and every
  node keeps at least its strongest link. ``method="lans"`` thresholds LANS
  p-values instead, as version 0.2 did.
