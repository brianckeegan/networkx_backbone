Concepts
========

This page provides a conceptual overview of backbone extraction and how the
methods in ``networkx-backbone`` are organized.

What is backbone extraction?
----------------------------

Real-world networks are often dense and noisy. Backbone extraction identifies
the most important substructure of a network by removing edges that are
redundant, statistically insignificant, or structurally unimportant. The result
is a sparser graph that preserves the essential structure of the original.

Taxonomy of methods
-------------------

The 65 functions in ``networkx-backbone`` are organized into nine modules based
on the approach they take. The method taxonomy aligns with the categories used
in ``netbone`` (Yassin et al., 2023; https://gitlab.liris.cnrs.fr/coregraphie/netbone).

Statistical methods
^^^^^^^^^^^^^^^^^^^

The :mod:`~networkx_backbone.statistical` module provides six methods that test
whether each edge's weight is statistically significant under a null model.
These methods produce a p-value or z-score for each edge.

- :func:`~networkx_backbone.disparity_filter` -- uniform null model (Serrano et al., 2009)
- :func:`~networkx_backbone.noise_corrected_filter` -- binomial null model (Coscia & Neffke, 2017)
- :func:`~networkx_backbone.marginal_likelihood_filter` -- binomial null considering both endpoints (Dianati, 2016)
- :func:`~networkx_backbone.ecm_filter` -- maximum-entropy null model (Gemmetto et al., 2017)
- :func:`~networkx_backbone.lans_filter` -- nonparametric empirical CDF (Foti et al., 2011)
- :func:`~networkx_backbone.multiple_linkage_analysis` -- local linkage significance (Van Nuffel et al., 2010; Yassin et al., 2023)

Structural methods
^^^^^^^^^^^^^^^^^^

The :mod:`~networkx_backbone.structural` module provides fourteen methods that use
topological properties of the network directly, without hypothesis testing.

- **Simple filters**: :func:`~networkx_backbone.global_threshold_filter`,
  :func:`~networkx_backbone.strongest_n_ties`,
  :func:`~networkx_backbone.global_sparsification`
- **Linkage/centrality filters**:
  :func:`~networkx_backbone.primary_linkage_analysis`,
  :func:`~networkx_backbone.edge_betweenness_filter`,
  :func:`~networkx_backbone.node_degree_filter`
- **Shortest-path methods**: :func:`~networkx_backbone.high_salience_skeleton`,
  :func:`~networkx_backbone.metric_backbone`,
  :func:`~networkx_backbone.ultrametric_backbone`
- **Normalization**: :func:`~networkx_backbone.doubly_stochastic_filter`
- **Index-based**: :func:`~networkx_backbone.h_backbone`
- **Community-based**: :func:`~networkx_backbone.modularity_backbone`
- **Constrained**: :func:`~networkx_backbone.planar_maximally_filtered_graph`,
  :func:`~networkx_backbone.maximum_spanning_tree_backbone`

Proximity methods
^^^^^^^^^^^^^^^^^

The :mod:`~networkx_backbone.proximity` module provides twelve methods that
score each edge based on how similar the neighborhoods of its endpoints are.
High-scoring edges are structurally embedded within communities, while
low-scoring edges tend to be bridges.

Methods include :func:`~networkx_backbone.jaccard_backbone`,
:func:`~networkx_backbone.dice_backbone`, :func:`~networkx_backbone.cosine_backbone`,
:func:`~networkx_backbone.adamic_adar_index`,
:func:`~networkx_backbone.resource_allocation_index`, and more.

Hybrid methods
^^^^^^^^^^^^^^

The :mod:`~networkx_backbone.hybrid` module contains
:func:`~networkx_backbone.glab_filter`, which combines global betweenness
information with local degree information (Zhang et al., 2014).

Bipartite methods
^^^^^^^^^^^^^^^^^

The :mod:`~networkx_backbone.bipartite` module provides methods for extracting
significant edges from bipartite graph projections:

- :func:`~networkx_backbone.simple_projection`,
  :func:`~networkx_backbone.hyper_projection`,
  :func:`~networkx_backbone.probs_projection`,
  :func:`~networkx_backbone.ycn_projection` -- weighted projection schemes
  (Coscia & Neffke, 2017)
- :func:`~networkx_backbone.sdsm` -- Stochastic Degree Sequence Model (analytical, Neal 2014)
- :func:`~networkx_backbone.fdsm` -- Fixed Degree Sequence Model (Monte Carlo, Neal et al. 2021)
- :func:`~networkx_backbone.fixedfill`, :func:`~networkx_backbone.fixedrow`,
  :func:`~networkx_backbone.fixedcol` -- fixed null-model variants
- :func:`~networkx_backbone.backbone_from_projection` /
  :func:`~networkx_backbone.backbone` -- high-level wrappers

Unweighted methods
^^^^^^^^^^^^^^^^^^

The :mod:`~networkx_backbone.unweighted` module provides sparsification methods
for unweighted graphs using a four-step pipeline of scoring, normalization,
filtering, and optional connectivity restoration:

- :func:`~networkx_backbone.sparsify` -- generic framework
- :func:`~networkx_backbone.lspar` -- Local Sparsification (Satuluri et al., 2011)
- :func:`~networkx_backbone.local_degree` -- Degree-based (Hamann et al., 2016)

The score-then-filter pattern
-----------------------------

Most backbone methods follow a two-step pattern:

1. **Score**: Apply a backbone method to annotate each edge with a score
   (p-value, similarity, salience, etc.). The method returns a copy of the
   graph with the score as an edge attribute.

2. **Filter**: Use a function from the :mod:`~networkx_backbone.filters` module
   to extract the backbone by selecting edges based on their score.

For example::

    # Step 1: Score edges
    H = nb.disparity_filter(G)  # adds "disparity_pvalue" attribute

    # Step 2: Filter edges
    backbone = nb.threshold_filter(H, "disparity_pvalue", 0.05)

Some methods return the backbone directly without a separate filter step.
These include :func:`~networkx_backbone.metric_backbone`,
:func:`~networkx_backbone.ultrametric_backbone`,
:func:`~networkx_backbone.maximum_spanning_tree_backbone`,
:func:`~networkx_backbone.planar_maximally_filtered_graph`,
:func:`~networkx_backbone.global_threshold_filter`, and
:func:`~networkx_backbone.strongest_n_ties`,
:func:`~networkx_backbone.global_sparsification`,
:func:`~networkx_backbone.primary_linkage_analysis`,
:func:`~networkx_backbone.edge_betweenness_filter`,
:func:`~networkx_backbone.node_degree_filter`, and
:func:`~networkx_backbone.multiple_linkage_analysis`.

Evaluation
----------

The :mod:`~networkx_backbone.measures` module provides metrics for evaluating
how well a backbone preserves the properties of the original graph:

- **Fraction metrics**: :func:`~networkx_backbone.node_fraction`,
  :func:`~networkx_backbone.edge_fraction`,
  :func:`~networkx_backbone.weight_fraction`
- **Connectivity**: :func:`~networkx_backbone.reachability`
- **Distribution preservation**: :func:`~networkx_backbone.ks_degree`,
  :func:`~networkx_backbone.ks_weight`
- **Comparison**: :func:`~networkx_backbone.compare_backbones`
