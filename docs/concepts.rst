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

The 87 functions in ``networkx-backbone`` are organized into ten modules based
on the approach they take. The method taxonomy aligns with the categories used
in ``netbone`` (Yassin et al., 2023; https://gitlab.liris.cnrs.fr/coregraphie/netbone),
extended with a hypergraph module for higher-order networks.

Statistical methods
^^^^^^^^^^^^^^^^^^^

The :mod:`~networkx_backbone.statistical` module provides eight methods that test
whether each edge's weight is statistically significant under a null model.
These methods produce a p-value or score for each edge, or a keep flag.

- :func:`~networkx_backbone.disparity_filter` -- uniform null model (Serrano et al., 2009)
- :func:`~networkx_backbone.noise_corrected_filter` -- Bayesian noise-corrected lift score with a standard deviation (Coscia & Neffke, 2017)
- :func:`~networkx_backbone.marginal_likelihood_filter` -- binomial null considering both endpoints (Dianati, 2016)
- :func:`~networkx_backbone.ecm_filter` -- maximum-entropy null model (Gemmetto et al., 2017)
- :func:`~networkx_backbone.lans_filter` -- nonparametric empirical CDF (Foti et al., 2011)
- :func:`~networkx_backbone.polya_filter` -- Pólya urn null with a reinforcement parameter that tolerates heterogeneous weights (Marcaccioli & Livan, 2019)
- :func:`~networkx_backbone.gloss_filter` -- global null that keeps the topology and draws weights from the observed weight distribution (Radicchi et al., 2011)
- :func:`~networkx_backbone.multiple_linkage_analysis` -- dominant links per node by fitting ideal distributions (Van Nuffel, 2007); ``method="lans"`` thresholds LANS p-values instead, as version 0.2 did

The noise-corrected backbone keeps an edge when its score exceeds ``delta``
standard deviations (``nc_score > delta * nc_sdev``, with ``delta = 1.64`` in
the paper); ``nc_pvalue`` expresses the same rule as a p-value.  Multiple
linkage analysis is not a significance test: for each node it chooses how many
links are dominant by comparing the node's sorted weight shares with ideal
"one dominant link", "two equal links", ... distributions (by the coefficient
of determination, as restated by Van Nuffel et al., 2010), whereas LANS gives
every link a p-value from its node's empirical weight distribution.

P-values can be corrected for multiple comparisons with
:func:`~networkx_backbone.adjust_pvalues` or the ``mtc`` argument of
:func:`~networkx_backbone.threshold_filter` (Bonferroni, Holm, Hochberg,
Benjamini-Hochberg, Benjamini-Yekutieli).  The ``disparity_filter``,
``noise_corrected_filter``, ``marginal_likelihood_filter``, ``ecm_filter``, and
``lans_filter`` methods also accept ``signed=True`` for a two-tailed test that
keeps significantly strong (``+1``) and significantly weak (``-1``) edges, each
tagged with a ``"sign"`` attribute.  The sign is ``+1`` when the upper-tail
p-value is smaller than the lower-tail one and ``-1`` otherwise, so an edge
whose two tails are exactly equal is signed ``-1``, as in R ``backbone``
3.0.4; the same rule applies to the signed bipartite null models.  With ``missing_as_zero=True`` they also
test every pair of nodes that has no edge, as an edge of weight 0, so a signed
backbone can contain pairs that interact significantly less than expected
(Backbone 3.0's ``missing_as_zero``).  The result then has one edge per tested
node pair, which takes :math:`O(n^2)` memory.

Each null model assumes something about the weights.  The disparity filter
assumes that a node's weight shares are uniform spacings, which holds for
independent exponential weights; ``disparity_filter(a=...)`` relaxes this to
a one-parameter family whose ``a`` can be fitted (``a="ml"``).  The marginal
likelihood, ECM, and noise-corrected filters treat weights as counts of
events, so their results depend on the unit, which ``unit=`` sets.
:func:`~networkx_backbone.transform_weights` applies a log, power, or other
transform before scoring.  :doc:`user_guide/weight_transforms` lists which
methods a rescaling or an increasing transform leaves unchanged.

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

Hypergraph methods
^^^^^^^^^^^^^^^^^^

The :mod:`~networkx_backbone.hypergraph` module backbones higher-order networks
(hypergraphs) directly, rather than dyadic graphs. Unlike the projection-based
:mod:`~networkx_backbone.bipartite` methods, which reduce a hypergraph to a
weighted pairwise graph, these methods return a sub-hypergraph (a subset of
hyperedges).

- :func:`~networkx_backbone.mdl_hypergraph_backbone` -- parameter-free,
  information-theoretic (minimum description length) backbone that prunes nested
  and redundant hyperedges, with an optional weighted extension (Kirkley,
  Felippe, Malizia & Battiston, 2026)
- :func:`~networkx_backbone.local_mdl_hypergraph_backbone` -- the local variant
  of the MDL backbone, which backbones each node's neighborhood separately and
  aggregates the results, like the disparity filter does for graphs
- :func:`~networkx_backbone.hypergraph_compression_ratio` -- inverse compression
  ratio achieved by the MDL backbone
- :func:`~networkx_backbone.intersection_graph` -- graph (or s-line graph)
  linking hyperedges that share at least *s* nodes
- :func:`~networkx_backbone.maximal_hyperedges` -- inclusion (toplex) reduction,
  keeping only hyperedges not contained in another
- :func:`~networkx_backbone.order_filter` -- select hyperedges by order (size)
- :func:`~networkx_backbone.s_components` -- s-connected components of a
  hypergraph
- :func:`~networkx_backbone.statistically_validated_hypergraph` and
  :func:`~networkx_backbone.statistically_validated_cores` -- statistical
  validation of recurring hyperedges/groups under a null model (Musciotto,
  Battiston & Mantegna, 2021)

Because a hypergraph backbone is a subset of hyperedges rather than a graph, this
module returns a :class:`~networkx_backbone.HypergraphBackbone` (or
:class:`~networkx_backbone.ValidatedHypergraph`) result instead of using the
:mod:`~networkx_backbone.filters` utilities.

The :mod:`~networkx_backbone.hypergraph_io` helpers convert hyperedge lists to and
from a NetworkX incidence bipartite graph (so the
:mod:`~networkx_backbone.bipartite` projection backbones apply to hypergraphs),
the HIF interchange format, and the ``xgi`` / ``HyperNetX`` / ``HypergraphX`` /
HAT hypergraph classes (optional, imported lazily).

Unweighted methods
^^^^^^^^^^^^^^^^^^

The :mod:`~networkx_backbone.unweighted` module provides sparsification methods
for unweighted graphs using a four-step pipeline of scoring, normalization,
filtering, and optional connectivity restoration:

- :func:`~networkx_backbone.sparsify` -- generic framework
- :func:`~networkx_backbone.lspar` -- Local Sparsification (Satuluri et al., 2011)
- :func:`~networkx_backbone.local_degree` -- Degree-based (Hamann et al., 2016)

Temporal networks
^^^^^^^^^^^^^^^^^

The :mod:`~networkx_backbone.temporal` module takes a network observed over
time in any of three forms: **snapshots** (a dict or list of graphs),
**events** (a multigraph with one timed edge per event), or **spans** (a
multigraph whose edges last from ``start`` to ``end``, read as the half-open
interval ``[start, end)``). :func:`~networkx_backbone.temporal_form` tells them
apart and :func:`~networkx_backbone.convert_temporal` converts between them.
Events and spans become snapshots by binning time into steps of width
``freq``. Converting snapshots to events or spans and back is lossless. See
:doc:`user_guide/temporal` for worked examples.

.. _temporal-eligibility:

Temporal backbones: presence and eligible steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The temporal backbone scores (:func:`~networkx_backbone.observation_fraction`,
:func:`~networkx_backbone.persistence`,
:func:`~networkx_backbone.backbone_stability`,
:func:`~networkx_backbone.weight_trend` and
:func:`~networkx_backbone.edge_persistence_hmm`) work on steps: the snapshots, or the
bins of width ``freq`` that events and spans fall into. A node is **present**
in a step if it is in that snapshot (isolated or not); after binning, only if
it has an event or span in that step. An edge that is missing from a step
means one of two things:

- both endpoints are present, so the edge could have been seen and was not:
  an **observed zero** (weight 0);
- an endpoint is absent, so nothing is known about the edge in that step.

An edge's **eligible steps** are the steps in which both of its endpoints are
present. The scores count only those, so a node that joins late is not
penalized for the steps before it existed. In this example ``"c"`` joins at
step 2:

.. code-block:: python

    >>> import networkx as nx
    >>> import networkx_backbone as nb
    >>> S = [
    ...     nx.Graph([("a", "b")]),
    ...     nx.Graph([("a", "b")]),
    ...     nx.Graph([("b", "c")]),
    ...     nx.Graph([("a", "b"), ("b", "c")]),
    ... ]
    >>> S[2].add_node("a")  # present at step 2, without edges
    >>> H = nb.observation_fraction(S)
    >>> H["a"]["b"]["obs_fraction"], H["a"]["b"]["n_eligible"]
    (0.75, 4)
    >>> H["b"]["c"]["obs_fraction"], H["b"]["c"]["n_eligible"]
    (1.0, 2)

``("a", "b")`` is missing at step 2 while both endpoints are present, an
observed zero, so it is seen in 3 of 4 eligible steps. ``("b", "c")`` is
eligible only from step 2, when ``"c"`` joins, and is seen in both of those
steps. Counting all four steps would give it 0.5 and treat the steps before
``"c"`` existed as evidence against the edge. A node that leaves early is
handled the same way: its edges are unknown, not zero, after it leaves.

Each score returns one static graph, the union of the observed edges with
score attributes (and ``H.graph["temporal"]``, which records the steps and
parameters), so the score-then-filter pattern below applies unchanged. See
the :doc:`tutorials/temporal_backbones` tutorial.

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

Many structural and unweighted methods expose boolean edge flags (for example
``metric_keep`` or ``sparsify_keep``). These still follow score-then-filter:

::

    scored = nb.metric_backbone(G)               # adds "metric_keep"
    backbone = nb.boolean_filter(scored, "metric_keep")

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
