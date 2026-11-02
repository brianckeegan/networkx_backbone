Changelog
=========

Unreleased
----------

*Documentation*

- One way to put light and dark figures on every kind of docs page (#126).
  The executed tutorials use a shared figure helper
  (``docs/_figures/tutorial.py``) that links the two image files lazily,
  with computed alt text and captions and a gallery thumbnail; all eight
  tutorials now use it instead of four different mechanisms, with no change
  to their narrative. Plain reST pages get build-time figures from a registry
  with one module per page (``docs/_figures/pages/``) and the new
  ``page-figure`` directive, which can also reuse a method-gallery figure;
  the first one is on :doc:`user_guide/workflow`. A new test requires a
  figure on every how-to and user-guide page, with the pages still to do
  listed explicitly. :doc:`contributing` explains how to write figures.
- Figures for five how-to tutorials (#121): the Davis incidence matrix
  beside its projection (:doc:`tutorials/bipartite_backbone`), hyperedges and
  their MDL backbones beside the pairwise projection
  (:doc:`tutorials/hypergraph_backbone`), edge activity over the time steps
  with observation fraction and persistence
  (:doc:`tutorials/temporal_backbones`), edges kept against alpha for each
  multiple-testing correction and the consensus vote counts
  (:doc:`tutorials/validating_backbones`), and an overview of one backbone
  per API module (:doc:`tutorials/graph_gallery`), plus method-gallery
  figures where they fit. The temporal and validating pages now link their
  narrative tutorials.

Version 0.3.0 (2026-11-02)
--------------------------

This release adds hypergraph backbones, the remaining Backbone 3.0 features,
the Pólya and GloSS filters, generalized null models and weight transforms,
and a new ``temporal`` module for networks observed over time (conversions
between snapshots, events, and spans, and collapsing them to a static
weighted graph). The public API is now type-annotated and dispatches to
NetworkX backends. It also fixes many correctness problems found in a full
code review that checked every method against its paper, and checks p-values
and backbones against R ``backbone`` 3.0.4 and netbone, which they now match
(signed backbones included) except where the documented conventions differ.
**Several methods now return different results**; see *Changes in results*
below. Open work is tracked in the :doc:`roadmap` and in GitHub issues.

**Renamed (breaking)**

- The package is now **nx-backbone**: install it with
  ``pip install nx-backbone`` and import it as ``import nx_backbone``. The
  repository moved to `brianckeegan/nx-backbone
  <https://github.com/brianckeegan/nx-backbone>`_, and the documentation to
  https://www.brianckeegan.com/nx-backbone/. This is a clean break: there is
  no ``networkx_backbone`` alias, so ``import networkx_backbone`` fails once
  the old package is uninstalled. The old ``networkx-backbone`` project on
  PyPI is not updated.
- To migrate, replace the distribution and the imports::

      pip uninstall networkx-backbone
      pip install nx-backbone
      # then, in your code:
      #   import networkx_backbone as nb  ->  import nx_backbone as nb
      #   from networkx_backbone import ...  ->  from nx_backbone import ...

  Function names, arguments and edge attributes are unchanged by the rename.
  The BibTeX key that :func:`~nx_backbone.describe_backbone` writes is now
  ``keegan_nx_backbone``.

**Supported versions (breaking)**

- Requires **Python 3.12 or newer** and **NetworkX 3.4 or newer**, matching
  NetworkX's own support window. The optional dependency minimums now match
  NetworkX's too: NumPy 2.2, SciPy 1.15, Matplotlib 3.10. A CI job tests
  these minimums exactly.

**New features**

*Statistical and structural methods*

- New :func:`~nx_backbone.gloss_filter`, the GloSS filter of Radicchi,
  Ramasco & Fortunato (2011). It tests each edge weight against a global null
  model that keeps the topology and draws weights from the network's own
  weight distribution, conditioning on both endpoints' degrees and strengths,
  and adds ``gloss_pvalue``. The p-values of the paper's Eq. (4) are computed
  exactly (integer weights) by direct convolution, stable at any degree;
  directed graphs use out- and in-degrees and strengths. Les Misérables: 3
  edges at alpha = 0.05 (#43).
- New :func:`~nx_backbone.polya_filter`: the Pólya urn filter of
  Marcaccioli & Livan (2019), adding ``polya_pvalue``. The null model is the
  beta-binomial urn of Eqs. (1)-(2), tested from both endpoints (out- and
  in-links for directed graphs). By default the reinforcement parameter ``a``
  is the maximum-likelihood estimate of the paper's Supplementary Note 5,
  stored in ``H.graph["polya_a"]``; non-integer weights use the paper's
  large-strength approximation, Eq. (6). Les Misérables: ``a`` = 0.340 and
  12 edges at ``polya_pvalue < 0.05`` (#42).
- ``disparity_filter(a=...)`` generalizes the uniform share null to a
  symmetric Dirichlet null with concentration ``1/a``, so a share is
  Beta(1/a, (k-1)/a) and the p-value is a regularized incomplete beta
  function. ``a=1`` (the default) is the original filter, bit for bit and
  without SciPy; ``a="ml"`` fits ``a`` by maximum likelihood and stores it in
  ``H.graph["disparity_a"]``. With the shares fixed, it is the
  large-strength limit of ``polya_filter``'s beta-binomial null. Les
  Misérables: ``a`` = 0.525 and 19 edges at alpha = 0.05 (9 at ``a=1``)
  (#55).
- New :func:`~nx_backbone.transform_weights` returns a copy of a graph
  with transformed weights, from a plain callable or any object with a
  scikit-learn style ``fit_transform`` (scikit-learn is not a dependency). It
  keeps each node's neighbor order, rejects non-finite or constant output,
  warns when a transform reorders weights or creates ties, and records the
  transform for ``describe_backbone``. The new user guide page
  :doc:`user_guide/weight_transforms` has a table of which methods are
  invariant to rescaling and to increasing transforms, each entry checked by
  a test (#55).
- ``unit=`` on ``marginal_likelihood_filter``, ``ecm_filter``, and
  ``noise_corrected_filter`` sets the weight of one count; the models use
  ``weight / unit``. The default ``unit=1`` leaves results unchanged. The
  three scorers now warn (``UserWarning``) when ``weight / unit`` is not an
  integer count, naming how they coerce it: MLF rounds counts below 0.5 to 0,
  the ECM raises counts below 1.5 to 1 (#55).
- ``describe_backbone`` reports ``a`` of the disparity filter (with the
  fitted value for ``a="ml"``), ``unit``, and transforms recorded by
  ``transform_weights`` (#55).
- A calibration test harness (``tests/test_calibration.py``) simulates
  weights from the disparity filter's null models and checks that endpoint
  p-values are uniform (#55).
- ``ecm_filter`` and ``noise_corrected_filter`` accept ``signed=True`` (#25).
- ``missing_as_zero=True`` on ``disparity_filter``, ``lans_filter``,
  ``marginal_likelihood_filter``, ``ecm_filter``, ``noise_corrected_filter``
  and ``backbone_from_weighted`` (Backbone 3.0). Node pairs without an edge
  are tested as weight-0 edges and returned with ``weight`` 0, so a signed
  backbone can keep pairs that interact significantly less than expected.
  Degrees, strengths, and fitted multipliers still come from the existing
  edges only, as in the R package; each scorer's Notes say how it scores a
  weight of 0. A ``RuntimeWarning`` is raised above one million added pairs
  (#26).
- ``glab_filter`` now supports directed graphs using the out- and
  in-involvement of Zhang et al. (2014), Section 2.3 (#12). On a symmetric
  ``DiGraph`` it gives the undirected p-values.
- ``doubly_stochastic_filter`` now also runs the second stage of Slater's
  two-stage algorithm: a boolean ``ds_keep`` edge attribute flags the edges
  added, in descending ``ds_weight`` order, until each connected component is
  connected (tie levels added together, so the result does not depend on
  insertion order). Extract it with ``boolean_filter(H, "ds_keep")``.
  ``ds_weight`` is unchanged by this stage (#44).
- ``doubly_stochastic_filter(smoothing=...)`` adds a pseudocount to every
  off-diagonal cell before Sinkhorn-Knopp, so the scaling converges on trees
  and sparse networks (``netbone`` uses ``1e-4``). The default ``0`` leaves the
  matrix unchanged (#54).
- ``local_path_index(weight=...)``; a ``h_bridge`` attribute from
  ``h_backbone``; ``statistically_validated_cores(max_groups=...)``.
- ``seed=`` on ``modularity_backbone``, ``sparsify``, and ``compare_graphs``.
- ``lspar(convention="r")`` reproduces R ``backbone``'s ``lspar``. R
  excludes the endpoints from the Jaccard denominator, uses dense ranks,
  ignores edges with Jaccard 0, and keeps ``floor(d**s)`` edges per node
  with no minimum of one. The reference tests show that it matches R exactly
  on the karate club and Les Misérables at s = 0.3, 0.5 and 0.7 (karate at
  s = 0.3: 30 edges, against 54 with the default). The default,
  ``convention="satuluri"``, is unchanged (#59).

*Bipartite and hypergraph methods*

- Hypergraph module: the parameter-free MDL hypergraph backbone (Kirkley et
  al., 2026) with ``"edge"``, ``"node"``, and ``"auto"`` optimizers;
  statistically validated hypergraphs and cores (SVH/SVC); maximal
  hyperedges, order filtering, s-components, and intersection graphs.
- New :func:`~nx_backbone.local_mdl_hypergraph_backbone`: the local
  MDL hypergraph backbone of Kirkley et al. (2026, Appendix E). It finds the
  MDL backbone of each node's neighborhood separately and combines them with
  the ``"union"``, ``"majority"`` (default), or ``"all"`` rule.
  :class:`~nx_backbone.HypergraphBackbone` gains ``n_hyperedges``,
  ``aggregation``, ``local_backbones``, and ``local_description_lengths``
  (#24).
- Backbone 3.0 parity: multiple-testing correction
  (:func:`~nx_backbone.adjust_pvalues`, ``threshold_filter(mtc=...)``),
  signed backbones (``signed=True``), and SDSM-EC edge constraints.
- ``backbone_from_projection(mtc=...)``; ``sdsm(pvalue_method=...)``;
  ``fdsm(burn_in=..., thin=...)``; ``bicm(tol=..., max_iter=...)``.
- ``backbone_from_projection(missing_as_zero=False)`` tests only the agent
  pairs that co-occur, as R ``backbone`` does by default. Pairs that never
  co-occur are not counted by ``mtc`` and are never retained, even as
  significantly weak with ``signed=True``. The default, ``True``, is unchanged
  and tests every pair, like R's ``missing_as_zero = TRUE``. The reference
  tests check both settings against R on Davis Southern Women, where they
  give different backbones in 13 of 48 settings at alpha = 0.06, 0.6 and
  0.8; at 0.05 none differs (#59).
- ``hyper_projection(newman=True)`` / ``projection="newman"`` gives Newman's
  :math:`1/(d(z)-1)` weighting, equal to
  ``nx.bipartite.collaboration_weighted_projected_graph``. The default
  hyperbolic weight, :math:`1/d(z)`, is unchanged and follows Coscia & Rossi
  (2019). Projection citations are corrected to Coscia & Rossi (2019,
  arXiv:1906.09081) (#13).

*Inputs and interoperability*

- New :func:`~nx_backbone.to_graph` converts a NumPy or SciPy sparse
  adjacency matrix (with ``nodelist`` and ``directed``) or a pandas edge-list
  ``DataFrame`` (``source``/``target``/``weight`` columns) to a graph with
  NetworkX's own converters, refusing an asymmetric matrix for an undirected
  graph and duplicate edge-list rows. ``backbone``, ``backbone_from_weighted``
  and ``backbone_from_unweighted`` accept these inputs directly and give the
  same backbone as the graph path; ``nx.to_pandas_edgelist`` returns the
  scored edges as a ``DataFrame``. The individual scorers keep graph-only
  signatures (#20).
- Bipartite functions accept NumPy, SciPy sparse, or pandas incidence
  matrices with ``agent_labels``/``artifact_labels``, and ``agent_nodes`` is
  optional (#20, bipartite part).
- ``hypergraph_io``: ``hypergraph_to_bipartite``, the HIF interchange format,
  and adapters for xgi, HyperNetX, HypergraphX, and HAT.
- The hypergraph adapters carry weights: ``to_xgi``, ``to_hypernetx``,
  ``to_hypergraphx``, and ``to_hat`` take ``weights=``, and the ``from_*``
  functions take ``return_weights=``. New extras ``[xgi]``, ``[hypernetx]``,
  ``[hypergraphx]``, ``[hat]``, and ``[interop]`` (all four) install the
  libraries; supported versions are in :doc:`install` (#15). Testing against
  the real libraries found and fixed these adapter bugs:

  - ``to_xgi`` failed on tuple node labels (xgi read a two-node hyperedge
    as a ``(members, edge_id)`` pair).
  - ``to_hat`` silently dropped repeated hyperedges (HAT's default
    ``compress=True``), and ``from_hat`` returned row indices instead of node
    labels. Labels now round-trip through ``H.nodes["Names"]``.
  - ``to_hypergraphx`` silently merged repeated hyperedges. It still merges
    them, since HypergraphX has no multi-edges, but now sums their weights,
    and it explains the ``TypeError`` raised for mixed-type node labels.
  - The HAT and HypergraphX HIF readers raised ``KeyError`` on
    ``write_hif`` output, which omitted the ``nodes`` and ``edges`` lists.
    ``write_hif`` now always writes both.
- ``multigraph_to_weighted(aggregate=...)``, which can sum, average, or take
  the max/min of parallel-edge weights.
- New module ``nx_backbone.temporal`` (first part of #67): temporal
  networks as snapshots (a dict or list of graphs), events (a multigraph
  with a ``time`` per edge), or spans (``start``/``end``, read as the
  half-open interval ``[start, end)``). :func:`~nx_backbone.temporal_form`
  detects the form; :func:`~nx_backbone.snapshots_from_events`,
  :func:`~nx_backbone.snapshots_from_spans`,
  :func:`~nx_backbone.events_from_snapshots`,
  :func:`~nx_backbone.spans_from_snapshots`,
  :func:`~nx_backbone.events_from_spans`,
  :func:`~nx_backbone.spans_from_events` and
  :func:`~nx_backbone.convert_temporal` convert between them, binning
  with ``freq=``/``origin=`` and combining weights with
  ``multigraph_to_weighted(aggregate=...)``. Snapshots to events or spans and
  back is lossless (empty snapshots, isolated nodes, and node, edge and graph
  attributes are kept in ``M.graph["temporal"]``). Numeric times and
  ``datetime`` times with ``timedelta`` steps need only NetworkX; frequency
  strings such as ``"1D"`` import pandas lazily. See
  :doc:`user_guide/temporal`.
- :func:`~nx_backbone.flatten_temporal` collapses snapshots, events or
  spans into one static weighted graph for the static backbone methods: one
  edge per node pair with the combined ``weight`` (``aggregate=`` sum, count,
  mean, max or min, or ``"duration"`` for spans, the length of the union of a
  pair's spans), ``n_obs``, ``first`` and ``last`` (and ``active_time`` for
  spans), optionally restricted to a half-open ``window=(t0, t1)``.
- :func:`~nx_backbone.from_event_table` and
  :func:`~nx_backbone.to_event_table` read and write pandas event (or
  span) tables, keeping duplicate rows as separate events (#70, pandas part).
- Temporal backbone scores, descriptive (no p-values), for any of the three
  forms (events and spans are binned with ``freq=``). Each returns one static
  graph, the union of the observed edges with score attributes and
  ``H.graph["temporal"]`` (method, input form, step times, ``n_steps``,
  ``freq``, ``origin``, parameters), so ``threshold_filter`` and the other
  filters apply. They count only an edge's *eligible* steps, those in which
  both endpoints are present: an absent edge is an observed zero there and
  unknown elsewhere, so a node that joins late is not penalized (#67).

  - :func:`~nx_backbone.observation_fraction`: ``obs_fraction``,
    ``n_obs``, ``n_eligible``, ``first_obs``, ``last_obs``; ``weight=`` and
    ``min_weight=`` count only heavier observations (#68).
  - :func:`~nx_backbone.persistence`: ``max_run``, ``mean_run``,
    ``n_runs`` and ``persistence``, the share of observed steps whose next
    eligible step is also observed; ``ineligible="break"`` lets unknown steps
    end a run (#68).
  - :func:`~nx_backbone.backbone_stability`: runs any static method
    (``disparity_filter``, ``lans_filter``, ...) on every step with the same
    cut and adds ``stability`` (``n_kept / n_eligible``), ``n_kept``,
    ``n_eligible``, ``first_kept``, ``last_kept``. The cut's attribute comes
    from the new ``describe.SCORE_KEYS`` registry (#69).
  - :func:`~nx_backbone.weight_trend` (needs numpy): ``weight_mean``,
    ``weight_std``, ``weight_cv``, ``weight_slope`` per step (and
    ``weight_slope_time`` per unit of numeric time), with an optional,
    descriptive slope p-value (SciPy) (#69).

  All but ``weight_trend`` work with NetworkX alone. ``describe_backbone``
  describes them, including the number of steps and the binning. New tutorial:
  :doc:`tutorials/temporal_backbones`; :ref:`temporal-eligibility` explains
  the eligible steps.
- New :func:`~nx_backbone.edge_persistence_hmm` (needs numpy): a
  two-state hidden Markov model of edge activity. Each edge switches between
  an inactive and an active state (transition probabilities ``a01`` and
  ``a10``) and is observed with a false-positive rate ``alpha`` and a miss
  rate ``beta``, all shared by the edges and fitted by Baum-Welch (EM) pooled
  over the edges, or fixed with ``params=``. The observations are, in each
  eligible step, the edge's presence (``observe="presence"``), whether a
  static method keeps it with ``backbone_stability``'s per-step cut
  (``observe="keep"``), or its p-value (``observe="pvalue"``: uniform when
  inactive, Beta(``a``, 1) when active, a modeling choice); ineligible steps
  carry no evidence. Exact forward-backward and Viterbi give per-edge
  ``hmm_posterior`` (a dict over the eligible steps), ``hmm_active_fraction``,
  ``hmm_score``, ``hmm_n_active`` and ``hmm_keep`` (active in at least
  ``min_steps`` steps); the fitted model is in ``H.graph["temporal"]["hmm"]``.
  ``alpha < 1 - beta`` is imposed by relabeling the states. ``hmm_score`` is
  a persistence-smoothed membership score, not a p-value. The temporal
  tutorial compares it with ``persistence`` (#73).
- ``edge_persistence_hmm`` records two diagnostics in
  ``H.graph["temporal"]["hmm"]``: ``n_active_steps``, the number of eligible
  steps the most likely state paths mark active, and
  ``unobserved_active_share``, the share of them in which the edge was not
  observed. Because the parameters are pooled over the edges, a mix of
  persistent and randomly flickering edges can be fitted with a large miss
  rate and slow switching, which gives the flickering edges more active
  steps than the persistent ones. A large share flags this. The Notes
  explain the trade-off and when to fix ``beta`` with ``params=``. A Beta
  prior on ``alpha`` and ``beta`` (MAP-EM) was evaluated on planted-truth
  simulations and not adopted. No fitted values change (#112).
- New :func:`~nx_backbone.kalman_backbone` (needs numpy and SciPy), a
  novel adaptation of the Kalman filter and Rauch-Tung-Striebel smoother to
  backboning (#72). It runs a static significance test (``disparity_filter``
  or ``glab_filter``; discrete tests such as ``noise_corrected_filter`` with
  ``discrete="raw"``) on every step, scoring
  eligible absent edges as weight 0 with ``missing_as_zero=True``, turns the
  p-values into probit scores, and smooths each edge's series with a
  local-level model (``mode="smoother"`` or the online ``"filter"``; unknown
  steps and unequal spacing are handled exactly). Per-step p-values use the
  exact null distribution of the smoothed level; the edge-level
  ``kalman_pvalue`` tests the largest smoothed z-score over the steps against
  a null simulated once per missingness pattern (``n_sim=``, ``seed=``). The
  process variance comes from ``q=``, ``half_life=`` (in steps) or a pooled
  maximum likelihood fit, recorded in ``H.graph["temporal"]["kalman"]``. Each
  edge also gets ``kalman_score``, ``kalman_n_obs`` and the per-step dicts
  ``kalman_pvalues``, ``kalman_levels`` and ``kalman_innovations``;
  :func:`~nx_backbone.kalman_snapshots` expands them into per-step
  graphs. Methods with discrete p-values (LANS, ECM, the marginal likelihood
  filter, Pólya, GloSS) raise unless ``discrete="raw"`` (valid but
  conservative), because none reports the ``P(X > x)`` that a mid-p transform
  needs. The disparity filter and GLAB give an edge the smaller of two
  endpoint p-values, which smoothing would turn into many false positives;
  the default ``sidak=True`` corrects for that, which keeps their smoothed
  tests valid but conservative. The docstring documents the risks: temporal
  leakage, autocorrelated confounders, and multiple testing over edges and
  steps.

*Reporting and documentation*

- New :func:`~nx_backbone.describe_backbone`: methods-section text and
  citations for a backbone, in text, Markdown, or BibTeX; the counterpart of
  Backbone 3.0's ``narrative`` (#23). It is backed by one method-to-BibTeX
  registry that covers every public scorer, and a test enforces that coverage.
- New tutorial, "Validating a backbone": multiple-testing correction,
  ``compare_backbones``, consensus, signed backbones, and reporting (#27).
- New narrative tutorial, "From a hairball to a backbone, end to end", the
  "Start here" entry point linked from the home page, the quick start, the
  tutorials index, and the statistical, filters, measures and describe API
  pages. A worked case study on Les Misérables and a seeded, simulated
  collaboration network with planted departments: why a global weight cut
  fails, scoring versus filtering and the disparity filter's null model for
  one node, choosing alpha with Bonferroni and Benjamini-Hochberg
  corrections, four backbones compared in small multiples with
  ``compare_backbones``, robustness to alpha and to the method
  (``consensus_backbone``), a signed backbone, and ``describe_backbone``
  output. It is a Sphinx-Gallery example, so its nine figures (light and dark
  variants, in the shared figure style) and every number are computed when
  the docs are built, and hidden checks fail the build if a claim in the
  text stops holding. The tutorials index is reorganized into "Start here",
  "By network type", and "Going further" (#89, #96).
- New narrative tutorial, "Structural and proximity backbones", executed in
  the example gallery with eight figures: what spanning trees, the PMFG, the
  metric and ultrametric backbones, salience, the h-backbone and the doubly
  stochastic filter guarantee (checked at build time) against the disparity
  filter; weights as strengths or distances; ties and which methods still
  depend on input order; proximity scores and sparsifiers for unweighted
  graphs, including ``lspar(convention="r")``; and Sinkhorn-Knopp convergence
  with ``smoothing=`` (#91).
- Added ``CITATION.cff``.

**Changes in results (bug fixes)**

The Les Misérables counts below are edges kept at the documented threshold.
Davis counts are significant woman-woman pairs at alpha = 0.05.

- ``doubly_stochastic_filter``: when Sinkhorn-Knopp does not converge,
  ``ds_weight`` depended on the orientation in which each edge was stored,
  and so on node insertion order (by up to 0.86 on Les Misérables). It is now
  the symmetric scaling :math:`\sqrt{D_{uv} D_{vu}}`, which equals the doubly
  stochastic limit whenever the iteration converges. On Les Misérables (no
  convergence) ``ds_keep`` now keeps 187 of 254 edges instead of 220 (#54).
- ``strongest_n_ties`` keeps ties together: each node selects every edge
  whose weight is at least its *n*-th largest weight, so a node can select
  more than *n* edges. Before, ties at the cutoff were broken by adjacency
  order, so ``G`` and ``G.copy()`` could give different backbones (on Les
  Misérables at ``n=2`` they differed in 23 edges). The result now depends only
  on the weights, as for ``lspar``/``local_degree``; on an unweighted graph
  every edge is kept. Les Misérables: 69 -> 108 edges (``n=1``), 113 -> 146
  (``n=2``), 146 -> 165 (``n=3``) (#60).
- ``global_sparsification`` and ``edge_betweenness_filter`` break ties by the
  sorted string form of an undirected edge's endpoints, so the result no
  longer depends on how each edge is stored (which follows node insertion
  order). Edge counts are unchanged; on Les Misérables at ``s=0.5``, 5
  (global sparsification) and 6 (edge betweenness) of the 127 kept edges
  swap for other edges tied with them (#60).
- ``noise_corrected_filter`` now implements the published Bayesian
  noise-corrected backbone (Coscia & Neffke, 2017). Its scores and standard
  deviations match the authors' reference code exactly. ``nc_score`` is the
  transformed lift, in (-1, 1), with new ``nc_sdev`` and ``nc_pvalue``
  attributes. Keep edges with ``nc_score > 1.64 * nc_sdev`` or
  ``nc_pvalue < 0.05``; thresholding ``nc_score`` at 2.0 now keeps nothing.
  The old z-score is available as ``method="zscore"``. Les Misérables: 98 ->
  195 edges (#9); see the next entry for the p-value.
- ``noise_corrected_filter``: ``nc_pvalue`` is now, by default, the exact
  binomial tail :math:`P(X \geq n_{ij})` of the method's null model,
  :math:`X \sim B(n_{..}, n_{i.} n_{.j} / n_{..}^2)` (the p-value of the
  paper's footnote 2), computed with the regularized incomplete beta
  function so that non-integer counts are not rounded. The previous
  ``nc_pvalue``, the normal tail of ``nc_score / nc_sdev`` (the paper's
  ``delta`` rule read as a z-test), is ``pvalue_method="delta"``. The
  ``delta`` rule divides the score by a standard deviation evaluated at the
  observed weight through a transform that flattens as the lift grows, so
  the ratio has no bound: on ``gnm_random_graph(30, 40, seed=1)`` with unit
  weights all 40 edges had p ≈ 1e-65 (median) and 4 are now below 0.05
  (median 0.12). On simulated graphs from the model's own null (2,000 unit
  interactions among 50 nodes, 20 seeds), the share of ``nc_pvalue < 0.05``
  was 0.24 (undirected) and 0.34 (directed) before and is 0.047 and 0.051
  now. ``nc_score`` and ``nc_sdev`` are unchanged. On a directed graph with
  integer weights the default p-values equal ``marginal_likelihood_filter``'s;
  on Les Misérables and the karate club the two keep the same edges.
  ``kalman_backbone`` now treats the default noise-corrected p-values as
  discrete (``discrete="raw"`` is needed). Les Misérables: 195 -> 126 edges
  (``unit=0.1``: 222 -> 215); karate club: 68 -> 39 (#105).
- ``multiple_linkage_analysis`` now fits each node's sorted weights to ideal
  dominance distributions by default (``method="dominance"``; Van Nuffel,
  2007, after Weaver, 1954) and adds ``mla_rank``. The previous LANS
  threshold is ``method="lans"``. The fit is the
  coefficient of determination over the zero-padded profile, as restated by
  Van Nuffel, Derudder & Witlox (2010), whose Table 1 the tests reproduce; a
  new ``deviation=`` keyword also offers ``"profile"`` (whole-profile squared
  deviation, the measure used earlier in this development cycle) and
  ``"weaver"`` (Weaver's variance over the candidate links). Les Misérables:
  109 -> 205 edges (``"profile"``: 223, ``"weaver"``: 253) (#10).
- ``sparsify(umst=True)`` now adds the union of all minimum spanning trees
  (Hamann et al., 2016) instead of one arbitrary spanning forest, so tied
  scores no longer make the result depend on node order (#32). The union is
  added whenever ``umst=True``, not only when filtering split a component.
- ``ecm_filter``: the solver never converged and every p-value was about 0,
  so every edge was kept (254 of 254). It now fits the ECM by maximum
  likelihood and uses the correct geometric tail; it keeps 72 edges and warns
  if the fit does not converge.
- ``marginal_likelihood_filter``: now uses Dianati's null,
  :math:`B(T, s_is_j/2T^2)`. The old formula depended on the order nodes were
  added. Les Misérables: 70 -> 126 edges.
- ``glab_filter``: now uses per-node shortest-path involvement and
  :math:`(1-I_{ij})^{(k_i-1)^c}` (Zhang et al., 2014) instead of global edge
  betweenness. Les Misérables: 5 -> 56 edges.
- ``disparity_filter``: the p-value is now :math:`(1 - p_{ij})^{k_i - 1}`
  (Serrano et al., 2009). Versions 0.1.0 and 0.2.0 computed
  :math:`1 - (k_i - 1)(1 - p_{ij})^{k_i - 2}`, the integrand of the paper's
  formula in place of its integral, so almost every edge looked significant.
  Les Misérables: 247 -> 9 edges. On directed graphs it also tests the
  target's incoming links.
- ``lans_filter`` on directed graphs also tests the target's incoming links
  (#11, contributed by @bfs2021 in #34).
- ``lans_filter(signed=True)``: the lower tail is now ``#{weights < w} / k``,
  the mirror image of the upper tail ``#{weights > w} / k``, as in R
  ``backbone`` 3.0.4. It was the empirical CDF ``#{weights <= w} / k``, which
  counts the link itself and is never below ``1 / k``, so a link could be
  significantly weak only at a node of degree above 40 (at alpha = 0.05). The
  two-sided p-values now equal R's. The one-tailed ``lans_pvalue`` is
  unchanged. Les Misérables signed backbone at alpha = 0.05: 108 positive / 0
  negative -> 108 positive / 86 negative (R: 68 / 126, the difference being
  40 ties, whose sign the next entry matches); karate: 45 / 0 -> 45 / 25
  (#58).
- Signed backbones sign an edge whose upper and lower tail p-values are
  exactly equal ``-1``, as R ``backbone`` 3.0.4 does (its ``.retain`` takes
  ``upper < lower`` as positive, in both its weighted and bipartite models);
  it was ``+1``. The rule is the same for every signed scorer
  (``disparity_filter``, ``marginal_likelihood_filter``, ``lans_filter``,
  ``ecm_filter``, ``noise_corrected_filter``, ``sdsm``, ``fdsm``,
  ``fixedfill``, ``fixedrow``, ``fixedcol`` and the ``backbone_from_*``
  wrappers). Two-sided p-values do not change. Only LANS ties are
  significant on the reference data, so only LANS backbones change, and its
  signed backbones now equal R's for every ``mtc``: Les Misérables at
  alpha = 0.05, 108 positive / 86 negative -> 68 / 126; karate, 45 / 25 ->
  30 / 40. In every other model a tie needs both tails of at least 0.5
  (two-sided p-value 1), so only the ``sign`` attribute of untestable or
  non-significant edges changes, for example ECM self-loops and, under the
  disparity filter, edges between two degree-1 nodes. Ties are now equal by
  construction rather than by floating-point rounding:
  ``noise_corrected_filter`` signs by the statistic, so only a score of
  exactly 0 is a tie, and ``fixedrow`` makes the two tails of a symmetric
  hypergeometric null observed at its mean exactly equal and no longer
  depends on the order of the two agents. Davis ``fixedrow``: all 7 such
  pairs are now ``-1`` (6 were ``+1``) (#59).
- ``bicm``/``sdsm``: the BiCM is now actually fitted rather than
  approximated with Chung-Lu, and p-values use exact Poisson-binomial tails
  (or RNA). SDSM-EC refits on the free cells. Davis: 3 -> 0 pairs.
- ``sdsm`` (and ``backbone_from_projection(method="sdsm")``):
  ``pvalue_method="auto"`` now computes exact Poisson-binomial tails whenever
  that is affordable, instead of switching to the refined normal
  approximation (RNA) above 200 artifacts. The RNA's far upper tail is
  anti-conservative when a pair's expected co-occurrence is small, as on
  sparse data. On a 2000 x 500 pure-noise matrix of density 0.01, the
  smallest p-value goes from 5.7e-37 to 3.9e-05 and Benjamini-Hochberg at
  0.05 keeps 1,498 -> 0 pairs. Under SDSM's own null (600 x 300, density
  0.01) the share of p-values below 1e-4 goes from 7.2e-04 to 1.1e-06. When
  exact tails would be too costly, ``"auto"`` keeps them for the pairs of
  degree classes with the smallest expected co-occurrence, uses the RNA for
  the rest and warns. ``pvalue_method="rna"`` is unchanged and still matches
  R ``backbone``. Davis and other inputs with at most 200 artifacts were
  already exact; their p-values change only in the last bits (#108).
- ``fdsm``: the null now preserves degrees (a curveball chain; previously
  197 of 200 samples broke the margins), and p-values are
  ``(count + 1) / (trials + 1)``. Davis: about 10 -> 5 pairs.
- ``fixedfill`` and ``fixedcol`` use their exact null distributions.
- ``backbone_from_projection`` and ``backbone()`` now threshold ``sdsm`` and
  ``fdsm`` at ``alpha``, as they already did for the other models.
- ``lspar``/``sparsify`` and ``local_degree``: each node ranks its own edges
  (Satuluri et al., 2011; Lindner et al., 2015), and ties are kept together.
  Les Misérables: 136 -> 174 (LSpar) and 135 -> 91 (local degree).
- ``h_backbone`` follows Zhang et al. (2018): unweighted betweenness on the
  full graph, normalized by N. When h = 0 no edges are kept. Les Misérables:
  22 -> 18.
- ``high_salience_skeleton`` splits credit among tied shortest paths, so
  saliences sum to n - 1.
- ``metric_backbone`` uses a relative tolerance; ``ultrametric_backbone`` no
  longer crashes on self-loops.
- Proximity indices exclude self-loops from neighborhoods and use out-degree
  for directed graphs.
- ``doubly_stochastic_filter`` warns when Sinkhorn-Knopp does not converge and
  no longer accepts a loose convergence check.
- ``reachability`` counts directed reachability correctly;
  ``compare_backbones`` forwards ``weight`` to every measure that accepts it.
- ``adjust_pvalues`` keeps NaN as NaN and excludes it from the number of tests
  (as R's ``p.adjust`` does), instead of making it significant.

**API conventions (NetworkX compatibility)**

- Every scorer raises ``NetworkXNotImplemented`` for multigraphs (use
  :func:`~nx_backbone.multigraph_to_weighted` first), and
  ``NetworkXError`` for missing, non-positive, NaN, or infinite weights.
- Bipartite functions accept generators for ``agent_nodes``, reject agent sets
  that are not one side of the bipartition, and raise ``AmbiguousSolution``
  when the partition cannot be inferred.
- Filters keep graph, node, and edge attributes and multigraph keys.
  ``fraction_filter`` ignores unscored elements, and ``consensus_backbone``
  works with node labels of mixed types and returns edges in a deterministic
  order.
- Functions that used a hardcoded ``seed=42`` now default to ``seed=None``,
  following NetworkX. Pass a seed for reproducible results.
- The graph-in, graph-out functions (statistical, structural, proximity,
  hybrid, and unweighted methods, ``multigraph_to_weighted``, and the
  threshold, fraction, and boolean filters) are dispatchable through
  ``nx._dispatchable``: they accept ``backend=`` and follow
  ``nx.config.backend_priority``, under dispatch names prefixed with
  ``backbone_`` so they cannot clash with NetworkX's own (two would have:
  ``adamic_adar_index`` and ``resource_allocation_index``). A new
  :doc:`user_guide/backends` guide shows how to use backends and records
  nx-parallel timings for the shortest-path methods. Bipartite, hypergraph,
  measure, and conversion functions are not dispatchable, since they take
  matrices or hyperedge lists or return numbers. The ``RuntimeWarning`` of
  ``ecm_filter`` and ``polya_filter`` now points at the caller's line (#19).
- SDSM-EC constraints that contradict the data (a ``required`` pair that is
  not an edge of ``B``, a ``prohibited`` pair that is) are now ignored with one
  ``RuntimeWarning`` per call, pointing at the caller's line, that counts and
  lists them, instead of raising ``NetworkXError``. This applies to ``sdsm``,
  ``backbone_from_projection`` and ``backbone``. Pairs naming nodes that are
  not an ``(agent, artifact)`` pair of ``B`` still raise, and so does a pair
  listed as both required and prohibited. When no constraint remains (all
  ignored, or empty lists), the p-values equal those of the unconstrained
  SDSM exactly. R ``backbone`` 3.0.4 codes constraints as cell values of the
  incidence matrix, so a constraint there replaces the observed value instead.

**Performance**

- Bipartite projections and null models use sparse incidence matrices and
  score agent pairs in row blocks, and ``backbone_from_projection`` builds
  edges only for significant pairs. On a 3000x800 projection this cuts run
  time from 40-76 s to under 2 s and peak memory from 2.2 GB to 0.23 GB;
  results are unchanged (#30).
- Exact SDSM tails truncate each Poisson-binomial convolution at the
  largest co-occurrence that can be observed (the smaller agent degree, or
  the observed count with edge constraints), which keeps every probability
  exact: the mass above the cut is accumulated as it crosses it. Building
  the tail tables is 8-60x faster (2000 x 2000 at density 0.01: 5.3 s ->
  0.18 s; 5000 x 5000 at 0.002: 19.7 s -> 0.35 s) (#108).
- Vectorized ``sdsm``/``fixed*``/``fdsm`` (``sdsm`` on 400x300: 6.4 s ->
  0.36 s); MDL ``"node"`` optimizer with a lazy heap (m = 2000: 8.3 s ->
  0.47 s); vectorized SVC p-values.
- ``ecm_filter`` fits the ECM over classes of nodes with the same degree and
  strength (out- and in-degree and strength for directed graphs), which is
  exact. Pair sums are computed in chunks, and the dense Newton polish is
  replaced by damped Newton steps (Newton-CG for large problems). On a
  2,000-node Barabási-Albert graph the fit drops from about 90 s and 600 MB
  to under 2 s and 50 MB; 20,000 nodes take about 16 s and 300 MB (about
  90 s for a directed version, which has 10,660 classes). The fit
  now also converges on directed graphs where the old solver stopped short,
  and nodes whose links all have weight 1 get the exact :math:`y_i = 0`.
  P-values on Les Misérables agree with the old solver to about 1e-8
  (#21).

**Infrastructure and documentation**

- New narrative tutorial, "Hypergraph backbones: keeping the groups that
  matter", a Sphinx-Gallery example executed at build time on the Davis
  Southern Women events and a synthetic meeting log. It covers what the
  pairwise projection loses, the MDL backbone's description-length
  trade-off and its optimizers, the local MDL aggregation rules, the
  statistically validated hypergraphs and cores, the structural helpers, and
  a HIF round trip, with six figures in light and dark variants. It is
  listed under "By network type" in the tutorials index and linked from the
  hypergraph API page and the how-to tutorial (#93, #96).
- A shared figure style for the docs, ``docs/_figures/style.py`` (docs only,
  not library API): color tokens checked with a palette validator against the
  docs theme's light and dark backgrounds (kept and removed edges, positive
  and negative edges, a one-hue ramp for scores, eight categorical colors in a
  fixed order), matplotlib defaults, a cached layout per dataset, a helper
  that draws a backbone over its original graph with a legend and the counts,
  and helpers that save light and dark variants and write the reST that shows
  the one matching the reader's theme. Sphinx-Gallery examples start from
  this style (#88, #96).
- Two narrative guides, executed when the docs are built (#95): *Coming from
  R backbone or netbone* maps each R ``backbone`` 3.0 and netbone function to
  nx-backbone, checks the four convention switches against the R
  fixtures, and reproduces R's fixed row backbone of the Davis Southern Women
  p-value by p-value; *Scaling up to large networks* covers sparse inputs,
  cost classes with run times measured at build time, the ECM solver's
  degree-strength classes, and the committed nx-parallel timings.
- The method gallery on the API pages is regenerated and redesigned. The
  images committed for 0.2.0 showed pre-fix results (the ECM image kept
  nearly every edge, where the fixed ECM keeps 72 of 254). The gallery is now
  generated on every docs build by ``scripts/generate_visualizations.py`` into
  the gitignored ``docs/_gallery/``; no gallery image is committed any more,
  and the build fails when a public method has no figure. Each figure draws the
  backbone over its original graph (kept edges in blue, removed edges faint),
  in one layout per dataset, under the method's own name, with a legend, light
  and dark variants, and the edges and nodes kept in the caption and alt text.
  New figures cover ``polya_filter``, ``gloss_filter``, a signed backbone
  (``fixedrow(signed=True)``, negative edges dashed), the MDL hypergraph
  backbones on Davis Southern Women as a hypergraph, and the temporal scores
  (including ``edge_persistence_hmm``) on the temporal tutorial's snapshots. The graph comparison example
  now draws the statistical and structural methods as small multiples, with a
  bar chart of the edges each keeps. The "Generate Visualization Gallery"
  workflow became a strict docs build on pull requests, and the deploy
  workflow no longer runs a separate gallery step. Generating the gallery
  takes about 15 s; with the lighter comparison example, a full docs build
  takes about 35 s instead of about 27 s (#88).
- New narrative tutorial, *Statistical null models side by side*: the seven
  statistical scorers on Les Misérables, with figures of p-value against
  weight, backbone overlap (checked against the hierarchy reported by Yassin
  et al., 2025), the KS distances of each backbone, the disparity filter's
  ``a=``, the count models' ``unit=``, ``transform_weights``, and signed
  backbones with the R tie convention. It is executed at build time (#90,
  #96).
- A narrative tutorial, *Bipartite projections and their backbones*, executed
  when the docs are built: two-mode data and why the raw projection
  overstates ties, the five projection weights, the five null models compared
  against the FDSM, the FDSM p-value floor and multiple testing, signed
  backbones, SDSM-EC constraints (including contradictory ones), a check
  against R ``backbone`` with ``missing_as_zero``, and sparse incidence
  input, with seven figures in light and dark variants. It is listed under
  "By network type" in the tutorials index (#92, #96).
- New narrative tutorial, *Temporal backbones, from snapshots to
  smoothing*: snapshots, events and spans and binning with ``freq=``;
  ``flatten_temporal`` by count and by duration, and what a static backbone
  of the flattened graph misses; observation fraction, persistence and
  backbone stability on a persistent block among random flicker; the Kalman
  backbone on a planted rising block (smoother against filter, the
  edge-level test, the Šidák correction and temporal leakage); the
  persistence HMM's posterior against raw presence; and a table for choosing
  among them. Five figures, executed at build time; it is listed under "By
  network type" in the tutorials index (#94, #96).
- The structural and proximity and the hypergraph tutorials no longer embed
  their figures in the page as base64 data: the figures are written at build
  time to the gitignored ``docs/_static/example_figures/`` and linked, and
  load lazily, so a reader fetches only the variant (light or dark) on
  screen. The three dense network drawings of the structural tutorial are
  PNG; the charts stay SVG. The pages shrink from 3.9 MB and 1.5 MB of HTML
  to 0.4 MB and 0.2 MB, both tutorials get a real gallery thumbnail, and the
  structural tutorial's captions, which the theme hid in light mode, show
  again (#96).
- The public API and the ``HypergraphBackbone``/``ValidatedHypergraph``
  result classes have type annotations, and the package ships a ``py.typed``
  marker (PEP 561), so mypy and pyright check calls into it. Functions whose
  return type depends on a flag (``bicm(return_labels=...)``,
  ``read_hif``/``from_*`` with ``return_weights=...``,
  ``compare_graphs(return_diff=...)``) have overloads. The decorators
  (``not_implemented_for``, ``py_random_state``, the backend dispatcher) are
  typed so they keep each signature, and dispatchable functions also accept
  ``backend=...``. NumPy, SciPy, pandas,
  and Matplotlib types are imported only for type checking, so the core
  package still needs only NetworkX at runtime. A new CI job runs
  non-strict mypy with the ``types-networkx`` stubs; the new ``typing`` extra
  pins both. The API pages keep plain signatures
  (``autodoc_typehints = "none"``); the docstrings still give the types. No
  behavior changes (#18).
- The roadmap page opens with a "Roadmap at a glance" section: a stacked bar
  chart of item status per section, a network map of sections and their items
  colored by status, and the same counts as a table. ``scripts/roadmap_figures.py``
  generates them from ``docs/roadmap.rst`` on every docs build, in light and dark
  variants, so they always match the page (#56).
- Docstrings are validated by the Sphinx build and by a new ``numpydoc lint``
  CI step. The checks are GL06, GL08, SS01, PR01, PR02, PR03, RT01, and EX01,
  configured in ``[tool.numpydoc_validation]``. numpydoc alone now renders
  docstrings; ``sphinx.ext.napoleon`` was removed (#17).
- The Les Misérables benchmark table is computed at docs build time from
  ``benchmarks/benchmarks/les_miserables.py``, the same definitions whose
  counts the tests pin (#28).
- An asv benchmark suite in ``benchmarks/`` covers the statistical, bipartite,
  MDL and SVH/SVC hypergraph, and shortest-path methods. It comes with a
  ``benchmark`` extra and a manually triggered Benchmarks workflow (#22).
- Cross-implementation reference tests (``tests/test_reference_implementations.py``)
  compare p-values and backbones with R ``backbone`` 3.0.4 and netbone 0.2.3
  on Les Misérables, the karate club, and Davis Southern Women, using JSON
  fixtures written by ``scripts/make_reference_fixtures.R`` and
  ``scripts/make_netbone_fixtures.py`` (no R or netbone needed to run them).
  Disparity, MLF, LANS (one-tailed), BiCM, SDSM (RNA), fixed fill, fixed row,
  ``p.adjust``, local degree, metric and ultrametric backbones match exactly.
  SDSM on Davis finds no significant pair at alpha = 0.05 in R either
  (smallest p-value 0.097 in R, 0.088 here with the exact tail), and FDSM finds
  the same 5 pairs. Documented differences: signed disparity and LANS lower
  tails, ``lspar`` conventions, ``global`` threshold strictness, ``mtc``
  counting non-co-occurring pairs, R's logistic-regression SDSM-EC fit, and
  netbone's noise-corrected total, GLAB
  and salience tie handling, h-bridge, and MLA. R's ``fdsm`` default draws
  samples with too few curveball trades on Davis, which biases its p-values
  upward by up to about 0.03 (#14).
- The ``multiple_linkage_analysis`` documentation no longer says
  ``method="lans"`` reproduces netbone: netbone 0.2.3's ``mlam`` is a
  correlation-based fit that neither method reproduces (#14).
- A decision note (``docs/design/reference-conventions.md``) covers the 14
  documented convention differences from R ``backbone`` and netbone. It
  measures each on Les Misérables, the karate club, and Davis Southern Women
  and recommends whether to keep ours, match the reference, or add an option.
  The R fixture gains ``lspar`` backbones for Les Misérables and Davis
  backbones at alphas where the ``mtc`` scope matters; every existing fixture
  value is unchanged. The ``multiple_linkage_analysis`` Notes now compare
  netbone's 134 Les Misérables links with the default's 205 (223 was the
  ``deviation="profile"`` count), and the ``global_threshold_filter`` Notes
  show how to reproduce R's strict threshold (#59).
- CI on every push and pull request: ruff lint and format checks, tests and
  doctests on Python 3.12-3.14, a job pinned to the oldest supported
  versions, and a NetworkX-only job that checks the optional dependencies
  stay optional. Test coverage is 97%.
- A CI job (and a step in the oldest-versions job) runs the dispatch tests
  (``tests/test_dispatch.py`` and ``tests/test_api_contract.py``, which call
  every dispatchable function) under NetworkX's dispatch test harness,
  ``NETWORKX_TEST_BACKEND=nx_loopback``, which checks that every dispatchable
  call gives the same result through a backend as without one (#19). The
  whole suite also passes under the harness.
- The test suite was condensed from 1,732 to 804 collected tests (with
  doctests) at unchanged line and branch coverage: sweeps over a registry
  (every method, style, scorer or dispatchable function) are one test per
  axis whose failure message names every failing case, and duplicated tests
  were merged. Doctests and coverage run on one Python version in CI.
- An "Interop" workflow (pull requests, weekly, and on demand) installs
  ``.[interop,test]`` and runs ``pytest -m interop``: round trips of
  hyperedges, weights, and node labels through the real xgi, HyperNetX,
  HypergraphX, and HAT, and HIF exchange with each library's own reader and
  writer. It is a separate, non-required workflow, so an upstream release
  cannot block unrelated pull requests (#15).
- A coverage evaluation of the 15 methods compared by Yassin et al. (2025)
  (``docs/design/yassin-2025-coverage.md``). The package covered 13 of them;
  the two it lacked, the Pólya urn and GloSS filters, were then added (#42,
  #43). The evaluation also notes where the covered methods behave
  differently from ``netbone`` (#29).
- The PyPI workflow builds from the release tag with its full history and
  fails unless the wheel and sdist carry the tag's version; manual runs no
  longer upload (they had published ``0.2.0.post1.dev9``). The conda recipe
  takes its version from the tag instead of a hardcoded 0.1.0. The
  ``Development Status`` classifier is now ``4 - Beta`` (#16).
- New :doc:`roadmap`; full numpydoc sections for the bipartite, hypergraph,
  and structural APIs; corrected citations (Van Nuffel, 2007; Lü et
  al., 2009; Lindner et al., 2015); the docs version is taken from the package
  metadata.


Version 0.2.0 (2026-02-14)
--------------------------

Tagged ``v0.2.0`` in git. PyPI has 0.1.0 and a development build
(``0.2.0.post1.dev9``), but no 0.2.0 release: ``pyproject.toml`` at the tag
still set ``version = "0.1.0"``, so the release workflow built 0.1.0 again,
and the development build came from a later manual run on ``main``.

- New :mod:`~nx_backbone.visualization` module:
  :func:`~nx_backbone.graph_difference`,
  :func:`~nx_backbone.compare_graphs`, and
  :func:`~nx_backbone.save_graph_comparison` draw an original graph
  and its backbone side by side.
- A Sphinx-Gallery visual comparison gallery and score-then-filter examples
  in the documentation.

Version 0.1.0 (Initial Release)
--------------------------------

First public release of ``networkx-backbone``.

**Features:**

- 47 backbone extraction functions across 8 modules
- Statistical methods: disparity filter, noise-corrected filter, marginal
  likelihood filter, ECM filter, LANS filter
- Structural methods: global threshold, strongest N ties, high salience
  skeleton, metric backbone, ultrametric backbone, doubly stochastic filter,
  h-backbone, modularity backbone, planar maximally filtered graph, maximum
  spanning tree backbone
- Proximity methods: neighborhood overlap, Jaccard, Dice, cosine, hub
  promoted/depressed indices, LHN local index, preferential attachment,
  Adamic-Adar, resource allocation, graph distance proximity, local path index
- Hybrid methods: GLAB filter
- Bipartite methods: SDSM, FDSM
- Unweighted sparsification: sparsify, LSpar, local degree
- Filtering utilities: threshold, fraction, boolean, consensus backbone
- Evaluation measures: node/edge/weight fractions, reachability, KS degree,
  KS weight, compare backbones
