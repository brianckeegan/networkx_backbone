Roadmap
=======

This page tracks where ``nx-backbone`` is headed: what was finished
recently, known gaps against the cited methods, and planned features. It
consolidates the design notes in ``docs/design/`` (the
`Backbone 3.0 coverage evaluation
<https://github.com/brianckeegan/nx-backbone/blob/main/docs/design/backbone-3.0-coverage.md>`_,
the `Yassin et al. (2025) coverage evaluation
<https://github.com/brianckeegan/nx-backbone/blob/main/docs/design/yassin-2025-coverage.md>`_,
the `hypergraph backboning proposal
<https://github.com/brianckeegan/nx-backbone/blob/main/docs/design/hypergraph-backboning.md>`_,
the `generalized null distributions report
<https://github.com/brianckeegan/nx-backbone/blob/main/docs/design/generalized-nulls.md>`_,
and the `temporal backbones proposal
<https://github.com/brianckeegan/nx-backbone/blob/main/docs/design/temporal-backbones.md>`_).
Each open item links to its GitHub issue; `#31 <https://github.com/brianckeegan/nx-backbone/issues/31>`_ tracks them all.
Contributions are welcome. Please comment on an issue before starting one
of the larger items.

Status key: **done**, **next** (planned for the next release), **planned**,
**idea** (open for discussion).

Roadmap at a glance
-------------------

These figures and the table below are generated from this page every time the
documentation is built, so they always match the lists that follow.

.. figure:: _static/roadmap/status_by_section-light.svg
   :figclass: only-light
   :class: only-light
   :width: 100%
   :alt: Horizontal stacked bar chart of roadmap items per section, split by
         status (idea, planned, next, done).

   Roadmap items per section, by status.

.. figure:: _static/roadmap/status_by_section-dark.svg
   :figclass: only-dark
   :class: only-dark
   :width: 100%
   :alt: Horizontal stacked bar chart of roadmap items per section, split by
         status (idea, planned, next, done).

   Roadmap items per section, by status.

.. figure:: _static/roadmap/roadmap_map-light.svg
   :figclass: only-light
   :class: only-light
   :width: 100%
   :alt: Network diagram: each roadmap section is a gray hub connected to its
         items, which are colored by status and labeled with their GitHub
         issue numbers.

   Each section (gray) and its items, colored by status and labeled with their
   GitHub issue numbers.

.. figure:: _static/roadmap/roadmap_map-dark.svg
   :figclass: only-dark
   :class: only-dark
   :width: 100%
   :alt: Network diagram: each roadmap section is a gray hub connected to its
         items, which are colored by status and labeled with their GitHub
         issue numbers.

   Each section (gray) and its items, colored by status and labeled with their
   GitHub issue numbers.

.. include:: _static/roadmap/status_table.inc

Recently completed
------------------

Method coverage
~~~~~~~~~~~~~~~

- **done**: Every backbone *model* in Neal's Backbone 3.0 R package, plus its
  cross-cutting features: multiple-testing correction (``mtc``), signed
  backbones, and SDSM-EC edge constraints.
- **done**: Hypergraph backbones: the MDL backbone (Kirkley et al., 2026),
  statistically validated hypergraphs and cores (SVH/SVC), structural
  reductions, and HIF/xgi/HyperNetX/HypergraphX/HAT interoperability.

Correctness review (2026-11)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A full code review checked each method against its paper. The fixes below
change numerical results, so they are listed in the :doc:`changelog` as
breaking changes:

- ``ecm_filter``: the solver did not converge and every p-value was about 0.
  It is now a maximum-likelihood fit with a Newton polish, using the geometric
  tail :math:`p_{ij}(y_iy_j)^{w-1}`.
- ``marginal_likelihood_filter``: now Dianati's null model
  :math:`B(T, s_is_j/2T^2)`. The old one was asymmetric and depended on the
  order nodes were added.
- ``glab_filter``: now uses per-node shortest-path involvement and
  :math:`(1-I_{ij})^{(k_i-1)^c}` (Zhang et al., 2014) instead of global edge
  betweenness.
- ``disparity_filter`` on directed graphs: now also tests the target's
  incoming links (Serrano et al., 2009).
- Bipartite: the FDSM null now preserves degrees (a curveball chain with
  burn-in, and p-values of ``(count + 1) / (trials + 1)``). The BiCM is fitted
  exactly, with exact or RNA Poisson-binomial tails. SDSM-EC refits on the
  free cells. ``fixedfill`` and ``fixedcol`` use their exact null
  distributions. ``backbone_from_projection`` thresholds every method at
  ``alpha``.
- See the :doc:`changelog` for the full bipartite, hypergraph, and structural
  lists.
- Shared conventions: every scorer rejects multigraphs, rejects NaN or
  infinite weights, works with node labels of mixed types, and preserves
  graph, node, and edge attributes through the filters.
- CI now runs tests on every push and pull request (Python 3.12-3.14 and a
  NetworkX-only job), along with ruff lint and format checks and doctests.

Next release (0.3.0)
--------------------

Correctness gaps found in review
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **done** (`#9 <https://github.com/brianckeegan/nx-backbone/issues/9>`_): ``noise_corrected_filter`` is a binomial z-score. Implement the
  full Bayesian noise-corrected backbone of Coscia & Neffke (2017): the
  posterior variance from the beta prior plus the delta method, and the
  transformed lift score. Keep the current score as an option for backward
  compatibility.
- **done** (`#10 <https://github.com/brianckeegan/nx-backbone/issues/10>`_): The dominance-fitting multiple
  linkage analysis is implemented (``method="dominance"``). Its fit is the
  coefficient of determination, checked against Table 1 of Van Nuffel,
  Derudder & Witlox (2010); the whole-profile sum of squares and Weaver's
  variance are available through ``deviation=``. The 2007 paper itself is
  paywalled and was not read.
- **done** (`#11 <https://github.com/brianckeegan/nx-backbone/issues/11>`_, contributed in
  `#34 <https://github.com/brianckeegan/nx-backbone/pull/34>`_): Directed ``lans_filter`` now also tests the
  target's in-links.
- **done** (`#12 <https://github.com/brianckeegan/nx-backbone/issues/12>`_): A directed GLAB variant, using the out- and in-involvement of
  Zhang et al. (2014), Section 2.3.
- **done** (`#13 <https://github.com/brianckeegan/nx-backbone/issues/13>`_): Check ``hyper_projection``'s weighting (:math:`1/d(z)` versus
  Newman's :math:`1/(d(z)-1)`) and the arXiv identifier cited for the
  projection methods against Coscia & Neffke (2017).
- **done** (`#58 <https://github.com/brianckeegan/nx-backbone/issues/58>`_): The signed ``lans_filter`` lower tail is now
  ``#{weights < w} / k``, the mirror image of the upper tail, as in R. Its
  two-sided p-values equal R's, and since #59 so do the signs of exact ties.
  Les Misérables: 0 -> 126 negative links, as in R (40 of them ties).
- **done** (`#60 <https://github.com/brianckeegan/nx-backbone/issues/60>`_): ``strongest_n_ties`` keeps edges tied at each node's
  cutoff together, so ``G``, ``G.copy()``, and any reordering of the same
  graph give the same backbone. ``global_sparsification`` and
  ``edge_betweenness_filter`` no longer depend on edge orientation.
  ``planar_maximally_filtered_graph`` and ``maximum_spanning_tree_backbone``
  (ties in weight), ``fraction_filter`` (ties in score, documented), and
  ``modularity_backbone`` (Louvain visiting order, even with a fixed seed)
  still depend on input order.
- **done** (`#59 <https://github.com/brianckeegan/nx-backbone/issues/59>`_): Decide, for each convention difference from
  R ``backbone`` and ``netbone`` that the reference tests document, whether
  to keep it, match the reference, or add an option. The
  `decision note
  <https://github.com/brianckeegan/nx-backbone/blob/main/docs/design/reference-conventions.md>`__
  measures each difference on the reference datasets and records the
  decisions: every difference is kept and documented, except the sign of an
  exact tie between the two tails, which now matches R (``-1``), so every
  signed weighted backbone equals R's on the fixtures. R-compatible options
  are available without changing any default:
  ``backbone_from_projection(missing_as_zero=False)`` and
  ``lspar(convention="r")``.
- **done** (`#105 <https://github.com/brianckeegan/nx-backbone/issues/105>`_): ``noise_corrected_filter``'s p-values were
  far too small on low-count graphs (pure-noise edges at p ≈ 1e-66),
  because the paper's ``delta`` rule is not a calibrated test. ``nc_pvalue``
  is now the exact binomial tail of the method's null model (the paper's
  footnote 2), checked by simulation; the ``delta`` rule's p-value is
  ``pvalue_method="delta"``. Les Misérables: 195 -> 126 edges.
- **done** (`#108 <https://github.com/brianckeegan/nx-backbone/issues/108>`_): ``sdsm(pvalue_method="auto")`` computes
  exact Poisson-binomial tails whenever that is affordable, instead of the
  refined normal approximation above 200 artifacts, whose far upper tail
  let pure-noise pairs pass a BH correction on sparse data. Truncated
  convolutions make the exact tails 8-60x cheaper; over budget, the pairs
  with the smallest expected co-occurrence keep exact tails and a warning
  names the rest. ``pvalue_method="rna"`` still matches R ``backbone``.

Validation
~~~~~~~~~~

- **done** (`#14 <https://github.com/brianckeegan/nx-backbone/issues/14>`_): Cross-implementation reference tests. P-values and backbones
  on Les Misérables, Davis Southern Women, and karate are pinned against R
  ``backbone`` 3.0.4 and ``netbone`` 0.2.3 in JSON fixtures, so the test suite
  needs neither R nor netbone.
- **done** (`#15 <https://github.com/brianckeegan/nx-backbone/issues/15>`_): An optional "Interop" CI workflow that installs xgi, HyperNetX,
  HypergraphX, and HAT (the ``[interop]`` extra) and runs the adapter and HIF
  round-trip tests against the real libraries on pull requests and weekly.
  Supported versions are listed in :doc:`install`.
- **done** (`#17 <https://github.com/brianckeegan/nx-backbone/issues/17>`_): Turn on numpydoc validation (``numpydoc_validation_checks``)
  in the docs build so missing Parameters, Returns, or Examples sections fail
  CI.

Documentation, tutorials, and figures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Longer, narrative documentation with figures, tracked in
`#96 <https://github.com/brianckeegan/nx-backbone/issues/96>`_. Each tutorial is executed when the docs are built, so its
numbers and figures cannot drift from the code, and uses one colorblind-safe
figure style.

- **done** (`#88 <https://github.com/brianckeegan/nx-backbone/issues/88>`_): Regenerate and redesign the per-method gallery
  on the API pages, whose images dated from 0.2.0 and showed pre-fix results,
  and generate it at build time so it cannot go stale. Every figure uses the
  shared figure style (``docs/_figures/style.py``) that the examples and
  tutorials also use.
- **done** (`#89 <https://github.com/brianckeegan/nx-backbone/issues/89>`_): Narrative tutorial: from a hairball to a
  backbone, end to end (why a backbone, scoring vs filtering, choosing alpha
  and corrections, comparing, validating, reporting), the "Start here" entry
  point of the tutorials.
- **done** (`#90 <https://github.com/brianckeegan/nx-backbone/issues/90>`_): Narrative tutorial: the statistical null models
  side by side, including generalized nulls and signed backbones.
- **done** (`#91 <https://github.com/brianckeegan/nx-backbone/issues/91>`_): Narrative tutorial: structural and proximity
  backbones, and when topology-based guarantees beat significance tests.
- **done** (`#92 <https://github.com/brianckeegan/nx-backbone/issues/92>`_): Narrative tutorial: bipartite projections and
  their backbones.
- **done** (`#93 <https://github.com/brianckeegan/nx-backbone/issues/93>`_): Narrative tutorial: hypergraph backbones
  (groups versus pairs, the MDL and local MDL backbones, SVH and SVC, the
  structural helpers, and HIF interchange), executed with six figures.
- **done** (`#94 <https://github.com/brianckeegan/nx-backbone/issues/94>`_): Narrative tutorial: temporal backbones, from
  snapshots to smoothing.
- **done** (`#95 <https://github.com/brianckeegan/nx-backbone/issues/95>`_): Narrative guides: coming from R ``backbone`` or
  ``netbone``, and scaling up to large networks.

Release engineering
~~~~~~~~~~~~~~~~~~~

- **next** (`#16 <https://github.com/brianckeegan/nx-backbone/issues/16>`_): Release ``v0.3.0``, the first
  release with this sprint's correctness fixes, new methods, and changes in
  results. ``v0.2.0`` (February 2026) is tagged in git, but PyPI has
  only 0.1.0 and a development build: the tag still had a static
  ``version = "0.1.0"``. The publishing workflows now build exactly the
  release tag's version and refuse to upload anything else, and the release
  notes are drafted in ``docs/release-notes/v0.3.0.md``. 0.3.0 is the first
  release as ``nx-backbone`` (#119), so PyPI needs a pending trusted
  publisher for the new project. What remains is that setup, dating the
  changelog, and publishing the GitHub Release, which uploads to PyPI and
  anaconda.org.
- **done** (`#18 <https://github.com/brianckeegan/nx-backbone/issues/18>`_): Add ``py.typed`` and type annotations for the public API,
  checked by a non-strict mypy CI job.

Planned features
----------------

API and ecosystem compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **done** (`#19 <https://github.com/brianckeegan/nx-backbone/issues/19>`_): NetworkX backend dispatching (``@nx._dispatchable``) so
  backends such as ``nx-cugraph`` and ``nx-parallel`` can accelerate the
  shortest-path-heavy methods (salience, metric backbone, GLAB). The
  graph-in, graph-out functions dispatch as ``backbone_*``, the tests pass
  under ``NETWORKX_TEST_BACKEND=nx_loopback``, and :doc:`user_guide/backends`
  records nx-parallel timings.
- **done** (`#20 <https://github.com/brianckeegan/nx-backbone/issues/20>`_): Accept SciPy sparse arrays and pandas edge lists as input:
  bipartite functions take incidence matrices, and ``backbone``,
  ``backbone_from_weighted`` and ``backbone_from_unweighted`` take adjacency
  matrices and edge lists via :func:`~nx_backbone.to_graph`, matching
  how NetworkX interoperates with the NumPy/SciPy/pandas stack.
- **done** (`#30 <https://github.com/brianckeegan/nx-backbone/issues/30>`_): Make the bipartite and projection code fully sparse
  (no dense agent-by-artifact matrices). The per-pair loops are already
  vectorized.
- **done** (`#21 <https://github.com/brianckeegan/nx-backbone/issues/21>`_): A sparse or blockwise ECM solver for graphs with more than
  about 2,000 nodes. The fit now runs over degree-strength classes with
  chunked pair sums and Newton or Newton-CG steps.
- **done** (`#22 <https://github.com/brianckeegan/nx-backbone/issues/22>`_): An ``asv`` benchmark suite, so performance regressions show up in
  review.

Methods
~~~~~~~

- **done** (`#23 <https://github.com/brianckeegan/nx-backbone/issues/23>`_): ``describe_backbone`` (Backbone 3.0's ``narrative``): methods
  text and citations for a chosen backbone and its parameters.
- **done** (`#32 <https://github.com/brianckeegan/nx-backbone/issues/32>`_): ``sparsify(umst=True)`` should add the union of
  all minimum spanning trees (Hamann et al., 2016), not a single spanning
  forest.
- **done** (`#24 <https://github.com/brianckeegan/nx-backbone/issues/24>`_): The local (per-node-neighborhood) MDL hypergraph backbone
  (Kirkley et al., 2026, Appendix E): ``local_mdl_hypergraph_backbone``.
- **done** (`#25 <https://github.com/brianckeegan/nx-backbone/issues/25>`_): Signed variants of ``ecm_filter`` and the noise-corrected
  filter, for parity with the other scorers' ``signed=True``.
- **done** (`#26 <https://github.com/brianckeegan/nx-backbone/issues/26>`_): A ``missing_as_zero`` toggle for weighted inputs (Backbone 3.0),
  for testing absent dyads.
- **done** (`#29 <https://github.com/brianckeegan/nx-backbone/issues/29>`_): Survey the methods in the Yassin et al. (2025)
  statistical and structural comparisons. With #42-#44 the package covers
  all 15 methods; see the `coverage table
  <https://github.com/brianckeegan/nx-backbone/blob/main/docs/design/yassin-2025-coverage.md>`__.
- **done** (`#42 <https://github.com/brianckeegan/nx-backbone/issues/42>`_): The Pólya urn filter (Marcaccioli & Livan, 2019),
  missing per the #29 survey, as :func:`~nx_backbone.polya_filter`.
  Its null model is unlike any method here, and Yassin et al. (2025) found
  it best preserves the weight distribution.
- **done** (`#43 <https://github.com/brianckeegan/nx-backbone/issues/43>`_): The GloSS filter (Radicchi et al., 2011), missing per the #29
  survey: :func:`~nx_backbone.gloss_filter`, a global null with
  weights drawn from the observed weight distribution over the fixed
  topology.
- **done** (`#44 <https://github.com/brianckeegan/nx-backbone/issues/44>`_): The connectivity stage of Slater's (2009) doubly stochastic
  backbone (add edges by ``ds_weight`` until the backbone is connected), the
  variant Yassin et al. (2025) evaluate. ``doubly_stochastic_filter`` now
  flags it as ``ds_keep``.
- **done** (`#54 <https://github.com/brianckeegan/nx-backbone/pull/54>`_): ``doubly_stochastic_filter(smoothing=...)`` adds
  a pseudocount so Sinkhorn-Knopp converges on sparse graphs, and
  ``ds_weight`` no longer depends on node insertion order when it does not
  converge.
- **done** (`#55 <https://github.com/brianckeegan/nx-backbone/issues/55>`_): Relax the distributional assumptions of the
  statistical nulls: a ``transform_weights`` preprocessing helper with an
  invariance table, a generalized ``disparity_filter(a=...)`` share null with
  a calibration test harness, and an explicit count ``unit=`` for MLF, ECM,
  and the noise-corrected filter.  Not scheduled: a continuous-weight ECM,
  an overdispersed MLF, and user-supplied weight distributions for GloSS.
- **planned** (`#76 <https://github.com/brianckeegan/nx-backbone/issues/76>`_): ``fdsm`` Monte Carlo resolution
  diagnostics: warn when ``1/(trials+1)`` makes family-wise significance
  unreachable, and report the p-value's Monte Carlo uncertainty.
- **idea** (`#77 <https://github.com/brianckeegan/nx-backbone/issues/77>`_): ``fdsm(refine="splitting")``: adaptive multilevel
  splitting for p-values below the Monte Carlo floor.

Temporal backbones
~~~~~~~~~~~~~~~~~~

A new top-level module, ``nx_backbone.temporal``, for filtering networks
observed over time. It accepts snapshots (a list or dict of graphs),
timestamped events, and time spans equally, bins events and spans into steps,
and returns one scored static graph, so the usual filters apply. The design and
the method proposals are in the `temporal backbones proposal
<https://github.com/brianckeegan/nx-backbone/blob/main/docs/design/temporal-backbones.md>`_;
`#78 <https://github.com/brianckeegan/nx-backbone/issues/78>`_ tracks the work.

- **done** (`#67 <https://github.com/brianckeegan/nx-backbone/issues/67>`_): Foundation: the three input forms, binning
  with ``freq=``, the node-presence rule (an absent edge is zero only when both
  endpoints are present), and the scored-static-graph output: ``temporal_form``,
  lossless conversions between the forms, ``flatten_temporal``, the shared
  presence and eligibility masks, and the API-contract ``kind="temporal"``.
- **done** (`#68 <https://github.com/brianckeegan/nx-backbone/issues/68>`_): ``observation_fraction`` and ``persistence``
  (runs of consecutive observed steps).
- **done** (`#69 <https://github.com/brianckeegan/nx-backbone/issues/69>`_): ``backbone_stability`` (how often any static
  scorer keeps an edge across steps) and ``weight_trend``.
- **planned** (`#70 <https://github.com/brianckeegan/nx-backbone/issues/70>`_): Interop with pandas event tables, DyNetX,
  teneto, and pathpy. Done so far: pandas event tables
  (``from_event_table``, ``to_event_table``). Remaining: the DyNetX, teneto,
  and pathpy adapters.
- **idea** (`#71 <https://github.com/brianckeegan/nx-backbone/issues/71>`_): Null-model significance tests for the simple
  filters.
- **done** (`#72 <https://github.com/brianckeegan/nx-backbone/issues/72>`_): ``kalman_backbone``: Kalman smoothing of
  per-step edge significance, tested against an exact null, with an
  edge-level test across steps and ``kalman_snapshots`` for per-step graphs.
- **done** (`#73 <https://github.com/brianckeegan/nx-backbone/issues/73>`_): ``edge_persistence_hmm``: a two-state hidden
  Markov model for persistent edge activity, observing presence, per-step
  keep flags, or p-values.
- **idea** (`#112 <https://github.com/brianckeegan/nx-backbone/issues/112>`_): Edge classes or a mixture over edges for
  ``edge_persistence_hmm``, so that persistent and flickering edges need not
  share one chain. Done for 0.3.0: Notes on the trade-off between miss rate
  and switching rate, and an ``unobserved_active_share`` diagnostic in the
  fitted record; a Beta prior on the error rates (MAP-EM) was evaluated and
  not adopted.
- **idea** (`#74 <https://github.com/brianckeegan/nx-backbone/issues/74>`_): ``wavelet_denoise``: wavelet shrinkage of
  edge-weight time series before backboning.
- **idea** (`#75 <https://github.com/brianckeegan/nx-backbone/issues/75>`_): ``poisson_lift_filter``: a Gamma-Poisson
  state-space filter for count weights.

Documentation
~~~~~~~~~~~~~

- **done** (`#27 <https://github.com/brianckeegan/nx-backbone/issues/27>`_): A "validating a backbone" tutorial showing the
  ``compare_backbones``, ``consensus_backbone``, and ``mtc`` workflow end to
  end.
- **done** (`#28 <https://github.com/brianckeegan/nx-backbone/issues/28>`_): Turn the Les Misérables benchmark table into an executed
  Sphinx-Gallery example, so its edge counts cannot go stale.
- **done** (`#56 <https://github.com/brianckeegan/nx-backbone/pull/56>`_): Generate the "Roadmap at a glance" figures and
  table on this page from the roadmap itself on every docs build.
