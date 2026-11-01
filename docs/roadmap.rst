Roadmap
=======

This page tracks where ``networkx-backbone`` is headed: what was finished
recently, known gaps against the cited methods, and planned features. It
consolidates the design notes in ``docs/design/`` (the
`Backbone 3.0 coverage evaluation
<https://github.com/brianckeegan/networkx_backbone/blob/main/docs/design/backbone-3.0-coverage.md>`_,
the `Yassin et al. (2025) coverage evaluation
<https://github.com/brianckeegan/networkx_backbone/blob/main/docs/design/yassin-2025-coverage.md>`_,
and the `hypergraph backboning proposal
<https://github.com/brianckeegan/networkx_backbone/blob/main/docs/design/hypergraph-backboning.md>`_).
Each open item links to its GitHub issue; `#31 <https://github.com/brianckeegan/networkx_backbone/issues/31>`_ tracks them all.
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

- **done** (`#9 <https://github.com/brianckeegan/networkx_backbone/issues/9>`_): ``noise_corrected_filter`` is a binomial z-score. Implement the
  full Bayesian noise-corrected backbone of Coscia & Neffke (2017): the
  posterior variance from the beta prior plus the delta method, and the
  transformed lift score. Keep the current score as an option for backward
  compatibility.
- **done** (`#10 <https://github.com/brianckeegan/networkx_backbone/issues/10>`_): The dominance-fitting multiple
  linkage analysis is implemented (``method="dominance"``). Its fit is the
  coefficient of determination, checked against Table 1 of Van Nuffel,
  Derudder & Witlox (2010); the whole-profile sum of squares and Weaver's
  variance are available through ``deviation=``. The 2007 paper itself is
  paywalled and was not read.
- **done** (`#11 <https://github.com/brianckeegan/networkx_backbone/issues/11>`_, contributed in
  `#34 <https://github.com/brianckeegan/networkx_backbone/pull/34>`_): Directed ``lans_filter`` now also tests the
  target's in-links.
- **done** (`#12 <https://github.com/brianckeegan/networkx_backbone/issues/12>`_): A directed GLAB variant, using the out- and in-involvement of
  Zhang et al. (2014), Section 2.3.
- **done** (`#13 <https://github.com/brianckeegan/networkx_backbone/issues/13>`_): Check ``hyper_projection``'s weighting (:math:`1/d(z)` versus
  Newman's :math:`1/(d(z)-1)`) and the arXiv identifier cited for the
  projection methods against Coscia & Neffke (2017).
- **next** (`#58 <https://github.com/brianckeegan/networkx_backbone/issues/58>`_): Signed ``lans_filter`` counts the edge itself
  in its lower tail, so almost no edge is significantly weak (Les Misérables:
  0 negative links, against 126 in R). Mirror the upper tail, as R does.
- **next** (`#60 <https://github.com/brianckeegan/networkx_backbone/issues/60>`_): ``strongest_n_ties`` breaks weight ties by
  adjacency order, so ``G`` and ``G.copy()`` can give different backbones.
  Make the result independent of node order.
- **planned** (`#59 <https://github.com/brianckeegan/networkx_backbone/issues/59>`_): Decide, for each convention difference from
  R ``backbone`` and ``netbone`` that the reference tests document, whether
  to keep it, match the reference, or add an option.

Validation
~~~~~~~~~~

- **done** (`#14 <https://github.com/brianckeegan/networkx_backbone/issues/14>`_): Cross-implementation reference tests. P-values and backbones
  on Les Misérables, Davis Southern Women, and karate are pinned against R
  ``backbone`` 3.0.4 and ``netbone`` 0.2.3 in JSON fixtures, so the test suite
  needs neither R nor netbone. Open follow-up: whether the signed LANS lower tail
  should exclude the link itself, as R's does.
- **done** (`#15 <https://github.com/brianckeegan/networkx_backbone/issues/15>`_): An optional "Interop" CI workflow that installs xgi, HyperNetX,
  HypergraphX, and HAT (the ``[interop]`` extra) and runs the adapter and HIF
  round-trip tests against the real libraries on pull requests and weekly.
  Supported versions are listed in :doc:`install`.
- **done** (`#17 <https://github.com/brianckeegan/networkx_backbone/issues/17>`_): Turn on numpydoc validation (``numpydoc_validation_checks``)
  in the docs build so missing Parameters, Returns, or Examples sections fail
  CI.

Release engineering
~~~~~~~~~~~~~~~~~~~

- **next** (`#16 <https://github.com/brianckeegan/networkx_backbone/issues/16>`_): Release ``v0.3.0``, the first
  release with this sprint's correctness fixes, new methods, and changes in
  results. ``v0.2.0`` (February 2026) is tagged in git, but PyPI has
  only 0.1.0 and a development build. Publish to PyPI and conda-forge from the existing workflows.
- **planned** (`#18 <https://github.com/brianckeegan/networkx_backbone/issues/18>`_): Add ``py.typed`` and type annotations for the public API.

Planned features
----------------

API and ecosystem compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **done** (`#19 <https://github.com/brianckeegan/networkx_backbone/issues/19>`_): NetworkX backend dispatching (``@nx._dispatchable``) so
  backends such as ``nx-cugraph`` and ``nx-parallel`` can accelerate the
  shortest-path-heavy methods (salience, metric backbone, GLAB). The
  graph-in, graph-out functions dispatch as ``backbone_*``, the tests pass
  under ``NETWORKX_TEST_BACKEND=nx_loopback``, and :doc:`user_guide/backends`
  records nx-parallel timings.
- **done** (`#20 <https://github.com/brianckeegan/networkx_backbone/issues/20>`_): Accept SciPy sparse arrays and pandas edge lists as input:
  bipartite functions take incidence matrices, and ``backbone``,
  ``backbone_from_weighted`` and ``backbone_from_unweighted`` take adjacency
  matrices and edge lists via :func:`~networkx_backbone.to_graph`, matching
  how NetworkX interoperates with the NumPy/SciPy/pandas stack.
- **done** (`#30 <https://github.com/brianckeegan/networkx_backbone/issues/30>`_): Make the bipartite and projection code fully sparse
  (no dense agent-by-artifact matrices). The per-pair loops are already
  vectorized.
- **done** (`#21 <https://github.com/brianckeegan/networkx_backbone/issues/21>`_): A sparse or blockwise ECM solver for graphs with more than
  about 2,000 nodes. The fit now runs over degree-strength classes with
  chunked pair sums and Newton or Newton-CG steps.
- **done** (`#22 <https://github.com/brianckeegan/networkx_backbone/issues/22>`_): An ``asv`` benchmark suite, so performance regressions show up in
  review.

Methods
~~~~~~~

- **done** (`#23 <https://github.com/brianckeegan/networkx_backbone/issues/23>`_): ``describe_backbone`` (Backbone 3.0's ``narrative``): methods
  text and citations for a chosen backbone and its parameters.
- **done** (`#32 <https://github.com/brianckeegan/networkx_backbone/issues/32>`_): ``sparsify(umst=True)`` should add the union of
  all minimum spanning trees (Hamann et al., 2016), not a single spanning
  forest.
- **done** (`#24 <https://github.com/brianckeegan/networkx_backbone/issues/24>`_): The local (per-node-neighborhood) MDL hypergraph backbone
  (Kirkley et al., 2026, Appendix E): ``local_mdl_hypergraph_backbone``.
- **done** (`#25 <https://github.com/brianckeegan/networkx_backbone/issues/25>`_): Signed variants of ``ecm_filter`` and the noise-corrected
  filter, for parity with the other scorers' ``signed=True``.
- **done** (`#26 <https://github.com/brianckeegan/networkx_backbone/issues/26>`_): A ``missing_as_zero`` toggle for weighted inputs (Backbone 3.0),
  for testing absent dyads.
- **done** (`#29 <https://github.com/brianckeegan/networkx_backbone/issues/29>`_): Survey the methods in the Yassin et al. (2025)
  statistical and structural comparisons. With #42-#44 the package covers
  all 15 methods; see the `coverage table
  <https://github.com/brianckeegan/networkx_backbone/blob/main/docs/design/yassin-2025-coverage.md>`__.
- **done** (`#42 <https://github.com/brianckeegan/networkx_backbone/issues/42>`_): The Pólya urn filter (Marcaccioli & Livan, 2019),
  missing per the #29 survey, as :func:`~networkx_backbone.polya_filter`.
  Its null model is unlike any method here, and Yassin et al. (2025) found
  it best preserves the weight distribution.
- **done** (`#43 <https://github.com/brianckeegan/networkx_backbone/issues/43>`_): The GloSS filter (Radicchi et al., 2011), missing per the #29
  survey: :func:`~networkx_backbone.gloss_filter`, a global null with
  weights drawn from the observed weight distribution over the fixed
  topology.
- **done** (`#44 <https://github.com/brianckeegan/networkx_backbone/issues/44>`_): The connectivity stage of Slater's (2009) doubly stochastic
  backbone (add edges by ``ds_weight`` until the backbone is connected), the
  variant Yassin et al. (2025) evaluate. ``doubly_stochastic_filter`` now
  flags it as ``ds_keep``.
- **done** (`#54 <https://github.com/brianckeegan/networkx_backbone/pull/54>`_): ``doubly_stochastic_filter(smoothing=...)`` adds
  a pseudocount so Sinkhorn-Knopp converges on sparse graphs, and
  ``ds_weight`` no longer depends on node insertion order when it does not
  converge.
- **planned** (`#55 <https://github.com/brianckeegan/networkx_backbone/issues/55>`_): Relax the distributional assumptions of the
  statistical nulls: a ``transform_weights`` preprocessing helper with an
  invariance table, a generalized ``disparity_filter(a=...)`` share null with
  a calibration test harness, and an explicit count ``unit=`` for MLF, ECM,
  and the noise-corrected filter.

Documentation
~~~~~~~~~~~~~

- **done** (`#27 <https://github.com/brianckeegan/networkx_backbone/issues/27>`_): A "validating a backbone" tutorial showing the
  ``compare_backbones``, ``consensus_backbone``, and ``mtc`` workflow end to
  end.
- **done** (`#28 <https://github.com/brianckeegan/networkx_backbone/issues/28>`_): Turn the Les Misérables benchmark table into an executed
  Sphinx-Gallery example, so its edge counts cannot go stale.
- **done** (`#56 <https://github.com/brianckeegan/networkx_backbone/pull/56>`_): Generate the "Roadmap at a glance" figures and
  table on this page from the roadmap itself on every docs build.
