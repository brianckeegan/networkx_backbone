Changelog
=========

Unreleased (0.2.0)
------------------

This release adds hypergraph backbones and the remaining Backbone 3.0
features. It also fixes many correctness problems found in a full code review
that checked every method against its paper. **Several methods now return
different results**; see *Changes in results* below. Remaining review findings
are tracked in the :doc:`roadmap` and in GitHub issues #9-#31.

**Supported versions (breaking)**

- Requires **Python 3.12 or newer** and **NetworkX 3.4 or newer**, matching
  NetworkX's own support window. The optional dependency minimums now match
  NetworkX's too: NumPy 2.2, SciPy 1.15, Matplotlib 3.10. A CI job tests
  these minimums exactly.

**New features**

- Hypergraph module: the parameter-free MDL hypergraph backbone (Kirkley et
  al., 2026) with ``"edge"``, ``"node"``, and ``"auto"`` optimizers;
  statistically validated hypergraphs and cores (SVH/SVC); maximal
  hyperedges, order filtering, s-components, and intersection graphs.
- ``hypergraph_io``: ``hypergraph_to_bipartite``, the HIF interchange format,
  and adapters for xgi, HyperNetX, HypergraphX, and HAT.
- Backbone 3.0 parity: multiple-testing correction
  (:func:`~networkx_backbone.adjust_pvalues`, ``threshold_filter(mtc=...)``),
  signed backbones (``signed=True``), and SDSM-EC edge constraints.
- ``backbone_from_projection(mtc=...)``; ``sdsm(pvalue_method=...)``;
  ``fdsm(burn_in=..., thin=...)``; ``bicm(tol=..., max_iter=...)``.
- ``seed=`` on ``modularity_backbone``, ``sparsify``, and ``compare_graphs``.
- ``multigraph_to_weighted(aggregate=...)``, which can sum, average, or take
  the max/min of parallel-edge weights.
- ``local_path_index(weight=...)``; a ``h_bridge`` attribute from
  ``h_backbone``; ``statistically_validated_cores(max_groups=...)``.

- New :func:`~networkx_backbone.describe_backbone`: methods-section text and
  citations for a backbone, in text, Markdown, or BibTeX; the counterpart of
  Backbone 3.0's ``narrative`` (#23). It is backed by one method-to-BibTeX
  registry that covers every public scorer, and a test enforces that coverage.
- Added ``CITATION.cff``.
- New tutorial, "Validating a backbone": multiple-testing correction,
  ``compare_backbones``, consensus, signed backbones, and reporting (#27).
- ``ecm_filter`` and ``noise_corrected_filter`` accept ``signed=True`` (#25).
- Bipartite functions accept NumPy, SciPy sparse, or pandas incidence
  matrices with ``agent_labels``/``artifact_labels``, and ``agent_nodes`` is
  optional (#20, bipartite part).
- ``hyper_projection(newman=True)`` / ``projection="newman"`` gives Newman's
  :math:`1/(d(z)-1)` weighting, equal to
  ``nx.bipartite.collaboration_weighted_projected_graph``. The default
  hyperbolic weight, :math:`1/d(z)`, is unchanged and follows Coscia & Rossi
  (2019). Projection citations are corrected to Coscia & Rossi (2019,
  arXiv:1906.09081) (#13).
- ``glab_filter`` now supports directed graphs using the out- and
  in-involvement of Zhang et al. (2014), Section 2.3 (#12). On a symmetric
  ``DiGraph`` it gives the undirected p-values.

**Changes in results (bug fixes)**

- ``noise_corrected_filter`` now implements the published Bayesian
  noise-corrected backbone (Coscia & Neffke, 2017). Its scores and standard
  deviations match the authors' reference code exactly. ``nc_score`` is the
  transformed lift, in (-1, 1), with new ``nc_sdev`` and ``nc_pvalue``
  attributes. Keep edges with ``nc_score > 1.64 * nc_sdev`` or
  ``nc_pvalue < 0.05``; thresholding ``nc_score`` at 2.0 now keeps nothing.
  The old z-score is available as ``method="zscore"``. Les Misérables: 98 ->
  195 edges (#9).
- ``multiple_linkage_analysis`` now fits each node's sorted weights to ideal
  dominance distributions by default (``method="dominance"``; Van Nuffel,
  2007, after Weaver, 1954) and adds ``mla_rank``. The previous LANS
  threshold, which netbone uses, is ``method="lans"``. The fit is the
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

The Les Misérables counts below are edges kept at the documented threshold.
Davis counts are significant woman-woman pairs at alpha = 0.05.

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
- ``disparity_filter`` on directed graphs also tests the target's incoming
  links.
- ``lans_filter`` on directed graphs also tests the target's incoming links
  (#11, contributed by @bfs2021 in #34).
- ``bicm``/``sdsm``: the BiCM is now actually fitted rather than
  approximated with Chung-Lu, and p-values use exact Poisson-binomial tails
  (or RNA). SDSM-EC refits on the free cells. Davis: 3 -> 0 pairs.
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
  :func:`~networkx_backbone.multigraph_to_weighted` first), and
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

**Performance**

- Bipartite projections and null models use sparse incidence matrices and
  score agent pairs in row blocks, and ``backbone_from_projection`` builds
  edges only for significant pairs. On a 3000x800 projection this cuts run
  time from 40-76 s to under 2 s and peak memory from 2.2 GB to 0.23 GB;
  results are unchanged (#30).

- Vectorized ``sdsm``/``fixed*``/``fdsm`` (``sdsm`` on 400x300: 6.4 s ->
  0.36 s); MDL ``"node"`` optimizer with a lazy heap (m = 2000: 8.3 s ->
  0.47 s); vectorized SVC p-values.

**Infrastructure and documentation**

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

- CI on every push and pull request: ruff lint and format checks, tests and
  doctests on Python 3.12-3.14, a job pinned to the oldest supported
  versions, and a NetworkX-only job that checks the optional dependencies
  stay optional. Test coverage is 97%.
- A coverage evaluation of the 15 methods compared by Yassin et al. (2025)
  (``docs/design/yassin-2025-coverage.md``). The package covers 13 of them;
  the Pólya urn and GloSS filters are missing. The evaluation also notes where
  the covered methods behave differently from ``netbone`` (#29).
- New :doc:`roadmap`; full numpydoc sections for the bipartite, hypergraph,
  and structural APIs; corrected citations (Van Nuffel, 2007; Lü et
  al., 2009; Lindner et al., 2015); the docs version is taken from the package
  metadata.


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
