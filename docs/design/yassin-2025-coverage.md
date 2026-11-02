# Coverage evaluation: Yassin et al. (2025) statistical and structural comparisons

- **Status:** Evaluation and gap proposals. All three follow-ups (§8) are
  now implemented: #42 (Pólya), #43 (GloSS), and #44 (doubly stochastic
  connectivity stage).
- **Tracking:** GitHub issue #29; open items are listed in `docs/roadmap.rst`
- **Papers:**
  - Yassin, A., Cherifi, H., Seba, H., & Togni, O. (2025). Backbone extraction
    through statistical edge filtering: A comparative study. *PLoS ONE*,
    20(1), e0316141. https://doi.org/10.1371/journal.pone.0316141
    (cited below as **[S]**)
  - Yassin, A., Cherifi, H., Seba, H., & Togni, O. (2025). Exploring weighted
    network backbone extraction: A comparative analysis of structural
    techniques. *PLoS ONE*, 20(5), e0322298.
    https://doi.org/10.1371/journal.pone.0322298 (cited below as **[T]**)

> **Sources.** Both papers were read in full from the PLOS ONE JATS XML.
> Implementation details come from two other sources. The first is the source
> of `netbone` 0.2.3 (the PyPI sdist), which [T] says it used (Sect 3.2).
> The second is the companion repository
> `al1yass2n/structural-backbone-methods-comparison`. Its
> `run backbone methods.py` calls the eight `netbone` functions with their
> default extraction rules on undirected graphs read with
> `nx.read_weighted_edgelist`. The `netbone` code was read, not run. [S] does
> not name its implementation. `netbone` 0.2.3 contains five of the seven
> methods [S] tests; it has no Pólya urn or GloSS filter.

---

## 1. Summary

The two papers evaluate **15 methods**: 7 statistical in [S] and 8 structural
in [T]. **`nx-backbone` now implements all 15.** When this evaluation
was written, two were missing, the Pólya urn filter (PF) and the Global
Statistical Significance filter (GloSS), both statistical methods from [S].
They were added as `polya_filter` (#42) and `gloss_filter` (#43).

The covered methods do not always give the same backbone as the papers. In
most cases the cause is the **extraction rule**, not the edge score. This
library scores edges and leaves the cut to `threshold_filter`,
`fraction_filter`, or `boolean_filter`. The papers use `netbone`'s built-in
cut for each method. The differences that change results:

- **Doubly stochastic (DS):** [T] keeps edges in descending score order until
  the backbone is connected. That is the second stage of Slater (2009).
  `doubly_stochastic_filter` now flags it as `ds_keep` (#44). `netbone` also
  adds 1e-4 to every cell before scaling; `smoothing=1e-4` reproduces that
  step (#54), but not `netbone`'s matrix construction.
- **High salience skeleton (HSS):** `netbone` keeps salience > 0.8.
  `high_salience_skeleton` returns only the score. It also gives tied
  shortest paths fractional credit; `netbone` uses one arbitrary path tree
  per root.
- **h-backbone (HB):** `netbone`'s bridge value is half of ours. It counts
  unordered pairs, uses weighted betweenness with weights read as distances,
  and rounds to three decimals.
- **Noise-corrected (NC):** [S] uses the binomial "approximation" p-value.
  `noise_corrected_filter` defaults to an exact binomial p-value too (since
  #105), with n.. = 2W trials on the symmetric weight matrix and the usual
  tail P(X >= w).

Details are in Section 4. The assessments of PF and GloSS made before they
were implemented are in Section 5.

## 2. What the papers evaluated

| | [S] statistical (PLoS ONE 20(1)) | [T] structural (PLoS ONE 20(5)) |
|---|---|---|
| Methods | DF, PF, MLF, NC, ECM, GloSS, LANS | MSP, PMFG, PLAM, HB, MB, UMB, DS, HSS |
| Networks | 27 real-world weighted networks | 33 real-world weighted networks |
| Extraction rule | alpha = 0.05, no correction / Bonferroni / FDR-BH; edges with p-value > alpha removed | each method's own rule (`netbone` defaults) |
| Similarity | Spearman correlation of p-values; overlap coefficient | Jaccard; overlap coefficient |
| Edge properties | Spearman (p-value vs weight, degree, betweenness) | point-biserial (in-backbone vs weight, degree, betweenness, and the weighted versions) |
| Global properties | edge, node, and weight fraction; weight entropy; reachability; number of components; transitivity | same |
| Distributions | two-sample KS on weight and degree (networks with >= 1000 edges) | same |
| Implementation | not named | `netbone` 0.2.3, Python 3.10 |

Both papers cite the Yassin et al. (2023) `netbone` paper ([S] ref 31,
[T] ref 98). The README and the concepts page already cite it.

## 3. Coverage table

Each citation is reproduced as the paper gives it, including any DOI. Where a
paper gives no DOI, none is added here.

| Abbr. | Method | Paper | Citation as given in the paper | `nx-backbone` | Status |
|---|---|---|---|---|---|
| DF | Disparity filter | [S] ref 12 | Serrano MA, Boguna M, Vespignani A. Extracting the multiscale backbone of complex weighted networks. *PNAS*. 2009;106:6483–6488. doi:10.1073/pnas.0808904106 | `disparity_filter` (`disparity_pvalue`); also `disparity` / `backbone_from_weighted(model="disparity")` | Covered |
| PF | Pólya urn filter | [S] ref 13 | Marcaccioli R, Livan G. A Pólya urn approach to information filtering in complex networks. *Nature Communications*. 2019;10:745. doi:10.1038/s41467-019-08667-3 | `polya_filter` (`polya_pvalue`) | Covered (#42) |
| MLF | Marginal likelihood filter | [S] ref 15 | Dianati N. Unwinding the hairball graph: Pruning algorithms for weighted complex networks. *Physical Review E*. 2016;93. doi:10.1103/PhysRevE.93.012304 | `marginal_likelihood_filter` (`ml_pvalue`); also `mlf` | Covered |
| NC | Noise-corrected filter | [S] ref 14 | Coscia M, Neffke FMH. Network Backboning with Noisy Data. IEEE; 2017. p. 425–436. *(no DOI given)* | `noise_corrected_filter` (`nc_pvalue`, `nc_score`) | Covered; uses a different p-value by default (§4) |
| ECM | Enhanced configuration model filter | [S] ref 16 | Gemmetto V, Cardillo A, Garlaschelli D. Irreducible network backbones: unbiased graph filtering via maximum entropy; 2017. https://arxiv.org/abs/1706.00230 | `ecm_filter` (`ecm_pvalue`) | Covered |
| GloSS | Global statistical significance filter | [S] ref 17 | Radicchi F, Ramasco JJ, Fortunato S. Information filtering in complex weighted networks. *Physical Review E*. 2011;83:046101. doi:10.1103/PhysRevE.83.046101 | `gloss_filter` (`gloss_pvalue`) | Covered (#43) |
| LANS | Locally adaptive network sparsification | [S] ref 18 | Foti NJ, Hughes JM, Rockmore DN. Nonparametric Sparsification of Complex Multiscale Networks. *PLoS ONE*. 2011;6:e16431. doi:10.1371/journal.pone.0016431 | `lans_filter` (`lans_pvalue`); also `lans` | Covered |
| MSP | Maximum spanning tree | [T] ref 52 | Kruskal JB Jr. On the shortest spanning subtree of a graph and the traveling salesman problem. *Proc Am Math Soc*. 1956;7:48–50. doi:10.2307/2033241 | `maximum_spanning_tree_backbone` (`mst_keep`) | Covered |
| PMFG | Planar maximally filtered graph | [T] ref 69 | Tumminello M, Aste T, Di Matteo T, Mantegna RN. A tool for filtering information in complex systems. *PNAS*. 2005;102(30):10421–6. doi:10.1073/pnas.0500298102 | `planar_maximally_filtered_graph` (`pmfg_keep`) | Covered |
| PLAM | Primary linkage analysis | [T] ref 70 | Nystuen J, Dacey M. A graph theory interpretation of nodal regions. Papers of the Regional Science Association; 2005. p. 7. doi:10.1111/j.1435-5597.1961.tb01769.x | `primary_linkage_analysis` (`primary_linkage_keep`) | Covered |
| HB | h-backbone | [T] ref 71 | Zhang RJ, Stanley HE, Ye FY. Extracting h-backbone as a core structure in weighted networks. *Sci Rep*. 2018;8(1):1–7. doi:10.1038/s41598-018-32430-1 | `h_backbone` (`h_backbone_keep`, `h_bridge`) | Covered; bridge value differs from `netbone` (§4) |
| MB | Metric (distance) backbone | [T] ref 62 | Simas T, Correia RB, Rocha LM. The distance backbone of complex networks. *J Complex Netw*. 2021;9:cnab021. doi:10.1093/comnet/cnab021 | `metric_backbone` (`metric_keep`) | Covered |
| UMB | Ultrametric (distance) backbone | [T] ref 62 | (same as MB) | `ultrametric_backbone` (`ultrametric_keep`) | Covered |
| DS | Doubly stochastic filter | [T] ref 61 | Slater PB. A two-stage algorithm for extracting the multiscale backbone of complex weighted networks. *PNAS*. 2009;106(26):E66. doi:10.1073/pnas.0904725106 | `doubly_stochastic_filter` (`ds_weight`, `ds_keep`) | Covered; connectivity stage added in #44 (§4) |
| HSS | High salience skeleton | [T] ref 56 | Grady D, Thiemann C, Brockmann D. Robust classification of salient links in complex networks. *Nat Commun*. 2012;3:864. doi:10.1038/ncomms1847 | `high_salience_skeleton` (`salience`) | Covered; no built-in 0.8 cut (§4) |

Notes on the citations:

- [T] names MSP inconsistently. Table 1 and Sect 2.1 call it "Maximum
  Spanning Tree", and the method maximizes total weight. Sect 4.1 and the
  abstract say "Minimum Spanning Tree". `netbone` calls
  `nx.maximum_spanning_tree`, and so does `maximum_spanning_tree_backbone`.
- [S] cites Kruskal (1956) with a different DOI
  (10.1090/S0002-9939-1956-0078686-7, [S] ref 21) from [T]. [S] also cites
  Nystuen & Dacey (1961) without a DOI ([S] ref 22).

## 4. Behavioral differences for covered methods

The comparison point is the `netbone` 0.2.3 source, which is what [T] ran.
For [S] it is the best available reference, because [S] names no
implementation. Undirected graphs are assumed throughout, as in both papers.

### Statistical methods ([S])

- **DF**: No difference in the score. Both compute
  `(1 - w/s)^(k-1)` at each endpoint and take the minimum. A degree-1
  endpoint is skipped in `netbone` and contributes p = 1 here, which is the
  same thing. [S] states the minimum rule explicitly.
- **MLF**: No difference in the score. `netbone` uses
  `binom_test(w, n=W, p=s_u s_v / (2 W^2), alternative="larger")`, the same
  upper tail P(X >= w) that `marginal_likelihood_filter` computes. Both are
  meant for integer weights; this package rounds weights.
- **NC**: [S] says it uses the "approximation method" of the noise-corrected
  filter. In `netbone` that is `1 - binom.cdf(w, W, s_u s_v / W^2)`: an exact
  binomial with W trials (W = total weight) and p = s_u s_v / W^2. Note
  that this is a *strict* upper tail, P(X > w).
  `noise_corrected_filter(method="zscore")` is the normal approximation to
  the same binomial, using P(X >= w). The default `method="bayesian"` is the
  full Coscia & Neffke (2017) score and posterior variance (#9).
  `netbone` also implements the Bayesian variant (`approximation=False`), but
  only as `score`/`nc_sdev` with no p-value. No option here reproduces
  `netbone`'s exact-binomial NC p-value. This is a small gap. The `netbone`
  reference fixtures planned in #14 would show how much it matters.
  *Update (#105):* the default `nc_pvalue` is now the exact binomial tail
  P(X >= w) with n.. = 2W trials on the symmetric weight matrix; the
  Bayesian score's normal p-value is `pvalue_method="delta"`. On Les
  Misérables, netbone's p-value keeps 124 edges at 0.05 and ours 126, with
  24 and 26 edges kept by only one of the two.
- **ECM**: Same model (Gemmetto et al. 2017). `netbone` fits it with SciPy
  optimizers using derivatives computed by JAX. This package uses L-BFGS-B
  with an analytic gradient. P-values therefore agree only up to solver
  tolerance. Above about 2,000 nodes see #21.
- **LANS**: [S] describes the p-value as the chance of drawing an edge with
  weight *equal to or greater than* the observed one. Read literally, the
  strongest link of a degree-k node would get p = 1/k. `netbone` and
  `lans_filter` both compute `1 - F(w)` instead, where `F` is the share of the
  node's links with weight <= w. Under that rule the strongest link gets
  p = 0, which follows Foti et al. Both take the minimum over the two
  endpoints. `lans_filter` matches `netbone`; only the prose in [S] differs.
- **Thresholding**: [S] removes edges with p > alpha, so it keeps p <= alpha.
  `threshold_filter(..., mode="below")` keeps p < alpha. The two differ only
  when a p-value equals alpha exactly. [S] applies Bonferroni and FDR-BH,
  available here as `threshold_filter(mtc="bonferroni")` and
  `threshold_filter(mtc="bh")`.

### Structural methods ([T])

- **MSP**: Same algorithm. Both call `nx.maximum_spanning_tree`, so tie-breaking
  is NetworkX's in both.
- **PMFG**: Same greedy algorithm. Edges are added heaviest first, an edge is
  kept only if the graph stays planar, and the loop stops at 3(n - 2) edges.
- **PLAM**: Same rule: each node keeps its heaviest link. On ties `netbone`
  keeps the first maximum in adjacency order, and
  `primary_linkage_analysis` keeps the neighbor whose `str()` sorts largest.
  For directed graphs this package uses out-links, following Nystuen &
  Dacey. `netbone` (and [T]) only handle undirected graphs.
- **MB / UMB**: Both convert weights to distances with `d = 1 / w`. The
  `distanceclosure` code bundled in `netbone` also has a `prox2dist`
  helper, `d = 1/p - 1`, but `netbone`'s wrappers do not use it. An edge is
  kept when its distance equals the
  shortest-path (sum) or minimax (max) distance. `netbone` uses the
  `distanceclosure` code; this package uses Dijkstra with a relative
  tolerance of 1e-9 on ties.
- **HB**: Both take the union of the h-strength network and the h-bridge
  network. They compute the bridge value `b(e) = eb(e) / N` differently:
  - `netbone` calls `nx.edge_betweenness_centrality(G, weight="weight",
    normalized=False)`. For undirected graphs that counts unordered pairs. It
    also treats the weight as a **distance**, so heavy edges are *longer*.
    The result is rounded to 3 decimals.
  - `h_backbone` uses unweighted (topological) betweenness over **ordered**
    pairs, following Zhang et al. (2018) (see the 0.3.0 changelog).
  - On a path of three nodes, `netbone` gives each edge
    b = 2/3 ≈ 0.667 and `h_backbone` gives 4/3. Both the bridge values and
    the h-bridge threshold differ, so the backbones can differ.
- **DS**: `netbone` follows Slater's two-stage algorithm:
  1. It adds 1e-4 to *every* entry of the dense matrix, including
     non-edges, and runs Sinkhorn-Knopp normalization.
  2. It then adds edges in decreasing order of normalized weight until the
     backbone is one connected component covering all nodes. That set is
     its `in_backbone`.

  If Sinkhorn-Knopp does not converge in 1000 iterations, `netbone` returns
  an empty result. `doubly_stochastic_filter` does not smooth the matrix,
  warns when it does not converge (and keeps the last iterate), and returns
  only `ds_weight`. **The connectivity stage that [T] evaluates has no
  equivalent here.** A user can get close with `fraction_filter`, but no
  function stops at connectivity. This is the largest behavioral gap among
  the covered methods. *(Resolved in #44: `ds_keep` flags the connectivity
  stage; the docstring Notes list the remaining differences from `netbone`.)* [T] attributes DS's distinctive results to this
  method: it best preserves weight and degree distributions and
  transitivity.
- **HSS**: Both use one shortest-path tree per root, with `1 / w` as the
  distance, and divide the count by N. They differ in two ways:
  - **Ties:** `netbone` builds each tree from
    `nx.single_source_dijkstra` paths. It picks one predecessor per node, so
    the result depends on node order. `high_salience_skeleton` splits a tie
    over k predecessors as 1/k credit each.
  - **Cut:** `netbone` sets `in_backbone` for salience > 0.8.
    [T] describes the method as keeping edges with salience "near 1". In
    this package the cut is left to the user, for
    example `threshold_filter(H, "salience", 0.8, mode="above")`. That rule
    keeps salience >= 0.8, not > 0.8.

## 5. Missing methods (now implemented)

> **Update.** Both methods below have since been implemented and checked
> against their papers: `polya_filter` (#42) and `gloss_filter` (#43). Their
> docstrings record how the open questions listed here were resolved. This
> section is kept as the original assessment.

### 5.1 Pólya urn filter (PF)

- **Citation:** Marcaccioli R, Livan G. A Pólya urn approach to information
  filtering in complex networks. *Nature Communications*. 2019;10:745.
  doi:10.1038/s41467-019-08667-3 ([S] ref 13).
- **What it does:** [S] describes the null model as follows. Edge weights
  build up through repeated interactions, and a node spreads its strength
  over its links through a Pólya urn process. A reinforcement parameter sets
  how strongly past interactions attract future ones. Each edge gets a
  p-value from each endpoint, and [S] keeps the smaller of the two.
- **Why it matters (from [S]):** PF is the most distinctive of the seven
  statistical methods:
  - Its p-values correlate with no other method's (Fig 3).
  - Its p-values are almost uncorrelated with edge weight, degree, and
    betweenness (Figs 5–7).
  - Its backbone has the weight distribution closest to the original
    network (Fig 15).

  No method in this package fills that role. Whether `disparity_filter` is
  the special case at one value of the parameter, and where the two differ,
  is for an implementer to work out from the original paper.
- **Before implementing, check against Marcaccioli & Livan (2019):**
  - the exact null distribution of an edge weight (we expect a beta-binomial
    tail, but have not verified it);
  - how the reinforcement parameter is chosen (fixed by the user or fitted
    to the network);
  - how the two endpoint p-values are combined in the original;
  - the treatment of directed graphs.

  [S] does not say which implementation it used, and `netbone` 0.2.3 has
  none. The authors' reference code, if published, would be the obvious
  cross-check (#14).
- **Input type:** weighted, undirected (directed if the original paper
  defines it). Integer (count) weights, like MLF and ECM.
- **Dependencies:** SciPy (`scipy.stats.betabinom` if the beta-binomial
  form is confirmed), imported lazily like the other statistical scorers. No
  new dependency.
- **Complexity:** A few tail evaluations per edge, so O(m) apart from the
  special functions. If the parameter is fitted by maximum likelihood, add
  an O(m) optimization over one scalar.
- **API sketch:** `polya_filter(G, weight="weight", a=None, signed=False)`
  adding `polya_pvalue`, used with `threshold_filter` and `mtc`, like the
  other scorers. Register it with `describe_backbone`.
- **Priority: medium-high.** The paper is well cited (*Nat Commun*), the
  model is distinct, [S] finds behavior no other method here has, and the
  implementation is small.

### 5.2 Global statistical significance filter (GloSS)

- **Citation:** Radicchi F, Ramasco JJ, Fortunato S. Information filtering
  in complex weighted networks. *Physical Review E*. 2011;83:046101.
  doi:10.1103/PhysRevE.83.046101 ([S] ref 17).
- **What it does:** [S] describes the null model as the observed topology
  kept fixed, with link weights drawn at random from the network's own
  empirical weight distribution, "effectively shuffl[ing] the observed link
  weights over the existing fixed network topology". The null is therefore
  global (one weight distribution for the whole network), while the test is
  per edge.
- **Evidence from [S]:**
  - Of the seven methods, GloSS extracted a non-trivial backbone least often:
    44% of networks, with or without FDR-BH (Table 2).
  - It never produced a single-component backbone (Fig 13).
  - It deviated most from the original degree distribution (Fig 16).
  - Its backbone is nested in the others: GloSS ⊂ NC ⊂ PF ⊂ LANS ⊂ MLF
    without correction (Fig 4).
- **Before implementing, check against Radicchi et al. (2011):**
  - the exact test statistic, and whether it conditions on node strength
    and degree;
  - how the null distribution is computed (analytically, for example by
    convolving the empirical weight distribution, or by Monte Carlo);
  - how continuous weights are handled (binning).

  This document does not state these, because neither [S] nor any source we
  could run confirms them.
- **Input type:** weighted, undirected (directed per the original).
- **Dependencies:** NumPy, and SciPy if FFT convolution is used. No new
  dependency.
- **Complexity:** Probably the most expensive of the seven statistical
  methods. An analytic null needs the distribution of sums of up to k_max
  draws from the empirical weight distribution. A Monte Carlo null needs
  R shuffles of m weights, O(R m), and should take a `seed=`.
- **API sketch:** `gloss_filter(G, weight="weight", ..., seed=None)` adding
  `gloss_pvalue`.
- **Priority: low-medium.** Adding it completes the seven-method set in [S],
  and it is a known global-null baseline. [S] found it the least useful in
  practice, and it costs more to implement and validate than PF.

## 6. Methods mentioned but not evaluated

The related-work sections of [S] and [T] mention further methods without
benchmarking them. They are listed here for completeness.

| Method | Mentioned in | Citation as given | `nx-backbone` |
|---|---|---|---|
| Global weight thresholding | [S], [T] (Dai et al. 2018 comparison) | no single citation given | `global_threshold_filter` |
| k-core decomposition | [T] ref 51 | Alvarez-Hamelin J, Dall' Asta L, Barrat A, Vespignani A. Large scale networks fingerprinting and visualization using the k-core decomposition. *Advances in Neural Information Processing Systems* 18. MIT Press; 2005. | not a backbone function; use `networkx.k_core` |
| Multiple linkage analysis | [T] ref 54 | Rushton G, Haggett P, Cliff A, Frey A. Locational analysis in human geography. *Geograph Rev*. 1980;70:112. doi:10.2307/214380 | `multiple_linkage_analysis` |
| High salience skeleton, global thresholding | [T] (Gomes Ferreira et al. 2022) | see §3 | covered |
| RECAST | [T] ref 57 | Vaz de Melo POS, Viana AC, Fiore M, Jaffrès-Runser K, Le Mouël F, Loureiro AAF, et al. RECAST: Telling apart social and random relationships in dynamic networks. *Perform Eval*. 2015;87:19–36. doi:10.1016/j.peva.2015.01.005 | missing; needs temporal input, out of scope |
| Tripartite backbone extraction | [T] ref 58 | Gomes Ferreira CH, Murai F, Couto da Silva AP, de Almeida JM, Trevisan M, Vassio L, et al. Unveiling community dynamics on instagram political network. *Proceedings of the 12th ACM Conference on Web Science*; 2020. p. 231–40. | missing; tripartite input, out of scope |
| Stochastic degree sequence model | [T] ref 59 | Neal ZP, Domagalski R, Sagan B. Comparing alternatives to the fixed degree sequence model for extracting the backbone of bipartite projections. *Sci Rep*. 2021;11(1):23929. doi:10.1038/s41598-021-03238-3 | `sdsm` |
| Fixed fill/row/column, FDSM, SDSM (Neal et al. comparison) | [S] refs 24–27 | see [S] | `fixedfill`, `fixedrow`, `fixedcol`, `fdsm`, `sdsm` |
| GLANB | [T] ref 63 | Zhang X, Zhang Z, Zhao H, Wang Q, Zhu J. Extracting the globally and locally adaptive backbone of complex networks. *PLOS ONE*. 2014;9(6):e100428. doi:10.1371/journal.pone.0100428 | `glab_filter` |

RECAST and tripartite extraction need temporal or tripartite data that this
package does not model, so no follow-up is proposed for them.

## 7. Evaluation measures

Both papers use the same backbone-level measures. Coverage in
`nx_backbone.measures`:

| Measure ([S], [T]) | Here |
|---|---|
| Edge, node, weight fraction | `edge_fraction`, `node_fraction`, `weight_fraction` |
| Reachability (fraction of node pairs joined by a path) | `reachability` |
| KS statistic on weight and degree distributions | `ks_weight`, `ks_degree` |
| Number of components, transitivity | NetworkX (`nx.number_connected_components`, `nx.transitivity`) |
| Weight entropy (Shannon, normalized by the original's) | not provided |
| Jaccard / overlap coefficient between backbones | not provided; one line with Python sets |

Weight entropy and the overlap coefficient are small additions, but they are
evaluation helpers, not methods, so they are not proposed as issues here.

## 8. Proposed follow-ups

All three are done: #42, #43, and #44.

1. **Pólya urn filter** (§5.1). Medium-high priority.
2. **GloSS filter** (§5.2). Low-medium priority.
3. **Doubly stochastic connectivity stage** (§4). Not a missing method, but
   it is the step [T] evaluates. It could be an option on
   `doubly_stochastic_filter` (for example a boolean `ds_keep` from the
   "add edges until connected" rule) or a small helper.

Cross-implementation checks of the remaining differences (NC approximation,
HB bridge scaling, HSS ties) belong with the `netbone` reference fixtures in
#14.
