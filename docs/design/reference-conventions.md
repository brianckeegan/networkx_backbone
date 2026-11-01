# Convention differences from R `backbone` and `netbone`

- **Status:** Decision note. The maintainer decides the defaults. Two
  opt-in options are implemented with defaults unchanged
  (`backbone_from_projection(missing_as_zero=False)`,
  `lspar(convention="r")`). Nothing else changes any result.
- **Tracking:** GitHub issue #59; roadmap entry in `docs/roadmap.rst`.
  The signed LANS lower tail is issue #58 and is not decided here, but item 2
  depends on it.
- **References:** R `backbone` 3.0.4 (CRAN) and `netbone` 0.2.3 (PyPI sdist).
  The fixtures are `tests/fixtures/r_backbone.json` and
  `tests/fixtures/netbone.json`, and the tests are in
  `tests/test_reference_implementations.py`.

> **How the numbers were measured.** Every count below was computed on
> 2026-11-01 from the fixtures and the current code: Les Misérables (77 nodes,
> 254 edges), the karate club (34 nodes, 78 edges) and Davis Southern Women
> (18 women, 153 pairs). Significance is at alpha = 0.05 unless stated
> otherwise. "Backbones identical" means that the same edges are kept with
> the same signs for every `mtc` in `none`, `bonferroni`, `holm` and `BH`.
> The R source of `backbone` 3.0.4 and the Python source of `netbone` 0.2.3
> were read to confirm each convention. R 4.3.3 with `backbone` 3.0.4 was run
> for item 3, item 4 and item 8. Regenerating `r_backbone.json` with it
> reproduces the existing fixture byte for byte, and the two additions
> (below) leave every existing value unchanged. `netbone` itself was not run
> for this note. For item 14 its doubly stochastic code was rewritten line
> by line and run with the current pandas.

---

## Decision table

| # | Difference | Measured impact on the fixtures | Recommendation |
|---|------------|---------------------------------|----------------|
| 1 | Signed disparity lower tail | p-values differ on 23/78 (karate) and 102/254 (Les Mis) edges. Without `mtc`, the backbones are identical for every alpha up to 0.27. | **Keep**, documented |
| 2 | Sign at exact ties | Today no retained edge is affected. After #58, 40 Les Mis and 15 karate LANS links would change sign. | **Decide together with #58**. Leaning toward matching R. |
| 3 | `mtc` counts non-co-occurring pairs | Davis: 14 of 153 pairs never co-occur, which scales adjusted p-values by up to 1.10. No backbone changes at 0.05. | **Keep the default**. Option added: `missing_as_zero=False`. |
| 4 | `lspar` conventions | Karate: R keeps 30/38/50 and we keep 54/61/70. Les Mis: R keeps 101/148/194 and we keep 154/174/217. | **Keep the default**. Option added: `convention="r"`. |
| 5 | `global` keeps `w >= t` (R: `w > t`) | Les Mis: 157 vs 107 edges at t = 2. Every integer threshold matters. | **Keep**, documented |
| 6 | SDSM-EC fit | p-values differ by up to 0.086. The backbones are identical (all empty). | **Keep** |
| 7 | `fixedcol` exact tail (R: RNA) | p-values differ by up to 0.0075 one-tailed and 0.015 two-sided. The backbones are identical. | **Keep** |
| 8 | `fdsm` mixing | R's default p-values sit up to 0.027 above a well-mixed reference. Both find the same 5 pairs. | **Keep**, no action |
| 9 | Noise-corrected total (netbone: W) | At delta = 1.64, ours keeps 68 vs 43 (karate) and 195 vs 139 (Les Mis). netbone's edges are a subset of ours. | **Keep** |
| 10 | GLAB tie weighting | p-values differ on 71/254 (Les Mis) and 59/78 (karate) edges. Backbones at 0.05 differ by at most 1 edge. | **Keep** |
| 11 | HSS tie splitting | Salience differs on 46 (Les Mis) and 40 (karate) edges. At salience >= 0.5 they keep 76 vs 76 and 29 vs 28. | **Keep** |
| 12 | h-bridge | Karate: ours keeps 15 and netbone 11 (9 shared). Les Mis: 18 vs 18 (13 shared). | **Keep** |
| 13 | MLA | netbone keeps 134 (Les Mis) and 45 (karate). We keep 205 and 76 (default `cod`), all of netbone's included. | **Keep**. Docstring count corrected. |
| 14 | Doubly stochastic | netbone scores only 101 of the 254 Les Mis edges. Rewritten from its source, it keeps 101 vs our 220 (96 shared). | **Keep**. Not worth matching. |

**Issue figures that did not check out** (details in each section):

- Item 1 says "identical on all three datasets". Only Les Misérables and
  karate are weighted; Davis has no disparity fixture. On those two the
  backbones are identical, but only because neither implementation keeps any
  negative edge at alpha = 0.05.
- Item 2 gives "the only link of a degree-1 node" as the example. It is one
  of three cases, and on karate the least common: 1 of 15 tied links. A node
  whose links all have equal weight accounts for 13 of 15.
- Item 8 gives "0.165 with 700, 0.157 with 7,000". Rerunning R gives 0.161
  with 700 trades. With 7,000 trades three runs give 0.166, 0.162 and 0.163.
  The well-mixed value is about 0.162, not 0.157, so p does not keep falling
  after about 700 trades.
- Item 12 says netbone's bridge value "is half of ours". The factor of two
  only accounts for netbone counting unordered pairs. netbone also uses
  weighted shortest paths, so the per-edge ratio is not 2: the median is 2.08
  on karate and 1.99 on Les Misérables.
- Item 13: verified (134, 223 with `deviation="profile"`, 205 with the
  default). The `multiple_linkage_analysis` docstring still said "134 versus
  223" for the default, and now gives 205.

All other issue numbers were verified, within rounding: 40 of 126, 30 vs 54,
0.09 and 0.008.

---

## R `backbone` 3.0.4

### 1. Signed disparity lower tail

**Difference.** R computes the lower tail as `1 - min(α_u, α_v)`, the
complement of the upper tail. `tests/test_reference_implementations.py`
asserts `lower == 1 - upper` for every R edge. That equals
`max(1 - α_u, 1 - α_v)`: an edge is weak only if it is weak *for both*
endpoints. We apply the OR rule to each tail, so ours is
`min(1 - α_u, 1 - α_v)`: weak for *either* endpoint, which mirrors how
"strong" is decided. Our lower tail is therefore never larger than R's.

**Impact.**

| | karate | Les Mis |
|---|---|---|
| Edges whose two-sided p differs | 23 / 78 | 102 / 254 |
| Largest difference | 0.45 | 0.70 |
| Sign differs (R +1, ours −1) | 6 | 26 |
| Smallest negative-edge p (ours / R) | 0.484 / 0.572 | 0.268 / 0.399 |
| Signed backbones at 0.05 (any `mtc`) | identical (0 edges) | identical (5 edges, all +1) |
| Signed backbones at alpha = 0.1, 0.2 (no `mtc`) | identical | identical |
| Signed backbones at alpha = 0.5 (no `mtc`) | ours 16 (2 −1), R 14 (0 −1) | ours 115 (51 −1), R 75 (11 −1) |

The disparity null rarely makes an edge significantly weak, so at
conventional alphas the two rules give the same backbone. Davis is bipartite
and has no disparity fixture.

**Options.**

- *Keep* (cost: none). The p-values stay different from R's for the 6/26
  edges whose sign differs. This is already documented in the
  `disparity_filter` Notes and asserted by `test_r_disparity_signed`.
- *Match R* (cost: about 5 lines plus test and pin updates). It changes
  `disparity_pvalue` and `sign` for `signed=True` only. It makes the two
  tails asymmetric: OR for strong, AND for weak.
- *Opt-in* (cost: a new parameter on `disparity_filter` and
  `backbone_from_weighted`, for a difference that does not show up below
  alpha = 0.27 on these data).

**Recommendation: keep.** Ours treats the two tails symmetrically. The
difference never changes a backbone at a usable alpha here, and an option
would add API for no visible gain.

### 2. Sign at exact ties

**Difference.** R's `.retain()` sets `upper_tail <- (p$upper < p$lower)`, so
an exact tie is signed −1. Our `_combine_tails` (weighted models) and
`_signed_pvalues` (projections) use `p_hi <= p_lo`, so a tie is +1. The
two-sided p-value is the same either way.

**When R's tails tie.** In R's LANS on Les Misérables, 48 edges have
`upper == lower`. There are three cases:

- 17 have a degree-1 endpoint.
- 15 touch a node whose links all have equal weight.
- 8 are the strongest link of one endpoint and the weakest of the other.

The other 8 fit none of these cases. 40 of the 48 ties have p = 0. The
other 8 have p between 0.18 and 0.67. On karate, 13 of 15 ties are equal-weight nodes,
1 is a degree-1 node and 1 is strongest-at-one, weakest-at-other. Every tie
with p = 0 is significant, and R calls it significantly *weak*:

| | karate | Les Mis |
|---|---|---|
| R negative LANS links (signed, every `mtc`) | 40 | 126 |
| … of which exact ties | 15 | 40 (verified) |
| Our positive LANS links vs R's | 45 vs 30 | 108 vs 68 |

The surplus of positive links (15, 40) is exactly those ties.

**Impact today: none.** With our current LANS lower tail (#58) the tails
rarely tie. Our own tails tie on 1 karate LANS edge, 1 Les Mis LANS edge
and 4 Davis `fixedrow` pairs, all with p >= 0.05. Disparity, MLF, SDSM,
`fixedfill` and `fixedcol` have no ties. Switching to R's rule today changes
no retained edge. It flips the `sign` attribute of those 6 non-significant
edges and pairs.

**Impact after #58.** If #58 mirrors R's lower tail, our LANS tails tie
wherever R's do, and this rule decides whether 40 Les Mis and 15 karate
links are +1 or −1.

**Options.**

- *Keep +1* (cost: none). After #58, `test_r_lans_signed` would still need
  to exclude ties.
- *Match R, −1* (cost: one character in each of two helpers, plus the test
  helpers `assert_signs` and `test_r_lans_signed`, which could then assert
  equality). Signed LANS backbones would become directly comparable to R's.
- *Sign 0, or do not retain ties*. A link that is both a node's strongest
  and its weakest is not evidence in either direction, but no reference
  does this, and it adds a third sign value.

**Recommendation: decide together with #58, leaning toward matching R.**
On its own the rule changes nothing today. With #58 it is the last
difference between our signed LANS and R's. Neither +1 nor −1 is more
correct for a tie, so matching the reference is the cheap tiebreaker.

### 3. `mtc` scope in `backbone_from_projection`

**Difference.** R's projection models leave a pair that never co-occurs
untested when `missing_as_zero = FALSE`, the default. Its p-value is `NA`,
`p.adjust(..., n = sum(!is.na(p)))` does not count it, and it is never
retained, not even as a significantly weak pair with `signed = TRUE`. We test
every pair, which matches R's `missing_as_zero = TRUE`.

**Impact on Davis.** 14 of the 153 pairs never co-occur. The adjusted
p-values of co-occurring pairs grow by up to 153/139 = 1.10 under our
default. For example, the smallest Bonferroni-adjusted `fixedfill` p-value
is 0.0643 for all pairs and 0.0584 for co-occurring pairs. No backbone in
the original fixture changes, because every `mtc` backbone at 0.05 is empty
in both conventions. No never-co-occurring pair is significant at 0.05.

To make the difference testable, the R fixture gained `davis.missing_as_zero`.
It holds R's `fixedfill` and `fixedrow` backbones with
`missing_as_zero = FALSE` and `TRUE` at alpha = 0.06, 0.6 and 0.8, for both
signs and the four `mtc` settings. 13 of the 48 settings differ between the
two scopes. For example, `fixedfill` with Bonferroni at 0.06 keeps 1 pair
under R's default and 0 under ours. `fixedfill` signed at 0.6 without `mtc`
keeps 46 vs 60, because the 14 extra pairs are significantly weak pairs that
never co-occur. Without `mtc`, an unsigned backbone does not depend on the
scope (a never-co-occurring pair has upper-tail p = 1).

**Implemented (opt-in).** `backbone_from_projection(..., missing_as_zero=False)`
(keyword-only; the default `True` keeps current results).
`test_r_projection_missing_as_zero` shows that `False` reproduces all 48 of
R's `FALSE` backbones and `True` all 48 of R's `TRUE` backbones exactly. The
name and meaning are R's own, and `backbone_from_weighted` already uses the
same name.

**Options.**

- *Keep the default `True`, with the option* (done; cost: none). The default
  is the conservative one: it counts more tests, and it keeps significantly
  weak never-co-occurring pairs when `signed=True`.
- *Flip the default to `False`* (cost: a change in results under **Changes in
  results**, though none at alpha = 0.05 on the fixtures). It would make both
  `backbone_*` wrappers default to R's behavior. Note that the defaults then
  agree in name but differ in effect: `backbone_from_weighted`'s
  `missing_as_zero=False` adds no pairs.

**Recommendation: keep the default** and keep the option. The default is
the stricter correction. Users who need R's numbers now have a one-argument
switch.

### 4. `lspar` conventions

**Difference.** R's `backbone_from_unweighted(model = "lspar")`, read from
`functions_unweighted.R`, differs from ours in four ways:

1. The Jaccard denominator is `|N(u) ∪ N(v)| - 2` (the endpoints are excluded).
2. Ranks are dense.
3. Edges with Jaccard 0 get rank 0. They are never kept and do not count
   toward `d'_u`.
4. A node keeps ranks `<= floor(d'_u ** s)`, with no minimum of one.

Ours: denominator `|N(u) ∪ N(v)|`, competition ranks over all edges, and
`max(1, ceil(d_u ** s))`.

**Impact.**

| | s = 0.3 | s = 0.5 | s = 0.7 |
|---|---|---|---|
| karate, R / ours | 30 / 54 | 38 / 61 | 50 / 70 |
| karate, R edges not in ours | 0 | 1 | 1 |
| Les Mis, R / ours | 101 / 154 | 148 / 174 | 194 / 217 |
| Les Mis, R edges not in ours | 1 | 8 | 12 |

Most of the difference comes from `floor` versus `max(1, ceil)`. Switching
that alone takes karate at s = 0.3 from 54 to 32 and Les Mis from 154 to
125. Ignoring zero-Jaccard edges is next (karate has 11, Les Mis 22).
Applied alone, the denominator and dense ranks change little, or even add
edges, but together with the other two they decide the last few edges. All
four are needed to reproduce R.

**Implemented (opt-in).** `lspar(G, s, convention="r")`, which also works
through `backbone_from_unweighted(..., convention="r")`. The default is
`convention="satuluri"`, which is unchanged. It reproduces R's backbone
exactly on karate and on the binary Les Misérables graph at s = 0.3, 0.5
and 0.7. Les Misérables was added to the fixture as `les_miserables.unweighted`.
`sparsify_score` is the critical exponent of R's rule, so `score <= s`
reproduces any other `s`.

**Options.**

- *Keep the default, with the option* (done; cost: about 40 lines).
- *Make R's conventions the default* (cost: a change in results; for
  example, the `EXPECTED_EDGES` pin for `lspar` in `tests/test_les_miserables.py`
  would change). It would also break the equivalence
  `lspar == sparsify(escore="jaccard", ...)`, unless `sparsify` gained the
  same conventions.

**Recommendation: keep the default** and keep the option. The default is the
rule the `lspar` docstring attributes to Satuluri et al.; the paper was not
re-read for this note. Under R's rule a node can keep no edges at all. A
degree-1 node's only edge has Jaccard 0, so neither endpoint keeps it, and a
leaf drops out of the backbone. Our `max(1, ...)` keeps it.

### 5. `global` threshold strictness

**Difference.** R keeps `w > t`. `global_threshold_filter` keeps `w >= t`,
consistent with `threshold_filter(mode="above")`.

**Impact.** All weights in both weighted datasets are integers, so every
integer threshold is affected:

| t | Les Mis `w > t` / `w >= t` | karate `w > t` / `w >= t` |
|---|---|---|
| 1 | 157 / 254 | 72 / 78 |
| 2 | 107 / 157 | 48 / 72 |
| 3 | 72 / 107 | 21 / 48 |
| 4 | 51 / 72 | 9 / 21 |

`test_r_global_threshold` reproduces R's backbones with
`threshold=math.nextafter(t, math.inf)`.

**Options.**

- *Keep* (cost: none). The `nextafter` recipe already reproduces R.
- *Match R* (cost: a change in results for every integer threshold, and an
  inconsistency with `threshold_filter`).
- *Opt-in `strict=True`* (cost: small, but it duplicates what `nextafter`
  does).

**Recommendation: keep.** The `global_threshold_filter` Notes now give the
`nextafter` recipe.

### 6. SDSM-EC fit

**Difference.** R fits the probabilities of the unconstrained cells with a
logistic regression on the row and column sums. We refit the BiCM on those
cells, so expected degrees equal observed ones exactly.

**Impact.** The p-values of co-occurring pairs differ by up to 0.086, with a
mean of 0.024 and a correlation of 0.993, comparing RNA with RNA. With the
default exact tail the largest difference is 0.088 (issue: "up to 0.09",
verified). The backbones are identical for all 8 signed/`mtc` settings, all
empty.

**Options.** *Keep* (cost: none). *Match R* (cost: implementing R's
regression fit, a second, approximate null model).

**Recommendation: keep.** The constrained BiCM is the maximum-entropy model
that SDSM-EC describes. R's regression only approximates it.

### 7. `fixedcol` exact tail

**Difference.** `fixedcol` computes the exact Poisson-binomial tail. R uses
the refined normal approximation (RNA).

**Impact.** All 153 pairs differ. The largest one-tailed difference is
0.0075 (issue: 0.008, verified) and the largest two-sided difference 0.015.
No sign differs, and the backbones are identical for all 8 settings.

**Options.** *Keep* (cost: none). *Opt-in `pvalue_method="rna"`*, as `sdsm`
has (cost: small; it would make `fixedcol` testable for equality with R).

**Recommendation: keep.** Exact is better. An RNA option is cheap if exact
R parity for `fixedcol` is ever wanted, but nothing needs it now.

### 8. `fdsm` mixing

**Difference.** R's `fdsm()` draws each sample independently with
`fastball(M, trades = 5 * nrow(M))`, confirmed in `functions_util.R`. On
Davis, rotated to 14 event rows, that is 70 trades. We run one curveball
chain.

**Impact.** The default p-values are biased upward: by up to 0.027
(Pearl Oglethorpe–Myra Liddel, 0.385 vs 0.358), and by 0.024 for Ruth
DeSand–Dorothy Murchison. Rerunning R for that pair (10,000 trials each):

| Trades | p |
|---|---|
| 70 (5 per row, R default) | 0.187 |
| 700 | 0.161 |
| 1,400 (fixture's `fdsm_mixed`) | 0.163 |
| 7,000 | 0.166; reruns 0.162, 0.163 (30,000 trials each) |
| ours, 50,000 curveball trials | 0.162 |

The issue's 0.19 is verified, its 0.165 at 700 trades is within Monte Carlo
error, and its 0.157 at 7,000 trades was not reproduced. The well-mixed value
is about 0.162. Both R variants and ours find the same 5 pairs at 0.05.

**Recommendation: keep; no action** (as the issue says).

---

## netbone 0.2.3

### 9. Noise-corrected total

**Difference.** netbone sums each undirected edge once for the total `W` but
at both endpoints for the strengths. Ours uses `2W`, so the strengths and the
total count the same thing. With netbone's total the score and
standard-deviation formulas are identical (`test_netbone_noise_corrected`).

**Impact.** Scores differ on every edge, by up to 0.34. The backbone,
`score > delta * sdev`:

| delta | karate ours / netbone | Les Mis ours / netbone |
|---|---|---|
| 1.28 | 70 / 48 | 209 / 150 |
| 1.64 | 68 / 43 | 195 / 139 |
| 2.32 | 60 / 40 | 177 / 122 |

netbone's edges are a subset of ours in every case.

**Recommendation: keep.** For an undirected graph the symmetric weight
matrix sums to `2W`, which is the total `n..` in the formula of the
`noise_corrected_filter` docstring. netbone's `W` halves every lift.

### 10. GLAB tie weighting

**Difference.** netbone weights a target by the number of tied shortest
paths. We use the paper's average over targets of the fraction of paths
through the link.

**Impact.**

| | Les Mis c = 0.5 | Les Mis c = 1.0 | karate c = 0.5 | karate c = 1.0 |
|---|---|---|---|---|
| Edges whose endpoints see no tie | 131 | 131 | 0 | 0 |
| p-values differing | 71 | 70 | 59 | 59 |
| Largest difference | 0.086 | 0.136 | 0.046 | 0.046 |
| Kept at 0.05, ours / netbone | 56 / 56 | 59 / 59 | 9 / 9 | 16 / 17 |

Equality on tie-free edges is verified on Les Misérables. Karate has no
tie-free edge, so the test checks Les Misérables only.

**Recommendation: keep.**

### 11. High salience skeleton

**Difference.** netbone credits the single path Dijkstra returns, so its
result depends on node order. We split the credit over tied predecessors.

**Impact.** Salience differs on 46 Les Mis edges (largest difference 0.048)
and 40 karate edges (0.43). Both lie within the tie bounds the test checks.
At salience >= 0.5 they keep 76 vs 76 edges (Les Mis) and 29 vs 28
(karate).

**Recommendation: keep.** Ours does not depend on node order.

### 12. h-backbone bridge

**Difference.** netbone's `h_bridge` is `round(eb_w / N, 3)`. `eb_w` is the
edge betweenness over *unordered* pairs, with the *weights as distances*,
so heavy edges count as long. Ours is `2 * eb / N`, where `eb` is the
unweighted betweenness over ordered pairs, following Zhang et al. (2018).
Both formulas were checked against the fixture's `h_bridge` values.

**Impact.** The per-edge ratio ours/netbone has a median of 2.08 (karate)
and 1.99 (Les Mis), not exactly 2. The h-strength part agrees. Backbones:
karate 15 (ours) vs 11 (netbone), 9 shared; Les Mis 18 vs 18, 13 shared.

**Recommendation: keep.** netbone's weighted distances invert the meaning
of the weights.

### 13. Multiple linkage analysis

**Difference.** netbone's `mlam` fits the ideal distributions by the squared
correlation, rounded to two decimals, and stops at the first fit that does
not improve.

**Impact.**

| | netbone | `cod` (default) | `profile` | `weaver` | `method="lans"` |
|---|---|---|---|---|---|
| Les Mis (254) | 134 | 205 | 223 | 253 | 109 |
| karate (78) | 45 | 76 | 76 | 76 | 45 |

Every netbone link is kept by all three deviations. The
`multiple_linkage_analysis` docstring compared netbone's 134 with 223, the
count for `deviation="profile"`. It now gives 205 for the default.

**Recommendation: keep.** The default follows the restatement by Van
Nuffel et al. (2010) that the docstring cites, which takes the cycle with
the highest coefficient of determination. netbone instead rounds and stops
at the first fit that does not improve.

### 14. Doubly stochastic

**Difference** (from `netbone/structural/doubly_stochastic.py`):

- netbone pivots the edge list one way, as source × target rather than a
  symmetric adjacency matrix, and adds 1e-4 to every cell of that pivot.
- It scales until the standard deviation of the row sums is below 1e-12,
  and returns an empty frame after 1,000 rounds.
- It keeps only cells with `source < target` (label order) and merges them
  back onto the original edge list. Edges stored with `source > target` are
  therefore dropped.
- It adds edges in descending score until the graph is connected.

`smoothing=1e-4` (#54) reproduces the pseudocount but not the construction.

**Impact** (netbone's code rewritten line by line, not netbone itself):

| | edges scored by netbone | netbone keeps | ours, `smoothing=1e-4` | shared |
|---|---|---|---|---|
| karate | 78 / 78 | 55 | 55 | 42 |
| Les Mis | 101 / 254 | 101 | 220 | 96 |

On Les Misérables, the label-order filter discards 153 edges before
scoring. netbone's backbone then never becomes connected and keeps all
101 edges. With `smoothing=0`, ours does not converge on either graph (it
warns) and keeps 45 (karate) and 187 (Les Mis).

**Recommendation: keep; do not add a netbone mode.** netbone's result
depends on how the edges are stored. A reference test is not useful until
netbone fixes this.

---

## Summary of changes made for #59

- `backbone_from_projection(missing_as_zero=...)`: keyword-only, default
  `True` (unchanged). Reference test: `test_r_projection_missing_as_zero`.
- `lspar(convention="r")`: default `"satuluri"` (unchanged). Reference test:
  `test_r_lspar_convention_r`.
- `scripts/make_reference_fixtures.R` and `r_backbone.json`: added
  `les_miserables.unweighted` (R `lspar`) and `davis.missing_as_zero`.
  Every existing value is unchanged.
- The `multiple_linkage_analysis` Notes give the correct default count (205).
- No default changed and no pinned number changed.
