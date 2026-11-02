# Design proposal: generalizing null distributions and weight preprocessing

- **Status:** Accepted in part. Tracking: GitHub issue #55.
- **Scope:** Research report. No library code was changed by this document. It
  was written before `polya_filter` (#42) and `gloss_filter` (#43) merged and
  before the ECM solver rewrite (#21); the #55 work re-verifies any number it
  relies on.

---

## Original report: Generalizing the distributional assumptions of nx-backbone

**Request (user):** "Several of the filtering methods assume an underlying distribution on the degree distributions or edge weights. Can these assumptions be softened or generalized to permit alternative distributions or preprocessing steps, like those present in scipy.stats and sklearn.preprocessing?"

**Short answer:** Yes, but only in a few specific, derivable places. A generic "fit any distribution and test" filter is not sound.

1. Add an opt-in preprocessing helper, `transform_weights`, that works with any `fit_transform` object (duck typing) or any plain callable. Document, method by method, whether the method's null hypothesis survives the transform.
2. Give `disparity_filter` one principled extra parameter: the Dirichlet concentration (`a = 1/c`). This is the large-strength limit of the Pólya filter. `a=1` is the current filter, and `a="ml"` fits it.
3. Make the count unit explicit (`unit=`) for the count-based nulls (MLF, ECM, NC), and stop coercing non-integer weights silently.
4. Later, once GloSS is merged, let it take a user-supplied scipy distribution in place of its empirical weight law. The paper's own null is "i.i.d. weights from a given law", so this swap is coherent.
5. Do not add a generic `distribution_filter`. With a global fit it is just a relabeled global threshold. With a per-node fit it is miscalibrated at typical degrees, as the simulations below show.

**Scope and provenance.**
- This was evaluated on `main`. The checkout advanced to `81f90fb` while I worked; that commit adds `missing_as_zero=` to the weighted scorers and `deviation="cod"` to MLA. The re-run checks gave identical numbers.
- `polya_filter` and `gloss_filter` are **not on main**. They exist only on the branches `claude/issue-42-polya-filter` and `claude/issue-43-gloss-filter`. I read their docstrings there and describe them as pending work.
- Every numerical claim marked "(checked)" comes from simulation scripts run during the research (not committed; the #55 calibration harness is their tested successor). They were run with Python 3, SciPy 1.18.1, NumPy 2.5.3 and scikit-learn 1.9.1. scikit-learn is installed in this environment, but it is not a dependency of the package.

---

## 1. Inventory: the assumption each scorer makes

Legend:
- **Weights in.** What the code accepts and how it coerces values.
- **Scale.** Results are invariant to `w → c·w`, c > 0.
- **Monotone.** Results are invariant to any strictly increasing `f(w)`.
- Invariance was checked on Les Misérables with ×10, √w and w² unless marked "by construction" or "untested".

### 1a. Statistical scorers (`statistical.py`)

| Scorer | Null / distributional assumption | Weights in | Scale | Monotone |
|---|---|---|---|---|
| `disparity_filter` (Serrano et al. 2009) | Per endpoint, the k shares `w/s` are uniform spacings of [0,1] ("k−1 points … uniform", Serrano eq. 1). So one share is **Beta(1, k−1)**, and `p = (1−w/s)^(k−1)`. Each endpoint is tested separately and the edge p-value is the min over endpoints (OR rule). Equivalent generative reading (my derivation, checked): i.i.d. **exponential** weights conditioned on node strength (Gamma–Dirichlet construction). | Positive, finite floats. No rounding. | yes | **no** (√w: 9→1 edges; w²: 9→26) |
| `noise_corrected_filter` (Coscia & Neffke 2017) | Weights are sums of unit interactions: binomial `N_ij ~ Bin(n.., P_ij)`. The paper's prior on `P_ij` is a Beta fitted by moments to a **hypergeometric** mean and variance. The score uses the delta method and a **normal** tail. `method="zscore"` is the legacy binomial z. | Positive floats. **Not rounded**, but `n..` is treated as a number of trials. | **no** (×10: 195→222) | no |
| `marginal_likelihood_filter` (Dianati 2016) | Integer weights are counts of unit edges whose endpoints are drawn ∝ strength. This gives **Bin(T, s_u s_v / 2T²)**. Dianati: "we assume all weights to be positive integers". | `round(w)`, Python round-half-even. A weight below 0.5 becomes 0, is dropped from `T`, and gets p = 1 silently (checked: w=0.4 → p=1.0). | **no** (×10: 126→215; ×0.1: 126→6) | no |
| `ecm_filter` (Gemmetto et al. 2017) | Maximum-entropy ensemble fixing ⟨k⟩ and ⟨s⟩ ("Bose–Fermi"). Given a link, the weight is **geometric**: `P(w ≥ w*) = p_ij (y_i y_j)^(w*−1)`. The paper "assume[s] integer weights". | `max(round(w), 1)`, so 0.4 becomes 1. This is **inconsistent** with MLF, which turns 0.4 into 0. | **no** (×10: 72→89) | no |
| `lans_filter` (Foti et al. 2011) | **Nonparametric.** p = 1 − F̂(w) over the node's own weights; Foti: "distribution-free". Not a calibrated p-value: every node's strongest link always gets p = 0. | Positive floats. | yes | **yes**, by construction and checked; holds even per node. |
| `multiple_linkage_analysis` (Van Nuffel 2007) | No probability model. Least-squares fit of the share profile to ideal k-equal-link profiles (`deviation="cod"/"profile"/"weaver"`). `method="lans"` uses LANS. | Positive floats. | yes | no |

### 1b. Pending branches (not on main)

| Scorer | Null / distributional assumption | Weights in | Scale | Monotone |
|---|---|---|---|---|
| `polya_filter` (Marcaccioli & Livan 2019) | Per endpoint, **beta-binomial**(s, 1/a, (k−1)/a). `a=0` gives Bin(s, 1/k); `a=1` is approximately disparity. `a=None` fits a by maximum likelihood (paper's Suppl. Note 5). Non-integer weights use the paper's large-strength approximation (Eq. 6). | Exact path needs integers and raises otherwise. | exact path: no; Eq. 6 path depends only on w/s, so yes | no |
| `gloss_filter` (Radicchi et al. 2011) | Topology fixed. Weights i.i.d. from the **empirical** P_obs(w), conditioned on both endpoints' strengths through convolutions. | Floats placed on a lattice (`resolution=`). | not run | no |

### 1c. Structural and hybrid scorers (`structural.py`, `hybrid.py`)

| Scorer | Null / distributional assumption | Weights in | Scale | Monotone |
|---|---|---|---|---|
| `global_threshold_filter` | None. | Any number. | Equivariant: the threshold must be transformed too. | Equivariant |
| `strongest_n_ties`, `global_sparsification`, `primary_linkage_analysis`, `maximum_spanning_tree_backbone`, `planar_maximally_filtered_graph` | None. These are rank or order based. | Any number (PMFG accepts negatives). | yes | **yes** (checked) |
| `ultrametric_backbone` | None. Minimax paths on 1/w. | Positive. | yes | **yes** (checked) |
| `metric_backbone`, `high_salience_skeleton`, `edge_betweenness_filter`, `glab_filter` | None. Shortest paths on **sums** of 1/w. GLAB also has a "uniform involvement" null, `(1−I)^((k−1)^c)`, applied to path involvement rather than to weights. | Positive. | yes | no |
| `doubly_stochastic_filter` | None. Sinkhorn scaling (invariant to D₁WD₂). | Nonnegative. | yes | no |
| `h_backbone` | None. The h-index of the raw weights: the integer h-strength depends on units. | Any number. | **no** | no |
| `modularity_backbone` | None. | Weighted modularity. | untested | untested |
| `node_degree_filter` | None. Ignores weights. | — | n/a | n/a |

### 1d. Proximity and unweighted scorers (`proximity.py`, `unweighted.py`)

| Scorer | Null / distributional assumption | Weights in | Scale | Monotone |
|---|---|---|---|---|
| `proximity.py` local indices; `sparsify`, `lspar`, `local_degree` | None. Topological only; weights are ignored. | — | n/a | n/a |
| `local_path_index(weight=…)` | None. | Uses A² + εA³ on the weights. | **no** (the mix of the A² and εA³ terms changes with scale) | no |

### 1e. Bipartite (`bipartite.py`)

| Scorer | Null / distributional assumption | Weights in | Scale | Monotone |
|---|---|---|---|---|
| `sdsm` (+ `bicm`) | BiCM maximum-entropy **Bernoulli** cells. Co-occurrence is **Poisson-binomial**, computed exactly or by RNA (Hong 2013). | **Binary only.** Matrices must be 0/1 or a ValueError is raised; graph edges count as 1. The `weight=` argument is unused. | n/a | n/a |
| `fdsm` | Exact degree sequences. Monte Carlo over curveball samples, p = (count+1)/(trials+1). | Binary. | n/a | n/a |
| `fixedfill` / `fixedrow` / `fixedcol` | Uniform over matrices with fixed fill, rows (**hypergeometric**) or columns (**Poisson-binomial**). | Binary. | n/a | n/a |

### 1f. Hypergraph (`hypergraph.py`)

| Scorer | Null / distributional assumption | Weights in | Scale | Monotone |
|---|---|---|---|---|
| `mdl_hypergraph_backbone`, `local_mdl_hypergraph_backbone` | MDL. Weights have an empirical-Bayes **Poisson or geometric** prior, selected with `prior=`. This is the library's existing precedent for a family switch. | Integers ≥ 1. | no | no |
| `statistically_validated_hypergraph` / `_cores` | **Binomial** `Bin(N, Π d_i/N)` per order. | Integer multiplicities; **raises** on non-integers (checked). This is the better precedent compared with MLF's silent rounding. | no | no |

### Facts behind the table (all checked)

**Disparity is exact only for exponential weights.** On an 8-regular graph with i.i.d. weights, I measured the per-endpoint rejection rate at nominal 0.05:

| Weight law | P(p ≤ 0.05) |
|---|---|
| exponential | 0.053 (KS 0.004) |
| Gamma(0.3) | 0.121 |
| lognormal(σ=1.5) | 0.099 |
| Gamma(3) | 0.004 |

With the matching Beta(c, (k−1)c) null, calibration is exact for both Gamma(0.3) and Gamma(3): 0.051, KS 0.005. This is consistent with:
- Radicchi et al. 2011, Fig. 1: disparity departs from uniform p-values for i.i.d. weights;
- Serrano et al. SI ("Networks with uncorrelated weights"): with uncorrelated weights, disparity reduces to a global threshold at ⟨ω⟩ ln α⁻¹, which is the exponential-tail form.

**The OR rule inflates type-I error.** Taking the min over endpoints is anti-conservative even when the null is exactly right: 0.081 at nominal 0.05 for exponential weights. This applies to every two-endpoint scorer (disparity, LANS, Pólya, GLAB). It is pre-existing, but any calibration test has to target the per-endpoint tails.

**The Pólya urn tends to the Dirichlet.** The beta-binomial(s, 1/a, (k−1)/a) tail at w ≈ 0.4s converges to `Beta(1/a,(k−1)/a).sf(w/s)`. With k=10, a=2:

| s | Beta-binomial tail |
|---|---|
| 10 | 0.0784 |
| 1000 | 0.03717 |
| 10⁵ | 0.036791 |

The Beta limit is 0.036787. Marcaccioli & Livan (Methods, below Eq. 8) state this explicitly: the disparity null "is ruled by a particular case of the Dirichlet distribution, which is known to be a limit case of the Beta-Binomial distribution as the number of draws goes to infinity".

**Count nulls are unit-dependent.** MLF, NC and ECM change under rescaling (numbers in the table). The unit is an implicit modeling choice today.

**Tie-breaking is copy-sensitive (incidental).** `strongest_n_ties` gives different edges on `G.copy()` than on `G` for Les Mis: 23 edges differ, and the identity transform does the same. `G.copy()` reorders adjacency and the method breaks ties by neighbor order. With `copy.deepcopy` the method is exactly invariant to ×10, √ and w². This matters for any helper that returns a transformed copy (P1).

---

## 2. Can each assumption be generalized? Three kinds of change

### (a) Swapping the null family

This changes the hypothesis being tested, so it needs a derivation.

| Assumption | Principled generalization? | Status |
|---|---|---|
| Disparity's uniform split, Beta(1,k−1) | **Yes.** Symmetric-Dirichlet shares give Beta(c, (k−1)c) per share. Generative reading: i.i.d. Gamma(c, θ) weights conditioned on node strength; θ cancels, so the method stays scale-free. It is the s→∞ limit of Pólya with a = 1/c (cited). Only the **symmetric** one-parameter family is admissible: exchangeable links force E[share] = 1/k, so d = (k−1)c. A general "share distribution" Beta(c, d) is not admissible (§5). | Proposal P2 |
| MLF's binomial over unit edges | **Partly.** For overdispersed counts, use a beta-binomial with a global intra-class correlation ρ: `W_uv ~ BetaBin(T, p_uv(1−ρ)/ρ, (1−p_uv)(1−ρ)/ρ)`. This keeps E[W_uv] = T p_uv, so strengths are preserved in expectation, and inflates the variance by `1+(T−1)ρ`. A negative binomial is the Poisson-limit analogue. **Neither is in Dianati 2016.** This would be a new method and needs its own write-up. The Pólya filter already provides a per-node overdispersion family. | P6 (defer) |
| ECM's geometric weights | **Yes, as a separate model.** The continuous maximum-entropy model with degree and strength constraints gives a conditional **exponential** weight (derivation in P5). Gemmetto et al. assume integers ("without loss of generality"). Continuous variants exist in the literature (e.g. Gabrielli et al. 2019; Parisi et al. 2020 "CReMa"), but I did **not read** them, so treat that pointer as **unverified**. | P5 (research) |
| NC's binomial with Beta prior | Coscia & Neffke fix the prior's moments from a hypergeometric argument. Another prior family would be a different method. The only sound generalization is to make the count unit explicit. | P3 |
| GloSS's empirical P_obs | **Already general.** Radicchi et al. state the null for any weight law ("once their distribution is defined one can always assign to each weight a proper likelihood"). A user-supplied parametric law fits naturally. | P4 |
| Pólya | Already a one-parameter family that tolerates heterogeneity. Nothing to add beyond the continuous Beta limit (P2). | — |
| LANS, MLA, rank- or path-based structural methods | No distribution to generalize. | — |
| SDSM / BiCM / FDSM / FFM / FRM / FCM | The **binary** model *is* the null. Valued bipartite data needs a different null model; that is out of scope here. | — |
| MDL hypergraph | Already has `prior={"poisson","geometric"}`. | — |

### (b) User-supplied scipy.stats distributions for tail p-values

Coherence depends on what the distribution is conditioned on.

- **Inside a model that already has a law slot** (GloSS P_obs, or the share law in disparity). Here a frozen distribution, or a one-parameter family fitted globally, is coherent. The test still conditions on degrees and strengths, so it remains a network null.
  - Fitting one or two global parameters on m ≫ 1 observations barely double-dips. Checked: disparity with a fitted ĉ gave P(p≤.05) = 0.050–0.053 and KS ≤ 0.008, even with n = 60 nodes. The Pólya branch already fits a_ML the same way.
- **As a raw marginal test `p = dist.sf(w)` on each edge.** This is incoherent as a backbone test; see §5.
  - **Global fit.** The p-value is monotone in w, so at any α the kept set is exactly a weight threshold. Checked: a global lognormal fit on Les Mis kept every edge with w ≥ 9 and dropped every edge with w ≤ 8.
  - **Per-node fit.** At typical degrees this double-dips or is miscalibrated. Checked, with i.i.d. lognormal weights and a normal fit on log w:
    - k = 4, in-sample fit: the smallest attainable p is 0.043, so the test is conservative and nothing can ever pass α = 0.01.
    - Leave-one-out plug-in: anti-conservative, 0.147 at k = 4 and 0.087 at k = 8.
    - Leave-one-out predictive t with k−2 degrees of freedom: calibrated (≈0.05). But that is a Grubbs-style outlier test, not a network null, and it only works for location-scale families.
- **Computational cost.** For GloSS with a parametric law, discretizing to the lattice is O(L) per law, and the convolutions stay O(k_max² L²). The lattice must cover up to the maximum strength, not just the maximum observed weight; this truncation is the main risk. The Beta tail in P2 costs one `betainc` per endpoint. The maximum-likelihood fit is a 1-D bounded optimization over Σ_i k_i log-shares, O(m) per evaluation.
- **Duck typing.** The new SciPy random-variable classes (`scipy.stats.Normal`, `make_distribution`) expose `cdf`/`ccdf`/`logccdf` but **no `sf`**. Classic frozen `rv_frozen` objects expose `sf`/`cdf`/`logsf` (checked). Any API should require only `cdf`, and prefer `logsf`/`logccdf` when present.

### (c) Preprocessing weights before scoring (sklearn-style)

sklearn's own docs: "quantile and power transforms are based on monotonic transformations … and thus preserve the rank of the values". Consequences by method:

- **Unaffected (monotone-invariant):** LANS, global_sparsification, strongest_n_ties, primary_linkage, MST, PMFG, ultrametric. Any strictly increasing transform (log, Box-Cox, Yeo-Johnson, quantile) is a no-op for these. New ties can change results: `QuantileTransformer` with `n_quantiles < m` and `KBinsDiscretizer` both create ties.
- **Scale-invariant but changed by nonlinear transforms:** disparity, MLA, doubly-stochastic, metric/HSS/edge-betweenness/GLAB.
  - For disparity, a power transform `w^λ` changes the share profile, and so changes the hypothesis: "shares of w^λ are uniform spacings".
  - That can be a legitimate modeling choice: e.g. log-flows for gravity-type data. It must be reported as such. On Les Mis it moves disparity from 9 edges to 0 (log1p) or 26 (w²).
  - `MinMaxScaler`/`StandardScaler`/`RobustScaler` are affine. The shift component changes every share, and centering produces non-positive weights, which the statistical scorers then reject. That is correct behavior.
- **Count nulls (MLF, ECM, NC, Pólya exact, SVH, MDL weights):** any continuous transform breaks the unit-count semantics.
  - Re-discretizing with `KBinsDiscretizer` does not restore them: bin codes 0…n−1 are ordinal labels, not event counts, and code 0 becomes a zero weight.
  - The only sound "preprocessing" is a change of physical unit (counts = w / unit). That is P3, not a transformer.
- **Gotchas the helper must catch:**
  - `Normalizer` works per sample. On the (m, 1) column of weights it maps every weight to 1.0.
  - `PowerTransformer` defaults to `standardize=True`, which produces negatives.
  - `QuantileTransformer(output_distribution="normal")` produces negatives.
  - `QuantileTransformer("uniform")` followed by disparity tests shares of ranks, which is meaningless; use LANS instead.

**Dependency recommendation.** Do **not** depend on scikit-learn.
- Accept any object with `fit_transform` (and optionally `transform`) by duck typing, applied to a (m, 1) float column. NumPy is imported lazily only on that path.
- Also accept any plain callable `f(w) -> float`, applied per edge in pure Python, so `math.log1p` works on a NetworkX-only install.
- This matches the package contract: NetworkX is the only hard dependency, numpy/scipy sit in the `[full]` extra, and `tests/test_minimal_deps.py` enforces this.

---

## 3. Proposals

All proposals keep the library's conventions:
- graph first;
- returns a scored copy with `<method>_pvalue`;
- NetworkX-style `weight=`;
- lazy optional imports;
- `@not_implemented_for("multigraph")`;
- defaults that reproduce current results bit for bit.

### P1 — `transform_weights` helper and invariance documentation (Priority: HIGH)

```python
def transform_weights(G, transformer, weight="weight", out=None):
    """Return a copy of G whose `out` edge attribute (default: `weight`) is transformer(weights)."""
```

**Behavior**

- `transformer` takes two forms:
  - An object with `fit_transform`: called once on `np.asarray(ws, float).reshape(-1, 1)` in `G.edges` order. The output must have shape (m, 1) or (m,), otherwise a ValueError is raised.
  - A callable without `fit_transform`: applied per edge in pure Python.
- The fitted object is not stored on the graph. Store `repr(transformer)` in `H.graph["weight_transforms"]` (append, so chains are recorded) so that `describe_backbone` can report it.
- **Copy semantics:** keep adjacency order (deepcopy-equivalent construction), because tie-breaking in `strongest_n_ties` depends on neighbor order (§1).
- **Validation:**
  - NaN/inf output raises `NetworkXError`.
  - A constant output when the input was not constant raises `ValueError` (catches `Normalizer`).
  - If the output does not preserve the observed weight order, emit a `UserWarning` ("not order preserving; rank-based methods will change").
  - If it creates new ties, emit a `UserWarning`.
  - Positivity is **not** checked here. The downstream scorers already reject non-positive weights with clear messages.
- No `by="node"` option. An undirected edge belongs to two nodes, and per-node normalization is exactly what disparity, LANS and MLA already do internally.
- Optional later step: a `transform=` keyword on `backbone_from_weighted` that calls this first.
- **Docs:** a "Weights and transforms" page with the invariance column of §1 and the do/don't list of §2(c) and §5.

**Tests that must not change:** everything. This is purely additive. Add it to `__all__`, the API contract registry, and the minimal-deps test for the callable path.

**Validation**
- Property tests on Les Mis for f ∈ {×10, √, log1p, w²}:
  - monotone-invariant methods return identical keep flags or p-values;
  - scale-invariant methods are identical under ×10;
  - count methods are documented as changing.
- `transform_weights(G, lambda w: w)` leaves every scorer unchanged. This includes `strongest_n_ties`, which pins down the copy-order issue.
- sklearn objects are tested only behind `pytest.importorskip("sklearn")`: `FunctionTransformer(np.log1p)`, `PowerTransformer(method="box-cox", standardize=False)`, `QuantileTransformer`. CI does not need sklearn.

**Risks:** users may read "transform then disparity" as the same hypothesis; the docstring must say it is not. Silent ties from quantile or binning transforms are handled by the warning.

### P2 — Dirichlet-concentration disparity, `disparity_filter(..., a=1.0)` (Priority: HIGH)

`a = 1/c` is the Pólya reinforcement parameter, so `disparity_filter(G, a=a)` is the large-strength limit of `polya_filter(G, a=a)`. That naming ties the two methods together. I recommend `a` over `concentration` for consistency with the Pólya branch. Avoid `alpha`, which already means the significance level elsewhere in the library.

**Formula.** For an endpoint with degree k ≥ 2, strength s and x = min(w/s, 1), with c = 1/a:

```
p_hi = P(X >= x),  X ~ Beta(c, (k-1)c)  =  betainc((k-1)c, c, 1-x)   # regularized I_{1-x}
p_lo = 1 - p_hi
```

- At `a == 1` this is exactly `(1-x)^(k-1)`. Keep the current pure-Python branch for `a == 1` so results stay bit-identical and no SciPy import is triggered.
- `a > 0` and finite. `a = 0` is the degenerate point mass at 1/k and is rejected with ValueError, as is a negative or NaN `a`.
- Degree-1 endpoints, the OR rule, directed out/in handling and `missing_as_zero` (x = 0 gives p_hi = 1, p_lo = 0) are all unchanged.

**`a="ml"`.** Maximize the symmetric-Dirichlet composite log-likelihood over every share profile with k_i ≥ 2. For an undirected graph that is each node's profile; for a directed graph, both the out-profiles and the in-profiles:

```
l(c) = Σ_i [ lnΓ(k_i c) − k_i lnΓ(c) + (c − 1) Σ_j ln(w_ij / s_i) ]
```

- Optimize log c on a bracket, e.g. [e⁻⁷, e⁵], with the bounded scalar method. Store the fitted `a = 1/ĉ` in `H.graph["disparity_a"]`.
- Weight-0 dyads from `missing_as_zero` are excluded, as they already are from degrees.
- This parallels the Pólya branch's `a_ML`. It uses only `scipy.special.gammaln`/`betainc` and `scipy.optimize`, imported lazily.

**Justification and citations**
- Serrano et al. eq. 1 is the k = c = 1 member.
- Marcaccioli & Livan (Methods) identify the disparity null as the Dirichlet limit of their beta-binomial.
- The Gamma–Dirichlet construction gives the generative reading: i.i.d. Gamma(c) weights conditioned on strength. This is standard probability; I verified it numerically rather than taking it from a cited source.
- Interpretation: c < 1 (a > 1) tolerates more heterogeneity and gives sparser backbones. On Les Mis, â = 0.525 (ĉ = 1.90) and the backbone has 19 edges at p < 0.05, against 9 at a = 1 (checked).

**Tests that must not change:**
- `tests/test_statistical.py`: the disparity formula, hub-star, directed and signed tests.
- `tests/test_les_miserables.py::EXPECTED_EDGES["disparity_filter"] == 9`.
- The doctests.
- `test_minimal_deps.py`: disparity stays pure Python at the default.

**Validation**
1. Known answers: `a=1` equals the closed form; `betainc` against `beta.sf`; hand-computed k = 3 values for c = 0.5 and c = 2.
2. Simulation-based calibration: i.i.d. Gamma(c) weights on random-regular and G(n, m) graphs, giving per-endpoint p-values. Require KS < 0.02 and P(p≤.05) within a binomial 99% CI. This mirrors `checks.py` and `checks2.py`.
3. Recovery of ĉ: bias < 2·SE for c ∈ {0.3, 1, 3}. Checked: 0.300±0.007, 1.005±0.029, 2.976±0.090.
4. Large-s agreement with `polya_filter` once #42 merges, e.g. |p_pólya − p_beta| → 0 as s grows.

**Risks**
- OR-rule inflation (≈1.6× nominal) remains. Document it rather than "fix" it.
- With `a="ml"`, the backbone is no longer a fixed-α test across graphs. `â` must be reported, and `describe_backbone` must print it.
- Degenerate profiles with all shares equal push ĉ → ∞. Clamp to the bracket and warn.

### P3 — Explicit count semantics: `unit=` and integer checks for MLF, ECM, NC (Priority: MEDIUM-HIGH)

```python
marginal_likelihood_filter(G, weight="weight", signed=False, missing_as_zero=False, unit=1)
ecm_filter(..., unit=1)
noise_corrected_filter(..., unit=1)
```

**Formula.** Counts are n_ij = w_ij / unit. MLF and ECM then apply the current coercion, `round(n)` and `max(round(n), 1)` respectively. NC uses n_ij unrounded, which is the current behavior.

- **Default `unit=1` changes nothing.** That covers the Les Mis pins 126 / 72 / 195 / 98, the NC reference-implementation values, the ECM geometric-tail tests and the R-backbone fixtures.
- **New warning.** Emit `UserWarning` when coercion changes a count by more than 1e-9 ("weights are not integer counts in units of `unit`; rounded"). It names the MLF w < 0.5 → 0 drop and the ECM w < 1.5 → 1 bump.
- **Possible later breaking change, for the maintainer to decide.** Unify coercion in a shared `_as_counts` helper and raise, as SVH already does, for weights whose count rounds to 0.
- Docs: state that p-values depend on `unit` (×10 moves MLF from 126 to 215 edges), and that `unit` should be the physical event size.

**Validation:** `unit=c` on a graph whose weights are c·(integers) gives identical p-values to the integer graph; a property test covers this. The default-path tests are unchanged.

**Risks:** warnings in user code that passes float counts. That is intended.

### P4 — GloSS with a user-supplied weight law: `gloss_filter(..., weight_distribution=None)` (Priority: MEDIUM-LOW, after #43 merges)

- `None` keeps the empirical P_obs, the current branch behavior.
- Otherwise the argument is any object with `cdf`: a frozen `rv_continuous`/`rv_discrete` or a new-infrastructure distribution. It is discretized on the existing lattice, `P(w_n) ∝ F(w_n + δ/2) − F(w_n − δ/2)`.
  - The lattice is extended to the maximum node strength, and the mass outside it is reported.
  - Discrete distributions with a `pmf` are used directly on an integer lattice.
- The posterior and p-value are unchanged (Radicchi Eqs. 1–4 as in the branch docstring), with P_obs replaced by P.
- **Coherence:** the null is the paper's own construction ("weights … randomly chosen from the … distributions", Sec. II B). Fitting P globally on the same weights is a low-dimensional fit, comparable to the empirical P_obs, which already uses the data.
- **Validation:**
  - SBC with weights drawn from P on fixed topologies (Radicchi Fig. 1 design: P(<α) = α).
  - Agreement with the empirical-P_obs path when P is the empirical law itself.
- **Risks:**
  - Lattice truncation for heavy tails.
  - O(k_max² L²) cost.
  - Gemmetto et al.'s critique of GloSS (weights decoupled from topology, so expected strength ∝ degree) applies whatever law is used, and must be stated.
- **Note (my derivation, unverified, not checked numerically).** For Gamma(c, θ) weights the one-endpoint posterior is P2's Beta. With both endpoints, θ no longer cancels: the posterior ∝ w^(c−1)(s_i−w)^((k_i−1)c−1)(s_j−w)^((k_j−1)c−1)e^(w/θ). So the two-endpoint GloSS with a Gamma law is not scale-free. Check this before documenting it.

### P5 — Continuous-weight ECM (Priority: LOW-MEDIUM, research first)

This is the maximum-entropy model with ⟨k_i⟩ and ⟨s_i⟩ constraints over w ∈ {0} ∪ (0, ∞), with Lebesgue reference measure on w > 0 (my derivation):

```
Z_ij = 1 + x_i x_j / (β_i + β_j)
p_ij = x_i x_j / (β_i + β_j + x_i x_j)
w | link ~ Exponential(rate β_i + β_j)
p-value = p_ij · exp(−(β_i + β_j) w_ij)
```

- Rescaling w → c·w maps β → β/c and x → x/√c, leaving the p-values unchanged. So unlike the integer ECM it has **no unit dependence**.
- API: `ecm_filter(..., weights="integer" | "continuous")`, default `"integer"`.
- The result depends on the choice of reference measure. The continuous ECM literature (Gabrielli et al. 2019; Parisi et al. 2020) is **unverified by me** and must be read before implementing.
- **Validation:** sample from the fitted model and run SBC, plus a constraint-reproduction test like the existing `test_fit_reproduces_degrees_and_strengths`.
- **Risks:** a new solver and its convergence; users mixing the two modes.

### P6 — Overdispersed MLF (beta-binomial / negative binomial) (Priority: LOW — defer)

The formula is in §2(a). It is not in Dianati and would be a new method. The Pólya filter (beta-binomial per endpoint) and P2 already answer "binomial is too tight" in a citable way. Revisit only if users need a joint (both-endpoint) overdispersed null.

### Supporting infrastructure — calibration harness (Priority: HIGH, ships with P2)

Add a test helper under `tests/`, not in the public API, that:
- simulates from a null (graph generator + weight law);
- collects **per-endpoint** tails, exposed through a private `_…_node_tails` function;
- asserts uniformity (KS and binomial CI at α ∈ {0.01, 0.05}), with a fixed seed and small sizes so CI stays fast.

It can be reused for Pólya, GloSS and P4/P5. Separately, assert that the edge-level OR-rule rejection rate is ≥ α, documenting the inflation.

### Changes to `describe_backbone` for any of P1–P5

Register the new parameters (`a`, `unit`, `weight_distribution`, and recorded transforms) in `describe.py` so the generated Methods text reports them. A transformed-weight backbone must say "on log-transformed weights".

---

## 4. Risks summary

- **Multiple testing.** Generalized nulls do not change the need for `threshold_filter(..., mtc=…)`. The OR rule (min over two endpoint tests) is anti-conservative before any correction: 0.081 at nominal 0.05 under an exactly correct null.
- **Fitted-parameter bias / double dipping.** Acceptable for one or two global parameters (ĉ, a_ML, a global P in GloSS); checked to stay calibrated. Unacceptable per node or per edge (§2b numbers).
- **Invalid combinations that should raise or warn:**
  - non-finite or constant transform output (raise);
  - non-order-preserving transform (warn);
  - `a ≤ 0` or NaN (raise);
  - `unit ≤ 0` (raise);
  - non-integer counts after `unit` scaling (warn now, raise later);
  - a distribution object without `cdf` (TypeError);
  - `weight_distribution` with no mass on the lattice (raise).
- **Interpretation drift.** "Transform then test" tests a different hypothesis. Docs and `describe_backbone` must say so.

---

## 5. Not recommended

1. **A generic `distribution_filter(G, dist, by="global")`.** `p = dist.sf(w)` is monotone in w, so the backbone at any α is exactly a global weight threshold: Les Mis with a fitted lognormal keeps exactly the edges with w ≥ 9. Calling these p-values "significance" is misleading. The test ignores degrees and strengths, which is the decoupling Gemmetto et al. criticize even in GloSS. `global_threshold_filter` already does this honestly.
2. **`distribution_filter(..., by="node")` with per-node parametric fits.** With typical degrees (k < 10):
   - in-sample fits cap the attainable p-value (k = 4: min p = 0.043);
   - leave-one-out plug-ins are anti-conservative (0.147 at k = 4);
   - the exactly calibrated fix (leave-one-out predictive t) is an outlier test for location-scale families, not a network null.

   LANS already gives the nonparametric per-node version honestly.
3. **Asymmetric or arbitrary "share distributions" in disparity** (Beta(c, d) with d ≠ (k−1)c, or a user-chosen Beta). Exchangeable links require E[share] = 1/k, and only the symmetric Dirichlet is consistent with shares summing to 1.
4. **Continuous transforms before count nulls** (log, Box-Cox, Yeo-Johnson, quantile, scaling before MLF, ECM, NC, Pólya-exact, SVH, MDL weights). They destroy the unit-count semantics; on Les Mis, log1p moves MLF from 126 to 45 edges. Use `unit=` (P3) for genuine unit changes.
5. **`KBinsDiscretizer` (or any binning) to "integerize" weights for count nulls.** Bin codes are ordinal labels, not event counts, and code 0 becomes a zero weight.
6. **`QuantileTransformer` / `StandardScaler` / `RobustScaler` / `Normalizer` / `PowerTransformer(standardize=True)` before disparity, Pólya or GloSS.** Their outputs are:
   - ranks, so the test is meaningless; use LANS;
   - non-positive values, which the scorers reject;
   - or constant values: `Normalizer` maps the weight column to all ones.
7. **Making scikit-learn a dependency.** Duck typing covers every use, and the package's minimal-dependency contract forbids it.
8. **Generalizing the bipartite nulls through weight transforms.** SDSM, FDSM, FFM, FRM and FCM are defined on binary incidence; valued data needs a different model.

---

## 6. Literature actually read (and what it says about variants)

- **Serrano, Boguñá & Vespignani 2009** (arXiv:0904.2389, the PNAS version).
  - Null: "normalized weights … produced by a random assignment from a uniform distribution", eq. 1.
  - The Methods use Υ(k) = kΣp² as a heterogeneity diagnostic, with null mean 2k/(k+1).
  - The SI shows that disparity ≈ global threshold at ⟨ω⟩ ln α⁻¹ for uncorrelated weights with a finite mean, and that this equivalence fails for infinite-mean power laws.
  - No alternative null family is proposed.
- **Marcaccioli & Livan 2019** (local copy). The Pólya urn / beta-binomial family, Eq. 1. a = 1 approximately recovers disparity (Eqs. 6–8). The disparity null is the Dirichlet limit of the beta-binomial. Non-integer weights use the Eq. 6 approximation.
- **Radicchi, Ramasco & Fortunato 2011** (local copy). GloSS null: fixed topology, i.i.d. weights from P_obs. Fig. 1 shows disparity's p-values are non-uniform for i.i.d. weights while GloSS's are uniform.
- **Gemmetto, Cardillo & Garlaschelli 2017** (arXiv:1706.00230). ECM (Bose–Fermi) "without loss of generality … integer weights". Critiques GloSS for decoupling weights from topology, and the WCM for producing near-complete graphs. No continuous variant in the parts I read.
- **Coscia & Neffke 2017** (arXiv:1701.07336). Weights are sums of unit interactions with a binomial likelihood, a Beta conjugate prior, and prior moments from a hypergeometric argument. "Particularly suited for applications in which edge strengths have a count-data structure." No alternative priors are proposed.
- **Dianati 2016** (arXiv:1503.04085). "We assume all weights to be positive integers"; unit edges are assigned independently. Positions MLF between disparity and GloSS.
- **Foti, Hughes & Rockmore 2011** (PLoS ONE HTML). LANS is "distribution-free … we rely only on the empirical distribution". Uses KS tests to show that node weight distributions differ from the disparity null.
- **scikit-learn preprocessing guide.** Quantile and power transforms are monotone and rank-preserving. Box-Cox needs strictly positive data. `PowerTransformer` standardizes by default. `Normalizer` scales samples, not features. `KBinsDiscretizer` strategies are uniform, quantile and kmeans.
- **scipy.stats reference.** `betabinom`, `nbinom`, `dirichlet`, `dirichlet_multinomial`, `poisson_binom`, `false_discovery_control`, `goodness_of_fit`, `monte_carlo_test`, and the new `make_distribution` infrastructure are all listed. The new classes have `ccdf` but not `sf` (checked locally).
- **Unverified / not read:**
  - the continuous maximum-entropy weighted models (Gabrielli et al. 2019; Parisi et al. 2020 "CReMa");
  - any negative-binomial backbone literature;
  - valued-bipartite backbone nulls.

---

## 7. Incidental findings (not part of this request)

1. `strongest_n_ties` results change under `G.copy()` on tied weights (Les Mis: 23 edges differ). Ties are broken by adjacency order, and `Graph.copy()` reorders adjacency. This affects any pipeline that copies the graph first.
2. MLF and ECM coerce sub-unit weights inconsistently: MLF turns 0.4 into 0, which is dropped with p = 1; ECM turns 0.4 into 1. Python's `round` is round-half-even (2.5 → 2, 0.5 → 0). Nothing warns. SVH raises in the same situation.
3. The main checkout has uncommitted modifications to `nx_backbone/structural.py`, presumably someone's WIP.
