# Design proposal: temporal backbones and filtering-inspired methods

- **Status:** Accepted for implementation in a new top-level module,
  `networkx_backbone.temporal`. Tracking: GitHub issue #78.
- **Issues:** #67 (foundation), #68 and #69 (simple filters), #70 (interop),
  #71 (null-model tests), #72 (P1 `kalman_backbone`), #73 (P2
  `edge_persistence_hmm`), #74 (P5, idea), #75 (P6, idea); not temporal:
  #76 (P3) and #77 (P4), both for `fdsm`.
- **Scope:** Research report. No library code was changed by this document.

## Decisions made after this report

The maintainer settled the module's conventions after reading this report. Where
they differ from Section 2 below, these decisions win:

- **Inputs.** Snapshots (a list or dict of graphs), timestamped events (a
  `MultiGraph` with `time=`), and spans (`start=`/`end=`) are accepted equally.
  Events and spans are binned into steps with `freq=`. Plain numbers work with
  NetworkX alone; datetimes and frequency strings need pandas (optional, lazy).
- **Absent edges.** A node is present in a step if it is in that snapshot or
  has an event or span there. An absent edge is an observed zero when both
  endpoints are present and unknown otherwise. This replaces Section 2's
  `missing="skip"` default.
- **Output.** One scored static graph: the union of observed edges with score
  attributes, so `threshold_filter` and the other filters work unchanged.
  Model-based methods also store per-step series as edge attributes. This
  replaces Section 2's list of per-snapshot graphs.
- **Simple filters first.** Descriptive scores (observation fraction,
  persistence, backbone stability, weight trend) ship before null-model tests.
- **Interop.** pandas event tables, DyNetX, teneto, and pathpy adapters.

---

## Original report: Filtering-inspired network backbone methods for `networkx_backbone`: proposals

**Scope.** I read five state-estimation and signal-filtering papers and checked which of their
techniques carry over to network backbone extraction in `networkx_backbone` (current `main`,
`d6a935b`). I read `README.md`, `docs/concepts.rst`, `docs/roadmap.rst`, `networkx_backbone/__init__.py`
(`__all__`), `bipartite.py` (`fdsm`, `fastball`, `_fdsm_tails`, `_curveball_trades`,
`backbone_from_projection`), `filters.py` (`threshold_filter`, `multigraph_to_weighted`,
`consensus_backbone`), `tests/test_api_contract.py` and the header of `describe.py`. I did not
modify the repository.

**Bottom line.** None of the papers is about networks. Three ideas carry over in a way that solves
a real, current gap in the library:

1. **Temporal backbones by state-space smoothing of per-snapshot significance.** The library has no
   temporal input at all.
2. **Exact finite-state (HMM) filtering of edge persistence.**
3. **Better Monte Carlo for `fdsm`'s small p-values.** The `(count+1)/(trials+1)` floor makes
   family-wise corrections unreachable at the default `trials` (demonstrated below).

Wavelet shrinkage carries over only as a temporal *denoising* preprocessing step. It is not a
backbone criterion. Particle filters, particle MCMC, ensemble Kalman and data assimilation, and
spectral-graph-wavelet thresholding are not recommended (reasons in the "Not recommended" section).

A runnable prototype backs the main proposal (P1): [`docs/design/prototypes/kalman_proto.py`](prototypes/kalman_proto.py)
(numpy and scipy only). Its results are quoted in P1.

---

## 0. Sources (as checked in the PDFs)

| Key | Paper | Notes |
|---|---|---|
| **FK** | Fearnhead, P. & Künsch, H. R. (2018). Particle Filters and Data Assimilation. *Annu. Rev. Stat. Appl.* 5:421–49. doi:10.1146/annurev-statistics-031017-100232 | Equation numbers below are FK's own. |
| **SV** | Schneider, K. & Vasilyev, O. V. (2010). Wavelet Methods in Computational Fluid Dynamics. *Annu. Rev. Fluid Mech.* 42:473–503. doi:10.1146/annurev-fluid-121108-145637 | Equations (1)–(13). |
| **AAK** | Anagnostaki, I., Anastassopoulos, V. & Koukiou, G. (2026). Applications of the Kalman Filter in Physical Processes: A Review. *Applied Sciences* 16(10), 4649. doi:10.3390/app16104649 (received 9 Mar 2026, published 8 May 2026) | Volume and year confirmed on the PDF: "Appl. Sci. 2026, 16, 4649". |
| **ZC** | Chen, Z. (2003). Bayesian Filtering: From Kalman Filters to Particle Filters, and Beyond. Technical report, Communications Research Laboratory, McMaster University (69 pp.) | PDF metadata: "Technical Report, 2003". The manuscript header says "MANUSCRIPT". The two-column extraction garbles some equation numbers, so where I am unsure I cite the section. |
| **WS** | Wills, A. G. & Schön, T. B. (2023). Sequential Monte Carlo: A Unified Review. *Annu. Rev. Control Robot. Auton. Syst.* 6:159–82. doi:10.1146/annurev-control-042920-015119 | Equations (1)–(80), Algorithm 1. |

**Citation discipline.** References outside the five papers are marked in two ways:

- **(exists)**: I verified title, authors and year in OpenAlex in this session. Any statement about
  the paper's *content* is from memory and was not re-read, so content claims carry **(unverified)**
  where they matter.
- **(unverified)**: not checked at all.

---

## 1. Summary of proposals

| # | Name (proposed API) | One line | Kind | Priority |
|---|---|---|---|---|
| P1 | `kalman_backbone(snapshots, method=..., ...)` | Smooth per-snapshot edge significance (probit-transformed p-values) with a local-level Kalman filter and Rauch–Tung–Striebel (RTS) smoother, and test against the **exact** null distribution of the linear smoother. | Novel adaptation of standard parts | **High** |
| P2 | `edge_persistence_hmm(snapshots, ...)` | Two-state hidden Markov model (HMM) per edge ("active" or "inactive"), solved by exact forward-backward with pooled Baum–Welch. Turns flickering presence or keep flags into a posterior probability of persistent activity. | Standard HMM, new use (temporal consensus) | **Medium** |
| P3 | `fdsm` Monte Carlo resolution diagnostics | Record `trials`, add a Monte Carlo error or upper bound, and warn when the p-value floor `1/(trials+1)` cannot pass `alpha/m` under the chosen `mtc`. | Established Monte Carlo practice | **High** (cheap) |
| P4 | `fdsm(..., refine="splitting")` | Sequential Monte Carlo (adaptive multilevel splitting) with curveball moves restricted to level sets. Gives accurate p-values far below `1/(trials+1)` for the pairs stuck at the floor. | Established rare-event method, novel application to FDSM | **Medium** |
| P5 | `wavelet_denoise(snapshots, ...)` | Haar wavelet shrinkage (Donoho–Johnstone universal threshold) of each edge's weight time series, as a *denoise-then-backbone* step, plus a dominant-time-scale diagnostic. | Standard signal processing, new use | **Low** |
| P6 | `poisson_lift_filter(snapshots, ...)` | Conjugate Gamma–Poisson local-level filter for count weights, with a null-model exposure `s_i s_j / 2W` per snapshot. Gives a dynamic "lift" and one-step predictive (innovation) p-values for bursts. | Standard count state-space model, novel combination | **Low** (idea) |

All six are absent from the library. `__all__` has no temporal or sequence input. `fdsm` has only plain
curveball Markov chain Monte Carlo with `(count+1)/(trials+1)`. Nothing is wavelet-based.

---

## 2. Shared infrastructure for the temporal proposals (P1, P2, P5, P6)

**Input type (proposal).** Every temporal function takes, as its first argument:

- `snapshots`: a sequence of NetworkX simple graphs `[G_1, …, G_T]`, all directed or all undirected,
  with a shared or overlapping node set. Multigraphs are rejected, as the other scorers already do.
- Optional keyword `times=None`: snapshot times, for unequally spaced snapshots. P1 scales the
  process variance by Δt. P5 uses a lifting-scheme Haar transform.
- A helper `snapshots_from_events(G, time="time", bins=...)` turns a timestamped `MultiGraph` or
  `MultiDiGraph` into snapshots. It reuses `multigraph_to_weighted(..., aggregate=...)` per bin.

**Missing versus zero (roadmap #26, `missing_as_zero`).** An edge absent from snapshot *t* can mean
"observed zero" or "not observed". The proposal is `missing={"skip", "zero"}`:

- `"skip"` (the default) treats it as an unobserved time step. Filters handle this natively with a
  prediction-only step.
- `"zero"` requires a per-snapshot scorer that can score zero weight. This ties into #26.

`"skip"` has a selection-bias risk: edges appear mostly when strong. The docs must say so.

**Output.** A list of `T` scored graph copies, one per snapshot. Each contains the union of edges
seen in any snapshot when `include_unobserved=True`; the default is only edges observed at *t*. Each
carries `<method>_pvalue`, `_score` and `_keep` attributes, so the usual idiom works:

```python
scored = nb.kalman_backbone(snaps, method="disparity_filter")
backbones = [nb.threshold_filter(H, "kalman_pvalue", 0.05, mtc="bh") for H in scored]
```

The graph-level attribute `H.graph["kalman"]` records the fitted parameters (for example, the
estimated `q`).

**Module and contracts.**

- A new module `networkx_backbone/temporal.py`. It must not be listed in `describe.HELPER_MODULES`.
- `tests/test_api_contract.py` needs a new `Spec.kind = "temporal"`. Today it assumes a single
  weighted, unweighted or bipartite graph.
- Each method needs a `describe.METHODS` entry and `REFERENCES`.
- numpy and scipy are imported lazily inside functions, consistent with the `[full]` extra.

---

## 3. Proposals

### P1. `kalman_backbone`: temporal smoothing of edge significance (high)

**Problem.** Analysts with snapshot sequences (monthly trade, daily contacts, yearly co-authorship)
currently have two choices:

- **Backbone each snapshot.** Results are noisy: edges flicker in and out, and power is low per
  snapshot.
- **Backbone the aggregate.** This destroys the dynamics: a tie that was strong in 2010–2012 and gone
  afterwards looks moderate.

There is no principled middle ground in the library. Temporal-backbone methods exist in the
literature: Kobayashi, Takaguchi & Barrat (2019, *Nat. Commun.*) (exists) and Nadini, Bongiorno, Rizzo
& Porfiri (2019, *Nonlinear Dyn.*) (exists). From memory, both build temporal null models of node
activity rather than smoothing per-snapshot evidence (unverified). A brief OpenAlex search found no
state-space or Kalman *backbone* method. Xu & Hero (2014, *IEEE JSTSP*, "Dynamic stochastic
blockmodels") (exists) applies Kalman-type filtering to block-level probabilities, from memory with
an extended Kalman filter (unverified). That is a precedent for Kalman filtering of network models,
not of edge backbones. **Treat P1 as a novel adaptation.**

**Idea.** Use any existing static scorer as the per-snapshot measurement. Convert its p-value to a
probit score

  y_{e,t} = Φ⁻¹(1 − p_{e,t}).

Under that scorer's null, y_{e,t} ~ N(0, 1). Model the latent standardized signal x_{e,t} with a
local-level state-space model, then smooth.

**Model** (scalar, per edge; the general linear-Gaussian form is FK Eqs 1–2, AAK Eqs 10–11, and
ZC's linear model in Sec. IV-B):

  x_t = x_{t−1} + η_t, η_t ~ N(0, q·Δt)  (state; AAK Eq 10 with A = 1)
  y_t = x_t + ε_t,  ε_t ~ N(0, r), r = 1  (measurement; AAK Eq 11 with H = 1)

The observation variance is **known** (r = 1) because y is a probit-transformed valid p-value. Only
the signal-to-noise ratio q is unknown.

**Algorithm.**

1. **Per-snapshot scoring.** Run the chosen scorer on each `G_t` (`disparity_filter`, `ecm_filter`,
   `lans_filter`, `noise_corrected_filter`, `marginal_likelihood_filter`, …). Read
   `<scorer>_pvalue`. For signed scorers, rebuild the one-sided upper tail from
   (two-sided p, `sign`). For discrete-p scorers (LANS, ECM, FDSM), use mid-p or randomized
   probability integral transforms (`pit="mid"|"random"`). Otherwise y is not N(0, 1) under the null.
2. **Forward Kalman filter**, vectorized over all edges at once. Prediction is
   m_{t|t−1} = m_{t−1}, P_{t|t−1} = P_{t−1} + q (AAK Eqs 33 and 36). The innovation is
   ν_t = y_t − m_{t|t−1} with variance S_t = P_{t|t−1} + 1 (AAK Eq 18 and §2.3.5). The gain is
   K_t = P_{t|t−1}/S_t (AAK Eq 31; FK §3.5; ZC Eq 38). The update is m_t = m_{t|t−1} + K_t ν_t,
   P_t = (1 − K_t) P_{t|t−1} (AAK Eqs 16 and 32). For missing y_t, K_t = 0.
3. **RTS smoother** (`mode="smoother"`, offline). ZC names the RTS smoother as the offline Kalman
   smoother (§IV-B remarks). FK §6 Eq 14 is the general backward recursion; its linear-Gaussian case
   is RTS. `mode="filter"` uses only the past, for streaming (WS Table 1: filtering versus
   smoothing densities).
4. **Exact null test.** This is the key point. With q fixed, the smoothed mean m_{t|T} = Σ_s a_{ts} y_s
   is **linear in y**, with weights that depend only on q and the edge's missingness pattern. Under
   H0 (x ≡ 0, so y_s iid N(0, 1)):

     m_{t|T} ~ N(0, Σ_s a_{ts}²)

   The p-value is therefore `kalman_pvalue` = 1 − Φ(m_{t|T} / sd₀,t), with sd₀,t² = Σ_s a_{ts}².
   Note that sd₀,t is **not** √P_{t|T}. The posterior variance answers a Bayesian question; using it
   gives miscalibrated p-values (see the validation results below). The null variance comes from an
   O(T) recursion. The filter part is the Joseph form P = (1 − K)P′(1 − K)ᵀ + KRKᵀ (AAK Eq 23),
   which holds for *any* gain and so gives the error variance under the null (true q = 0) even
   though the filter assumes q > 0:
   - v_t = (1 − K_t)² v_{t−1} + K_t²
   - then backwards, with G_t = P_t / P_{t+1|t}:
     V_t = (1 − G_t)² v_t + G_t² V_{t+1} + 2G_t(1 − G_t) v_t (1 − K_{t+1}) χ_{t+1}, and
     χ_t = (1 − G_t) + G_t (1 − K_{t+1}) χ_{t+1}, with V_T = v_T and χ_T = 1.

   The prototype checks this recursion against brute force (running the smoother on unit vectors).
   They agree to 2·10⁻¹⁶ with and without missing snapshots.
5. **Estimating q.** Maximize the pooled Gaussian log-likelihood from the prediction-error
   decomposition, log p(y_{1:T}) = Σ_t log N(ν_t; 0, S_t) (WS Eq 75; FK Eq 6). Pool across all edges
   with a 1-D bounded search over log q. Alternatively, the user sets
   `half_life=` (in snapshots), which maps to q through the steady-state gain. Many backbone users
   will prefer this, since pooled maximum likelihood over mostly-null edges pushes q toward 0.
6. **Extra outputs.** `kalman_score` = m/sd₀ (the z-statistic), `kalman_level` = m_{t|T},
   `kalman_level_sd` = √P_{t|T} (Bayesian uncertainty, for plotting), and
   `kalman_innovation` = ν_t/√S_t. The last is a burst or change indicator, an "innovation-based
   test" in the sense the brief asked about. It is calibrated only under the fitted model, not under
   the null. Whiteness of standardized innovations serves as a model diagnostic: ZC §IV-B,
   footnote 29, notes that the optimality of a filter can be judged by how white its innovations are.

**Honest framing.** For fixed q, P1 is a **time-localized weighted Stouffer combination** of
per-snapshot evidence. Its kernel is close to a two-sided exponential whose width is set by q.
Two limits make this explicit: q → 0 gives plain Stouffer over all observed snapshots, and q → ∞
gives the per-snapshot test. Both follow from the recursions; they are listed as validation tests
(b) below but were not run in the prototype. What the Kalman machinery adds over an ad hoc moving window:

- a likelihood-based choice of bandwidth (step 5)
- exact handling of missing snapshots and unequal spacing
- an O(E·T) online version
- exact null calibration that survives missingness

**Optional robust variant** (`robust=True`). Use a two-component Gaussian-sum observation model, a
core plus a wide outlier component. This follows the Gaussian Sum Filter used for noisy hits in
AAK §8.2 and the robust filters of FK §3.7.3. One-off spikes are downweighted. The smoother then
becomes nonlinear in y, so the exact-null property is lost. Null p-values then come from
simulation. Under H0 the y's are iid N(0, 1), so a single shared simulation per (T, missingness
pattern) suffices.

**Dependencies.** numpy and scipy (`scipy.stats.norm`, `scipy.optimize.minimize_scalar`), lazily
imported, all in `[full]`. statsmodels' `UnobservedComponents` could serve as a *test-only* cross-check
and should not be a dependency.

**Complexity.** O(T · cost(scorer)) for the per-snapshot scoring, which dominates. Then O(E·T) per
filter and smoother pass, and O(I·E·T) for a q search with I evaluations. Memory is O(E·T) floats.

**Validation plan.**

- **(a) Null calibration.** iid N(0, 1) inputs give uniform p-values (KS test), including with
  missing snapshots. Prototype, E = 20,000 edges, T = 40, q = 0.05: P(p < 0.05) = 0.0498,
  P(p < 0.01) = 0.0102; with 30% of snapshots missing, 0.0511. Using √P_{t|T} instead of the exact
  null sd gives 0.0138, which is miscalibrated (conservative).
- **(b) Limits.** As q → 0, the result equals Stouffer over the observed snapshots. As q → ∞,
  `kalman_pvalue` equals the input p-values.
- **(c) Power.** With signal μ = 0.8 in snapshots 10–19 only: rejection at t = 15 is 0.73 for P1
  versus 0.20 per snapshot.
- **(d) End-to-end.** Use synthetic weighted SBM snapshots with a planted block whose weights rise
  for a window. Run `disparity_filter` per snapshot versus `kalman_backbone(method="disparity_filter")`
  and measure recovery of the planted edges.
- **(e) Invariances.** Node relabeling, edge order, and time reversal (smoother output reverses).
- **(f) Optional cross-check.** Filter and smoother means against statsmodels in an optional test.

**Risks and open questions.**

- **Temporal leakage.** The p-value is exact for the *edge-global* null (no signal at any time). It
  is not exact for a time-local null (no signal at t, signal elsewhere). In the prototype, 10
  snapshots after a signal window ended, the rejection rate was 0.067 against a nominal 0.05.
  Document this, or offer `mode="filter"`, which leaks only from the past.
- **Cross-time dependence under the null.** Static nulls condition on per-snapshot margins, but
  dyad-level confounders that persist over time make y positively autocorrelated, which is
  anti-conservative. Mitigation: a whiteness diagnostic, and an AR(1)-observation-noise option.
- **Discrete p-values.** Handled by mid-p or randomized transforms; the effect must be tested.
- **Multiple testing over E×T tests.** Is per-snapshot `mtc` enough, or is an edge-level summary
  test needed? For example, the max over t of z, with its null distribution from the same shared
  simulation.
- **Return type.** A list of graphs versus a single union graph with per-time attributes.

**Priority: High.** It fills the largest gap (no temporal input at all), reuses every existing scorer,
adds no new dependency, and is exact, cheap and easy to test. The prototype already works.

---

### P2. `edge_persistence_hmm`: exact two-state filtering of edge activity (medium)

**Problem.** Binary temporal data and backbone membership over time both flicker. Examples are
contact or presence snapshots, and any scorer's `_keep` flag computed per snapshot. Analysts use
ad hoc rules such as "present in ≥ k% of snapshots", or `consensus_backbone`, which is a hard
intersection across methods rather than over time. Neither accounts for **persistence** (an active
tie tends to stay active) or **observation error** (a tie can be missed or spuriously recorded).

**Method.** A per-edge HMM with hidden state s_t ∈ {0 inactive, 1 active}:

- **Transition matrix** [[1 − a₀₁, a₀₁], [a₁₀, 1 − a₁₀]], shared across edges.
- **Emission** for an observed presence or keep flag o_t: P(o = 1 | s = 1) = 1 − β (miss rate β) and
  P(o = 1 | s = 0) = α (false-positive rate α). A missing o_t has emission 1.

For a finite discrete state, the grid-based (point-mass) filter is **exact**. ZC §V-C, Eqs 50–52,
gives the prediction and update for discrete states. ZC §VII-E covers HMM filters, the Baum–Welch
algorithm for state probabilities given all data, and Viterbi for the maximum a posteriori path.
FK §6.1 says the forward-backward particle smoother "is based on the same ideas that are used in the
forward-backward algorithm for discrete-state hidden Markov models and linear Gaussian state-space
models". The parameters (a₀₁, a₁₀, α, β) are estimated
by Baum–Welch (EM, as in WS Eqs 76–77 with exact rather than particle smoothing), **pooled across
all edges**, or fixed by the user.

**Outputs.** On each snapshot copy: `hmm_score` = P(s_t = 1 | o_{1:T}) and `hmm_keep` (Viterbi
state, or `hmm_score > 0.5`). On `H.graph`: the fitted parameters. An optional
`observation="pvalue"` uses continuous emissions: p ~ Uniform under s = 0 and Beta(a, 1) under
s = 1. This is a beta-uniform mixture, a known idea, citation not checked (unverified).

**Why existing methods fall short.** Thresholded presence frequency ignores ordering. A tie present
in snapshots 1–10 is treated like one present in 10 random snapshots. The HMM fills one-snapshot
dropouts inside active spells and suppresses isolated blips. The interacting-multiple-model idea in
ZC §VII-C and AAK §3 (switching between modes, with weights from posterior mode probabilities) is
exactly this, made exact by the discrete state.

**Precedents.** Dynamic network models with link persistence exist, for example Zhang, Moore &
Newman (2017, *EPJ B*, "Random graph models for dynamic networks") (exists) and Barucca, Lillo,
Mazzarisi & Tantari (2018, *J. Stat. Mech.*, "Disentangling group and link persistence in dynamic
stochastic block models") (exists). From memory, they include Markov link-persistence mechanisms
(unverified). Using an HMM as a *backbone-membership smoother* is, as far as I know, a novel use
(unverified).

**Dependencies.** numpy only. It could be pure Python, but vectorizing over edges needs numpy.

**Complexity.** O(E·T·4) per forward-backward pass, times the number of EM iterations (typically
tens). Memory O(E·T).

**Validation plan.**

- Simulate edges from known (a₀₁, a₁₀, α, β). Check parameter recovery and that posterior state
  accuracy matches the Bayes rate.
- Check against brute-force enumeration over all 2^T state paths for T ≤ 10.
- Degenerate cases: α = β = 0 reproduces the observations exactly. a₀₁ = a₁₀ = 0.5 (no persistence)
  reduces to a per-snapshot posterior.
- Apply it to per-snapshot `disparity_filter` keep flags on a synthetic SBM sequence and measure
  stability, for example the Jaccard index between consecutive backbones.

**Risks.**

- **Label switching and identifiability.** Impose α < 1 − β.
- **Heterogeneity.** Shared parameters across very different edges form a mixture. An option is
  per-node-pair-class parameters, for example by degree class.
- **Applied to `_keep` flags**, the "emission" error rates are not physical measurement errors. The
  docs must frame `hmm_score` as a persistence-smoothed membership score, not a p-value.

**Priority: Medium.** It is exact, cheap and easy to explain, and it complements P1 for binary data.
It is less central than P1 because it smooths decisions rather than evidence.

---

### P3. `fdsm` Monte Carlo resolution diagnostics (high, small)

**Problem, demonstrated on `main`.** `fdsm` p-values are `(count+1)/(trials+1)`
(`_fdsm_tails`, `bipartite.py`), so their minimum is 1/(trials+1). Family-wise corrections need
p < alpha/m for m pairs. When trials < m/alpha − 1, **no** pair can pass Holm or Bonferroni,
whatever the data.

I ran this on a synthetic 60 × 80 incidence matrix with a planted 10-agent block (scratch run on
`main`). With `trials=1000`:

- 1,770 pairs; alpha/m = 2.8·10⁻⁵, floor = 10⁻³.
- 44 pairs sat at the floor.
- `threshold_filter(..., mtc="holm")` kept **0** edges.
- `mtc="bh"` kept 44, only because 44 tied floor values pass the BH step-up together.
- `sdsm` on the same data with Holm kept 2 (minimum p 2.3·10⁻⁵).

On Davis Southern Women the floor does not bind: the minimum fdsm p was ≈ 0.008.

**Proposal.**

1. `fdsm` stores `H.graph["fdsm_trials"] = trials` and adds `fdsm_pvalue_upper`, a one-sided
   Clopper–Pearson upper bound on the true p, or `fdsm_pvalue_se` = √(p(1−p)/trials). This follows
   the Monte Carlo error rate O(N^{-1/2}) in ZC §V-G, Eq 57 and the central-limit discussion there.
2. `backbone_from_projection(method="fdsm", mtc=...)`, which already knows `alpha` and `mtc`, and
   `threshold_filter` (when `fdsm_trials` is present in `H.graph`) emit a `RuntimeWarning` when
   1/(trials+1) ≥ alpha/m under an FWER method. The warning reports the `trials` needed
   (≈ m/alpha). For BH it notes that ties at the floor drive the result.
3. Document `sdsm` as the analytic alternative and P4 as the precise one.

Sequential stopping (Besag & Clifford 1991, *Biometrika*, "Sequential Monte Carlo p-values")
(exists) saves little here: each curveball sample scores *all* pairs at once, so per-pair early
stopping does not reduce cost. I mention it only to explain why it is not proposed.

**Dependencies.** None new. scipy's `beta.ppf` gives Clopper–Pearson.

**Complexity.** O(m) extra.

**Validation.**

- The warning fires exactly when floor ≥ alpha/m.
- Coverage of `fdsm_pvalue_upper`: simulate binomial counts with known p and check the empirical
  coverage is at least nominal.
- API-contract update: the new attribute lies in [0, 1].

**Risk.** Warning fatigue. Fire it only for FWER methods, or when ≥ 1 pair sits at the floor.

**Priority: High.** It prevents silently empty backbones at a cost of a few lines. The papers only
motivate it, through the Monte Carlo error discussion. It is not a new method.

---

### P4. `fdsm(..., refine="splitting")`: SMC rare-event p-values for floor pairs (medium)

**Problem.** P3 detects the floor; P4 removes it. Raising `trials` to m/alpha costs
O((m/alpha)·(S·d + Σ_z d(z)²)) (`_COMPLEXITY["fdsm"]`), which is infeasible for thousands of agents.

**Method.** This is adaptive multilevel splitting, an SMC sampler whose intermediate targets are
level sets. FK §3.4: sequential Monte Carlo samples "a sequence of intermediate distributions
π₀, π₁, …, π_n = π", with resampling, propagation by kernels P_t that leave π_t invariant, and
reweighting. Del Moral, Doucet & Jasra (2006, *JRSS-B*) (exists) is the reference FK give. Cérou &
Guyader (2007, *Stoch. Anal. Appl.*, "Adaptive multilevel splitting for rare event analysis")
(exists) is the rare-event specialization; its details are from memory (unverified).

For a pair (i, j) with observed co-occurrence k* and current estimate at the floor:

1. Let Ω be the binary matrices with the observed margins, and π₀ the uniform distribution on Ω (the
   FDSM null). Draw N particles from the existing curveball chain (`_fdsm_row_sets`).
2. Choose levels ℓ₁ < … < ℓ_L = k* adaptively: ℓ_{l} is the (1−ρ) quantile of C_ij over the current
   particles, with ρ ≈ 0.1–0.5. The targets are π_l = uniform on Ω_l = {C_ij ≥ ℓ_l}, so the weight
   function is the indicator 1{C_ij ≥ ℓ_l} (FK Eq 10, with g an indicator).
3. At each level:
   - record ρ̂_l, the surviving fraction;
   - **resample** the survivors (FK §3.1, step 1, with systematic resampling as FK recommend in §3.1);
   - **move** each particle with curveball trades, rejecting any trade that leaves Ω_l. The curveball
     kernel is symmetric with uniform stationary law on Ω (Carstens 2015, *Phys. Rev. E*, "Proof of
     uniform sampling … for the fast Curveball algorithm") (exists). Restricting a symmetric
     proposal to Ω_l by rejection leaves uniform-on-Ω_l invariant. Trades that involve neither
     row i nor row j never change C_ij = |row_i ∩ row_j| and are always accepted.
4. The estimate is p̂ = Π_l ρ̂_l. This is the SMC normalizing-constant estimator, the analogue of the
   unbiased likelihood estimate in FK §3.1, step 4. Its relative variance grows like
   Σ_l (1−ρ_l)/(ρ_l N), which is roughly logarithmic in 1/p, whereas plain Monte Carlo grows like 1/p
   (compare the importance-sampling variance in ZC Eqs 61–62, and footnote 40: the zero-variance
   proposal is q ∝ |f|·p, which here means concentrating on the tail event).
   - Example: p = 10⁻⁸, ρ = 0.1 gives L = 8 levels; N = 1000 particles gives a relative error of
     about √(8·9/1000) ≈ 0.27.

**API.** `fdsm(..., refine=None | "splitting", refine_particles=1000, refine_level_frac=0.1)`.
Only pairs whose plain-Monte-Carlo count is 0 (or ≤ c) are refined. Their `fdsm_pvalue` is replaced
and `fdsm_refined=True` is set. Report `fdsm_pvalue_se` from repeated independent runs or the
variance formula above.

**Alternative considered.** Sequential importance sampling for 0-1 tables with fixed margins
(Chen, Diaconis, Holmes & Liu 2005, *JASA*) (exists) samples tables column by column, with
importance weights (ZC §V-G.1, Eqs 58–60) and effective-sample-size monitoring (ZC §VI-A, N_eff).
It gives independent samples with no burn-in. But it does not cure the rare-event floor unless the
proposal is tilted toward {C_ij ≥ k*} per pair. It needs a new sampler, and its weights degenerate
for large tables (FK §5.1). I deferred it in favor of splitting, which reuses
`_curveball_trades` unchanged apart from the constraint check.

**Dependencies.** numpy and scipy, as now.

**Complexity.** Per refined pair: O(L · N · M · d), where M is the moves per level and d the cost
of a curveball trade. This is roughly 10⁴–10⁵ trades per level for N = 1000, so it is practical for
tens to hundreds of floor pairs, not for all m pairs. The pairs are independent, so it parallelizes
trivially.

**Validation.**

- **Exact enumeration** of all fixed-margin matrices for tiny cases (for example 5 × 6 with modest
  margins) gives exact tail probabilities. Check that the mean of p̂ over repeated runs matches them
  (unbiasedness) and that the empirical relative error matches the variance formula.
- Agreement with plain `fdsm` at very large `trials` for p ≈ 10⁻⁴.
- Monotonicity: larger k* gives smaller p̂.

**Risks.**

- **Irreducibility.** The curveball chain restricted to Ω_l may not be irreducible. If Ω_l splits
  into components that single trades cannot connect, the estimate is biased. The enumeration tests
  must probe this; a mitigation is to allow compound moves.
- **Cost per pair.**
- **Value.** The value depends on how often users need exact FDSM at small alpha. Neal, Domagalski &
  Sagan (2021), already cited in the README, compare alternatives to FDSM; from memory they find
  SDSM a close approximation (unverified). For many users, `sdsm` is the practical answer.
- **GloSS-style global nulls** (Radicchi, Ramasco & Fortunato 2011, *Phys. Rev. E*) (exists) are not
  in the library. The splitting estimator is generic for any simulation null with a level-set move
  kernel, but adding GloSS is a separate method request not motivated by these papers.

**Priority: Medium.** A real correctness and power gap (P3's demonstration), solid theory and reuse
of existing code, but niche and with an irreducibility risk that needs careful testing.

---

### P5. `wavelet_denoise`: Haar shrinkage of edge-weight time series (low)

**Problem.** Edge-weight time series mix persistent structure, abrupt regime changes and noise. P1's
smoother is a *linear* filter, so it blurs jumps. SV §5.2 makes the same contrast in turbulence:
"what distinguishes CVS from LES is the use of wavelet threshold filters, which depend on the
instantaneous flow realization, whereas LES uses linear low-pass filters". Nonlinear wavelet
thresholding keeps localized jumps sharp (SV §2.2; SV §1 on localized, intermittent structure).

**Method.** For each edge e, take its weight series w_{e,1..T}, with zero for absent edges.

1. **Optional variance stabilization** for counts: the Anscombe transform 2√(w + 3/8), after which
   Poisson noise has variance ≈ 1 (Anscombe 1948, *Biometrika*) (exists). This makes SV §4.1's
   additive white Gaussian noise model approximately valid with a known σ = 1.
2. **Orthonormal Haar transform** of the series (SV Eqs 1–2). It is implemented natively with
   pairwise averages and differences, O(T). For T not a power of two, or for unequal `times`, use
   lifting-scheme (second-generation) Haar. SV §2.1 notes that second-generation wavelets (Sweldens
   1996) handle "arbitrary boundary conditions and irregular sampling intervals".
3. **Hard-threshold the detail coefficients** at ε = σ√(2 ln N) (SV Eq 3 and §4.1, CVE threshold
   "motivated by denoising theory (Donoho & Johnstone 1994)"). The coarsest scaling coefficient is
   never thresholded (SV Eq 3 keeps all ū).
   - **σ.** Known (1) after Anscombe. Otherwise it is estimated from the pooled finest-scale
     coefficients, either iteratively as in SV §4.1 (which cites the iterative algorithm of
     Azzalini et al. 2005 but does not spell it out) or by the standard median-absolute-deviation
     rule (unverified detail).
   - **N.** Choose N = E·T, the total number of coefficients across all edges, so that with high
     probability no pure-noise coefficient survives anywhere. This uses the universal-threshold
     property of Donoho & Johnstone (1994, *Biometrika*, "Ideal spatial adaptation by wavelet
     shrinkage") (exists). SV only cite it for optimality.
4. **Reconstruct.** The coherent part is w^C (SV Eq 11 split, ω = ω_C + ω_I). With orthonormal
   wavelets the energy splits additively (SV §4.1, enstrophy conservation).

**Outputs.** Per snapshot copy:

- `wavelet_weight` = the denoised weight, back-transformed if stabilized.
- Per edge, repeated on each copy: `wavelet_keep` = True if any detail coefficient survives (the edge
  has temporal structure beyond noise), and `wavelet_scale` = the scale with the most coherent detail
  energy. SV §4.2 treats wavelet coefficients as "generalized increments" for scale-dependent
  statistics. This diagnostic helps choose the snapshot aggregation window.

**Intended use.** Denoise first, then backbone:
`[nb.disparity_filter(H, weight="wavelet_weight") for H in nb.wavelet_denoise(snaps)]`, or feed P1.

**Why low priority, honestly.** Wavelet shrinkage is **not a backbone criterion**. The scaling
coefficient (the edge's mean level) is never thresholded, so a weak constant edge and a strong
constant edge are both untouched. Deciding which is significant remains the static scorer's job.
Other limits:

- Typical T is 10–100, so there are only a few dyadic scales.
- Anscombe is poor for mean counts below about 4. Many edges are sparse.
- P1 already covers the main use (temporal pooling) with exact null calibration.

**Dependencies.** None for Haar (numpy). PyWavelets could be an optional lazy import for other
families (`wavelet="db4"`) behind a new `[wavelet]` extra. I recommend **not** adding it at first:
Haar is the natural basis for piecewise-constant tie strength, and an extra dependency is not
justified until users ask for smoother wavelets.

**Complexity.** O(E·T).

**Validation.**

- Pure noise (Gaussian, or Poisson with Anscombe) gives `wavelet_keep` false for ≥ 95% of edges.
- Piecewise-constant series with one jump are recovered: jump location and level, versus P1's
  blurred estimate.
- Perfect reconstruction with zero threshold.
- Energy additivity (Parseval).

**Risks.** The meaning of N, and therefore of the threshold, across edges. Zero-filling absent
edges. Boundary handling for odd T.

**Priority: Low.**

---

### P6. `poisson_lift_filter`: conjugate count state-space with null exposure (low, idea)

**Problem.** For integer weights (counts of interactions), a natural dynamic analogue of the
noise-corrected or "lift" view is this: is the edge stronger than its endpoints' current activity
predicts, and is it *becoming* stronger than its own history predicts?

**Method.** y_{e,t} ~ Poisson(λ_{e,t} · e_{e,t}), with exposure e_{e,t} = s_i s_j / (2W) taken from
snapshot t's strengths. This is the configuration-style expectation also behind
`marginal_likelihood_filter`'s binomial null. λ is the latent lift, with Gamma prior and
discount-factor evolution. The conjugate family keeps the filter finite-dimensional:

- ZC §IV-C.1, Theorem 1 (Eq 47): a fixed finite-dimensional filter exists if and only if the
  unnormalized conditional density is exponential-family.
- ZC §VII-A: the conjugate latent process (Vidoni).
- The Gamma–Poisson local-level form is Harvey & Fernandes (1989, *JBES*) (exists); its details are
  from memory (unverified).

The recursions are:

- a_{t|t−1} = δ a_{t−1}, b_{t|t−1} = δ b_{t−1}
- a_t = a_{t|t−1} + y_t, b_t = b_{t|t−1} + e_t
- One-step predictive: y_t | y_{1:t−1} ~ NegBin(a_{t|t−1}, b_{t|t−1}/(b_{t|t−1} + e_t)). This is
  FK Eq 6's predictive p(y_t | y_{1:t−1}) in closed form.

**Outputs.** `plift_level` = a_t/b_t (the dynamic lift). `plift_pvalue` = the mid-p upper tail of the
one-step predictive at the observed y_t, an innovation or burst test analogous to AAK Eq 18.
`plift_score` = the predictive z. δ is estimated by pooled predictive likelihood (WS Eq 75).

**Honest caveat.** `plift_pvalue` measures surprise **relative to the fitted dynamic model**. It is
not a test against a no-signal null. For a backbone test under H0 λ ≡ 1, use the per-snapshot
Poisson tail and feed it to P1 instead. P6's distinct value is the interpretable lift trajectory
and burst detection. If the Gamma discount model is inadequate (for example, a log-normal random
walk for log λ), the right numerical method is a 1-D **grid (point-mass) filter** (ZC §V-C), not a
particle filter (see "Not recommended").

**Dependencies.** scipy (`nbinom`). **Complexity.** O(E·T).

**Validation.**

- Under data simulated from the model, the probability integral transforms of the predictive are
  uniform.
- δ = 1 with constant exposure reduces to the static Poisson-Gamma posterior.
- Detect planted bursts in synthetic count snapshots.

**Risks.** Exposure is undefined when an endpoint has zero strength in a snapshot. Overdispersion.
Overlap with P1.

**Priority: Low (idea).** Worth a design note, not an implementation now.

---

## 4. Not recommended

| Technique (source) | Why rejected |
|---|---|
| **Particle filters for edge states**: bootstrap, SIR and auxiliary filters (FK §§3.1–3.2, Eqs 9–11; WS §§3.1–3.5, Algorithm 1, Eq 70; ZC §VI) | Each edge's latent state is scalar, and the useful models are linear-Gaussian (P1), conjugate (P6) or finite-state (P2), all with *exact* filters. For non-conjugate 1-D models a grid filter is better: ZC §V-G notes that the Monte Carlo error O(N^{-1/2}) is slower than deterministic quadrature's O(N^{-1}) in one dimension. A cost of E·T·N in Python is prohibitive. A *joint* all-edges state is high-dimensional, where particle filters collapse (FK §5.1; WS §3.6: impractical beyond n_x ≈ 10). |
| **Particle MCMC**: particle marginal Metropolis–Hastings and particle Gibbs (FK §7.2; WS §4.1, Eq 80) | Built for intractable likelihoods. Here the likelihood is exact via the Kalman prediction-error decomposition (WS Eq 75 evaluated in closed form). Not needed. |
| **Ensemble Kalman filter / data assimilation as "null prior + observed weights"** (FK §3.5, Eq 12, §5.2–5.3, §6.2; AAK §4.1; this was candidate (d) in the brief) | A static graph has no dynamical model to propagate an ensemble. With H = I, the Kalman update of (null expectation, observation) is plain Gaussian shrinkage in closed form and needs no ensemble. Its useful core is already inside P1 (the null enters through per-snapshot z-scores) and P6 (the exposure). Using null-model samples (for example FDSM draws) as an ensemble to estimate an edge-by-edge covariance would mean n_ens ≪ E samples. That is exactly the covariance-instability regime FK §5.2 describes (spurious off-diagonal terms, needing inflation or tapering). The output would be a shrunken weight with no significance interpretation. Localization on graph distance (FK §5.3) solves a problem we do not have. |
| **Extended, unscented, cubature and Kalman–Bucy filters; MMAE and IMM banks; quantum Kalman filters** (AAK §3, §§5–8; ZC §IV-C, §VII-C) | The proposed models are linear or conjugate, so there is no nonlinearity to linearize. Snapshot data are discrete-time. IMM's mode switching is captured exactly by P2's discrete HMM. The Gaussian Sum Filter idea survives only as P1's optional robust variant. |
| **Quasi-Monte Carlo filters, transport filters, differentiable particle filters, twisting potentials** (FK §3.3, §3.7.1; WS §5) | No corresponding network problem. Twisting is conceptually P4's "concentrate on the tail" idea, but gives no additional practical algorithm here. |
| **Plain sequential importance sampling for fixed-margin tables** (Chen et al. 2005; ZC §V-G.1, §VI-A) | Considered as the P4 alternative and deferred: it needs a new sampler, has no rare-event benefit without per-pair tilting, and its weights degenerate on large tables. |
| **Spectral graph wavelet thresholding of edge or node signals** (Hammond, Vandergheynst & Gribonval 2011, *ACHA*) (exists; **not** covered by SV) | SV covers only tensor-product and second-generation wavelets on meshes and spheres, and calls non-tensorial wavelets on unstructured meshes immature (§3.6). On a graph, thresholding small wavelet coefficients keeps regionally smooth weight patterns and *removes edges whose weights are locally anomalous* compared with their neighbors. A strong tie among weak neighbors is precisely what local backbones (disparity, LANS) want to *keep*. It also needs a signal domain (the line graph, with Σk² vertices, for edge signals) and has no null model. Rejected for backbones; possibly useful for denoising node attributes, which is out of scope. |
| **Adaptive wavelet Galerkin and collocation methods, safety (adjacent) zones, CVS, SCALES, Brinkman penalization, space-time wavelets** (SV §§3, 5) | These are PDE-solver machinery with no network analogue. Space-time wavelets on graph × time would require the rejected graph wavelets. |
| **Naming** | Avoid "multiscale" for P5. Serrano et al.'s "multiscale backbone" (`disparity_filter`) means weight heterogeneity, not wavelet scales. |

---

## 5. Paper-by-paper relevance

| Paper | Verdict | What was used | Proposals |
|---|---|---|---|
| **FK** (Fearnhead & Künsch 2018) | **Substantial.** The most useful of the five. | Eqs 1–2 (state space); Eq 6 (one-step predictive, used for likelihood and innovations); Eqs 7–8 (predict and update); §3.1 (resampling, unbiased normalizing constant); §3.4 (SMC over intermediate distributions); §3.5 (Kalman update and gain); §3.7.3 (robust filters); §6.1 and Eq 14 (forward-backward smoothing; discrete HMM analogue). Its negative results (§5.1 weight collapse, §5.2 EnKF covariance instability) underpin several rejections. | P1, P2, P4; rejections |
| **WS** (Wills & Schön 2023) | **Modest.** It largely duplicates FK; no technique is uniquely taken from it. | Eqs 30–33 (Bayes filter); Eq 37 (importance-sampling identity); Eq 75 (prediction-error likelihood, used for estimating q and δ); Eqs 76–77 (EM, the pattern for Baum–Welch); Table 1 (filter versus smoother modes); §3.6 (dimension caveat). | P1, P2, P6 (supporting); rejections |
| **AAK** (Anagnostaki et al. 2026) | **Low.** A textbook derivation plus physics applications. The applications (thermo-fluids, electromagnetism, high-energy physics, quantum, astrophysics) offer nothing transferable. | Eqs 10–11, 16, 18 (innovation), S_k (§2.3.5), Eq 31 (gain), Eq 32, Eqs 33 and 36. **Eq 23 (Joseph form, valid for any gain)** is genuinely useful: it gives P1's exact null variance in filter mode. §8.2 Gaussian Sum Filter suggests P1's robust option. §3 IMM is the conceptual link to P2. | P1 (supporting); P2 (conceptual) |
| **ZC** (Chen 2003) | **Moderate.** The broadest toolbox. | Eqs 23–25 (recursion); §IV-B (Kalman, gain Eq 38, RTS remark, footnote 29 on innovation whiteness as a diagnostic); §IV-C.1 Theorem 1 / Eq 47 and §VII-A (conjugate finite-dimensional filters); §V-C Eqs 50–52 and §VII-E (exact grid and HMM filters, Baum–Welch, Viterbi); §V-G Eq 57 (Monte Carlo error rate) and Eqs 58–62 with footnote 40 (importance sampling, zero-variance proposal); §VI-A (N_eff); §VII-C (IMM). | P1, P2, P3, P4, P6; rejections |
| **SV** (Schneider & Vasilyev 2010) | **Low–moderate.** Only the denoising part transfers, and only to edge *time series*. All CFD numerics are irrelevant, and graph wavelets are not covered. | Eqs 1–4 (expansion, thresholding filter, error bound); §2.2 (Donoho–Johnstone denoising); §4.1 (CVE threshold σ√(2 ln N), iterative σ estimation, coherent/incoherent split Eq 11); §2.1 (second-generation wavelets for irregular sampling); §4.2 (scale-dependent statistics); §5.2 (nonlinear versus linear filtering); §3.6 (non-tensorial wavelets immature, which supports rejecting graph wavelets). | P5; rejections |

---

## 6. Suggested sequencing

1. **P3** (one small PR): diagnostics and warning. It prevents silent empty backbones today.
2. **P1** together with the shared temporal infrastructure (Section 2): the `temporal` module,
   `snapshots_from_events`, the API-contract `kind="temporal"`, `describe` entries, and a tutorial
   on synthetic SBM snapshots. Settle the return type and the `missing` default with the maintainer
   first; this connects to roadmap #26.
3. **P2**, reusing the snapshot-panel helper from P1.
4. **P4**, behind `refine=`, after the exact-enumeration test harness exists.
5. **P5 and P6** only on user demand. File P6 as an "idea" item on the roadmap, alongside #26 and
   #29.

## 7. Prototype

[`docs/design/prototypes/kalman_proto.py`](prototypes/kalman_proto.py) (needs numpy and scipy) contains:

- a vectorized local-level filter and RTS smoother;
- the brute-force null sd and the O(T) null-variance recursion;
- the calibration, missingness and power checks;
- a pooled log-likelihood scan over q.

The outputs quoted in P1 come from this script with seed 0.
