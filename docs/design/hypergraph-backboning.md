# Design proposal: hypergraph network backboning

- **Status:** Draft / for discussion
- **Scope:** Evaluation + proposed integration plan (no library code changed by this document)
- **Trigger:** Request to evaluate incorporating hypergraph network backboning
  methods (in the spirit of arXiv:2606.00893) into `networkx-backbone`, with
  optional interoperability for the `xgi`, `hypergraphx`, `HyperNetX`, and
  `Hypergraph-Analysis-Toolbox` libraries, and an assessment of which hypergraph
  backbone methods are genuinely distinct (sui generis) versus generalizations of
  existing methods.

---

## 1. Summary and recommendation

**Recommendation: incorporate hypergraph backboning in phases, treating it as two
distinct problems, add optional interoperability with the major hypergraph
libraries (centered on the HIF interchange format), and port a small set of
genuinely hypergraph-native ("sui generis") methods that the current ensemble
cannot express.**

Hypergraph backboning in the literature splits into two families with *different
output types*:

| Family | What it does | Output | Library readiness |
|--------|--------------|--------|-------------------|
| **A — Projection backboning** | Hypergraph → weighted pairwise graph → null-model edge test | a normal graph | **~90% already built** (verified working today) |
| **B — Direct hyperedge filtering** | Keep statistically/structurally significant *hyperedges* | a sub-hypergraph | **new surface** (needs representation + hyperedge filters + an output type) |

Key conclusions:

1. **Family A already works** with today's API: a hypergraph's incidence matrix
   *is* a bipartite graph (nodes ↔ hyperedges), and the `bipartite` module already
   runs degree-preserving null models (`sdsm`, `fdsm`, `fixedrow`, ...) on it
   (verified end-to-end, §5).
2. **Most weighted-hypergraph backbones are generalizations** of methods we already
   ship and become reachable for free once a hypergraph can enter the pipeline
   (§6).
3. **A few methods are genuinely sui generis** — they have no clean dyadic analog
   and the current ensemble cannot produce them. These are worth porting: the
   **Statistically Validated Hypergraph (SVH)** and **Statistically Validated
   Cores (SVC)** filters, **toplex/inclusion (encapsulation) reduction**,
   **s-connectivity / s-line-graph backbones**, and **order-resolved hyperedge
   filtering** (§6).
4. **All four target libraries already converge on a common substrate** (an
   incidence/bipartite representation) and on the **HIF** JSON interchange format,
   so optional interoperability is cheap and need not add any *required*
   dependency (§7).

Proposed phasing (details in §9):

- **Phase 0** — surface Family A + add input adapters (HIF + per-library
  converters) so hypergraphs from any of the four libraries can enter the
  pipeline. Near-zero risk.
- **Phase 1** — a `hypergraph` module implementing the sui generis methods (SVH,
  SVC, toplex/inclusion, s-line/s-connectivity, order-resolved), natively on an
  incidence substrate, returning a sub-hypergraph and able to export back to the
  external libraries / HIF.
- **Out of scope** — hypergraph neural networks / representation learning; making
  any third-party hypergraph library a *required* dependency.

## 2. Note on the source paper (arXiv:2606.00893)

The specific paper could not be retrieved while preparing this proposal:

- The execution environment's network egress is allow-listed and excludes
  `arxiv.org`, `export.arxiv.org`, and the relevant docs hosts ("Host not in
  allowlist"); arXiv additionally returns HTTP 403 to automated fetchers.
- The ID `2606.00893` corresponds to **June 2026** and was only ~1 day old at the
  time of writing, so it is not yet indexed by web search or paper hubs.

Consequently this proposal is grounded in (a) a full reading of this library,
(b) the source code and READMEs of the four target libraries (fetched from
`raw.githubusercontent.com`, which *is* reachable), and (c) the established
hypergraph-backboning literature any such paper builds on (see §11) — **not** on
the paper's exact formulation. §10 lists the specific details to confirm against
the paper before implementing Phase 1.

## 3. Background: what "hypergraph network backboning" means

A hypergraph `H = (V, E)` has hyperedges `e ⊆ V` that may join more than two
nodes. "Backboning" means keeping only its most informative structure. Two
distinct families exist:

- **Family A — projection backboning.** Represent `H` as an incidence
  (node × hyperedge) structure, project to a node–node weighted graph, then apply
  a degree-preserving null model to decide which *pairwise* links are significant.
  Lineage: Neal's `backbone` R package (Backbone 3.0, PLOS One 2026, explicitly
  supports hypergraph-projection backbones) and Coscia & Neffke (2017). Output is
  an ordinary graph and fits the existing score-then-filter idiom.
- **Family B — direct hyperedge filtering.** Keep the *hyperedges themselves* that
  are significant (statistically over-expressed, or structurally essential).
  Reference: Musciotto, Battiston & Mantegna, "Detecting informative higher-order
  interactions in statistically validated hypergraphs," *Communications Physics*
  (2021). Output is a *subset of hyperedges* (a sub-hypergraph), which does **not**
  map onto the current graph-in/graph-out filter functions.

Given the phrase "hypergraph network backboning," the source paper is most
plausibly in Family B.

## 4. Current library capabilities relevant to this

- **Score-then-filter pattern.** Methods annotate edges with a score
  (e.g. `disparity_pvalue`) and return a copy of the graph; `threshold_filter` /
  `boolean_filter` / `fraction_filter` then extract the subgraph.
- **Dependency policy.** Core requires only `networkx>=3.0`; `numpy`/`scipy` are
  the optional `[full]` extra, imported lazily inside functions.
- **The `bipartite` module already is an incidence engine.**
  `_bipartite_projection_matrix(B, agent_nodes)` builds the binary incidence matrix
  `R` (agents × artifacts) and the co-occurrence `R @ R.T`; on top of it the module
  provides `sdsm`, `fdsm`, `fixedfill`/`fixedrow`/`fixedcol`, plus reusable
  randomizers `fastball` and `_random_bipartite_matrix`, and `bicm` probabilities.

A hypergraph encoded with nodes in one partition and hyperedges in the other is
*exactly* the input these functions already expect.

## 5. Key finding: Family A already works today

```python
import networkx as nx
import networkx_backbone as nb

hyperedges = {"H1": [1, 2, 3], "H2": [1, 2, 3], "H3": [3, 4, 5, 6], "H4": [5, 6]}
nodes = sorted({v for members in hyperedges.values() for v in members})

B = nx.Graph()                                 # incidence bipartite graph
B.add_nodes_from(nodes, bipartite=0)           # nodes
B.add_nodes_from(hyperedges, bipartite=1)      # hyperedges as "artifacts"
for h, members in hyperedges.items():
    for v in members:
        B.add_edge(v, h)

scored   = nb.sdsm(B, agent_nodes=nodes, projection="hyper")
backbone = nb.threshold_filter(scored, "sdsm_pvalue", 0.30, mode="below")
# backbone.edges() -> [(1, 2), (4, 5), (4, 6), (5, 6)]   (verified)
```

`fdsm(...)` works identically with Monte-Carlo, exactly-degree-preserving nulls.
A large slice of "hypergraph backboning" is therefore a **latent, undocumented
capability** today.

## 6. Method inventory: generalizations vs. sui generis methods

The user's question — are hypergraph backbone methods just generalizations of what
we have, or are some genuinely distinct? — resolves as **"mostly generalizations,
plus a short list of genuinely hypergraph-native methods."**

### 6.1 Generalizations (already reachable, or trivially so)

These reduce to existing methods once a hypergraph enters the pipeline as its
incidence/bipartite form or via an expansion. **No new algorithms needed** beyond
the Phase 0 adapters.

| Hypergraph method | Reduces to | Notes |
|-------------------|-----------|-------|
| Hyperedge global-weight threshold | `global_threshold_filter` | trivial |
| Degree-preserving projection null (SDSM/FDSM/fixed*) | `sdsm`/`fdsm`/`fixed*` on incidence | **already works** (§5) |
| Disparity / MLF / LANS / noise-corrected / ECM on the projection | the matching statistical filter | applied to the projected graph; the only wrinkle is the "which incident node's disparity" choice, identical to the bipartite case |
| Clique-expansion or line-graph + any graph backbone | existing graph methods | apply a transform, then any current method |

### 6.2 Sui generis methods (no clean dyadic analog — worth porting)

These cannot be produced by the current ensemble, primarily because (a) the
hypothesis or structure is defined over *groups of arbitrary size*, and (b) the
output is a *sub-hypergraph*, not a graph.

| Method | Type | Reference / reference impl | Why it is distinct |
|--------|------|----------------------------|--------------------|
| **Statistically Validated Hypergraph (SVH)** | statistical | Musciotto+ 2021; HGX `get_svh` | Tests each *hyperedge of order k* for over-expression under a node-degree-preserving null, with FDR correction across tests. The hypothesis is about a k-node group, not a dyad — irreducible to pairwise filtering. |
| **Statistically Validated Cores / significant interacting groups (SVC)** | statistical | HGX `get_svc` | Validates significant *groups* (cores) order-by-order, including groups not present as a single hyperedge. Complements SVH. |
| **Toplex / inclusion (encapsulation) reduction** | structural | HNX `toplexes()`; XGI `encapsulation_dag` | Keep only maximal hyperedges (or filter nested/encapsulated ones). A "subset-of" relation between edges has no analog in simple graphs. |
| **s-connectivity / s-line-graph backbone** | structural | HNX `s_components`, `s_connected_components`, s-line graph | Two hyperedges are *s-adjacent* if they share ≥ s nodes. Backbones that preserve s-components, or that backbone the (weighted) s-line graph, form a family parameterized by `s` with no dyadic counterpart. |
| **Order-resolved hyperedge filtering** | structural / utility | building block of SVH | Score/keep hyperedges per order (size). Meaningful only because hyperedges have variable arity; also the substrate for SVH/SVC. |

**Characterization.** The *statistical* sui generis methods (SVH/SVC) are best
described as higher-order descendants of statistically-validated-network ideas,
but the group-level hypothesis and sub-hypergraph output make them irreducible to
any pairwise backbone in our ensemble. The *structural* ones (toplex/inclusion,
s-connectivity, order-resolved) are genuinely unique to set systems.

**Recommendation.** Port SVH and SVC first (statistical; likely the source paper's
family), then the structural trio. Implement them **natively** on our incidence
substrate (numpy/scipy only), so they require no third-party hypergraph library;
use HGX/HNX as references for correctness fixtures and as optional fast paths
(§7).

## 7. Optional interoperability with hypergraph libraries

The four target libraries differ in focus but **converge on the same substrate**
— every one can produce/consume an incidence matrix and/or a bipartite
representation, and **all four support the HIF JSON interchange format**. This
makes optional interop cheap and keeps the core `networkx`-only.

### 7.1 What each library exposes (verified from source)

| Library | Hypergraph type(s) | → incidence / bipartite (into our pipeline) | ← construct (from our output) | HIF |
|---------|--------------------|---------------------------------------------|-------------------------------|-----|
| **XGI** (`xgi`) | `Hypergraph`, `DiHypergraph`, `SimplicialComplex` | `xgi.to_bipartite_graph(H)`, `xgi.to_incidence_matrix(H)` | `xgi.from_bipartite_graph(B)`, `xgi.from_incidence_matrix(M)` | `xgi.read_hif` / `xgi.write_hif` |
| **HyperNetX** (`hypernetx`) | `Hypergraph` | `H.bipartite()`, `H.incidence_matrix()`, `H.incidence_dict` | `Hypergraph.from_bipartite(B)`, `H.restrict_to_edges(keep)` | supported |
| **HypergraphX** (`hypergraphx`) | `Hypergraph`, `Temporal/Directed/Multiplex` | `H.binary_incidence_matrix(return_mapping=True)` | `Hypergraph(edge_list=...)` | supported |
| **HAT** (`HAT`) | `Hypergraph` (tensor/incidence) | `H.incidence_matrix` | `Hypergraph(incidence_matrix=...)` | import/export |

Notes:
- **HNX `restrict_to_edges`** is the natural way to return a sub-hypergraph backbone
  in HNX terms; **XGI `encapsulation_dag`** and **HNX `toplexes`** directly support
  the inclusion-reduction method (§6.2).
- **HAT** is tensor/controllability/entropy-focused; it contributes interop value
  (and an incidence matrix), not new backbone methods.

### 7.2 Proposed interop design

Two complementary, fully optional layers — neither becomes a hard dependency
(adapters lazily import the third party and raise a friendly `ImportError` if it
is absent; HIF needs only the stdlib `json`):

1. **HIF as the primary hub (recommended).** Add dependency-free `read_hif(path)`
   / `write_hif(H, path)` that parse/emit the HIF JSON schema into our internal
   incidence form. Because XGI, HGX, HNX, and HAT all read/write HIF themselves,
   this yields universal round-tripping with *zero* third-party dependencies and
   minimal maintenance.
2. **Thin direct adapters (ergonomic convenience).** `from_xgi`/`to_xgi`,
   `from_hypernetx`/`to_hypernetx`, `from_hypergraphx`/`to_hypergraphx`,
   `from_hat`/`to_hat`. Each is ~10–20 lines because each library already exposes
   incidence/bipartite converters (table above). Example sketch:

   ```python
   def from_xgi(H):
       """Convert an xgi.Hypergraph to our incidence bipartite graph (lazy import)."""
       import xgi  # optional; raises ImportError with install hint if missing
       return xgi.to_bipartite_graph(H)        # already a NetworkX bipartite graph

   def to_hypernetx(hyperedges):
       import hypernetx as hnx
       return hnx.Hypergraph(hyperedges)
   ```

3. **Packaging.** Add extras so users can opt in:
   `pip install networkx-backbone[xgi|hypernetx|hypergraphx|hat]` (and a `hif`
   extra is unnecessary — HIF is stdlib-only). Core install is unchanged.

This means a user can take a hypergraph from *any* of the four libraries, run our
backbone methods, and hand the result back to their library of choice.

## 8. Gap analysis and challenges

1. **No native hypergraph type in NetworkX.** A representation must be chosen
   (§9.1). This is the central decision.
2. **Output-type mismatch for Family B / sui generis methods.** A sub-hypergraph
   cannot flow through `threshold_filter`. Options: return the kept hyperedges
   (list/`frozenset`s), annotate the *hyperedge* nodes of the bipartite encoding
   and filter those (preserving the idiom), and/or return an external-library
   object (`restrict_to_edges`, etc.).
3. **Dependency policy.** Keep `networkx`-only core; sui generis methods need only
   `numpy`/`scipy` (already the `[full]` extra). The four libraries stay *optional*
   extras; HIF needs only the stdlib.
4. **Multiple testing and cost.** SVH/SVC test one hypothesis per candidate group,
   so they need FDR/Bonferroni correction (HGX uses FDR) and benefit from the
   existing Monte-Carlo randomizers and per-order guardrails.
5. **Scope discipline.** Stay within classical backboning. Hypergraph neural
   networks are out of scope.

## 9. Proposed design and phased plan

### 9.1 Hypergraph representation

Primary internal representation: **the incidence form** — a list of hyperedges
(tuples/`frozenset`s) plus a node ordering, with the equivalent incidence bipartite
graph and incidence matrix available on demand. Rationale: zero new dependencies,
reuses the entire `bipartite` engine, matches the substrate all four libraries
share, and round-trips through HIF.

| Option | Pros | Cons |
|--------|------|------|
| **Incidence form / bipartite graph** (recommended) | no new deps; reuses null models; matches every target library + HIF | hyperedge identity lives in labels |
| Lightweight `list[frozenset]` + matrix | natural sub-hypergraph output | minor plumbing |
| Direct dependence on one library's type | rich features | violates dependency policy; picks a winner |

### 9.2 Phase 0 — surface Family A + input adapters

- Converters: `hypergraph_to_bipartite(hyperedges)`, `incidence_to_bipartite(M)`,
  `read_hif`/`write_hif`, and the four `from_*`/`to_*` adapters (§7).
- A tutorial showing SDSM/FDSM/fixed-model hypergraph projection backbones and
  ingestion from each library.
- Tests + a note in `docs/concepts.rst`. No changes to existing functions.

### 9.3 Phase 1 — `hypergraph` module (sui generis methods)

Native implementations on the incidence substrate, preserving score-then-filter.
Indicative API (finalize against the paper and HGX for SVH/SVC):

```python
# networkx_backbone/hypergraph.py
def svh(hyperedges, max_order=None, alpha=0.05, correction="fdr", seed=None): ...
def svc(hyperedges, min_order=2, max_order=None, alpha=0.05, correction="fdr"): ...
def toplex_backbone(hyperedges): ...               # inclusion / encapsulation reduction
def s_line_backbone(hyperedges, s=1, method="disparity", **kw): ...  # backbone the s-line graph
def order_filter(hyperedges, min_order=2, max_order=None): ...

def hyperedge_filter(scored, score="svh_pvalue", alpha=0.05): ...   # -> sub-hypergraph
```

- Reuse `fastball` / `_random_bipartite_matrix` for null ensembles; lazy
  `numpy`/`scipy` imports per the existing convention.
- Output: the kept hyperedges; optionally also the bipartite encoding (so existing
  graph filters apply) and/or an external-library object via the §7 adapters.

### 9.4 Testing strategy

- Phase 0: round-trip converter tests for each library + HIF; equivalence test that
  `hypergraph_to_bipartite(...) + sdsm` matches the manual recipe.
- Phase 1: known-answer tests on tiny hypergraphs (planted over-represented group
  retained; random groups not); determinism via `seed`; FDR behavior; degenerate
  inputs; **cross-validation against HGX `get_svh`/`get_svc` and HNX `toplexes`**
  where those libraries are installed (skipped otherwise).

## 10. Open questions to confirm against arXiv:2606.00893

1. Which family — direct hyperedge filtering (B) or a projection method (A)?
2. The exact null model (node-degree-preserving, hyperedge-size-preserving, both).
3. Test statistic and the multiple-comparison correction used.
4. Output: a sub-hypergraph, a validated projection, or both.
5. Whether it coincides with SVH/SVC or is a distinct method to add to §6.2.

## 11. References

- Musciotto, Battiston & Mantegna — Detecting informative higher-order
  interactions in statistically validated hypergraphs — Communications Physics
  (2021). https://www.nature.com/articles/s42005-021-00710-4 (arXiv:2103.16484)
- Backbone 3.0: An R package for extracting network backbones — PLOS One (2026).
  https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0349258
- Coscia & Neffke — Network backboning with noisy data — arXiv:1906.09081.
  https://arxiv.org/pdf/1906.09081
- HIF: The hypergraph interchange format for higher-order networks — Network
  Science / arXiv:2507.11520. https://arxiv.org/html/2507.11520v1 ·
  standard: https://github.com/HIF-org/HIF-standard
- Hypergraphx (HGX) — J. Complex Networks (2023), arXiv:2303.15356.
  https://github.com/HGX-Team/hypergraphx
- HyperNetX (HNX) — arXiv:2310.11626. https://github.com/pnnl/HyperNetX
- XGI — https://github.com/xgi-org/xgi
- Hypergraph Analysis Toolbox (HAT) — PLOS Comput. Biol. (2023).
  https://github.com/Jpickard1/Hypergraph-Analysis-Toolbox
