# Design proposal: hypergraph network backboning

- **Status:** Draft / for discussion
- **Scope:** Evaluation + proposed integration plan (no library code changed by this document)
- **Trigger:** Request to evaluate incorporating hypergraph network backboning
  methods (in the spirit of arXiv:2606.00893) into `networkx-backbone`.

---

## 1. Summary and recommendation

**Recommendation: incorporate hypergraph backboning, in two clearly separated
phases, treating it as two distinct problems.**

Hypergraph backboning in the literature splits into two families with *different
output types*:

| Family | What it does | Output | Library readiness |
|--------|--------------|--------|-------------------|
| **A — Projection backboning** | Hypergraph → weighted pairwise graph → null-model edge test | a normal graph | **~90% already built** (verified working today) |
| **B — Direct hyperedge filtering** | Keep statistically over-represented *hyperedges* | a sub-hypergraph | **new surface** (needs representation + hyperedge filters + an output type) |

The key finding is that **Family A already works** with today's API, because a
hypergraph's incidence matrix *is* a bipartite graph (nodes ↔ hyperedges) and the
`bipartite` module already builds that matrix and runs degree-preserving null
models (`sdsm`, `fdsm`, `fixedrow`, ...) on it. The main gap for Family A is
ergonomics and documentation, not algorithms.

Family B (statistically validated hypergraphs) is genuinely new for the library
and is the more likely home of the source paper's contribution. It is tractable
by reusing existing machinery (`fastball`, `bicm`, `_bipartite_projection_matrix`)
but requires one real design decision: **how to represent a hypergraph**, since
NetworkX has no native hypergraph type.

Proposed phasing:

- **Phase 0** — surface and document Family A (tiny adapters + tutorial). Near-zero risk.
- **Phase 1** — add a `hypergraph` module for Family B (hyperedge-level significance).
- **Out of scope** — hypergraph neural networks / representation learning; making
  any third-party hypergraph library a *required* dependency.

## 2. Note on the source paper (arXiv:2606.00893)

The specific paper could not be retrieved while preparing this proposal:

- The execution environment's network egress is allow-listed and excludes
  `arxiv.org`, `export.arxiv.org`, `huggingface.co`, and `alphaxiv.org`
  ("Host not in allowlist").
- arXiv additionally returns HTTP 403 to automated fetchers.
- The ID `2606.00893` corresponds to **June 2026** and was only ~1 day old at the
  time of writing, so it is not yet indexed by web search or the HuggingFace
  papers hub.

Consequently this proposal is grounded in (a) a full reading of this library and
(b) the established hypergraph-backboning literature any such paper builds on
(see [References](#7-references)) — **not** on the paper's exact formulation.
Section 6 lists the specific details to confirm against the paper before
implementing Phase 1.

## 3. Background: what "hypergraph network backboning" means

A hypergraph `H = (V, E)` has hyperedges `e ⊆ V` that may join more than two
nodes. "Backboning" a hypergraph means keeping only its most informative
structure. Two distinct families exist:

### Family A — projection backboning

Represent `H` as an incidence (node × hyperedge) structure, project to a
node–node weighted graph, then apply a null model that preserves degree
sequences to decide which *pairwise* links are statistically significant. This is
the lineage of:

- Neal's `backbone` R package — **Backbone 3.0 (PLOS One, 2026)** explicitly
  supports "networks whose weights are the product of bipartite or hypergraph
  projection (including stochastic and fixed degree sequence models)".
- **Coscia & Neffke (2017)** (already cited in this library's README).

The output is an ordinary graph, so it fits the existing score-then-filter idiom.

### Family B — statistically validated hypergraphs

Keep the *hyperedges themselves* that are over-expressed relative to a
configuration-model null (preserving node degrees / hyperedge sizes), discarding
redundant or noisy higher-order groups. Reference: **Musciotto, Battiston &
Mantegna, "Detecting informative higher-order interactions in statistically
validated hypergraphs," Communications Physics (2021)** (arXiv:2103.16484).

Given the phrase "hypergraph network backboning," the source paper is most
plausibly in Family B (likely refining the null model, the multiple-testing
correction, or computational efficiency). The output is a *subset of hyperedges*,
which does **not** map onto the current graph-in/graph-out filter functions.

## 4. Current library capabilities relevant to this

- **Score-then-filter pattern.** Methods annotate edges with a score
  (e.g. `disparity_pvalue`) and return a copy of the graph; `threshold_filter` /
  `boolean_filter` / `fraction_filter` then extract the subgraph. This is the
  central design idiom (see `docs/concepts.rst`).
- **Dependency policy.** Core requires only `networkx>=3.0`; `numpy`/`scipy` are
  the optional `[full]` extra and are imported lazily inside functions.
- **The `bipartite` module already is an incidence engine.**
  `_bipartite_projection_matrix(B, agent_nodes)` builds the binary incidence
  matrix `R` (agents × artifacts) and the co-occurrence matrix `R @ R.T`. On top
  of it the module provides degree-preserving null models — `sdsm` (analytic
  Poisson-binomial), `fdsm` (Monte-Carlo, exact degree preservation),
  `fixedfill` / `fixedrow` / `fixedcol` — plus reusable randomizers `fastball`
  and `_random_bipartite_matrix`, and `bicm` probabilities.

A hypergraph encoded with nodes in one partition and hyperedges in the other is
*exactly* the input these functions already expect.

## 5. Key finding: Family A already works today

Encoding a hypergraph as its incidence bipartite graph and running the existing
SDSM/FDSM backbone produces a node–node projection backbone with no new code:

```python
import networkx as nx
import networkx_backbone as nb

hyperedges = {"H1": [1, 2, 3], "H2": [1, 2, 3], "H3": [3, 4, 5, 6], "H4": [5, 6]}
nodes = sorted({v for members in hyperedges.values() for v in members})

# Hypergraph -> incidence bipartite graph (nodes | hyperedges)
B = nx.Graph()
B.add_nodes_from(nodes, bipartite=0)          # nodes
B.add_nodes_from(hyperedges, bipartite=1)     # hyperedges as "artifacts"
for h, members in hyperedges.items():
    for v in members:
        B.add_edge(v, h)

scored   = nb.sdsm(B, agent_nodes=nodes, projection="hyper")
backbone = nb.threshold_filter(scored, "sdsm_pvalue", 0.30, mode="below")
# backbone.edges() -> [(1, 2), (4, 5), (4, 6), (5, 6)]   (verified)
```

`fdsm(B, agent_nodes=nodes, trials=500, seed=0)` works identically with
Monte-Carlo, exactly-degree-preserving null models. In other words, a large slice
of "hypergraph backboning" is a **latent, undocumented capability** of the
library today. Surfacing it is cheap and high value.

## 6. Gap analysis and challenges

1. **No native hypergraph type in NetworkX.** A representation must be chosen
   (Section 7.1). This is the central decision.
2. **Output-type mismatch for Family B.** A sub-hypergraph cannot flow through
   `threshold_filter`, which returns a graph. Either add a small parallel filter
   or annotate the *artifact* (hyperedge) nodes of the bipartite encoding and
   filter those — preserving the idiom.
3. **Dependency policy.** Keep `networkx`-only core; hyperedge null models need
   only `numpy`/`scipy` (already the `[full]` extra). `xgi` / `HyperNetX` should
   be **optional interop**, never required.
4. **Multiple testing and cost.** Family B tests one hypothesis per candidate
   hyperedge, so it needs a correction (Bonferroni/FDR) and benefits from the
   existing Monte-Carlo randomizers (`fastball`, `_random_bipartite_matrix`).
5. **Scope discipline.** Stay within classical backboning. Hypergraph neural
   networks (the bulk of recent "hypergraph" literature) are out of scope.

## 7. Proposed design

### 7.1 Hypergraph representation

Recommended primary representation: **the incidence bipartite graph** (and/or an
equivalent incidence matrix / list of `frozenset` hyperedges). Rationale: zero new
dependencies, reuses the entire `bipartite` engine, and is consistent with the
library's NetworkX-centric design.

| Option | Pros | Cons |
|--------|------|------|
| **Incidence bipartite graph** (recommended) | no new deps; reuses null models; matches library style | hyperedge identity lives in node labels |
| **Lightweight internal form** (`list[frozenset]` + incidence matrix) | natural for Family B output; explicit | small amount of new plumbing |
| **Optional `xgi` / `HyperNetX` interop** | convenient for users already in that ecosystem | must stay optional; extra maintenance |

Plan: use the incidence representation internally; offer thin converters to/from
`xgi`/`HyperNetX` behind lazy imports for users who have them.

### 7.2 Phase 0 — surface Family A (docs + small adapters)

Add converter helpers and a tutorial so the already-working capability is
discoverable. Sketch:

```python
def hypergraph_to_bipartite(hyperedges, node_partition=0):
    """Build an incidence bipartite graph from an iterable/mapping of hyperedges.

    Returns (B, nodes) so the result can be passed straight to sdsm/fdsm/...
    """

def incidence_to_bipartite(matrix, node_labels=None, edge_labels=None):
    """Build an incidence bipartite graph from a binary node x hyperedge matrix."""
```

Deliverables: the two converters, a `docs/tutorials/` page demonstrating SDSM /
FDSM / fixed-model hypergraph projection backbones, tests, and a short note in
`docs/concepts.rst`. No changes to existing functions.

### 7.3 Phase 1 — `hypergraph` module for Family B

A new module providing hyperedge-level significance, preserving score-then-filter.
Indicative API (to be finalized against the paper):

```python
# networkx_backbone/hypergraph.py
def statistically_validated_hypergraph(hyperedges, null="configuration",
                                       trials=1000, correction="fdr", seed=None):
    """Score each candidate hyperedge with a p-value under a node-degree-
    preserving null model. Returns the hyperedges annotated with `svh_pvalue`."""

def hyperedge_filter(scored_hyperedges, score="svh_pvalue", alpha=0.05):
    """Keep hyperedges whose corrected p-value passes `alpha` (sub-hypergraph)."""
```

Implementation reuses `fastball` / `_random_bipartite_matrix` for the null
ensemble and follows the existing lazy-`numpy`/`scipy` import convention. Output
is the kept hyperedges; optionally also return the bipartite encoding so the
existing graph filters apply unchanged.

### 7.4 Testing strategy

- Phase 0: round-trip converter tests; equivalence test showing
  `hypergraph_to_bipartite(...)` + `sdsm` matches the manual bipartite recipe.
- Phase 1: known-answer tests on tiny hypergraphs (a planted over-represented
  group must be retained; random groups must not); determinism via `seed`;
  multiple-testing-correction behavior; degenerate inputs (empty, singletons).
- Mirror reference outputs from the paper / Musciotto et al. where available.

## 8. Open questions to confirm against arXiv:2606.00893

1. Which family — direct hyperedge filtering (B) or a projection method (A)?
2. The exact null model (node-degree-preserving, hyperedge-size-preserving, or both).
3. Test statistic and the multiple-comparison correction used.
4. Output: a sub-hypergraph, a validated projection, or both.
5. Whether a reference implementation exists to mirror for test fixtures.

## 9. References

- Backbone 3.0: An R package for extracting network backbones — PLOS One (2026).
  https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0349258
- Musciotto, Battiston & Mantegna — Detecting informative higher-order
  interactions in statistically validated hypergraphs — Communications Physics
  (2021). https://www.nature.com/articles/s42005-021-00710-4 (arXiv:2103.16484)
- Stochastic Degree Sequence Model with Edge Constraints (SDSM-EC) —
  arXiv:2307.12828. https://arxiv.org/pdf/2307.12828
- Fast nonparametric inference of network backbones — arXiv:2409.06417.
  https://arxiv.org/pdf/2409.06417
- Coscia & Neffke — Network backboning with noisy data — arXiv:1906.09081.
  https://arxiv.org/pdf/1906.09081
- `xgi` (Comple**X** Group Interactions): https://github.com/xgi-org/xgi
- `HyperNetX`: https://github.com/pnnl/HyperNetX
