Bipartite Projection Backbones
===============================

This tutorial demonstrates weighted bipartite projections (simple, hyper, ProbS,
YCN) and backbone scoring with SDSM/FDSM, then extracts significant edges with
filtering.

What is a bipartite projection backbone?
-----------------------------------------

In many real-world datasets, relationships are bipartite: people attend events,
authors write papers, users rate products. To study relationships among one
set of nodes (for example, people), we project the bipartite graph into a
unipartite graph where two people are connected if they share an event.
However, this projection often produces a dense graph with many spurious
connections. Backbone methods identify which co-occurrences are statistically
significant.

Setup
-----

::

    import networkx as nx
    import networkx_backbone as nb

    # Davis Southern Women graph from NetworkX social generators
    B = nx.davis_southern_women_graph()

    # Choose the women partition as the "agent" nodes
    women_nodes = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    event_nodes = [n for n, d in B.nodes(data=True) if d["bipartite"] == 1]

    print(f"Bipartite graph: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")
    print(f"Women: {len(women_nodes)}, events: {len(event_nodes)}")

Weighted projections
--------------------

Coscia & Rossi (2019) compare four ways to weight a projection, all built from
the shared artifacts ``z`` of two agents ``u`` and ``v`` (``d`` is a degree):

- **simple**: the number of shared artifacts, ``|Γ(u) ∩ Γ(v)|`` (equal to
  ``nx.bipartite.weighted_projected_graph``);
- **hyper** (hyperbolic): ``Σ 1/d(z)``, so a small shared event counts for more
  than a large one;
- **ProbS**: the probability that a two-step random walk from ``u`` (to an
  artifact, then to an agent) ends at ``v``, ``(1/d(u)) Σ 1/d(z)``, averaged
  over the two directions;
- **YCN** (Yildirim & Coscia, 2014): the stationary flow ``π_u T_uv`` of that
  random walk. On a connected graph it equals the hyperbolic weight divided by
  the number of bipartite edges.

::

    G_simple = nb.simple_projection(B, women_nodes)
    G_hyper = nb.hyper_projection(B, women_nodes)
    G_probs = nb.probs_projection(B, women_nodes)  # symmetrized ProbS
    G_ycn = nb.ycn_projection(B, women_nodes)      # symmetrized YCN

    print("Simple edges:", G_simple.number_of_edges())
    print("Hyper edges:", G_hyper.number_of_edges())
    print("ProbS edges:", G_probs.number_of_edges())
    print("YCN edges:", G_ycn.number_of_edges())

Coscia & Rossi's hyperbolic weight discounts each artifact by ``d(z)``. Newman's
(2001) original collaboration weight discounts by ``d(z) - 1``, the number of
*other* co-authors, and is what NetworkX's
``nx.bipartite.collaboration_weighted_projected_graph`` computes. The two
differ on every Davis pair (Olivia and Sylvia: 1/12 versus 1/11), so pick one
deliberately::

    G_newman = nb.hyper_projection(B, women_nodes, newman=True)

You can also dispatch by name (``"simple"``, ``"hyper"``, ``"newman"``,
``"probs"``, ``"ycn"``), and pass ``directed=True`` for the asymmetric ProbS and
YCN scores::

    G = nb.bipartite_projection(B, women_nodes, method="probs")
    D = nb.bipartite_projection(B, women_nodes, method="probs", directed=True)

SDSM: Stochastic Degree Sequence Model
----------------------------------------

The SDSM (Neal, 2014) uses an analytical approximation (Poisson-binomial via
normal distribution) to compute p-values for each co-occurrence. It returns the
full projection with ``"sdsm_pvalue"`` on each edge. You can choose which
projection weights to attach to those edges via ``projection=``::

    H = nb.sdsm(B, agent_nodes=women_nodes, projection="hyper")
    backbone = nb.threshold_filter(H, "sdsm_pvalue", 0.05, mode="below")
    print(f"SDSM backbone: {backbone.number_of_nodes()} nodes, {backbone.number_of_edges()} edges")

    # Examine p-values
    for u, v, data in H.edges(data=True):
        print(f"  Edge ({u}, {v}): p-value = {data['sdsm_pvalue']:.4f}")

FDSM: Fixed Degree Sequence Model
-----------------------------------

The FDSM (Neal et al., 2021) uses Monte Carlo simulation to compute p-values.
It preserves the exact degree sequence of the bipartite graph in each random
trial. It also returns the full projection::

    H = nb.fdsm(B, agent_nodes=women_nodes, trials=1000, seed=42, projection="ycn")
    backbone = nb.threshold_filter(H, "fdsm_pvalue", 0.05, mode="below")
    print(f"FDSM backbone: {backbone.number_of_nodes()} nodes, {backbone.number_of_edges()} edges")

    # Examine p-values
    for u, v, data in H.edges(data=True):
        print(f"  Edge ({u}, {v}): p-value = {data['fdsm_pvalue']:.4f}")

Additional null models
----------------------

Fixed-fill, fixed-row, and fixed-column models are also available::

    bb_fill = nb.fixedfill(B, women_nodes, alpha=0.05)
    bb_row = nb.fixedrow(B, women_nodes, alpha=0.05)
    bb_col = nb.fixedcol(B, women_nodes, alpha=0.05)

You can also use the high-level projection wrapper::

    bb = nb.backbone_from_projection(
        B,
        women_nodes,
        method="fixedrow",
        alpha=0.05,
        projection="probs",
    )

The wrapper scores every agent pair and returns only the edges significant at
``alpha``, for every ``method`` (including ``"sdsm"`` and ``"fdsm"``). Pass
``mtc`` to correct for the number of pairs tested, and ``signed=True`` to keep
significantly weak pairs too::

    bb = nb.backbone_from_projection(B, women_nodes, method="sdsm", alpha=0.05, mtc="bh")

Calling :func:`~networkx_backbone.sdsm` or :func:`~networkx_backbone.fdsm`
directly instead returns the full scored projection, so you can choose the
threshold yourself.

Comparing SDSM and FDSM
-------------------------

- **SDSM** is faster (analytical) and works well for large networks. It fits
  the bipartite configuration model (BiCM), in which node degrees hold in
  expectation rather than exactly. It then evaluates each co-occurrence count
  with the exact Poisson-binomial distribution, or with its refined normal
  approximation (RNA) for large inputs, as R ``backbone`` does.

- **FDSM** is more conservative and statistically rigorous. It preserves the
  exact degree sequence by sampling a curveball Markov chain that starts from
  the observed matrix (tuned with ``burn_in`` and ``thin``), but requires more
  computation. P-values are ``(count + 1) / (trials + 1)``, so they are never
  exactly 0. Increase ``trials`` for more precise p-values.

::

    sdsm_scores = nb.sdsm(B, agent_nodes=women_nodes)
    fdsm_scores = nb.fdsm(B, agent_nodes=women_nodes, trials=1000, seed=42)

    sdsm_backbone = nb.threshold_filter(sdsm_scores, "sdsm_pvalue", 0.05, mode="below")
    fdsm_backbone = nb.threshold_filter(fdsm_scores, "fdsm_pvalue", 0.05, mode="below")

    print(f"SDSM edges: {sdsm_backbone.number_of_edges()}")
    print(f"FDSM edges: {fdsm_backbone.number_of_edges()}")

Partition selection note
------------------------

``nx.davis_southern_women_graph()`` includes a ``"bipartite"`` node attribute,
which makes partition selection straightforward. In general, pass whichever
partition you want to project as ``agent_nodes``; if you leave it out, it is
taken from the ``"bipartite"`` attribute (value 0) or from a two-coloring of a
connected graph.

Incidence matrices and large data
---------------------------------

Every function in this tutorial also accepts an agents x artifacts incidence
matrix in place of the graph, as R ``backbone`` does: a NumPy array, a SciPy
sparse array or matrix, or a pandas ``DataFrame`` (whose index and columns
become the labels). Rows are the agents; name them with ``agent_labels`` (and
the columns with ``artifact_labels``, which ``prohibited``/``required`` refer
to). The results are identical to the graph path::

    women = sorted(women_nodes, key=str)
    events = sorted(event_nodes, key=str)
    M = nx.bipartite.biadjacency_matrix(B, row_order=women, column_order=events)
    bb = nb.backbone_from_projection(
        M, method="sdsm", alpha=0.05, agent_labels=women, artifact_labels=events
    )

The incidence is kept as a sparse matrix, co-occurrences come from sparse
products, and agent pairs are scored in blocks of rows, so no dense agents x
agents matrix is built. ``backbone_from_projection`` only turns significant
pairs into edges, which keeps memory bounded for tens of thousands of agents
(with ``mtc`` it also keeps one float per pair for the correction). By
contrast, :func:`~networkx_backbone.sdsm` and :func:`~networkx_backbone.fdsm`
return *every* pair as an edge, which is only practical for a few thousand
agents::

    import scipy.sparse as sp

    M = sp.random_array((20_000, 5_000), density=0.01, format="csr")
    M.data[:] = 1
    bb = nb.backbone_from_projection(M, method="sdsm", alpha=0.01)  # ~0.5 min

Signed backbones
----------------

By default the null-model tests are one-tailed and retain only
*significantly strong* co-occurrences. Pass ``signed=True`` to ``sdsm``,
``fdsm``, ``fixedfill``, ``fixedrow``, or ``fixedcol`` for a **two-tailed** test:
the stored p-value becomes two-sided and each edge gains a ``"sign"`` attribute
(``+1`` significantly strong, ``-1`` significantly weak)::

    H = nb.sdsm(B, agent_nodes=women_nodes, signed=True)
    backbone = nb.threshold_filter(H, "sdsm_pvalue", 0.05, mode="below")
    weak = [(u, v) for u, v, d in backbone.edges(data=True) if d["sign"] == -1]

Edge constraints (SDSM-EC)
--------------------------

The Stochastic Degree Sequence Model with Edge Constraints (Neal & Neal, 2023)
lets you fix some cells of the null model: ``prohibited`` edges cannot occur
(null probability 0) and ``required`` edges must occur (null probability 1).
Pass them to ``sdsm`` as iterables of ``(agent, artifact)`` pairs::

    H = nb.sdsm(
        B,
        agent_nodes=women_nodes,
        prohibited=[(women_nodes[0], event_nodes[13])],
        required=[(women_nodes[1], event_nodes[1])],
    )

A constraint must agree with the data: a ``prohibited`` pair must not be an
edge of ``B`` and a ``required`` pair must be one. Pairs that contradict the
data are ignored with a ``RuntimeWarning`` that lists them; pairs naming nodes
that are not an ``(agent, artifact)`` pair of ``B``, or listed as both
required and prohibited, raise ``NetworkXError``.

References
----------

- Coscia, M., & Rossi, L. (2019). *The impact of projection and backboning on
  network topologies*. Proc. IEEE/ACM ASONAM.
  https://doi.org/10.1145/3341161.3342862 (arXiv:1906.09081)
- Newman, M. E. J. (2001). *Scientific collaboration networks. II. Shortest
  paths, weighted networks, and centrality*. Physical Review E, 64, 016132.
- Yildirim, M. A., & Coscia, M. (2014). *Using random walks to generate
  associations between objects*. PLoS ONE, 9(8), e104813.
- Zhou, T., Ren, J., Medo, M., & Zhang, Y.-C. (2007). *Bipartite network
  projection and personal recommendation*. Physical Review E, 76, 046115.
- Neal, Z. P. (2014). *The backbone of bipartite projections*. Social Networks,
  39, 84-97.
- Neal, Z. P., Domagalski, R., & Sagan, B. (2021). *Comparing alternatives to the
  fixed degree sequence model for extracting the backbone of bipartite
  projections*. Scientific Reports, 11, 23929.
- Saracco, F., Di Clemente, R., Gabrielli, A., & Squartini, T. (2015).
  *Randomizing bipartite networks: the case of the World Trade Web*. Scientific
  Reports, 5, 10595.
- Godard, K., & Neal, Z. P. (2022). *fastball: A fast algorithm to sample
  bipartite graphs with fixed degree sequences*. J. Complex Networks, 10(6),
  cnac049.
- Neal, Z. P., & Neal, J. W. (2023). *Stochastic Degree Sequence Model with Edge
  Constraints (SDSM-EC) for Backbone Extraction*. Complex Networks 12, 127-136.
