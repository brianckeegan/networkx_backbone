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

Following Coscia & Neffke (2017, arXiv:1906.09081), you can project the
bipartite graph into weighted one-mode networks using different weighting
schemes::

    G_simple = nb.simple_projection(B, women_nodes)
    G_hyper = nb.hyper_projection(B, women_nodes)
    G_probs = nb.probs_projection(B, women_nodes)  # symmetrized ProbS
    G_ycn = nb.ycn_projection(B, women_nodes)      # symmetrized YCN

    print("Simple edges:", G_simple.number_of_edges())
    print("Hyper edges:", G_hyper.number_of_edges())
    print("ProbS edges:", G_probs.number_of_edges())
    print("YCN edges:", G_ycn.number_of_edges())

You can also dispatch by name::

    G = nb.bipartite_projection(B, women_nodes, method="probs")

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

Comparing SDSM and FDSM
-------------------------

- **SDSM** is faster (analytical) and works well for large networks. It uses
  a stochastic degree sequence model where node degrees are treated as
  expectations rather than fixed values.

- **FDSM** is more conservative and statistically rigorous. It preserves the
  exact degree sequence through simulation, but requires more computation.
  Increase ``trials`` for more precise p-values.

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
partition you want to project as ``agent_nodes``.

References
----------

- Coscia, M., & Neffke, F. M. (2017). *Network backboning with noisy data*.
  https://arxiv.org/abs/1906.09081
