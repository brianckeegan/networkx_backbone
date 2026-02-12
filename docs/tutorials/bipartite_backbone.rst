Bipartite Projection Backbones
===============================

This tutorial demonstrates how to extract significant edges from bipartite
graph projections using the SDSM and FDSM methods.

What is a bipartite projection backbone?
-----------------------------------------

In many real-world datasets, relationships are bipartite: authors write papers,
users rate products, students take classes. To study relationships among one
set of nodes (e.g., authors), we project the bipartite graph into a
unipartite graph where two nodes are connected if they share an artifact
(e.g., co-authorship). However, this projection often produces a dense graph
with many spurious connections. Backbone methods identify which co-occurrences
are statistically significant.

Setup
-----

::

    import networkx as nx
    import networkx_backbone as nb

    # Create a bipartite graph
    # Agents (authors): 1, 2, 3, 4
    # Artifacts (papers): "a", "b", "c", "d"
    B = nx.Graph()
    B.add_edges_from([
        (1, "a"), (1, "b"), (1, "c"),
        (2, "a"), (2, "b"),
        (3, "b"), (3, "c"), (3, "d"),
        (4, "c"), (4, "d"),
    ])

    agent_nodes = [1, 2, 3, 4]
    print(f"Bipartite graph: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")

SDSM: Stochastic Degree Sequence Model
----------------------------------------

The SDSM (Neal, 2014) uses an analytical approximation (Poisson-binomial via
normal distribution) to compute p-values for each co-occurrence. It is fast
because it avoids simulation::

    backbone = nb.sdsm(B, agent_nodes=agent_nodes, alpha=0.05)
    print(f"SDSM backbone: {backbone.number_of_nodes()} nodes, {backbone.number_of_edges()} edges")

    # Examine p-values
    for u, v, data in backbone.edges(data=True):
        print(f"  Edge ({u}, {v}): p-value = {data['sdsm_pvalue']:.4f}")

FDSM: Fixed Degree Sequence Model
-----------------------------------

The FDSM (Neal et al., 2021) uses Monte Carlo simulation to compute p-values.
It preserves the exact degree sequence of the bipartite graph in each random
trial, making it more conservative than SDSM::

    backbone = nb.fdsm(B, agent_nodes=agent_nodes, alpha=0.05, trials=1000, seed=42)
    print(f"FDSM backbone: {backbone.number_of_nodes()} nodes, {backbone.number_of_edges()} edges")

    # Examine p-values
    for u, v, data in backbone.edges(data=True):
        print(f"  Edge ({u}, {v}): p-value = {data['fdsm_pvalue']:.4f}")

Comparing SDSM and FDSM
-------------------------

- **SDSM** is faster (analytical) and works well for large networks. It uses
  a stochastic degree sequence model where node degrees are treated as
  expectations rather than fixed values.

- **FDSM** is more conservative and statistically rigorous. It preserves the
  exact degree sequence through simulation, but requires more computation.
  Increase ``trials`` for more precise p-values.

::

    sdsm_backbone = nb.sdsm(B, agent_nodes=agent_nodes, alpha=0.05)
    fdsm_backbone = nb.fdsm(B, agent_nodes=agent_nodes, alpha=0.05, trials=1000, seed=42)

    print(f"SDSM edges: {sdsm_backbone.number_of_edges()}")
    print(f"FDSM edges: {fdsm_backbone.number_of_edges()}")

Weighted bipartite graphs
--------------------------

Both methods support weighted bipartite graphs via the ``weight`` parameter.
When provided, co-occurrence counts are replaced by sums of edge weights::

    # Add weights to the bipartite graph
    for u, v in B.edges():
        B[u][v]["weight"] = 2.0

    backbone = nb.sdsm(B, agent_nodes=agent_nodes, alpha=0.05, weight="weight")
