Statistical Backbone Extraction
================================

This tutorial demonstrates all five statistical backbone extraction methods.
Each method tests whether an edge's weight is statistically significant under
a null model.

Setup
-----

We use a weighted graph for this tutorial::

    import networkx as nx
    import networkx_backbone as nb

    # Create a weighted graph
    G = nx.karate_club_graph()
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0

    print(f"Original: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

Disparity filter
----------------

The disparity filter (Serrano et al., 2009) tests each edge against a null
model where a node's total strength is uniformly distributed across its edges.
Edges with unexpectedly large weight receive low p-values::

    H = nb.disparity_filter(G)

    # Each edge now has a "disparity_pvalue" attribute
    u, v = list(H.edges())[0]
    print(f"Edge ({u}, {v}): p-value = {H[u][v]['disparity_pvalue']:.4f}")

    # Filter at alpha = 0.05
    backbone = nb.threshold_filter(H, "disparity_pvalue", 0.05)
    print(f"Disparity backbone: {backbone.number_of_edges()} edges")

Noise-corrected filter
----------------------

The noise-corrected filter (Coscia & Neffke, 2017) uses a binomial framework
to model edge weights. It produces z-scores rather than p-values -- higher
z-scores indicate more significant edges::

    H = nb.noise_corrected_filter(G)

    # Each edge now has an "nc_score" attribute (z-score)
    u, v = list(H.edges())[0]
    print(f"Edge ({u}, {v}): z-score = {H[u][v]['nc_score']:.4f}")

    # Filter: keep edges with z-score above a threshold
    backbone = nb.threshold_filter(H, "nc_score", 2.0, mode="above")
    print(f"Noise-corrected backbone: {backbone.number_of_edges()} edges")

Marginal likelihood filter
--------------------------

The marginal likelihood filter (Dianati, 2016) considers both endpoints in a
binomial null model and treats weights as integer counts::

    H = nb.marginal_likelihood_filter(G)
    backbone = nb.threshold_filter(H, "ml_pvalue", 0.05)
    print(f"Marginal likelihood backbone: {backbone.number_of_edges()} edges")

ECM filter
----------

The Enhanced Configuration Model (Gemmetto et al., 2017) uses a maximum-entropy
null model that preserves both degree and strength sequences. This is the most
principled null model but also the most computationally expensive::

    H = nb.ecm_filter(G)
    backbone = nb.threshold_filter(H, "ecm_pvalue", 0.05)
    print(f"ECM backbone: {backbone.number_of_edges()} edges")

LANS filter
-----------

Locally Adaptive Network Sparsification (Foti et al., 2011) uses nonparametric
empirical CDFs instead of parametric distributions. This makes no distributional
assumptions::

    H = nb.lans_filter(G)
    backbone = nb.threshold_filter(H, "lans_pvalue", 0.05)
    print(f"LANS backbone: {backbone.number_of_edges()} edges")

Comparing all statistical methods
----------------------------------

Use :func:`~networkx_backbone.compare_backbones` to compare the results::

    backbones = {
        "disparity": nb.threshold_filter(
            nb.disparity_filter(G), "disparity_pvalue", 0.05
        ),
        "noise_corrected": nb.threshold_filter(
            nb.noise_corrected_filter(G), "nc_score", 2.0, mode="above"
        ),
        "marginal_likelihood": nb.threshold_filter(
            nb.marginal_likelihood_filter(G), "ml_pvalue", 0.05
        ),
        "ecm": nb.threshold_filter(
            nb.ecm_filter(G), "ecm_pvalue", 0.05
        ),
        "lans": nb.threshold_filter(
            nb.lans_filter(G), "lans_pvalue", 0.05
        ),
    }

    results = nb.compare_backbones(G, backbones)
    for name, metrics in results.items():
        ef = metrics["edge_fraction"]
        nf = metrics["node_fraction"]
        print(f"{name:25s}: edges={ef:.1%}, nodes={nf:.1%}")
