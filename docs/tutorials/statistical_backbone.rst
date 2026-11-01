Statistical Backbone Extraction
================================

This tutorial demonstrates six statistical backbone extraction methods.
Each method tests whether an edge's weight is statistically significant under
a null model.

Setup
-----

We use a weighted graph for this tutorial::

    import networkx as nx
    import networkx_backbone as nb

    # Create a weighted graph
    G = nx.les_miserables_graph()

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

The noise-corrected backbone (Coscia & Neffke, 2017) compares each edge's
weight with the weight expected from its endpoints' strengths.  The score
``nc_score`` is the *lift* (observed over expected weight) mapped to
``(-1, 1)``: 0 means exactly as expected, positive means stronger.  Because
weights are noisy counts, the method also estimates the score's standard
deviation ``nc_sdev`` from a Bayesian (beta-binomial) model, and keeps an edge
when ``nc_score > delta * nc_sdev``.  The paper uses ``delta = 1.64``, roughly
a one-tailed test at the 5% level; ``nc_pvalue`` is the matching
normal-approximation p-value, so it works with
:func:`~networkx_backbone.threshold_filter` and multiple-testing corrections::

    H = nb.noise_corrected_filter(G)

    u, v = list(H.edges())[0]
    d = H[u][v]
    print(f"Edge ({u}, {v}): score = {d['nc_score']:.3f} +/- {d['nc_sdev']:.3f}")

    # The paper's rule, delta = 1.64
    kept = [(u, v) for u, v, d in H.edges(data=True)
            if d["nc_score"] > 1.64 * d["nc_sdev"]]
    print(f"Noise-corrected backbone (delta=1.64): {len(kept)} edges")

    # The same rule as a p-value threshold (delta = 1.645 <=> p < 0.05)
    backbone = nb.threshold_filter(H, "nc_pvalue", 0.05)
    print(f"Noise-corrected backbone: {backbone.number_of_edges()} edges")

An undirected graph is treated as a symmetric weight matrix, so each edge is
counted in both directions, as in the authors' reference implementation.  The
noise-corrected backbone is permissive on small, dense graphs such as Les
Misérables (195 of 254 edges at ``delta = 1.64``); it is designed for large,
noisy count data.  The binomial z-score of networkx-backbone 0.2 and earlier is
still available as ``method="zscore"``.

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

Multiple linkage analysis
-------------------------

Multiple linkage analysis (Van Nuffel, 2007) decides how many of each
node's links are *dominant*.  It sorts the node's link weights in decreasing
order, normalizes them to shares, and compares that profile by least squares
with ideal distributions: one link with all the weight, two equal links, three
equal links, and so on.  The node keeps the links of the best-fitting ideal
distribution, and ``mla_rank`` records how many that is.  A node with one
overwhelming link keeps one; a node whose weight is spread evenly keeps all of
them::

    H = nb.multiple_linkage_analysis(G)
    backbone = nb.boolean_filter(H, "mla_keep")
    print(f"MLA backbone: {backbone.number_of_edges()} edges")
    print("Dominant links of Valjean:", H.nodes["Valjean"]["mla_rank"])

Directed graphs use each node's out-links.  In an undirected graph an edge is
kept when it is dominant for *either* endpoint, so every node keeps at least
its strongest link.

This differs from LANS: LANS gives each link a p-value from the empirical
distribution of its node's weights, while the dominance fit chooses a number
of links per node and has no significance level.  ``method="lans"`` keeps
links whose LANS p-value is at most ``alpha`` instead (the behavior of
networkx-backbone 0.2 and earlier)::

    H = nb.multiple_linkage_analysis(G, alpha=0.05, method="lans")

Comparing all statistical methods
----------------------------------

Use :func:`~networkx_backbone.compare_backbones` to compare the results::

    backbones = {
        "disparity": nb.threshold_filter(
            nb.disparity_filter(G), "disparity_pvalue", 0.05
        ),
        "noise_corrected": nb.threshold_filter(
            nb.noise_corrected_filter(G), "nc_pvalue", 0.05
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
        "mla": nb.boolean_filter(
            nb.multiple_linkage_analysis(G), "mla_keep"
        ),
    }

    results = nb.compare_backbones(G, backbones)
    for name, metrics in results.items():
        ef = metrics["edge_fraction"]
        nf = metrics["node_fraction"]
        print(f"{name:25s}: edges={ef:.1%}, nodes={nf:.1%}")

Multiple-testing correction
---------------------------

Because a statistical backbone tests every edge, correcting for multiple
comparisons avoids retaining edges by chance. Pass ``mtc`` to
:func:`~networkx_backbone.threshold_filter` (or use
:func:`~networkx_backbone.adjust_pvalues` directly). Supported corrections match
R's ``p.adjust``: ``"bonferroni"``, ``"holm"``, ``"hochberg"``, ``"bh"``/``"fdr"``
(Benjamini-Hochberg), and ``"by"`` (Benjamini-Yekutieli)::

    scored = nb.disparity_filter(G)

    raw = nb.threshold_filter(scored, "disparity_pvalue", 0.05)
    fdr = nb.threshold_filter(scored, "disparity_pvalue", 0.05, mtc="bh")
    bonf = nb.threshold_filter(scored, "disparity_pvalue", 0.05, mtc="bonferroni")

    print(f"raw={raw.number_of_edges()}, BH={fdr.number_of_edges()}, "
          f"Bonferroni={bonf.number_of_edges()}")

Signed backbones
----------------

By default the statistical filters run a one-tailed test that keeps only
*significantly strong* edges. With ``signed=True``, the
``disparity_filter``, ``noise_corrected_filter``, ``marginal_likelihood_filter``,
``ecm_filter``, and ``lans_filter`` run a **two-tailed** test: the stored
p-value becomes two-sided, and each edge gains a ``"sign"`` attribute that is
``+1`` for a significantly strong edge and ``-1`` for a significantly weak one.
For the ECM the lower tail is :math:`P(w_{ij} \le w) = 1 - p_{ij} (y_i y_j)^w`
under the fitted model; for the noise-corrected backbone both tails come from
the normal approximation to ``nc_score / nc_sdev``::

    signed = nb.disparity_filter(G, signed=True)
    backbone = nb.threshold_filter(signed, "disparity_pvalue", 0.05, mtc="bh")

    positive = [(u, v) for u, v, d in backbone.edges(data=True) if d["sign"] == 1]
    negative = [(u, v) for u, v, d in backbone.edges(data=True) if d["sign"] == -1]
    print(f"strong (+): {len(positive)}, weak (-): {len(negative)}")

References
----------

- Serrano, M. A., Boguna, M., & Vespignani, A. (2009). *Extracting the multiscale
  backbone of complex weighted networks*. PNAS, 106(16), 6483-6488.
- Coscia, M., & Neffke, F. M. (2017). *Network backboning with noisy data*.
  Proc. IEEE ICDE, 425-436.
- Dianati, N. (2016). *Unwinding the hairball graph: Pruning algorithms for
  weighted complex networks*. Physical Review E, 93, 012304.
- Gemmetto, V., Cardillo, A., & Garlaschelli, D. (2017). *Irreducible network
  backbones: unbiased graph filtering via maximum entropy*. arXiv:1706.00230.
- Foti, N. J., Hughes, J. M., & Rockmore, D. N. (2011). *Nonparametric
  sparsification of complex multiscale networks*. PLOS ONE, 6(2), e16431.
- Van Nuffel, N. (2007). *Determination of the number of significant flows in
  origin-destination specific analysis: the case of commuting in Flanders*.
  Regional Studies, 41(4), 509-524.
- Weaver, J. C. (1954). *Crop-combination regions in the Middle West*.
  Geographical Review, 44(2), 175-200.
- Benjamini, Y., & Hochberg, Y. (1995). *Controlling the false discovery rate*.
  J. Royal Statistical Society B, 57(1), 289-300.
- Benjamini, Y., & Yekutieli, D. (2001). *The control of the false discovery rate
  in multiple testing under dependency*. Annals of Statistics, 29(4), 1165-1188.
