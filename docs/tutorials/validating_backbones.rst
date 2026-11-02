Validating a Backbone
=====================

The other tutorials show how to score edges and filter them. This one covers
the steps that come after: checking that a backbone is not an artifact of
running hundreds of tests at once, comparing it with other backbones,
finding the edges that every method agrees on, reading a signed backbone,
and reporting what you did. See the narrative tutorial,
:doc:`/auto_examples/start_here/plot_hairball_to_backbone`, and its section
:ref:`Validating <hb-validate>` for the same steps told as one case study.

The examples use the Les Misérables co-appearance network (77 characters,
254 weighted edges). Every code block on this page runs as part of the test
suite (``tests/test_describe.py``), so the printed output is what you will
see. The statistical tests and the Kolmogorov-Smirnov measures need SciPy
(``pip install nx-backbone[full]``).

>>> import networkx as nx
>>> import nx_backbone as nb
>>> G = nx.les_miserables_graph()
>>> G.number_of_nodes(), G.number_of_edges()
(77, 254)

1. Why per-edge tests need correcting
-------------------------------------

A statistical backbone runs one hypothesis test per edge. At
:math:`\alpha = 0.05`, about 5% of edges are expected to pass even if
*no* edge is truly significant: here, roughly :math:`0.05 \times 254
\approx 13` false positives. A multiple-testing correction adjusts the
p-values for the number of tests before they are compared with
:math:`\alpha`. :func:`~nx_backbone.threshold_filter` applies one with
its ``mtc`` argument (the corrections of R's ``p.adjust``, see
:func:`~nx_backbone.adjust_pvalues`):

- ``"bonferroni"`` and ``"holm"`` control the *family-wise error rate*, the
  chance of keeping even one false edge;
- ``"bh"`` (Benjamini-Hochberg) and ``"by"`` control the *false discovery
  rate*, the expected share of false edges among those kept.

Score the edges with the disparity filter, then filter at the same
:math:`\alpha` with each correction:

>>> scored = nb.disparity_filter(G)
>>> for mtc in ["none", "bonferroni", "holm", "bh"]:
...     bb = nb.threshold_filter(scored, "disparity_pvalue", 0.05, mtc=mtc)
...     print(f"{mtc:>10}: {bb.number_of_edges()} edges")
      none: 9 edges
bonferroni: 0 edges
      holm: 0 edges
        bh: 0 edges

The uncorrected backbone keeps 9 edges, fewer than the 13 that chance alone
would produce, and no correction keeps any. The smallest p-values show why:

>>> pvalues = sorted(d["disparity_pvalue"] for _, _, d in scored.edges(data=True))
>>> [round(p, 4) for p in pvalues[:5]]
[0.0005, 0.0008, 0.0081, 0.0113, 0.0173]
>>> [round(p, 3) for p in sorted(nb.adjust_pvalues(pvalues, "bh"))[:5]]
[0.096, 0.096, 0.684, 0.717, 0.72]

Even the strongest evidence does not survive 254 tests. On this small
network, the disparity filter's null model leaves too little power for any
edge to be significant, so the 9-edge backbone should not be reported as a
set of significant ties.

Correction matters less for a method with more power. The marginal
likelihood filter tests each edge against both endpoints' strengths at
once:

>>> ml = nb.marginal_likelihood_filter(G)
>>> for mtc in ["none", "holm", "bh"]:
...     bb = nb.threshold_filter(ml, "ml_pvalue", 0.05, mtc=mtc)
...     print(f"{mtc:>4}: {bb.number_of_edges()} edges")
none: 126 edges
holm: 21 edges
  bh: 100 edges

Over a range of :math:`\alpha`, the two filters behave very differently:

.. page-figure:: validating_backbones/edges-vs-alpha

Choose the correction, and :math:`\alpha`, before looking at the result,
and report both. Holm is the right choice when every retained edge must be
trustworthy; BH keeps more edges at the cost of a controlled share of false
ones, which is usually acceptable when the backbone is used to describe the
overall structure.

2. Comparing methods
--------------------

Different methods answer different questions, so compare a few before
settling on one. Build several backbones of the same graph:

>>> la = nb.lans_filter(G)
>>> backbones = {
...     "disparity": nb.threshold_filter(scored, "disparity_pvalue", 0.05),
...     "mlf (BH)": nb.threshold_filter(ml, "ml_pvalue", 0.05, mtc="bh"),
...     "lans (BH)": nb.threshold_filter(la, "lans_pvalue", 0.05, mtc="bh"),
...     "metric": nb.boolean_filter(nb.metric_backbone(G), "metric_keep"),
...     "salience": nb.threshold_filter(
...         nb.high_salience_skeleton(G), "salience", 0.5, mode="above"
...     ),
... }

:func:`~nx_backbone.compare_backbones` evaluates each backbone with
any measures that take ``(original, backbone)``.
:func:`~nx_backbone.reachability` takes a single graph, so wrap it:

>>> def backbone_reachability(original, backbone):
...     return nb.reachability(backbone)
>>> measures = [
...     nb.node_fraction,
...     nb.edge_fraction,
...     nb.weight_fraction,
...     backbone_reachability,
...     nb.ks_degree,
...     nb.ks_weight,
... ]
>>> results = nb.compare_backbones(G, backbones, measures=measures)
>>> columns = ["nodes", "edges", "weight", "reach", "KS deg", "KS w"]
>>> print(f"{'method':10}", " ".join(f"{c:>6}" for c in columns))
method      nodes  edges weight  reach KS deg   KS w
>>> for name, row in results.items():
...     print(f"{name:10}", " ".join(f"{v:6.3f}" for v in row.values()))
disparity   0.130  0.035  0.171  0.010  0.870  0.838
mlf (BH)    0.844  0.394  0.605  0.203  0.403  0.292
lans (BH)   1.000  0.425  0.523  0.461  0.429  0.113
metric      1.000  0.642  0.828  1.000  0.208  0.173
salience    1.000  0.299  0.404  1.000  0.571  0.132

How to read the columns:

- **nodes, edges, weight** -- the share of non-isolated nodes, edges and
  total weight kept (:func:`~nx_backbone.node_fraction`,
  :func:`~nx_backbone.edge_fraction`,
  :func:`~nx_backbone.weight_fraction`). A good backbone keeps few
  edges but much of the weight.
- **reach** -- the share of node pairs still joined by a path
  (:func:`~nx_backbone.reachability`); the original network is
  connected, so it scores 1. The metric backbone and the salience skeleton
  keep the network connected; the statistical backbones break it apart.
- **KS deg, KS w** -- Kolmogorov-Smirnov distances between the degree and
  weight distributions of the original and the backbone
  (:func:`~nx_backbone.ks_degree`, :func:`~nx_backbone.ks_weight`);
  lower means the distribution is better preserved.

No single row wins. The metric backbone preserves connectivity and the
degree distribution but keeps almost two-thirds of the edges; the
statistical backbones are sparser but fragment the network. See
:doc:`../user_guide/evaluation` for more on each measure.

3. Robustness: the edges every method keeps
-------------------------------------------

Edges kept by several methods with different assumptions are less likely to
be artefacts of any one null model.
:func:`~nx_backbone.consensus_backbone` keeps the edges present in
every backbone it is given:

>>> core = nb.consensus_backbone(
...     backbones["mlf (BH)"], backbones["lans (BH)"], backbones["metric"]
... )
>>> core.number_of_edges(), core.number_of_nodes()
(73, 65)

Counting how many backbones contain each edge shows how stable the result
is, rather than a single yes-or-no answer:

>>> from collections import Counter
>>> votes = Counter()
>>> for name in ["mlf (BH)", "lans (BH)", "metric", "salience"]:
...     votes.update(frozenset(e) for e in backbones[name].edges())
>>> sorted(Counter(votes.values()).items())
[(1, 19), (2, 41), (3, 70), (4, 34)]

Thirty-four edges are kept by all four methods; nineteen are kept by only
one and deserve less weight in any interpretation.

.. page-figure:: validating_backbones/vote-counts
   :width: 70%

Varying a parameter of a single method is less informative: thresholds at
several values of :math:`\alpha` are nested, so their consensus is simply
the strictest backbone. Vary the null model (as above) or the correction
instead:

>>> for alpha in [0.01, 0.05, 0.1]:
...     sizes = [
...         nb.threshold_filter(ml, "ml_pvalue", alpha, mtc=mtc).number_of_edges()
...         for mtc in ["none", "bh"]
...     ]
...     print(f"alpha={alpha}: {sizes[0]} uncorrected, {sizes[1]} with BH")
alpha=0.01: 80 uncorrected, 37 with BH
alpha=0.05: 126 uncorrected, 100 with BH
alpha=0.1: 155 uncorrected, 126 with BH

A backbone whose size changes this much with :math:`\alpha` should be
reported with its sensitivity, not as a single number.

4. Signed backbones
-------------------

The tests above are one-tailed: they keep edges that are *stronger* than the
null model expects. With ``signed=True``, the disparity, marginal
likelihood and LANS filters (and the bipartite null models ``sdsm``,
``fdsm``, ``fixedfill``, ``fixedrow`` and ``fixedcol``) run a two-tailed
test. An edge can then be significantly strong *or* significantly weak, and
each scored edge gets a ``sign`` attribute: ``+1`` for stronger than
expected, ``-1`` for weaker. An edge whose upper and lower tails are exactly
equal (for example the only link of a degree-1 node under LANS) is signed
``-1``, as in R ``backbone``.

>>> signed = nb.marginal_likelihood_filter(G, signed=True)
>>> signed_bb = nb.threshold_filter(signed, "ml_pvalue", 0.05)
>>> sorted(Counter(d["sign"] for _, _, d in signed_bb.edges(data=True)).items())
[(-1, 1), (1, 105)]
>>> [(u, v, d["weight"]) for u, v, d in signed_bb.edges(data=True) if d["sign"] < 0]
[('Valjean', 'Bossuet', 1)]

Valjean has the largest strength in the network (158) and Bossuet one of
the larger ones (66), so the null model, which pairs endpoints in
proportion to their strengths, expects them to co-appear about six times.
They co-appear once, so the edge is significantly weak. A negative edge is a statement about an *expected*
relationship that did not happen, which is why signed backbones are
popular for bipartite projections (for example, legislators who co-sponsor
less often than chance).

Keep the sign when analyzing the backbone, for example by splitting it into
its positive and negative parts:

>>> positive = nb.boolean_filter(
...     nb.threshold_filter(signed, "ml_pvalue", 0.05),
...     "sign",
... )
>>> positive.number_of_edges()
106

:func:`~nx_backbone.boolean_filter` keeps edges whose attribute is
*truthy*, and ``-1`` is truthy, so this kept every edge. Compare the sign
explicitly instead:

>>> positive = signed_bb.edge_subgraph(
...     (u, v) for u, v, d in signed_bb.edges(data=True) if d["sign"] > 0
... )
>>> positive.number_of_edges()
105

The two-tailed p-value is twice the smaller one-tailed p-value, so a
signed test keeps fewer strong edges than the one-tailed test at the same
:math:`\alpha` (106 rather than 126 here) and should be corrected in the
same way.

5. Reporting the backbone
-------------------------

A methods section should name the method and its null model, give
:math:`\alpha` and the correction, say how much of the network was kept
and cite the method and the software.
:func:`~nx_backbone.describe_backbone` writes that paragraph from the
same arguments you passed to the filter (its version number comes from the
installed package):

>>> bb = backbones["mlf (BH)"]
>>> print(nb.describe_backbone(
...     "marginal_likelihood_filter", alpha=0.05, mtc="bh", original=G, backbone=bb
... ))  # doctest: +ELLIPSIS
We extracted the backbone with the marginal likelihood filter (Dianati, 2016), which tests each edge's weight against a binomial null model in which unit edges choose their endpoints with probability proportional to node strength. Edges were kept if their p-value was below alpha = 0.05 after a Benjamini-Hochberg false discovery rate correction for multiple comparisons (Benjamini & Hochberg, 1995). The backbone retains 100 of 254 edges (39.4%) and 65 of 77 non-isolated nodes (84.4%), and 60.5% of the total edge weight. Backbones were computed with nx-backbone version ... (Keegan, n.d.).
<BLANKLINE>
References
<BLANKLINE>
Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B, 57(1), 289-300.
Dianati, N. (2016). Unwinding the hairball graph: Pruning algorithms for weighted complex networks. Physical Review E, 93, 012304. https://doi.org/10.1103/PhysRevE.93.012304
Keegan, B. C. (n.d.). nx-backbone: Backbone extraction algorithms for complex networks (Version ...) [Python package]. https://github.com/brianckeegan/nx-backbone

``style="markdown"`` returns the same text formatted for Markdown, and
``style="bibtex"`` returns the BibTeX entries for your reference manager:

>>> print(nb.describe_backbone(
...     "marginal_likelihood_filter", alpha=0.05, mtc="bh", style="bibtex"
... ))  # doctest: +ELLIPSIS
@article{dianati2016,
  author = {Dianati, N.},
  title = {Unwinding the hairball graph: Pruning algorithms for weighted complex networks},
  journal = {Physical Review E},
  volume = {93},
  pages = {012304},
  year = {2016},
  doi = {10.1103/PhysRevE.93.012304}
}
<BLANKLINE>
@article{benjamini1995,
...
@misc{keegan_nx_backbone,
...
}

For a signed backbone, pass ``signed=True`` and the paragraph also reports
how many retained edges are positive and negative. Pass any other
parameters you set (such as ``s`` for :func:`~nx_backbone.lspar`) as
keyword arguments and they are listed too. The ``CITATION.cff`` file in
the repository holds the same citation metadata for the package.

Checklist
---------

Before reporting a backbone:

#. Fix :math:`\alpha` and the correction in advance; report both.
#. Check that the backbone is not empty, or close to what chance alone
   would produce, after correction.
#. Compare it with at least one method based on a different idea, using
   the fraction, reachability and KS measures.
#. Report which edges are robust across methods (consensus or vote
   counts).
#. For a signed backbone, analyze the positive and negative edges
   separately.
#. Generate the methods text and citations with
   :func:`~nx_backbone.describe_backbone`.
