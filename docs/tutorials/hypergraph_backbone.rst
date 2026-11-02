Hypergraph Backbones
====================

This tutorial demonstrates backbone extraction for **hypergraphs** -- networks
whose edges (hyperedges) join an arbitrary number of nodes, capturing
higher-order interactions that pairwise graphs cannot. It covers the
information-theoretic (MDL) backbone, structural reductions, statistical
validation, and interoperability with the higher-order ecosystem.

For the same methods told as one case study, with figures of the
description-length trade-off, the local aggregation rules and the statistical
validation, see :doc:`/auto_examples/hypergraphs/plot_hypergraph_backbones`.

What is a hypergraph backbone?
------------------------------

Real higher-order datasets -- co-authorship teams, group chats, protein
complexes -- are often dense and redundant, with smaller interactions nested
inside larger ones. A hypergraph backbone keeps only the essential hyperedges.
Unlike the :doc:`bipartite_backbone` methods (which project a hypergraph to a
pairwise graph), these methods return a **sub-hypergraph**: a subset of the
original hyperedges.

A hypergraph is represented as a plain iterable of hyperedges, each an iterable
of node labels::

    import networkx_backbone as nb

    H = [
        (1, 2, 3, 4),   # a 4-node interaction
        (1, 2, 3),      # nested inside the first hyperedge (redundant)
        (2, 3, 4),      # also nested
        (8, 9),         # a separate pairwise interaction
    ]

MDL backbone (parameter-free)
-----------------------------

:func:`~networkx_backbone.mdl_hypergraph_backbone` implements the
information-theoretic method of Kirkley, Felippe, Malizia & Battiston (2026). It
keeps a minimal set of "parent" hyperedges from which the remaining "child"
hyperedges can be reconstructed via overlap and nestedness, by minimizing a
two-part description length. It is **fully nonparametric** for unweighted
hypergraphs::

    result = nb.mdl_hypergraph_backbone(H)

    print(result.backbone)            # [frozenset({1, 2, 3, 4}), frozenset({8, 9})]
    print(result.compression_ratio)   # inverse compression ratio eta in [0, 1]
    print(result.assignment)          # parent -> [pruned child hyperedges]

The result is a :class:`~networkx_backbone.HypergraphBackbone`. The inverse
compression ratio ``eta`` (also available via
:func:`~networkx_backbone.hypergraph_compression_ratio`) is near 0 for highly
redundant hypergraphs and equals 1 when no compression is possible.

Choosing the optimizer
~~~~~~~~~~~~~~~~~~~~~~~

Exact minimization is combinatorial, so the backbone is found with a greedy
heuristic over the intersection graph. Two schemes are available (Appendix D of
the paper):

- ``method="edge"`` adds parent--child links by decreasing gain;
- ``method="node"`` adds backbone parents by decreasing total gain;
- ``method="auto"`` (the default) runs **both** and keeps the lower description
  length, exactly as the paper does.

::

    edge = nb.mdl_hypergraph_backbone(H, method="edge")
    node = nb.mdl_hypergraph_backbone(H, method="node")
    auto = nb.mdl_hypergraph_backbone(H, method="auto")

    print(edge.description_length, node.description_length, auto.description_length)

``"auto"`` never does worse than either scheme. Use ``"edge"`` for the fastest
single pass on large hypergraphs.

Weighted hypergraphs
~~~~~~~~~~~~~~~~~~~~~

When hyperedges carry integer weights (interaction strengths), the backbone
balances topology against weight through a single knob ``gamma`` in ``(0, 1]``.
``gamma=1`` ignores weights; ``gamma`` near 0 makes it costly to leave a
high-weight hyperedge out of the backbone::

    weights = [1, 1, 5, 1]   # the {2, 3, 4} interaction is strong
    result = nb.mdl_hypergraph_backbone(H, weights=weights, gamma=0.1)

Local MDL backbone
~~~~~~~~~~~~~~~~~~

The MDL backbone is *global*: a hyperedge is dropped whenever some
overlapping hyperedge anywhere in the hypergraph encodes it cheaply. Appendix E
of the paper also gives a **local** variant, the higher-order analogue of the
disparity filter. :func:`~networkx_backbone.local_mdl_hypergraph_backbone`
takes each node's *neighborhood* (the hyperedges that contain it), finds the
MDL backbone of that neighborhood on its own, and then combines the
neighborhood backbones.

In this example, a core interaction has two nested sub-interactions and two
"satellite" interactions, each joining part of the core to three nodes that
appear nowhere else::

    H2 = [
        (0, 1, 2, 3, 4, 5),        # a core interaction
        (0, 1, 2, 3, 4),           # nested inside the core
        (0, 1, 2, 3, 5),           # nested inside the core
        (0, 1, 2, 3, 6, 7, 8),     # the core's 0-3 plus nodes 6, 7, 8
        (2, 3, 4, 5, 9, 10, 11),   # the core's 2-5 plus nodes 9, 10, 11
    ]

    nb.mdl_hypergraph_backbone(H2).backbone
    # [frozenset({0, 1, 2, 3, 4, 5}), frozenset({2, 3, 4, 5, 9, 10, 11})]

    local = nb.local_mdl_hypergraph_backbone(H2)
    local.backbone
    # [frozenset({0, 1, 2, 3, 4, 5}), frozenset({0, 1, 2, 3, 6, 7, 8}),
    #  frozenset({2, 3, 4, 5, 9, 10, 11})]

The global backbone encodes the satellite ``{0, 1, 2, 3, 6, 7, 8}`` as a child
of another hyperedge. In the neighborhoods of nodes 6, 7, and 8 it is the only
hyperedge, so it is part of their local backbones, and the local backbone keeps
it. Both backbones drop the two nested sub-interactions. The result records
each node's neighborhood backbone and its description length::

    local.local_backbones[6]               # [frozenset({0, 1, 2, 3, 6, 7, 8})]
    local.local_backbones[0]               # the core and the first satellite
    local.local_description_lengths[0]     # L(G_0, B_0*) in bits

A hyperedge can be in the neighborhood backbones of some of its nodes and not
others. The ``aggregation`` argument sets how many must agree:

- ``"union"`` keeps a hyperedge if **any** of its nodes' neighborhood backbones
  contains it. Every node stays in the backbone, but the backbone is dense.
- ``"majority"`` (the default, used for Table II of the paper) keeps it if
  **at least half** of them do. For pairwise edges, this matches the disparity
  filter's rule of keeping an edge that either endpoint keeps.
- ``"all"`` keeps it only if **every** one of them does. This is the most
  conservative rule, and the backbone is often very sparse.

::

    for rule in ["union", "majority", "all"]:
        result = nb.local_mdl_hypergraph_backbone(H2, aggregation=rule)
        print(rule, len(result.backbone), round(result.compression_ratio, 2))
    # union 3 0.84
    # majority 3 0.84
    # all 2 0.85

The compression ratio of a local backbone is the paper's
:math:`\eta^{(\mathrm{local})}`. It sums the neighborhood description lengths
and divides by the cost of sending each neighborhood without compression. Local
backbones are usually denser than the global one and compress less (0.84 here
against 0.79 for the global backbone). Choose the local backbone when you care
more about keeping each node's important interactions than about overall
compression. ``weights``, ``gamma``, ``prior``, and ``method`` work as for
:func:`~networkx_backbone.mdl_hypergraph_backbone` and apply within each
neighborhood.

Structural reductions
---------------------

Several purely structural backbones have no analog in pairwise graphs:

- :func:`~networkx_backbone.maximal_hyperedges` -- inclusion (toplex) reduction:
  drop every hyperedge contained in another::

    nb.maximal_hyperedges(H)        # [frozenset({1, 2, 3, 4}), frozenset({8, 9})]

- :func:`~networkx_backbone.order_filter` -- keep hyperedges by order (size)::

    nb.order_filter(H, min_order=3)  # interactions of three or more nodes

- :func:`~networkx_backbone.s_components` -- group hyperedges into s-connected
  components (those sharing at least ``s`` nodes)::

    nb.s_components(H, s=2)

Statistical validation (SVH / SVC)
----------------------------------

:func:`~networkx_backbone.statistically_validated_hypergraph` keeps hyperedges
that **recur** more often than expected under a null model preserving node
activity (Musciotto, Battiston & Mantegna, 2021). Repeated hyperedges (or
integer ``weights``) are treated as interaction counts::

    events = [(1, 2)] * 5 + [(3, 4)] * 100      # {1, 2} co-occurs 5x; {3, 4} is background
    result = nb.statistically_validated_hypergraph(events, alpha=0.05)
    print(result.validated)                     # [frozenset({1, 2})]
    print(result.pvalues[frozenset({1, 2})])

:func:`~networkx_backbone.statistically_validated_cores` additionally validates
significant sub-groups (cores), attributing significance to the largest
validated group. Both require ``scipy``.

Interoperability
----------------

Convert a hypergraph to its incidence bipartite graph to reuse the projection
backbones (:doc:`bipartite_backbone`) such as SDSM and FDSM::

    B, nodes = nb.hypergraph_to_bipartite(H)
    scored = nb.sdsm(B, agent_nodes=nodes)
    projection_backbone = nb.threshold_filter(scored, "sdsm_pvalue", 0.05, mode="below")

Exchange hypergraphs with the wider ecosystem through the HIF interchange format
(standard library only) or the optional ``xgi``, ``HyperNetX``, ``HypergraphX``,
and HAT adapters (imported lazily)::

    nb.write_hif(H, "hypergraph.hif")
    edges = nb.read_hif("hypergraph.hif")

    # With the relevant library installed:
    # hg = nb.to_xgi(H);  edges = nb.from_hypernetx(hnx_hypergraph)

References
----------

- Kirkley, A., Felippe, H., Malizia, F., & Battiston, F. (2026). *Hypergraph
  backboning*. arXiv:2606.00893.
- Musciotto, F., Battiston, F., & Mantegna, R. N. (2021). *Detecting informative
  higher-order interactions in statistically validated hypergraphs*.
  Communications Physics, 4, 218.
