"""
Hybrid backbone extraction methods.

These methods combine statistical and structural approaches.

Methods
-------
glab_filter
    Zhang et al. (2014) -- Globally and Locally Adaptive Backbone.
    Tests each link's shortest-path involvement against a local,
    degree-dependent null model.
"""

from __future__ import annotations

import math

import networkx as nx

from networkx_backbone._docstrings import append_complexity_docstrings
from networkx_backbone._typing import not_implemented_for

__all__ = ["glab_filter"]


@not_implemented_for("multigraph")
def glab_filter(G: nx.Graph, weight: str | None = "weight", c: float = 0.5) -> nx.Graph:
    r"""Compute GLAB (Globally and Locally Adaptive Backbone) p-values.

    The GLAB method [1]_ (called GLANB in the paper) combines global and local
    information.  For every node *i* and incident link *(i, j)* it computes the
    *involvement* of the link: the fraction of shortest paths from *i* to every
    other node that leave *i* through *(i, j)*,

    .. math::

        I_{ij} = \frac{1}{N - 1} \sum_{s \neq i}
                 \frac{g_{is}(i, j)}{g_{is}},

    where :math:`g_{is}` is the number of shortest paths between *i* and *s*
    and :math:`g_{is}(i, j)` the number of those that use link *(i, j)*.  Link
    lengths are ``1 / weight``, so heavy links are short.  Involvements sum to
    one over each node's links.  Under a null model in which a node's
    involvement is spread uniformly over its :math:`k_i` links, the
    statistical importance is

    .. math::

        SI_{ij} = (1 - I_{ij})^{(k_i - 1)^c}, \qquad k_i \geq 2,

    and the p-value of an undirected link is :math:`\min(SI_{ij}, SI_{ji})`.
    When one endpoint has degree 1, only the other endpoint's value is used.

    For a directed graph the paper (Section 2.3, Eqs. 5-6) splits each link
    *(i, j)* into an out-link of *i* and an in-link of *j*.  The
    *out-involvement* :math:`I^{(out)}_{ij}` is the fraction of shortest
    paths *from* *i* that leave through *(i, j)*, normalized by the number
    of nodes reachable from *i*; the *in-involvement* :math:`I^{(in)}_{ij}`
    is the fraction of shortest paths *to* *j* that arrive through
    *(i, j)*, normalized by the number of nodes that can reach *j*.  Then

    .. math::

        SI^{(out)}_{ij} = (1 - I^{(out)}_{ij})^{(k^{(out)}_i - 1)^c},
        \qquad
        SI^{(in)}_{ij} = (1 - I^{(in)}_{ij})^{(k^{(in)}_j - 1)^c},

    each defined when the degree in its exponent is at least 2, and the
    p-value of the directed link is the minimum of the two.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph, directed or undirected.
    weight : string or None, optional (default="weight")
        Edge attribute holding positive weights; shortest paths use
        ``1 / weight`` as link length.  If ``None``, every link has length 1.
    c : float, optional (default=0.5)
        Non-negative exponent controlling how strongly the degree of a node
        enters the null model.  ``c=0`` ignores degree.

    Returns
    -------
    H : networkx.Graph or networkx.DiGraph
        A copy of *G* with a ``"glab_pvalue"`` edge attribute.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is a multigraph.
    NetworkXError
        If ``weight`` is given and an edge has a missing, non-positive, or
        non-finite weight.
    ValueError
        If ``c`` is negative.

    See Also
    --------
    disparity_filter : Purely local significance test.
    high_salience_skeleton : Purely global shortest-path-tree method.

    Notes
    -----
    In a disconnected graph the involvement of node *i* is normalized by the
    number of nodes reachable from *i* rather than :math:`N - 1`, so the
    involvements of each node still sum to one.  A link whose endpoints both
    have degree 1 (an isolated dyad) has no null distribution and gets a
    p-value of 1.

    In a directed graph, :math:`I^{(out)}` is computed by a shortest-path
    search from *i* along out-links and :math:`I^{(in)}` by a search from *j*
    along reversed links.  A link whose source has out-degree below 2 uses
    only :math:`SI^{(in)}`, one whose target has in-degree below 2 uses only
    :math:`SI^{(out)}`, and a link with neither gets a p-value of 1, which
    mirrors the undirected degree-1 rule.  On a symmetric directed graph
    (every link present in both directions with equal weight), out- and
    in-involvement of *(i, j)* equal the undirected :math:`I_{ij}` and
    :math:`I_{ji}`, so both directions of each link get the undirected
    p-value.

    **Comparison with netbone.**  netbone's ``glanb`` counts shortest paths:
    a target reached by *g* tied paths weighs *g* times in the involvement,
    whose denominator is the total number of paths.  Here every reachable
    target weighs the same, as in the formula above.  The two agree when no
    shortest paths tie.

    References
    ----------
    .. [1] Zhang, X., Zhang, Z., Zhao, H., Wang, Q., & Zhu, J. (2014).
       Extracting the globally and locally adaptive backbone of complex
       networks. *PLoS ONE*, 9(6), e100428.
       https://doi.org/10.1371/journal.pone.0100428

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import glab_filter, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = glab_filter(G)
    >>> backbone = threshold_filter(H, "glab_pvalue", 0.05, mode="below")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    if c < 0:
        raise ValueError(f"c must be non-negative, got {c}")

    H = G.copy()
    if weight is not None:
        for u, v, data in G.edges(data=True):
            w = data.get(weight)
            if w is None or not (0 < w < math.inf):
                raise nx.NetworkXError(
                    f"Edge ({u}, {v}) has a missing, non-positive, or non-finite "
                    f"{weight!r} attribute."
                )

        def length(u, v, data):
            return 1.0 / data[weight]

    else:
        length = None  # type: ignore[assignment]

    if G.is_directed():
        # SI^(out)_ij from i's out-links; SI^(in)_ij from j's in-links, i.e.
        # j's out-links in the reversed graph (stored there as (j, i)).
        si_out = _node_importance(G, dict(G.out_degree()), length, c)  # type: ignore[attr-defined]
        si_in = _node_importance(G.reverse(copy=False), dict(G.in_degree()), length, c)  # type: ignore[attr-defined]
        for u, v, data in H.edges(data=True):
            values = [
                d[key] for d, key in ((si_out, (u, v)), (si_in, (v, u))) if key in d
            ]
            data["glab_pvalue"] = min(values) if values else 1.0
        return H

    si = _node_importance(G, dict(G.degree()), length, c)
    for u, v, data in H.edges(data=True):
        values = [si[key] for key in ((u, v), (v, u)) if key in si]
        data["glab_pvalue"] = min(values) if values else 1.0

    return H


def _node_importance(G, degree, length, c):
    """Return ``{(i, j): SI_ij}`` for every node *i* with ``degree[i] >= 2``.

    ``SI_ij = (1 - I_ij) ** ((degree[i] - 1) ** c)`` is the importance of
    link ``(i, j)`` seen from *i*, where ``I_ij`` is the involvement over
    the links leaving *i* in *G* (its successors if *G* is directed).
    """
    si = {}
    for i in G:
        if degree[i] < 2:
            continue
        involvement = _involvement(G, i, length)
        exponent = (degree[i] - 1) ** c
        for j, inv in involvement.items():
            si[(i, j)] = max(0.0, 1.0 - inv) ** exponent
    return si


def _involvement(G, source, length):
    """Fraction of shortest paths from ``source`` that leave via each neighbor.

    Returns ``{neighbor: I_source,neighbor}``, normalized by the number of
    nodes reachable from ``source``.  For a directed *G* the neighbors are
    the successors of ``source`` and paths follow edge directions.
    """
    if length is None:
        pred = nx.predecessor(G, source)
        order = list(pred)  # BFS discovery order, nondecreasing distance
    else:
        pred, dist = nx.dijkstra_predecessor_and_distance(G, source, weight=length)
        order = sorted(dist, key=dist.__getitem__)

    sigma = {source: 1}
    # first_hop[v][j]: number of shortest source->v paths whose first link is
    # (source, j).
    first_hop = {}
    for v in order:
        if v == source:
            continue
        sigma[v] = sum(sigma[p] for p in pred[v])
        counts = {}
        for p in pred[v]:
            if p == source:
                counts[v] = counts.get(v, 0) + 1
            else:
                for j, cnt in first_hop[p].items():
                    counts[j] = counts.get(j, 0) + cnt
        first_hop[v] = counts

    reachable = len(order) - 1
    involvement = dict.fromkeys(G[source], 0.0)
    if reachable == 0:
        return involvement
    for v, counts in first_hop.items():
        total = sigma[v]
        for j, cnt in counts.items():
            involvement[j] += cnt / total
    return {j: val / reachable for j, val in involvement.items()}


_COMPLEXITY = {
    "glab_filter": {
        "time": "O(n m + n^2 log n)",
        "space": "O(n k_max + m)",
        "notes": "One single-source shortest-path search per node.",
    }
}

append_complexity_docstrings(globals(), _COMPLEXITY)
