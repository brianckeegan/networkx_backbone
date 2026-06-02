"""
Hypergraph backbone methods.

These methods operate directly on hypergraphs (collections of hyperedges, where
each hyperedge is a set of nodes of arbitrary size) rather than on dyadic graphs.

A hyperedge is given as any iterable of node labels; a hypergraph is an iterable
of hyperedges, for example ``[(1, 2, 3), (2, 4), (1, 2, 3, 5)]``.  NetworkX has
no native hypergraph type, so inputs and outputs use plain Python collections
(the backbone is returned as a list of :class:`frozenset` hyperedges).

Methods
-------
mdl_hypergraph_backbone
    Parameter-free information-theoretic (MDL) backbone that prunes nested and
    redundant hyperedges (Kirkley, Felippe, Malizia & Battiston, 2026).
hypergraph_compression_ratio
    Inverse compression ratio ``eta`` achieved by the MDL backbone.
intersection_graph
    Graph linking hyperedges that share at least one node.
"""

import math
from dataclasses import dataclass, field

import networkx as nx

from networkx_backbone._docstrings import append_complexity_docstrings

__all__ = [
    "intersection_graph",
    "mdl_hypergraph_backbone",
    "hypergraph_compression_ratio",
    "HypergraphBackbone",
]

_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Information-theoretic primitives (all codelengths in bits, log base 2)
# ---------------------------------------------------------------------------


def _log2(x):
    return math.log2(x)


def _log2_factorial(n):
    """log2(n!) via the log-gamma function (stdlib, no numpy/scipy needed)."""
    return math.lgamma(n + 1.0) / _LN2


def _log2_binom(n, k):
    """log2 of the binomial coefficient C(n, k)."""
    if k < 0 or k > n or n < 0:
        return float("-inf")
    return (math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)) / _LN2


def _parent_codelength(size, n_orders, n_nodes):
    """H(p): bits to transmit a parent (backbone) hyperedge.  Eq. (1)."""
    return _log2(n_orders) + _log2_binom(n_nodes, size)


def _child_codelength(size_c, size_p, overlap, n_orders, n_nodes):
    """H(c|p): bits to transmit a child hyperedge from its parent.  Eq. (3)."""
    return (
        _log2(n_orders)
        + _log2(min(size_p, size_c))
        + _log2_binom(size_p, overlap)
        + _log2_binom(n_nodes - size_p, size_c - overlap)
    )


def _reduced_mutual_information(size_p, size_c, overlap, n_nodes):
    """R(c, p): reduced mutual information between hyperedges.  Eq. (11).

    Symmetric in the two hyperedge sizes.  Larger overlap / nestedness gives a
    larger value, so the description length favours assigning highly overlapping
    hyperedges as children of a shared parent.
    """
    p_not_c = size_p - overlap
    c_not_p = size_c - overlap
    rest = n_nodes - size_p - c_not_p  # = n_nodes - |p ∪ c|
    log2_multinomial = (
        math.lgamma(n_nodes + 1.0)
        - math.lgamma(overlap + 1.0)
        - math.lgamma(p_not_c + 1.0)
        - math.lgamma(c_not_p + 1.0)
        - math.lgamma(rest + 1.0)
    ) / _LN2
    return (
        _log2_binom(n_nodes, size_p)
        + _log2_binom(n_nodes, size_c)
        - log2_multinomial
        - _log2(min(size_p, size_c))
    )


# ---------------------------------------------------------------------------
# Weight model (empirical-Bayes Poisson / Geometric prior).  Sec. III.
# ---------------------------------------------------------------------------


def _expected_weight(role, gamma, mean_weight):
    """Expected weight mu_b for a parent (role=1) or child (role=0).  Eq. (15)."""
    return 1.0 + 2.0 * (gamma ** (1 - role)) * (mean_weight - 1.0) / (1.0 + gamma)


def _weight_codelength(weight, role, gamma, mean_weight, prior):
    """L(w, b): bits to transmit a hyperedge weight given its role.  Eqs (16)-(17)."""
    mu = _expected_weight(role, gamma, mean_weight)
    if prior == "poisson":
        if mu - 1.0 <= 0.0:
            return 0.0
        return (
            (mu - 1.0) / _LN2
            - (weight - 1.0) * _log2(mu - 1.0)
            + _log2_factorial(weight - 1.0)
        )
    if prior == "geometric":
        if mu <= 1.0:
            return 0.0
        return _log2(mu) + (weight - 1.0) * _log2(mu / (mu - 1.0))
    raise ValueError(f"prior must be 'poisson' or 'geometric', got {prior!r}")


def _weight_term(weight, gamma, mean_weight, prior):
    """L(w, 1) - L(w, 0): weight-dependent reward for keeping an edge as a parent.

    Negative for high-weight hyperedges when ``gamma < 1``, which discourages
    demoting them to children (i.e. encourages keeping them in the backbone).
    Zero when ``gamma == 1`` or the hypergraph is effectively unweighted.
    """
    return _weight_codelength(weight, 1, gamma, mean_weight, prior) - _weight_codelength(
        weight, 0, gamma, mean_weight, prior
    )


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _normalize_hyperedges(hyperedges, weights):
    """Validate input and collapse duplicate / empty hyperedges.

    Returns ``(edges, weights)`` where ``edges`` is a list of unique
    :class:`frozenset` hyperedges and ``weights`` a parallel list of floats.
    Duplicate hyperedges are merged; their weights are summed.
    """
    raw = [frozenset(e) for e in hyperedges]
    if weights is not None:
        weights = [float(w) for w in weights]
        if len(weights) != len(raw):
            raise ValueError("weights must have the same length as hyperedges")
        if any(w < 1.0 for w in weights):
            raise ValueError("weights must be >= 1")

    edges = []
    out_weights = []
    index = {}
    for i, fs in enumerate(raw):
        if len(fs) == 0:
            continue
        w = weights[i] if weights is not None else 1.0
        if fs in index:
            if weights is not None:
                out_weights[index[fs]] += w
        else:
            index[fs] = len(edges)
            edges.append(fs)
            out_weights.append(w)
    return edges, out_weights


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class HypergraphBackbone:
    """Result of :func:`mdl_hypergraph_backbone`.

    Attributes
    ----------
    backbone : list of frozenset
        The retained ("parent") hyperedges forming the structural backbone.
    assignment : dict
        Mapping from each parent hyperedge to the list of child hyperedges it
        encodes (its star in the intersection graph).  Parents with no children
        are absent from this mapping.
    n_nodes : int
        Number of distinct nodes in the input hypergraph.
    n_orders : int
        Number of distinct hyperedge sizes (orders) in the input.
    weighted : bool
        Whether edge weights influenced the backbone.
    gamma : float
        Weight/topology trade-off used (only meaningful when ``weighted``).
    prior : str
        Weight prior used (``"poisson"`` or ``"geometric"``).
    description_length : float
        Description length ``L(G, B*)`` of the input given the backbone (bits).
    baseline_description_length : float
        Description length ``L(G, G)`` with no backbone (bits).
    compression_ratio : float
        Inverse compression ratio ``eta = L(G, B*) / L(G, G)`` in ``[0, 1]``.
        Smaller means more redundant structure was removed.
    """

    backbone: list = field(default_factory=list)
    assignment: dict = field(default_factory=dict)
    n_nodes: int = 0
    n_orders: int = 0
    weighted: bool = False
    gamma: float = 1.0
    prior: str = "poisson"
    description_length: float = 0.0
    baseline_description_length: float = 0.0
    compression_ratio: float = 1.0

    def __len__(self):
        return len(self.backbone)

    @property
    def n_input_hyperedges(self):
        """Total hyperedges = backbone parents + all children."""
        return len(self.backbone) + sum(len(v) for v in self.assignment.values())

    @property
    def fraction_kept(self):
        """Fraction of (unique) input hyperedges retained in the backbone."""
        total = self.n_input_hyperedges
        return len(self.backbone) / total if total else 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def intersection_graph(hyperedges, weight="overlap"):
    """Build the intersection graph of a hypergraph.

    Each hyperedge becomes a node (labelled by its integer index after
    deduplication); two hyperedges are linked when they share at least one node.
    This is the structure over which MDL parent--child relationships are formed
    (Kirkley et al. 2026, Appendix D).

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Each hyperedge is an iterable of node labels.
    weight : string, optional (default="overlap")
        Edge attribute name used to store the overlap size ``|e_i ∩ e_j|``.

    Returns
    -------
    I : networkx.Graph
        Graph on hyperedge indices.  Each node has a ``"members"`` attribute
        (the hyperedge as a :class:`frozenset`); each edge stores the overlap
        size under *weight*.

    Examples
    --------
    >>> from networkx_backbone import intersection_graph
    >>> I = intersection_graph([(1, 2, 3), (2, 3, 4), (5, 6)])
    >>> I.number_of_nodes()
    3
    >>> I[0][1]["overlap"]
    2
    """
    edges, _ = _normalize_hyperedges(hyperedges, None)
    graph = nx.Graph()
    for i, e in enumerate(edges):
        graph.add_node(i, members=e)

    node_to_edges = {}
    for i, e in enumerate(edges):
        for v in e:
            node_to_edges.setdefault(v, []).append(i)

    overlaps = {}
    for incident in node_to_edges.values():
        m = len(incident)
        if m < 2:
            continue
        for a in range(m):
            for b in range(a + 1, m):
                i, j = incident[a], incident[b]
                key = (i, j) if i < j else (j, i)
                overlaps[key] = overlaps.get(key, 0) + 1

    for (i, j), o in overlaps.items():
        graph.add_edge(i, j, **{weight: o})
    return graph


def mdl_hypergraph_backbone(
    hyperedges,
    weights=None,
    gamma=1.0,
    prior="poisson",
    method="edge",
):
    """Extract a hypergraph backbone via minimum description length (MDL).

    Implements the parameter-free information-theoretic backboning method of
    Kirkley, Felippe, Malizia & Battiston [1]_.  The backbone ``B`` is a subset
    of hyperedges ("parents") chosen so that the remaining "child" hyperedges can
    be cheaply reconstructed from a parent they overlap, exploiting the nested
    and redundant structure unique to higher-order networks.  The optimal
    backbone minimises the two-part description length
    ``L(G, B) = L(B) + L(G|B)`` (Eqs (1)-(6)); equivalently it maximises the
    total parent--child reduced mutual information (Eq. (12)).

    The method is **fully nonparametric** for unweighted hypergraphs.  Edge
    weights are incorporated through an empirical-Bayes prior with a single knob
    *gamma* trading off weight against topology (Sec. III).

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Each hyperedge is an iterable of node labels (treated as
        a set; repeated nodes within a hyperedge are ignored).  Duplicate
        hyperedges are merged.
    weights : iterable of numbers or None, optional (default=None)
        Optional per-hyperedge weights (each ``>= 1``), parallel to
        *hyperedges*.  ``None`` means an unweighted hypergraph.
    gamma : float, optional (default=1.0)
        Weight/topology trade-off in ``(0, 1]`` (Eq. (14)).  ``gamma=1`` ignores
        weights (recovers the unweighted backbone); ``gamma -> 0`` makes it
        increasingly costly to leave a high-weight hyperedge out of the backbone.
        Ignored when *weights* is ``None``.
    prior : {"poisson", "geometric"}, optional (default="poisson")
        Weight prior family (Eqs (16)-(17)).  Ignored when *weights* is ``None``.
    method : {"edge"}, optional (default="edge")
        Greedy optimiser.  Only the "edge"-addition scheme (the better-performing
        one in [1]_) is currently implemented.

    Returns
    -------
    result : HypergraphBackbone
        The backbone hyperedges, the parent--child assignment, and compression
        diagnostics (see :class:`HypergraphBackbone`).

    Raises
    ------
    ValueError
        If *gamma* is not in ``(0, 1]``, *prior* is unknown, *weights* has the
        wrong length or contains a value ``< 1``, or *method* is unsupported.

    Notes
    -----
    Exact minimisation is combinatorial; this uses the greedy "edge" heuristic of
    [1]_, which forms a maximum-reward partition of the intersection graph into
    disjoint stars (each child attached to a single parent).  On small inputs the
    greedy compression matches exhaustive search.

    References
    ----------
    .. [1] Kirkley, A., Felippe, H., Malizia, F., & Battiston, F. (2026).
       Hypergraph backboning. arXiv:2606.00893.

    Examples
    --------
    >>> from networkx_backbone import mdl_hypergraph_backbone
    >>> # A 4-node hyperedge with two nested (redundant) sub-hyperedges.
    >>> G = [(1, 2, 3, 4), (1, 2, 3), (2, 3, 4), (8, 9)]
    >>> result = mdl_hypergraph_backbone(G)
    >>> frozenset({1, 2, 3, 4}) in result.backbone
    True
    >>> result.compression_ratio <= 1.0
    True
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    if prior not in ("poisson", "geometric"):
        raise ValueError(f"prior must be 'poisson' or 'geometric', got {prior!r}")
    if method != "edge":
        raise ValueError(
            f"method={method!r} is not supported; only 'edge' is implemented"
        )

    edges, weight_list = _normalize_hyperedges(hyperedges, weights)
    n_edges = len(edges)

    result = HypergraphBackbone(gamma=gamma, prior=prior)
    if n_edges == 0:
        return result

    all_nodes = set()
    sizes = set()
    for e in edges:
        all_nodes.update(e)
        sizes.add(len(e))
    n_nodes = len(all_nodes)
    n_orders = len(sizes)
    result.n_nodes = n_nodes
    result.n_orders = n_orders

    mean_weight = sum(weight_list) / n_edges
    weighted = weights is not None and any(w != 1.0 for w in weight_list)
    result.weighted = weighted

    # Pairwise overlaps via node neighbourhoods (Appendix D).
    node_to_edges = {}
    for i, e in enumerate(edges):
        for v in e:
            node_to_edges.setdefault(v, []).append(i)
    overlaps = {}
    for incident in node_to_edges.values():
        m = len(incident)
        if m < 2:
            continue
        for a in range(m):
            for b in range(a + 1, m):
                i, j = incident[a], incident[b]
                key = (i, j) if i < j else (j, i)
                overlaps[key] = overlaps.get(key, 0) + 1

    # Precompute per-edge weight terms (0 when unweighted).
    if weighted:
        wterm = [_weight_term(w, gamma, mean_weight, prior) for w in weight_list]
    else:
        wterm = [0.0] * n_edges

    # Candidate parent->child moves; gain = R(c, p) + weight_term(c).
    candidates = []
    for (i, j), o in overlaps.items():
        si, sj = len(edges[i]), len(edges[j])
        rmi = _reduced_mutual_information(si, sj, o, n_nodes)
        gain_i_child = rmi + wterm[i]  # i becomes child of parent j
        gain_j_child = rmi + wterm[j]  # j becomes child of parent i
        if gain_i_child > 0.0:
            candidates.append((gain_i_child, sj, si, i, j))
        if gain_j_child > 0.0:
            candidates.append((gain_j_child, si, sj, j, i))

    # Highest gain first; prefer the larger hyperedge as parent on ties.
    candidates.sort(key=lambda t: (-t[0], -t[1], t[2], t[3], t[4]))

    UNDECIDED, PARENT, CHILD = 0, 1, 2
    role = [UNDECIDED] * n_edges
    parent_of = {}
    for _gain, _psize, _csize, c, p in candidates:
        if role[c] != UNDECIDED or role[p] == CHILD:
            continue
        role[c] = CHILD
        parent_of[c] = p
        if role[p] == UNDECIDED:
            role[p] = PARENT

    backbone_idx = [i for i in range(n_edges) if role[i] != CHILD]
    result.backbone = [edges[i] for i in backbone_idx]

    assignment = {}
    for c, p in parent_of.items():
        assignment.setdefault(edges[p], []).append(edges[c])
    result.assignment = assignment

    # Description lengths and compression ratio.
    dl = 0.0
    dl0 = 0.0
    for i, e in enumerate(edges):
        size = len(e)
        dl0 += _parent_codelength(size, n_orders, n_nodes)
        if weighted:
            dl0 += _weight_codelength(weight_list[i], 1, gamma, mean_weight, prior)
        if role[i] == CHILD:
            p = parent_of[i]
            o = len(e & edges[p])
            dl += _child_codelength(size, len(edges[p]), o, n_orders, n_nodes)
            if weighted:
                dl += _weight_codelength(weight_list[i], 0, gamma, mean_weight, prior)
        else:
            dl += _parent_codelength(size, n_orders, n_nodes)
            if weighted:
                dl += _weight_codelength(weight_list[i], 1, gamma, mean_weight, prior)

    result.description_length = dl
    result.baseline_description_length = dl0
    result.compression_ratio = dl / dl0 if dl0 > 0 else 1.0
    return result


def hypergraph_compression_ratio(
    hyperedges, weights=None, gamma=1.0, prior="poisson"
):
    """Inverse compression ratio ``eta`` of the MDL backbone (Eq. (8)).

    A convenience wrapper returning only
    :attr:`HypergraphBackbone.compression_ratio` from
    :func:`mdl_hypergraph_backbone`.  ``eta`` lies in ``[0, 1]``; values near 0
    indicate a highly redundant (compressible) hypergraph, while ``eta = 1``
    indicates no compressible structure.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.
    weights : iterable of numbers or None, optional (default=None)
        Optional per-hyperedge weights.
    gamma : float, optional (default=1.0)
        Weight/topology trade-off (see :func:`mdl_hypergraph_backbone`).
    prior : {"poisson", "geometric"}, optional (default="poisson")
        Weight prior family.

    Returns
    -------
    eta : float
        Inverse compression ratio in ``[0, 1]``.

    Examples
    --------
    >>> from networkx_backbone import hypergraph_compression_ratio
    >>> eta = hypergraph_compression_ratio([(1, 2, 3, 4), (1, 2, 3), (2, 3, 4)])
    >>> 0.0 <= eta <= 1.0
    True
    """
    return mdl_hypergraph_backbone(
        hyperedges, weights=weights, gamma=gamma, prior=prior
    ).compression_ratio


_COMPLEXITY = {
    "intersection_graph": {
        "time": "O(sum_i |G_i|^2)",
        "space": "O(m + P)",
        "notes": "G_i=hyperedges incident to node i, m=hyperedges, P=overlapping pairs.",
    },
    "mdl_hypergraph_backbone": {
        "time": "O(sum_i |G_i|^2 + P log P)",
        "space": "O(m + P)",
        "notes": "Bottleneck is building the intersection graph; P=overlapping pairs.",
    },
    "hypergraph_compression_ratio": {
        "time": "O(sum_i |G_i|^2 + P log P)",
        "space": "O(m + P)",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
