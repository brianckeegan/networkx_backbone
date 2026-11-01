"""
Hypergraph ingestion and interoperability adapters.

``networkx_backbone`` represents a hypergraph as a plain iterable of hyperedges
(each an iterable of node labels), which every hypergraph backbone method in
:mod:`networkx_backbone.hypergraph` consumes.  This module converts that
representation to and from:

- a **NetworkX incidence bipartite graph**, so bipartite projection backbones
  (:func:`~networkx_backbone.sdsm`, :func:`~networkx_backbone.fdsm`, ...) can be
  applied to hypergraphs;
- the **HIF** (Hypergraph Interchange Format) JSON standard;
- the **xgi**, **HyperNetX**, **HypergraphX**, and **Hypergraph Analysis
  Toolbox (HAT)** hypergraph classes.

The third-party libraries are imported lazily; none is a required dependency.
HIF is the recommended interchange path because all four libraries read and
write it, so it needs only the standard library.
"""

import importlib
import json
import os

import networkx as nx

from networkx_backbone.hypergraph import _sorted_labels

__all__ = [
    "hypergraph_to_bipartite",
    "read_hif",
    "write_hif",
    "from_xgi",
    "to_xgi",
    "from_hypernetx",
    "to_hypernetx",
    "from_hypergraphx",
    "to_hypergraphx",
    "from_hat",
    "to_hat",
]

#: HIF ``network-type`` values describing an undirected hypergraph.
_UNDIRECTED_HIF_TYPES = ("undirected", "asc")


def _as_hyperedges(hyperedges):
    """Normalize to a list of non-empty :class:`frozenset` hyperedges (order kept)."""
    out = []
    for e in hyperedges:
        fs = frozenset(e)
        if fs:
            out.append(fs)
    return out


def _require(module, pip_name=None):
    """Import an optional dependency or raise a helpful ImportError."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"'{module}' is required for this conversion; install it with "
            f"`pip install {pip_name or module}`"
        ) from exc


# ---------------------------------------------------------------------------
# NetworkX incidence bipartite graph
# ---------------------------------------------------------------------------


def hypergraph_to_bipartite(hyperedges, edge_prefix="he"):
    """Convert a hypergraph to its incidence bipartite graph.

    Nodes of the hypergraph form one partition (``bipartite=0``); hyperedges form
    the other (``bipartite=1``).  The ``i``-th (non-empty) hyperedge becomes the
    node ``(edge_prefix, i)`` -- a tuple label, so it cannot be confused with a
    string node label such as ``"he0"`` -- carrying a ``"members"`` attribute
    with the hyperedge as a :class:`frozenset`.  The result is exactly the input
    expected by the bipartite projection backbones, so a hypergraph can be
    backboned with, e.g., :func:`~networkx_backbone.sdsm` or
    :func:`~networkx_backbone.fdsm`.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Empty hyperedges are skipped.
    edge_prefix : hashable, optional (default="he")
        First element of the ``(edge_prefix, i)`` hyperedge-node labels.

    Returns
    -------
    B : networkx.Graph
        The incidence bipartite graph.
    nodes : list
        The hypergraph's nodes (the agent partition), suitable to pass as
        ``agent_nodes`` to the bipartite methods.  Sorted in natural order when
        the labels are mutually comparable, otherwise by ``repr``.

    Raises
    ------
    ValueError
        If a hyperedge label ``(edge_prefix, i)`` coincides with a hypergraph
        node label (choose a different *edge_prefix*).

    See Also
    --------
    sdsm, fdsm : Bipartite projection backbones that consume the result.
    intersection_graph : Hyperedge-level (line-graph) view of a hypergraph.

    Examples
    --------
    >>> import networkx as nx
    >>> import networkx_backbone as nb
    >>> B, nodes = nb.hypergraph_to_bipartite([(1, 2, 3), (2, 3, 4), (5, 6)])
    >>> nx.is_bipartite(B)
    True
    >>> nodes
    [1, 2, 3, 4, 5, 6]
    >>> sorted(B[("he", 0)])
    [1, 2, 3]
    >>> scored = nb.sdsm(B, agent_nodes=nodes)
    """
    edges = _as_hyperedges(hyperedges)
    nodes = _sorted_labels({v for e in edges for v in e})
    node_set = set(nodes)

    edge_labels = [(edge_prefix, i) for i in range(len(edges))]
    clashes = [label for label in edge_labels if label in node_set]
    if clashes:
        raise ValueError(
            f"hyperedge label(s) {clashes[:3]!r} collide with hypergraph node "
            "labels; pass a different edge_prefix"
        )

    B = nx.Graph()
    B.add_nodes_from(nodes, bipartite=0)
    for label, e in zip(edge_labels, edges):
        B.add_node(label, bipartite=1, members=e)
        for v in e:
            B.add_edge(v, label)
    return B, nodes


# ---------------------------------------------------------------------------
# HIF (Hypergraph Interchange Format)
# ---------------------------------------------------------------------------


def _to_builtin(value):
    """Convert numpy scalars (also inside tuples) to builtin Python values."""
    if isinstance(value, tuple):
        return tuple(_to_builtin(v) for v in value)
    if type(value).__module__ == "numpy" and getattr(value, "ndim", None) == 0:
        return value.item()
    return value


def _from_json(value):
    """Make a JSON-decoded identifier hashable (JSON arrays -> tuples)."""
    if isinstance(value, list):
        return tuple(_from_json(v) for v in value)
    return value


def _check_network_type(network_type):
    if network_type in _UNDIRECTED_HIF_TYPES:
        return
    if network_type == "directed":
        raise nx.NetworkXNotImplemented(
            "directed hypergraphs are not supported; networkx_backbone "
            "hypergraphs are undirected (HIF network-type 'undirected' or 'asc')"
        )
    raise ValueError(
        "HIF network-type must be 'undirected', 'asc', or 'directed', "
        f"got {network_type!r}"
    )


def write_hif(hyperedges, target=None, weights=None, network_type="undirected"):
    """Serialise a hypergraph to the HIF (Hypergraph Interchange Format) standard.

    HIF is a JSON format shared by xgi, HyperNetX, HypergraphX, and HAT, making it
    the most portable interchange path.  Numpy scalar node labels and weights are
    converted to builtin Python numbers so the result is JSON-serialisable.
    Tuple node labels are written as JSON arrays and restored as tuples by
    :func:`read_hif`.

    The output always has ``nodes`` and ``edges`` lists, which the HAT and
    HypergraphX readers require.  Hyperedge weights go in the edge records'
    ``weight`` field, as the HIF schema specifies.  HyperNetX and HAT read
    them; xgi's reader keeps only ``attrs`` and HypergraphX's ignores weights,
    so for those two use ``to_xgi(*read_hif(path, return_weights=True))`` (or
    :func:`to_hypergraphx`) instead of the library's own reader.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Empty hyperedges are skipped.
    target : str, path, file-like, or None, optional (default=None)
        Where to write.  If ``None``, the HIF dictionary is returned instead of
        being written.
    weights : iterable of numbers or None, optional (default=None)
        Optional per-hyperedge weights, written to the HIF ``edges`` list.
    network_type : {"undirected", "asc"}, optional (default="undirected")
        Value for the HIF ``network-type`` field.

    Returns
    -------
    result : dict or target
        The HIF dictionary if *target* is ``None``; otherwise *target*.

    Raises
    ------
    ValueError
        If *weights* does not match *hyperedges* in length, or *network_type*
        is not a valid HIF network type.
    networkx.NetworkXNotImplemented
        If *network_type* is ``"directed"`` (hyperedges here are undirected).

    See Also
    --------
    read_hif : The inverse operation.

    Examples
    --------
    >>> import networkx_backbone as nb
    >>> hif = nb.write_hif([(1, 2, 3), (3, 4)], weights=[2, 1])
    >>> hif["network-type"]
    'undirected'
    >>> len(hif["incidences"])
    5
    >>> hif["edges"]
    [{'edge': 0, 'weight': 2}, {'edge': 1, 'weight': 1}]
    >>> hif["nodes"][:2]
    [{'node': 1}, {'node': 2}]
    """
    _check_network_type(network_type)
    raw = list(hyperedges)
    if weights is not None:
        weights = list(weights)
        if len(weights) != len(raw):
            raise ValueError("weights must have the same length as hyperedges")

    incidences = []
    node_records = {}
    edge_records = []
    edge_id = 0
    for i, e in enumerate(raw):
        members = frozenset(_to_builtin(v) for v in e)
        if not members:
            continue
        for node in _sorted_labels(members):
            incidences.append({"edge": edge_id, "node": node})
            node_records.setdefault(node, {"node": node})
        record = {"edge": edge_id}
        if weights is not None:
            record["weight"] = _to_builtin(weights[i])
        edge_records.append(record)
        edge_id += 1

    # ``nodes`` and ``edges`` are optional in the spec, but the HAT and
    # HypergraphX readers fail without them, so both are always written.
    hif = {
        "network-type": network_type,
        "metadata": {},
        "incidences": incidences,
        "nodes": list(node_records.values()),
        "edges": edge_records,
    }

    if target is None:
        return hif
    if hasattr(target, "write"):
        json.dump(hif, target)
        return target
    with open(target, "w") as handle:
        json.dump(hif, handle)
    return target


def read_hif(source, return_weights=False):
    """Read a hypergraph from the HIF (Hypergraph Interchange Format) standard.

    JSON arrays used as node or edge identifiers are converted to tuples so they
    are hashable (this restores tuple node labels written by :func:`write_hif`).

    Parameters
    ----------
    source : dict, str, path, or file-like
        A parsed HIF dictionary, a path to a HIF JSON file, or an open file.
    return_weights : bool, optional (default=False)
        If ``True``, also return per-hyperedge weights, read from each ``edges``
        record's ``weight`` (or ``attrs["weight"]``) and defaulting to 1.

    Returns
    -------
    hyperedges : list of frozenset
        The hyperedges, in order of first appearance among the incidences.
    weights : list of numbers
        Returned only when *return_weights* is ``True``.

    Raises
    ------
    TypeError
        If *source* is not a dict, path, or file-like object.
    networkx.NetworkXNotImplemented
        If the HIF ``network-type`` is ``"directed"`` (directed hyperedges would
        otherwise silently lose their head/tail orientation).
    ValueError
        If the HIF ``network-type`` is not a recognized value.

    See Also
    --------
    write_hif : The inverse operation.

    Examples
    --------
    >>> import networkx_backbone as nb
    >>> hif = nb.write_hif([(1, 2, 3), (3, 4)], weights=[2, 1])
    >>> edges, weights = nb.read_hif(hif, return_weights=True)
    >>> [sorted(e) for e in edges], weights
    ([[1, 2, 3], [3, 4]], [2, 1])
    """
    if isinstance(source, dict):
        hif = source
    elif isinstance(source, (str, os.PathLike)):
        with open(source) as handle:
            hif = json.load(handle)
    elif hasattr(source, "read"):
        hif = json.load(source)
    else:
        raise TypeError("source must be a dict, path, or file-like object")

    _check_network_type(hif.get("network-type", "undirected"))

    groups = {}
    for incidence in hif.get("incidences", []):
        edge = _from_json(incidence["edge"])
        groups.setdefault(edge, set()).add(_from_json(incidence["node"]))

    edge_ids = list(groups)
    hyperedges = [frozenset(groups[eid]) for eid in edge_ids]
    if not return_weights:
        return hyperedges

    weight_map = {}
    for rec in hif.get("edges", []):
        weight = rec.get("weight")
        if weight is None:
            weight = (rec.get("attrs") or {}).get("weight", 1)
        weight_map[_from_json(rec["edge"])] = weight
    weights = [weight_map.get(eid, 1) for eid in edge_ids]
    return hyperedges, weights


# ---------------------------------------------------------------------------
# Shared helpers for the library adapters
# ---------------------------------------------------------------------------


def _edges_and_weights(hyperedges, weights):
    """Drop empty hyperedges, keeping *weights* (if any) aligned with the rest."""
    raw = list(hyperedges)
    if weights is not None:
        weights = list(weights)
        if len(weights) != len(raw):
            raise ValueError("weights must have the same length as hyperedges")
    edges, kept = [], []
    for i, e in enumerate(raw):
        fs = frozenset(e)
        if fs:
            edges.append(fs)
            if weights is not None:
                kept.append(_to_builtin(weights[i]))
    return edges, (kept if weights is not None else None)


def _result(edges, weights, return_weights):
    if return_weights:
        return edges, [_to_builtin(w) for w in weights]
    return edges


# ---------------------------------------------------------------------------
# xgi
# ---------------------------------------------------------------------------


def from_xgi(H, return_weights=False):
    """Convert an :class:`xgi.Hypergraph` to a list of :class:`frozenset` hyperedges.

    Parameters
    ----------
    H : xgi.Hypergraph
        The xgi hypergraph (read via ``H.edges.members(dtype=dict)``).
    return_weights : bool, optional (default=False)
        If ``True``, also return each edge's ``"weight"`` attribute
        (default 1).

    Returns
    -------
    hyperedges : list of frozenset
        One frozenset per xgi hyperedge, in xgi's edge order.  Multi-edges are
        kept; edge IDs are discarded.
    weights : list of numbers
        Returned only when *return_weights* is ``True``.

    See Also
    --------
    to_xgi : The inverse conversion.
    read_hif : Library-independent interchange via HIF.

    Examples
    --------
    >>> import xgi  # doctest: +SKIP
    >>> import networkx_backbone as nb
    >>> nb.from_xgi(xgi.Hypergraph([[1, 2, 3], [3, 4]]))  # doctest: +SKIP
    [frozenset({1, 2, 3}), frozenset({3, 4})]
    """
    members = H.edges.members(dtype=dict)
    edges = [frozenset(m) for m in members.values()]
    if not return_weights:
        return edges
    return _result(edges, [H.edges[e].get("weight", 1) for e in members], True)


def to_xgi(hyperedges, weights=None):
    """Convert hyperedges to an :class:`xgi.Hypergraph`.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Empty hyperedges are skipped; the remaining edges get
        the IDs ``0, 1, 2, ...``.  Repeated hyperedges become xgi multi-edges.
    weights : iterable of numbers or None, optional (default=None)
        Per-hyperedge weights (aligned with *hyperedges*), stored as the
        ``"weight"`` edge attribute.

    Returns
    -------
    H : xgi.Hypergraph
        The equivalent xgi hypergraph.

    Raises
    ------
    ImportError
        If ``xgi`` is not installed.
    ValueError
        If *weights* does not match *hyperedges* in length.

    See Also
    --------
    from_xgi : The inverse conversion.
    write_hif : Library-independent interchange via HIF.

    Examples
    --------
    >>> import networkx_backbone as nb
    >>> H = nb.to_xgi([(1, 2, 3), (3, 4)], weights=[2.0, 1.0])  # doctest: +SKIP
    >>> H.num_edges  # doctest: +SKIP
    2
    """
    xgi = _require("xgi")
    edges, weights = _edges_and_weights(hyperedges, weights)
    H = xgi.Hypergraph()
    # The {id: members} form is unambiguous: from a list, xgi reads a 2-tuple
    # of tuple labels as a (members, edge_id) pair.
    H.add_edges_from({i: set(e) for i, e in enumerate(edges)})
    if weights is not None:
        H.set_edge_attributes(dict(enumerate(weights)), name="weight")
    return H


# ---------------------------------------------------------------------------
# HyperNetX
# ---------------------------------------------------------------------------


def from_hypernetx(H, return_weights=False):
    """Convert a :class:`hypernetx.Hypergraph` to a list of :class:`frozenset` hyperedges.

    Parameters
    ----------
    H : hypernetx.Hypergraph
        The HyperNetX hypergraph (read via ``H.incidence_dict``).
    return_weights : bool, optional (default=False)
        If ``True``, also return each edge's HyperNetX ``weight`` property
        (HyperNetX defaults it to 1 and stores it as a float).

    Returns
    -------
    hyperedges : list of frozenset
        One frozenset per HyperNetX edge, in ``incidence_dict`` order (edge
        names are discarded).
    weights : list of numbers
        Returned only when *return_weights* is ``True``.

    See Also
    --------
    to_hypernetx : The inverse conversion.
    read_hif : Library-independent interchange via HIF.

    Examples
    --------
    >>> import hypernetx as hnx  # doctest: +SKIP
    >>> import networkx_backbone as nb
    >>> nb.from_hypernetx(hnx.Hypergraph({"a": [1, 2], "b": [2, 3]}))  # doctest: +SKIP
    [frozenset({1, 2}), frozenset({2, 3})]
    """
    incidence = H.incidence_dict
    edges = [frozenset(members) for members in incidence.values()]
    if not return_weights:
        return edges
    return _result(edges, [H.edges[e].weight for e in incidence], True)


def to_hypernetx(hyperedges, weights=None):
    """Convert hyperedges to a :class:`hypernetx.Hypergraph`.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Empty hyperedges are skipped; the remaining edges are
        named ``0, 1, 2, ...``.
    weights : iterable of numbers or None, optional (default=None)
        Per-hyperedge weights (aligned with *hyperedges*), stored as the
        HyperNetX edge ``weight`` property.

    Returns
    -------
    H : hypernetx.Hypergraph
        The equivalent HyperNetX hypergraph.

    Raises
    ------
    ImportError
        If ``hypernetx`` is not installed.
    ValueError
        If *weights* does not match *hyperedges* in length.

    See Also
    --------
    from_hypernetx : The inverse conversion.
    write_hif : Library-independent interchange via HIF.

    Examples
    --------
    >>> import networkx_backbone as nb
    >>> H = nb.to_hypernetx([(1, 2, 3), (3, 4)])  # doctest: +SKIP
    >>> sorted(H.incidence_dict)  # doctest: +SKIP
    [0, 1]
    """
    hnx = _require("hypernetx")
    edges, weights = _edges_and_weights(hyperedges, weights)
    setsystem = {i: set(e) for i, e in enumerate(edges)}
    if weights is None:
        return hnx.Hypergraph(setsystem)
    properties = {i: {"weight": w} for i, w in enumerate(weights)}
    return hnx.Hypergraph(setsystem, edge_properties=properties)


# ---------------------------------------------------------------------------
# HypergraphX
# ---------------------------------------------------------------------------


def from_hypergraphx(H, return_weights=False):
    """Convert a ``hypergraphx.Hypergraph`` to a list of :class:`frozenset` hyperedges.

    Parameters
    ----------
    H : hypergraphx.Hypergraph
        The HypergraphX hypergraph (read via ``H.get_edges()``).
    return_weights : bool, optional (default=False)
        If ``True``, also return each edge's weight (``H.get_weight``; 1 in an
        unweighted HypergraphX hypergraph).

    Returns
    -------
    hyperedges : list of frozenset
        One frozenset per HypergraphX hyperedge.
    weights : list of numbers
        Returned only when *return_weights* is ``True``.

    See Also
    --------
    to_hypergraphx : The inverse conversion.
    read_hif : Library-independent interchange via HIF.

    Examples
    --------
    >>> import hypergraphx as hgx  # doctest: +SKIP
    >>> import networkx_backbone as nb
    >>> nb.from_hypergraphx(hgx.Hypergraph(edge_list=[(1, 2), (2, 3, 4)]))  # doctest: +SKIP
    [frozenset({1, 2}), frozenset({2, 3, 4})]
    """
    raw = list(H.get_edges())
    edges = [frozenset(e) for e in raw]
    if not return_weights:
        return edges
    return _result(edges, [H.get_weight(e) for e in raw], True)


def to_hypergraphx(hyperedges, weights=None):
    """Convert hyperedges to a ``hypergraphx.Hypergraph``.

    HypergraphX has no multi-edges, so repeated hyperedges are merged into one
    edge whose weight is the sum of theirs (each counts 1 when *weights* is
    ``None``).  The result is weighted when *weights* is given or a hyperedge
    repeats, and unweighted otherwise.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Empty hyperedges are skipped.  HypergraphX sorts the
        nodes of each hyperedge, so node labels must be mutually comparable.
    weights : iterable of numbers or None, optional (default=None)
        Per-hyperedge weights (aligned with *hyperedges*).

    Returns
    -------
    H : hypergraphx.Hypergraph
        The equivalent HypergraphX hypergraph (built from an ``edge_list`` of
        tuples).

    Raises
    ------
    ImportError
        If ``hypergraphx`` is not installed.
    ValueError
        If *weights* does not match *hyperedges* in length.
    TypeError
        If the node labels of a hyperedge cannot be sorted (for example, a mix
        of ``int`` and ``str``).

    See Also
    --------
    from_hypergraphx : The inverse conversion.
    write_hif : Library-independent interchange via HIF.

    Examples
    --------
    >>> import networkx_backbone as nb
    >>> H = nb.to_hypergraphx([(1, 2, 3), (3, 4), (3, 4)])  # doctest: +SKIP
    >>> nb.from_hypergraphx(H, return_weights=True)  # doctest: +SKIP
    ([frozenset({1, 2, 3}), frozenset({3, 4})], [1, 2])
    """
    hgx = _require("hypergraphx")
    edges, weights = _edges_and_weights(hyperedges, weights)
    merged = {}
    for e, w in zip(edges, weights if weights is not None else [1] * len(edges)):
        merged[e] = merged.get(e, 0) + w
    edge_list = [tuple(e) for e in merged]
    weighted = weights is not None or len(merged) < len(edges)
    try:
        if weighted:
            return hgx.Hypergraph(
                edge_list=edge_list, weighted=True, weights=list(merged.values())
            )
        return hgx.Hypergraph(edge_list=edge_list, weighted=False)
    except TypeError as exc:
        raise TypeError(
            "hypergraphx sorts the nodes of each hyperedge, so node labels must "
            "be mutually comparable"
        ) from exc


# ---------------------------------------------------------------------------
# Hypergraph Analysis Toolbox (HAT)
# ---------------------------------------------------------------------------


def _hat_incidence_matrix(H):
    for attr in ("IM", "incidence_matrix", "incidenceMatrix"):
        value = getattr(H, attr, None)
        if value is None:
            continue
        value = value() if callable(value) else value
        if hasattr(value, "toarray"):  # scipy.sparse
            value = value.toarray()
        return value
    raise AttributeError("could not find an incidence matrix on the HAT hypergraph")


def _hat_column(H, table, column):
    """Return a column of HAT's ``nodes``/``edges`` table as a list, or None."""
    frame = getattr(H, table, None)
    if column not in getattr(frame, "columns", ()):
        return None
    return list(frame[column])


def from_hat(H, return_weights=False):
    """Convert a HAT ``Hypergraph`` to a list of :class:`frozenset` hyperedges.

    HAT stores nodes by position (row of the incidence matrix).  Node labels
    are read from the ``"Names"`` column of ``H.nodes`` when it exists (HAT
    fills it when built from labeled edges, and :func:`to_hat` sets it), else
    from its ``"Nodes"`` column (row indices, or the HIF node IDs for a
    hypergraph from ``HAT.Hypergraph.from_hif``), else the row indices.

    Parameters
    ----------
    H : HAT.Hypergraph
        The HAT hypergraph.  Its node-by-edge incidence matrix is read from the
        ``IM``, ``incidence_matrix``, or ``incidenceMatrix`` attribute (dense or
        scipy sparse; a method is called).
    return_weights : bool, optional (default=False)
        If ``True``, also return the ``"weight"`` column of ``H.edges``
        (default 1).

    Returns
    -------
    hyperedges : list of frozenset
        One frozenset per non-empty incidence-matrix column.
    weights : list of numbers
        Returned only when *return_weights* is ``True``.

    Raises
    ------
    ImportError
        If ``numpy`` is not installed.
    AttributeError
        If no incidence matrix can be found on *H*.

    See Also
    --------
    to_hat : The inverse conversion.
    read_hif : Library-independent interchange via HIF.

    Examples
    --------
    >>> import networkx_backbone as nb
    >>> H = nb.to_hat([("a", "b"), ("b", "c", "d")])  # doctest: +SKIP
    >>> nb.from_hat(H)  # doctest: +SKIP
    [frozenset({'a', 'b'}), frozenset({'b', 'c', 'd'})]
    """
    np = _require("numpy")
    matrix = np.asarray(_hat_incidence_matrix(H))
    names = _hat_column(H, "nodes", "Names")
    if names is None:
        # Row positions, or the HIF node IDs when read with Hypergraph.from_hif.
        names = _hat_column(H, "nodes", "Nodes")
    label = (lambda r: _to_builtin(names[r])) if names is not None else int
    all_weights = _hat_column(H, "edges", "weight")
    hyperedges, weights = [], []
    for col in range(matrix.shape[1]):
        members = frozenset(label(r) for r in np.flatnonzero(matrix[:, col]))
        if members:
            hyperedges.append(members)
            weights.append(1 if all_weights is None else all_weights[col])
    return _result(hyperedges, weights, return_weights)


def to_hat(hyperedges, weights=None):
    """Convert hyperedges to a HAT ``Hypergraph`` via an incidence matrix.

    HAT is positional: node ``i`` is row ``i`` of the incidence matrix.  Rows
    follow sorted label order (natural order when the labels are mutually
    comparable, otherwise by ``repr``), and the labels are kept in the
    ``"Names"`` column of ``H.nodes``, where :func:`from_hat` finds them.
    Repeated hyperedges are kept (HAT's own duplicate removal is turned off).

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Empty hyperedges are skipped.
    weights : iterable of numbers or None, optional (default=None)
        Per-hyperedge weights (aligned with *hyperedges*), stored in the
        ``"weight"`` column of ``H.edges`` (HAT's ``edge_weights``).

    Returns
    -------
    H : HAT.Hypergraph
        A HAT hypergraph built from the ``nodes x hyperedges`` 0/1 incidence
        matrix (``HAT.Hypergraph(incidence_matrix=..., compress=False)``).

    Raises
    ------
    ImportError
        If ``numpy`` or ``HAT`` (``pip install HypergraphAnalysisToolbox``) is
        not installed.
    ValueError
        If *weights* does not match *hyperedges* in length.

    See Also
    --------
    from_hat : The inverse conversion.
    write_hif : Library-independent interchange via HIF.

    Examples
    --------
    >>> import networkx_backbone as nb
    >>> H = nb.to_hat([("a", "b"), ("b", "c", "d")], weights=[2, 1])  # doctest: +SKIP
    >>> list(H.edge_weights)  # doctest: +SKIP
    [2, 1]
    """
    np = _require("numpy")
    hat = _require("HAT", pip_name="HypergraphAnalysisToolbox")

    edges, weights = _edges_and_weights(hyperedges, weights)
    nodes = _sorted_labels({v for e in edges for v in e})
    index = {v: i for i, v in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(edges)), dtype=int)
    for j, e in enumerate(edges):
        for v in e:
            matrix[index[v], j] = 1
    # compress=False keeps repeated hyperedges (HAT drops duplicate columns).
    H = hat.Hypergraph(incidence_matrix=matrix, compress=False)
    names = np.empty(len(nodes), dtype=object)
    for i, v in enumerate(nodes):  # element-wise, so tuple labels stay tuples
        names[i] = v
    H.nodes["Names"] = names
    if weights is not None:
        H.edges["weight"] = weights
    return H
