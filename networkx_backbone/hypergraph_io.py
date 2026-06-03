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


def _as_hyperedges(hyperedges):
    """Normalise to a list of non-empty :class:`frozenset` hyperedges (order kept)."""
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
    except ImportError as exc:  # pragma: no cover - exercised when lib absent
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
    the other (``bipartite=1``, labelled ``f"{edge_prefix}{i}"`` with a
    ``"members"`` attribute).  The result is exactly the input expected by the
    bipartite projection backbones, so a hypergraph can be backboned with, e.g.,
    :func:`~networkx_backbone.sdsm` or :func:`~networkx_backbone.fdsm`.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.
    edge_prefix : str, optional (default="he")
        Prefix for the generated hyperedge-node labels.

    Returns
    -------
    B : networkx.Graph
        The incidence bipartite graph.
    nodes : list
        The hypergraph's nodes (the agent partition), suitable to pass as
        ``agent_nodes`` to the bipartite methods.

    Examples
    --------
    >>> import networkx_backbone as nb
    >>> B, nodes = nb.hypergraph_to_bipartite([(1, 2, 3), (2, 3, 4), (5, 6)])
    >>> import networkx as nx
    >>> nx.is_bipartite(B)
    True
    >>> scored = nb.sdsm(B, agent_nodes=nodes)
    """
    edges = _as_hyperedges(hyperedges)
    nodes = sorted({v for e in edges for v in e}, key=repr)

    B = nx.Graph()
    B.add_nodes_from(nodes, bipartite=0)
    for i, e in enumerate(edges):
        edge_node = f"{edge_prefix}{i}"
        B.add_node(edge_node, bipartite=1, members=e)
        for v in e:
            B.add_edge(v, edge_node)
    return B, nodes


# ---------------------------------------------------------------------------
# HIF (Hypergraph Interchange Format)
# ---------------------------------------------------------------------------


def write_hif(hyperedges, target=None, weights=None, network_type="undirected"):
    """Serialise a hypergraph to the HIF (Hypergraph Interchange Format) standard.

    HIF is a JSON format shared by xgi, HyperNetX, HypergraphX, and HAT, making it
    the most portable interchange path.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.
    target : str, path, file-like, or None, optional (default=None)
        Where to write.  If ``None``, the HIF dictionary is returned instead of
        being written.
    weights : iterable of numbers or None, optional (default=None)
        Optional per-hyperedge weights, written to the HIF ``edges`` list.
    network_type : str, optional (default="undirected")
        Value for the HIF ``network-type`` field.

    Returns
    -------
    result : dict or target
        The HIF dictionary if *target* is ``None``; otherwise *target*.
    """
    raw = list(hyperedges)
    if weights is not None:
        weights = list(weights)
        if len(weights) != len(raw):
            raise ValueError("weights must have the same length as hyperedges")

    incidences = []
    edge_records = []
    edge_id = 0
    for i, e in enumerate(raw):
        members = frozenset(e)
        if not members:
            continue
        for node in members:
            incidences.append({"edge": edge_id, "node": node})
        if weights is not None:
            edge_records.append({"edge": edge_id, "weight": weights[i]})
        edge_id += 1

    hif = {"network-type": network_type, "metadata": {}, "incidences": incidences}
    if edge_records:
        hif["edges"] = edge_records

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

    Parameters
    ----------
    source : dict, str, path, or file-like
        A parsed HIF dictionary, a path to a HIF JSON file, or an open file.
    return_weights : bool, optional (default=False)
        If ``True``, also return per-hyperedge weights (defaulting to 1 for edges
        without an explicit weight).

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

    groups = {}
    for incidence in hif.get("incidences", []):
        groups.setdefault(incidence["edge"], set()).add(incidence["node"])

    edge_ids = list(groups)
    hyperedges = [frozenset(groups[eid]) for eid in edge_ids]
    if not return_weights:
        return hyperedges

    weight_map = {rec["edge"]: rec.get("weight", 1) for rec in hif.get("edges", [])}
    weights = [weight_map.get(eid, 1) for eid in edge_ids]
    return hyperedges, weights


# ---------------------------------------------------------------------------
# xgi
# ---------------------------------------------------------------------------


def from_xgi(H):
    """Convert an :class:`xgi.Hypergraph` to a list of :class:`frozenset` hyperedges."""
    return [frozenset(members) for members in H.edges.members()]


def to_xgi(hyperedges):
    """Convert hyperedges to an :class:`xgi.Hypergraph` (requires ``xgi``)."""
    xgi = _require("xgi")
    return xgi.Hypergraph([set(e) for e in _as_hyperedges(hyperedges)])


# ---------------------------------------------------------------------------
# HyperNetX
# ---------------------------------------------------------------------------


def from_hypernetx(H):
    """Convert a :class:`hypernetx.Hypergraph` to a list of :class:`frozenset` hyperedges."""
    return [frozenset(members) for members in H.incidence_dict.values()]


def to_hypernetx(hyperedges):
    """Convert hyperedges to a :class:`hypernetx.Hypergraph` (requires ``hypernetx``)."""
    hnx = _require("hypernetx")
    return hnx.Hypergraph(
        {i: set(e) for i, e in enumerate(_as_hyperedges(hyperedges))}
    )


# ---------------------------------------------------------------------------
# HypergraphX
# ---------------------------------------------------------------------------


def from_hypergraphx(H):
    """Convert a ``hypergraphx.Hypergraph`` to a list of :class:`frozenset` hyperedges."""
    return [frozenset(e) for e in H.get_edges()]


def to_hypergraphx(hyperedges):
    """Convert hyperedges to a ``hypergraphx.Hypergraph`` (requires ``hypergraphx``)."""
    hgx = _require("hypergraphx")
    return hgx.Hypergraph(edge_list=[tuple(e) for e in _as_hyperedges(hyperedges)])


# ---------------------------------------------------------------------------
# Hypergraph Analysis Toolbox (HAT)
# ---------------------------------------------------------------------------


def _hat_incidence_matrix(H):
    for attr in ("IM", "incidence_matrix", "incidenceMatrix"):
        value = getattr(H, attr, None)
        if value is None:
            continue
        return value() if callable(value) else value
    raise AttributeError("could not find an incidence matrix on the HAT hypergraph")


def from_hat(H):
    """Convert a HAT ``Hypergraph`` to a list of :class:`frozenset` hyperedges.

    Node labels are positional (row indices of the incidence matrix), as HAT is
    incidence/tensor-based.
    """
    np = _require("numpy")
    matrix = np.asarray(_hat_incidence_matrix(H))
    hyperedges = []
    for col in range(matrix.shape[1]):
        members = frozenset(int(r) for r in np.flatnonzero(matrix[:, col]))
        if members:
            hyperedges.append(members)
    return hyperedges


def to_hat(hyperedges):
    """Convert hyperedges to a HAT ``Hypergraph`` via an incidence matrix.

    Node labels are replaced by contiguous integer indices (HAT is positional).
    Requires ``numpy`` and ``HAT``.
    """
    np = _require("numpy")
    hat = _require("HAT", pip_name="HypergraphAnalysisToolbox")

    edges = _as_hyperedges(hyperedges)
    nodes = sorted({v for e in edges for v in e}, key=repr)
    index = {v: i for i, v in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(edges)), dtype=int)
    for j, e in enumerate(edges):
        for v in e:
            matrix[index[v], j] = 1
    return hat.Hypergraph(incidence_matrix=matrix)
