"""
Conversion from NumPy, SciPy sparse and pandas data to NetworkX graphs.

Backbone methods score the edges of a NetworkX graph and return a graph.  Data
held as a weighted adjacency matrix or as an edge-list ``DataFrame`` can be
turned into a graph with :func:`~networkx_backbone.to_graph`, which wraps
NetworkX's own converters
(:func:`~networkx.convert_matrix.from_numpy_array`,
:func:`~networkx.convert_matrix.from_scipy_sparse_array` and
:func:`~networkx.convert_matrix.from_pandas_edgelist`) and adds the checks a
backbone needs: an undirected graph must come from a symmetric matrix, and an
edge list must not list the same node pair twice.  The dispatch wrappers
:func:`~networkx_backbone.backbone`,
:func:`~networkx_backbone.backbone_from_weighted` and
:func:`~networkx_backbone.backbone_from_unweighted` call it for you.  The
individual scorers keep graph-only signatures, as NetworkX functions do:
convert once with :func:`~networkx_backbone.to_graph` and pass the graph to
as many scorers as you like.

For the way back, :func:`~networkx.convert_matrix.to_pandas_edgelist` returns
a backbone, or a scored graph, as a tidy ``DataFrame`` with ``source`` and
``target`` columns and one column per edge attribute (weights and scores)::

    >>> import networkx as nx
    >>> import networkx_backbone as nb
    >>> H = nb.disparity_filter(nx.les_miserables_graph())
    >>> df = nx.to_pandas_edgelist(H)  # doctest: +SKIP
    >>> list(df.columns)  # doctest: +SKIP
    ['source', 'target', 'weight', 'disparity_pvalue']

NumPy, SciPy and pandas are optional: they are only needed for the matching
input type and are never imported by :func:`~networkx_backbone.to_graph` for
graph input.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import networkx as nx

from networkx_backbone._docstrings import append_complexity_docstrings

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from typing import TypeGuard

    import pandas as pd

    from networkx_backbone._typing import GraphInput

__all__ = ["to_graph"]


def _is_dataframe(data: object) -> TypeGuard[pd.DataFrame]:
    """Return True for a pandas ``DataFrame`` without importing pandas."""
    pd = sys.modules.get("pandas")
    return pd is not None and isinstance(data, pd.DataFrame)


def _is_sparse(data):
    """Return True for a SciPy sparse array or matrix without importing SciPy."""
    sp = sys.modules.get("scipy.sparse")
    return sp is not None and sp.issparse(data)


def _check_nodelist(nodelist, n):
    nodelist = list(nodelist)
    if len(nodelist) != n:
        raise ValueError(
            f"nodelist has {len(nodelist)} entries but the adjacency matrix has {n}."
        )
    try:
        unique = len(set(nodelist)) == n
    except TypeError as err:
        raise ValueError("nodelist must contain hashable nodes.") from err
    if not unique:
        raise ValueError("nodelist must not contain duplicates.")
    return nodelist


def _from_adjacency(data, nodelist, directed, weight):
    """Graph from a square NumPy array (or array-like) or SciPy sparse matrix."""
    import numpy as np

    sparse = _is_sparse(data)
    if sparse:
        import scipy.sparse as sp

        A = sp.csr_array(data, copy=True)
        # Explicit zeros are not edges, as for a dense array.
        A.sum_duplicates()
        A.eliminate_zeros()
    else:
        A = np.asarray(data)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(
            f"An adjacency matrix must be square; got shape {tuple(A.shape)}."
        )
    if A.dtype.kind == "b":
        A = A.astype(np.int64)
    elif A.dtype.kind not in "iuf":
        raise ValueError(
            f"The adjacency matrix must be real-valued; got dtype {A.dtype}."
        )
    n = A.shape[0]
    if nodelist is not None:
        nodelist = _check_nodelist(nodelist, n)
    if not directed:
        asymmetric = (A != A.T).nnz if sparse else not np.array_equal(A, A.T)
        if asymmetric:
            raise ValueError(
                "The adjacency matrix is not symmetric. Pass directed=True to "
                "read row i, column j as an edge from node i to node j."
            )

    create_using = nx.DiGraph if directed else nx.Graph
    attr = "weight" if weight is None else weight
    if sparse:
        G = nx.from_scipy_sparse_array(
            A, create_using=create_using, edge_attribute=attr
        )
    else:
        G = nx.from_numpy_array(A, create_using=create_using, edge_attr=attr)
    if weight is None:
        for _, _, d in G.edges(data=True):
            d.clear()
    if nodelist is not None:
        G = nx.relabel_nodes(G, dict(enumerate(nodelist)), copy=True)
    return G


def _from_edgelist(df, source, target, directed, weight):
    """Graph from a pandas edge-list ``DataFrame``."""
    missing = [c for c in (source, target) if c not in df.columns]
    if missing:
        raise ValueError(
            f"The edge-list DataFrame has no column(s) {missing}; name the "
            "node columns with source= and target=. For an adjacency-matrix "
            "DataFrame use nx.from_pandas_adjacency(df), or pass "
            "df.to_numpy() with nodelist=df.index."
        )
    edge_attr = weight if weight is not None and weight in df.columns else None
    G = nx.from_pandas_edgelist(
        df,
        source=source,
        target=target,
        edge_attr=edge_attr,
        create_using=nx.DiGraph if directed else nx.Graph,
    )
    duplicates = len(df) - G.number_of_edges()
    if duplicates:
        pair = "(source, target) pair" if directed else "node pair"
        raise ValueError(
            f"The edge list has {duplicates} duplicate row(s) for the same "
            f"{pair}. Aggregate them first, e.g. with "
            "df.groupby([source, target], as_index=False).sum() (and, for an "
            "undirected graph, after ordering each pair), or build a "
            "MultiGraph with nx.from_pandas_edgelist and collapse it with "
            "multigraph_to_weighted."
        )
    return G


def to_graph(
    data: GraphInput,
    *,
    nodelist: Sequence[Hashable] | None = None,
    directed: bool = False,
    source: str = "source",
    target: str = "target",
    weight: str | None = "weight",
) -> nx.Graph:
    """Return a NetworkX graph for a graph, adjacency matrix or edge list.

    Uses NetworkX's own converters:
    :func:`~networkx.convert_matrix.from_numpy_array` for a NumPy array,
    :func:`~networkx.convert_matrix.from_scipy_sparse_array` for a SciPy
    sparse array or matrix, and
    :func:`~networkx.convert_matrix.from_pandas_edgelist` for a pandas
    ``DataFrame``.  Unlike those converters, it refuses input that would
    silently lose edges: an asymmetric matrix for an undirected graph, or an
    edge list that lists a node pair more than once.

    Parameters
    ----------
    data : graph, array-like, sparse array or DataFrame
        A NetworkX graph (returned unchanged, and the other arguments are
        ignored); a square adjacency matrix (NumPy array or array-like, or
        SciPy sparse array or matrix) whose nonzero entry ``A[i, j]`` is an
        edge of weight ``A[i, j]``; or a pandas edge-list ``DataFrame`` with
        one row per edge.
    nodelist : sequence or None, optional (default=None)
        Node labels for the rows and columns of an adjacency matrix (default
        ``0, 1, ..., n - 1``).  Isolated nodes are kept.
    directed : bool, optional (default=False)
        Build a :class:`~networkx.DiGraph`, with ``A[i, j]`` (or a row) the
        edge from ``i`` (``source``) to ``j`` (``target``).  If ``False`` a
        :class:`~networkx.Graph` is built and an adjacency matrix must be
        symmetric.
    source, target : str, optional (default="source", "target")
        Edge-list columns holding the two endpoints of each edge.
    weight : str or None, optional (default="weight")
        Edge attribute that stores the matrix entries, and edge-list column
        copied to that attribute (when present).  If ``None`` no edge
        attributes are stored.

    Returns
    -------
    G : graph
        A :class:`~networkx.Graph` or :class:`~networkx.DiGraph`, or *data*
        itself if it is already a graph.

    Raises
    ------
    TypeError
        If *data* is not a graph, an array, a sparse array or a ``DataFrame``.
    ValueError
        If an adjacency matrix is not square, not real-valued, or (with
        ``directed=False``) not symmetric; if *nodelist* does not match it;
        or if an edge list lacks the *source* / *target* columns or has
        duplicate node pairs.

    See Also
    --------
    networkx.convert.to_networkx_graph,
    networkx.convert_matrix.to_pandas_edgelist, backbone

    Examples
    --------
    A symmetric NumPy adjacency matrix with node labels:

    >>> import numpy as np
    >>> import networkx_backbone as nb
    >>> A = np.array([[0, 3, 1], [3, 0, 2], [1, 2, 0]])
    >>> G = nb.to_graph(A, nodelist=["a", "b", "c"])
    >>> sorted(G.edges(data="weight"))
    [('a', 'b', 3), ('a', 'c', 1), ('b', 'c', 2)]

    The same network as a pandas edge list with custom column names (pandas
    is optional, so this example is not run in the test suite):

    >>> import pandas as pd  # doctest: +SKIP
    >>> df = pd.DataFrame(
    ...     {"from": ["a", "a", "b"], "to": ["b", "c", "c"], "w": [3, 1, 2]}
    ... )  # doctest: +SKIP
    >>> G = nb.to_graph(df, source="from", target="to", weight="w")  # doctest: +SKIP
    >>> sorted(G.edges(data="w"))  # doctest: +SKIP
    [('a', 'b', 3), ('a', 'c', 1), ('b', 'c', 2)]

    A SciPy sparse adjacency matrix gives the same backbone as the graph it
    came from:

    >>> import networkx as nx
    >>> G = nx.les_miserables_graph()
    >>> nodes = list(G)
    >>> A = nx.to_scipy_sparse_array(G, nodelist=nodes)
    >>> H = nb.disparity_filter(nb.to_graph(A, nodelist=nodes))
    >>> dict(H.edges) == dict(nb.disparity_filter(G).edges)
    True
    """
    if isinstance(data, nx.Graph):
        return data
    if _is_dataframe(data):
        return _from_edgelist(data, source, target, directed, weight)
    if (
        _is_sparse(data)
        or hasattr(data, "__array__")
        or isinstance(data, (list, tuple))
    ):
        return _from_adjacency(data, nodelist, directed, weight)
    raise TypeError(
        "Expected a NetworkX graph, a square adjacency matrix (NumPy array or "
        "SciPy sparse array) or a pandas edge-list DataFrame; got "
        f"{type(data).__name__}."
    )


_COMPLEXITY = {
    "to_graph": {
        "time": "O(n + m)",
        "space": "O(n + m)",
        "notes": "Sparse input; a dense n x n array is O(n^2).",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
