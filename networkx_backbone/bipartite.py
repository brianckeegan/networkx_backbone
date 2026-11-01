"""
Bipartite projection backbone methods.

These methods operate on bipartite networks and extract significant edges
in the one-mode projection among "agent" nodes.

Every function accepts either an undirected bipartite NetworkX graph ``B``
with its agent nodes, or an agents x artifacts incidence matrix (a NumPy
array, a SciPy sparse array or matrix, or a pandas ``DataFrame``) with
optional ``agent_labels`` / ``artifact_labels``, as R ``backbone`` does.
Internally the incidence is kept as a SciPy CSR sparse array and agent
co-occurrences are computed with sparse products, one block of agent rows at
a time, so no dense agents x agents matrix is ever formed.

Methods
-------
simple_projection
    Shared-neighbor (co-occurrence count) projection.
hyper_projection
    Hyperbolic projection, ``1 / d(z)`` per shared artifact (or Newman's
    ``1 / (d(z) - 1)`` with ``newman=True``).
probs_projection
    ProbS (mass-diffusion) random-walk projection.
ycn_projection
    YCN stationary random-walk flow projection.
sdsm
    Stochastic Degree Sequence Model (Neal 2014).
fdsm
    Fixed Degree Sequence Model (Neal et al. 2021).
fixedfill, fixedrow, fixedcol
    Fixed fill / row / column null models (Neal et al. 2021).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast, overload

import networkx as nx

from networkx_backbone._docstrings import append_complexity_docstrings

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Sequence

    import numpy as np
    import numpy.typing as npt

    from networkx_backbone._typing import GraphInput, IdentityDecorator, Seed

__all__ = [
    # Bipartite projection methods
    "simple_projection",
    "hyper_projection",
    "probs_projection",
    "ycn_projection",
    "bipartite_projection",
    # Projection methods
    "sdsm",
    "fdsm",
    "fixedfill",
    "fixedrow",
    "fixedcol",
    # Utilities
    "bicm",
    "fastball",
    # High-level wrappers
    "backbone_from_projection",
    "backbone_from_weighted",
    "backbone_from_unweighted",
    "backbone",
]

# Agent-pair cells evaluated per block of agent rows (8 MB per float64 array);
# bounds the memory used for co-occurrence counts, weights and p-values.
_MAX_BLOCK_CELLS = 1 << 20

_PROJECTIONS = ("simple", "hyper", "newman", "probs", "ycn")
_PROJECTION_ALIASES = {"hyperbolic": "hyper", "collaboration": "newman"}
_PROJECTION_ERROR = (
    "Unknown projection method. Choose one of: "
    "'simple', 'hyper', 'newman', 'probs', 'ycn'."
)


# =====================================================================
# Input handling: graphs and incidence matrices
# =====================================================================


def _reject_directed_or_multigraph(B):
    """Raise for directed or multigraph *graph* input; matrices pass through.

    Plays the role of :func:`networkx.utils.not_implemented_for`, which
    cannot be used here because ``B`` may also be an incidence matrix.
    """
    if isinstance(B, nx.Graph):
        if B.is_directed():
            raise nx.NetworkXNotImplemented("not implemented for directed type")
        if B.is_multigraph():
            raise nx.NetworkXNotImplemented("not implemented for multigraph type")
    return B


# The cast keeps the decorated functions' signatures for type checkers
# (types-networkx types ``argmap`` as returning ``Callable[..., Any]``).
_graph_or_matrix = cast(
    "IdentityDecorator", nx.utils.argmap(_reject_directed_or_multigraph, 0)
)


def _validate_bipartite(B, agent_nodes):
    """Validate that ``agent_nodes`` is one side of the bipartite graph ``B``.

    Returns ``agent_nodes`` materialized as a list, so that generators are
    consumed exactly once.
    """
    agent_nodes = list(agent_nodes)
    if not nx.is_bipartite(B):
        raise nx.NetworkXError("Input graph B is not bipartite.")
    try:
        agent_set = set(agent_nodes)
    except TypeError as err:
        raise nx.NetworkXError("agent_nodes must contain hashable nodes.") from err
    if not agent_set.issubset(B):
        raise nx.NetworkXError("agent_nodes contains nodes not in B.")
    for u, v in B.edges():
        u_in = u in agent_set
        if u_in == (v in agent_set):
            where = "inside" if u_in else "outside"
            raise nx.NetworkXError(
                "agent_nodes is not one side of the bipartition: edge "
                f"({u!r}, {v!r}) has both endpoints {where} agent_nodes."
            )
    return agent_nodes


def _infer_agent_nodes(B):
    """Infer the agent partition of ``B``.

    Nodes with ``bipartite == 0`` are agents when that node attribute is set.
    Otherwise a two-coloring is used, which is only unique for connected
    graphs; disconnected graphs raise :exc:`networkx.AmbiguousSolution`.
    """
    attr_agents = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]
    if attr_agents:
        return sorted(attr_agents, key=str)
    if B.number_of_nodes() == 0:
        return []
    if not nx.is_bipartite(B):
        raise nx.NetworkXError("Input graph B is not bipartite.")
    if not nx.is_connected(B):
        raise nx.AmbiguousSolution(
            "B is disconnected, so its bipartition is ambiguous. Pass "
            "agent_nodes explicitly or set the 'bipartite' node attribute "
            "(0 for agents, 1 for artifacts)."
        )
    colors = nx.algorithms.bipartite.color(B)
    color0 = [n for n, c in colors.items() if c == 0]
    return sorted(color0, key=str)


class _Incidence(NamedTuple):
    """Agents, artifacts and their binary incidence as a sorted CSR array."""

    agents: list
    artifacts: list
    R: object  # scipy.sparse.csr_array of int64 0/1 values, sorted indices


def _check_labels(labels, n, name):
    if labels is None:
        return list(range(n))
    labels = list(labels)
    if len(labels) != n:
        raise ValueError(
            f"{name} has {len(labels)} entries but the incidence matrix has {n}."
        )
    try:
        unique = len(set(labels)) == n
    except TypeError as err:
        raise ValueError(f"{name} must be hashable.") from err
    if not unique:
        raise ValueError(f"{name} must not contain duplicates.")
    return labels


def _binary_csr(M):
    """Convert an array-like or sparse incidence matrix to a binary CSR array."""
    import numpy as np
    import scipy.sparse as sp

    if sp.issparse(M):
        if M.ndim != 2:
            raise ValueError("The incidence matrix B must be two-dimensional.")
        A = sp.csr_array(M, copy=True)
        A.sum_duplicates()
        A.eliminate_zeros()
        values = A.data
    else:
        A = np.asarray(M)
        if A.ndim != 2:
            raise ValueError("The incidence matrix B must be two-dimensional.")
        values = A
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError("The incidence matrix B must be numeric.") from err
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("The incidence matrix B must be binary (0/1).")
    if sp.issparse(A):
        ones = np.ones(A.nnz, dtype=np.int64)
        return sp.csr_array((ones, A.indices, A.indptr), shape=A.shape)
    return sp.csr_array(values.astype(np.int64))


def _incidence(B, agent_nodes=None, agent_labels=None, artifact_labels=None):
    """Return the :class:`_Incidence` of a bipartite graph or incidence matrix.

    Agents and artifacts are ordered by ``str`` of their label (as the
    graph path always has been), so a graph and the incidence matrix of the
    same graph give bit-identical results.
    """
    import numpy as np
    import scipy.sparse as sp

    if isinstance(B, nx.Graph):
        if agent_labels is not None or artifact_labels is not None:
            raise ValueError(
                "agent_labels and artifact_labels are only used when B is an "
                "incidence matrix; for a graph, pass agent_nodes."
            )
        if agent_nodes is None:
            agent_nodes = _infer_agent_nodes(B)
        agent_set = set(_validate_bipartite(B, agent_nodes))
        agents = sorted(agent_set, key=str)
        artifacts = sorted(set(B) - agent_set, key=str)
        a_idx = {v: i for i, v in enumerate(agents)}
        f_idx = {v: i for i, v in enumerate(artifacts)}
        rows, cols = [], []
        for u, v in B.edges():
            if u in a_idx:
                rows.append(a_idx[u])
                cols.append(f_idx[v])
            else:
                rows.append(a_idx[v])
                cols.append(f_idx[u])
        R = sp.csr_array(
            (
                np.ones(len(rows), dtype=np.int64),
                (np.asarray(rows, dtype=np.intp), np.asarray(cols, dtype=np.intp)),
            ),
            shape=(len(agents), len(artifacts)),
        )
    else:
        if agent_nodes is not None:
            raise ValueError(
                "agent_nodes selects one side of a bipartite graph; when B is "
                "an incidence matrix its rows are the agents. Use agent_labels "
                "to name them."
            )
        # pandas DataFrame: use its index and columns as default labels.
        if hasattr(B, "index") and hasattr(B, "columns"):
            if agent_labels is None:
                agent_labels = list(B.index)
            if artifact_labels is None:
                artifact_labels = list(B.columns)
        R = _binary_csr(B)
        agents = _check_labels(agent_labels, R.shape[0], "agent_labels")
        artifacts = _check_labels(artifact_labels, R.shape[1], "artifact_labels")
        row_order = sorted(range(len(agents)), key=lambda i: str(agents[i]))
        col_order = sorted(range(len(artifacts)), key=lambda i: str(artifacts[i]))
        R = R[np.asarray(row_order, dtype=np.intp)][
            :, np.asarray(col_order, dtype=np.intp)
        ]
        R = sp.csr_array(R)
        agents = [agents[i] for i in row_order]
        artifacts = [artifacts[i] for i in col_order]
    R.sum_duplicates()
    R.sort_indices()
    return _Incidence(agents, artifacts, R)


def _degrees(R: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(agent_degrees, artifact_degrees)`` of a sparse incidence."""
    import numpy as np

    row_deg = np.asarray(R.sum(axis=1)).ravel().astype(np.int64)
    col_deg = np.asarray(R.sum(axis=0)).ravel().astype(np.int64)
    return row_deg, col_deg


def _as_generator(seed):
    """Return a :class:`numpy.random.Generator` for int/Generator/RandomState/None."""
    import numpy as np

    if isinstance(seed, np.random.RandomState):
        return np.random.Generator(seed._bit_generator)
    return np.random.default_rng(seed)


# =====================================================================
# Agent-pair blocks, p-values and multiple-testing correction
# =====================================================================


def _row_blocks(n):
    """Yield ``(start, stop)`` agent-row ranges of at most ``_MAX_BLOCK_CELLS`` cells."""
    step = max(1, _MAX_BLOCK_CELLS // max(n, 1))
    for start in range(0, n, step):
        yield start, min(start + step, n)


def _upper_positions(start, stop, n):
    """Row-major ``(local_row, col)`` cells of a row block with ``col > row``.

    Over consecutive blocks this enumerates the pairs in the order of
    ``np.triu_indices(n, 1)``.
    """
    import numpy as np

    mask = np.arange(n)[None, :] > np.arange(start, stop)[:, None]
    return np.nonzero(mask)


def _signed_pvalues(p_hi, p_lo):
    """Vectorized two-sided p-value and sign from upper/lower tails.

    ``sign`` is ``+1`` for a significantly strong (over-represented)
    co-occurrence and ``-1`` for a significantly weak (under-represented) one.
    """
    import numpy as np

    p_hi = np.clip(np.asarray(p_hi, dtype=float), 0.0, 1.0)
    p_lo = np.clip(np.asarray(p_lo, dtype=float), 0.0, 1.0)
    pval = np.minimum(1.0, 2.0 * np.minimum(p_hi, p_lo))
    sign = np.where(p_hi <= p_lo, 1, -1)
    return pval, sign


def _adjust_array(p, method):
    """Vectorized :func:`~networkx_backbone.adjust_pvalues` for a float array.

    Bit-for-bit identical to the list version (same tie order and the same
    floating-point operations), but fast enough for millions of agent pairs.
    """
    import numpy as np

    from networkx_backbone.filters import _MTC_METHODS

    method_l = method.lower()
    if method_l not in _MTC_METHODS:
        raise ValueError(f"method must be one of {_MTC_METHODS}, got {method!r}")
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0 or method_l == "none":
        out = p.copy()
    elif method_l == "bonferroni":
        out = n * p
    elif method_l == "holm":
        order = np.argsort(p, kind="stable")
        vals = (n - np.arange(n)) * p[order]
        out = np.empty(n)
        out[order] = np.minimum(1.0, np.maximum.accumulate(np.maximum(vals, 0.0)))
    else:
        order = np.argsort(-p, kind="stable")
        j = np.arange(n)
        if method_l == "hochberg":
            vals = (j + 1) * p[order]
        elif method_l in ("bh", "fdr"):
            vals = (n / (n - j)) * p[order]
        else:  # Benjamini-Yekutieli; harmonic sum in the list version's order
            c = sum(1.0 / k for k in range(1, n + 1))
            vals = c * (n / (n - j)) * p[order]
        out = np.empty(n)
        out[order] = np.minimum(1.0, np.minimum.accumulate(vals))
    return np.clip(out, 0.0, 1.0)


# =====================================================================
# Poisson-binomial helpers
# =====================================================================


def _poisson_binomial_pmf(probs):
    """Exact Poisson-binomial pmf by sequential convolution.

    ``probs`` has shape ``(..., n)``; the result has shape ``(..., n + 1)``
    where ``pmf[..., k] = P(X = k)`` for ``X`` the sum of independent
    Bernoulli(``probs[..., t]``) variables.
    """
    import numpy as np

    q = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    n = q.shape[-1]
    pmf = np.zeros((*q.shape[:-1], n + 1), dtype=float)
    pmf[..., 0] = 1.0
    for t in range(n):
        p = q[..., t : t + 1]
        pmf[..., 1 : t + 2] = pmf[..., 1 : t + 2] * (1.0 - p) + pmf[..., 0 : t + 1] * p
        pmf[..., 0] *= 1.0 - p[..., 0]
    return pmf


def _pmf_tails(pmf, k):
    """Return ``(P(X >= k), P(X <= k))`` for 1-D ``pmf`` and integer array ``k``."""
    import numpy as np

    k = np.asarray(k, dtype=int)
    n = len(pmf) - 1
    sf = np.cumsum(pmf[::-1])[::-1]  # sf[j] = P(X >= j)
    cdf = np.cumsum(pmf)  # cdf[j] = P(X <= j)
    upper = np.where(k <= 0, 1.0, np.where(k > n, 0.0, sf[np.clip(k, 0, n)]))
    lower = np.where(k < 0, 0.0, np.where(k >= n, 1.0, cdf[np.clip(k, 0, n)]))
    return np.clip(upper, 0.0, 1.0), np.clip(lower, 0.0, 1.0)


def _poisson_binomial_sf(probs, k):
    """Exact Poisson-binomial upper tail ``P(X >= k)`` (via convolution)."""
    import numpy as np

    upper, _ = _pmf_tails(_poisson_binomial_pmf(np.ravel(probs)), k)
    return upper if np.ndim(k) else float(upper)


def _poisson_binomial_cdf(probs, k):
    """Exact Poisson-binomial lower tail ``P(X <= k)`` (via convolution)."""
    import numpy as np

    _, lower = _pmf_tails(_poisson_binomial_pmf(np.ravel(probs)), k)
    return lower if np.ndim(k) else float(lower)


def _rna_tails(mu, s2, s3, k):
    """Refined normal approximation (Hong 2013) to Poisson-binomial tails.

    Parameters are vectorized: ``mu = sum p``, ``s2 = sum p^2``,
    ``s3 = sum p^3`` for each test, and ``k`` the observed count.  Returns
    ``(P(X >= k), P(X <= k))`` where ``P(X <= k) ~ G((k + 0.5 - mu) / sigma)``
    with ``G(x) = Phi(x) + gamma (1 - x^2) phi(x) / 6``.
    """
    import numpy as np
    from scipy.stats import norm

    mu = np.asarray(mu, dtype=float)
    k = np.asarray(k, dtype=float)
    var = np.asarray(mu - s2, dtype=float)
    skew = np.asarray(mu - 3.0 * s2 + 2.0 * s3, dtype=float)  # sum p(1-p)(1-2p)
    degenerate = var <= 1e-12
    sigma = np.sqrt(np.where(degenerate, 1.0, var))
    gamma = np.where(degenerate, 0.0, skew / sigma**3)

    x_hi = (k - 0.5 - mu) / sigma  # P(X >= k) = 1 - P(X <= k - 1)
    upper = norm.sf(x_hi) - gamma * (1.0 - x_hi**2) * norm.pdf(x_hi) / 6.0
    x_lo = (k + 0.5 - mu) / sigma
    lower = norm.cdf(x_lo) + gamma * (1.0 - x_lo**2) * norm.pdf(x_lo) / 6.0

    # Degenerate case: X is (numerically) the constant round(mu).
    m0 = np.round(mu)
    upper = np.where(degenerate, (k <= m0).astype(float), upper)
    lower = np.where(degenerate, (k >= m0).astype(float), lower)
    upper = np.where(k <= 0, 1.0, upper)
    lower = np.where(k < 0, 0.0, lower)
    return np.clip(upper, 0.0, 1.0), np.clip(lower, 0.0, 1.0)


def _resolve_pvalue_method(method, n_pairs, n_artifacts):
    """Resolve ``pvalue_method="auto"`` to ``"exact"`` or ``"rna"``."""
    if method == "auto":
        nf = n_artifacts
        cheap = nf <= 200 and n_pairs * (nf + 1) * (nf + 1) <= 2e8
        return "exact" if cheap else "rna"
    if method not in ("exact", "rna"):
        raise ValueError("pvalue_method must be one of 'auto', 'exact', 'rna'.")
    return method


def _pairwise_pb_tails(P, k, method="auto", need_lower=False, pairs=None):
    """Poisson-binomial tails for agent pairs under edge probabilities ``P``.

    For agents ``i < j`` the co-occurrence count is a sum of independent
    Bernoulli(``P[i, t] * P[j, t]``) variables.  ``pairs`` is an
    ``(iu, ju)`` tuple of pair indices (default: every pair, in
    ``np.triu_indices`` order) aligned with the observed counts ``k``.
    """
    import numpy as np

    na, nf = P.shape
    iu, ju = np.triu_indices(na, 1) if pairs is None else pairs
    n_pairs = len(iu)
    method = _resolve_pvalue_method(method, n_pairs, nf)
    if method == "rna":
        # Products are taken one block of distinct rows at a time.
        rows, pos = np.unique(iu, return_inverse=True)
        mu = (P[rows] @ P.T)[pos, ju]
        P2 = P * P
        s2 = (P2[rows] @ P2.T)[pos, ju]
        P3 = P2 * P
        s3 = (P3[rows] @ P3.T)[pos, ju]
        upper, lower = _rna_tails(mu, s2, s3, k)
        return upper, (lower if need_lower else None)

    upper = np.empty(n_pairs, dtype=float)
    lower = np.empty(n_pairs, dtype=float) if need_lower else None
    k = np.asarray(k, dtype=int)
    chunk = max(1, int(2_000_000 // (nf + 1)))
    for start in range(0, n_pairs, chunk):
        stop = min(start + chunk, n_pairs)
        pmf = _poisson_binomial_pmf(P[iu[start:stop]] * P[ju[start:stop]])
        kk = np.clip(k[start:stop], 0, nf)
        rows = np.arange(stop - start)
        sf = np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1]
        upper[start:stop] = sf[rows, kk]
        if need_lower:
            lower[start:stop] = np.cumsum(pmf, axis=1)[rows, kk]
    upper = np.clip(upper, 0.0, 1.0)
    if need_lower:
        lower = np.clip(lower, 0.0, 1.0)
    return upper, lower


# =====================================================================
# Projection weightings
# =====================================================================


def _projection_name(method):
    method_l = method.lower()
    method_l = _PROJECTION_ALIASES.get(method_l, method_l)
    if method_l not in _PROJECTIONS:
        raise ValueError(_PROJECTION_ERROR)
    return method_l


class _ProjectionWeights:
    """Row blocks of a projection weight matrix, from a sparse incidence ``R``.

    With ``H = R diag(w) R^T`` (``w_z = 1 / d(z)``, or ``1 / (d(z) - 1)`` for
    Newman's variant) the weights are ``R R^T`` (simple), ``H`` (hyper /
    newman), ``H_uv / d(u)`` (ProbS) and ``pi_u H_uv / d(u)`` (YCN); the
    symmetric versions average the two directions.  Only one block of agent
    rows is densified at a time.
    """

    def __init__(self, R, method="simple", directed=False, max_iter=1000, tol=1e-12):
        import numpy as np
        import scipy.sparse as sp

        self.method = _projection_name(method)
        if directed and self.method not in ("probs", "ycn"):
            raise ValueError(f"{method} projection does not support directed=True")
        self.directed = directed
        self.R = R
        self.RT = sp.csr_array(R.T)
        row_deg, col_deg = _degrees(R)
        self.n = R.shape[0]
        if self.method != "simple":
            discount = col_deg - (1.0 if self.method == "newman" else 0.0)
            w = np.zeros(len(col_deg), dtype=float)
            np.divide(1.0, discount, out=w, where=discount > 0)
            self.Rw = sp.csr_array(R @ sp.diags_array(w))
        inv_deg = np.zeros(self.n, dtype=float)
        np.divide(1.0, row_deg, out=inv_deg, where=row_deg > 0)
        if self.method == "probs":
            self.scale = inv_deg
        elif self.method == "ycn":
            self.scale = self._stationary(row_deg, inv_deg, max_iter, tol) * inv_deg

    def _stationary(self, row_deg, inv_deg, max_iter, tol):
        """Stationary distribution of the two-step walk ``T = D^-1 H``.

        Power iteration from the uniform vector, as in Yildirim & Coscia
        (2014, Eq. 4).  Agents without artifacts keep their mass (``T_uu = 1``).
        """
        import numpy as np

        n = self.n
        if n == 0:
            return np.zeros(0)
        isolated = row_deg <= 0
        pi = np.full(n, 1.0 / n)
        for _ in range(max_iter):
            nxt = self.Rw @ (self.RT @ (pi * inv_deg))  # pi @ T, H symmetric
            nxt[isolated] += pi[isolated]
            if np.abs(nxt - pi).sum() <= tol:
                pi = nxt
                break
            pi = nxt
        total = pi.sum()
        if total > 0:
            pi = pi / total
        return pi

    def cooccurrence(self, start, stop):
        """Dense block ``(R R^T)[start:stop]`` of co-occurrence counts."""
        return (self.R[start:stop] @ self.RT).toarray()

    def block(self, start, stop, counts=None):
        """Dense weight block ``W[start:stop]`` with a zero diagonal."""
        import numpy as np

        if self.method == "simple":
            if counts is None:
                counts = self.cooccurrence(start, stop)
            W = counts.astype(float)
        else:
            W = (self.Rw[start:stop] @ self.RT).toarray()
            if self.method in ("probs", "ycn"):
                if self.directed:
                    W *= self.scale[start:stop, None]
                else:
                    W *= 0.5 * (self.scale[start:stop, None] + self.scale[None, :])
        rows = np.arange(stop - start)
        W[rows, rows + start] = 0.0
        return W


def _projection_graph(inc, method, weight="weight", directed=False, **kwargs):
    """Weighted projection graph on the agents of ``inc``."""
    import numpy as np

    weights = _ProjectionWeights(inc.R, method, directed=directed, **kwargs)
    agents = inc.agents
    n = len(agents)
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(agents)
    for start, stop in _row_blocks(n):
        W = weights.block(start, stop)
        keep = W > 0
        if not directed:
            keep &= np.arange(n)[None, :] > np.arange(start, stop)[:, None]
        r, c = np.nonzero(keep)
        G.add_edges_from(
            (agents[start + i], agents[j], {weight: w})
            for i, j, w in zip(r.tolist(), c.tolist(), W[r, c].tolist())
        )
    return G


@_graph_or_matrix
def simple_projection(
    B: GraphInput,
    agent_nodes: Iterable[Hashable] | None = None,
    weight: str = "weight",
    *,
    agent_labels: Sequence[Hashable] | None = None,
    artifact_labels: Sequence[Hashable] | None = None,
) -> nx.Graph:
    r"""Project a bipartite graph using shared-neighbor counts.

    The weight of agents ``u`` and ``v`` is the number of artifacts they
    share [1]_:

    .. math::

        w_{uv} = |\Gamma(u) \cap \Gamma(v)|

    This equals :func:`~networkx.algorithms.bipartite.projection.weighted_projected_graph`.

    Parameters
    ----------
    B : graph or array-like
        An undirected bipartite NetworkX graph, or an agents x artifacts
        binary incidence matrix (NumPy array, SciPy sparse array or matrix,
        or pandas ``DataFrame``).
    agent_nodes : iterable or None, optional (default=None)
        Nodes in the "agent" partition of a graph *B* (one full side of the
        bipartition).  If ``None`` they are inferred as in
        :func:`backbone_from_projection`.  Must be ``None`` for a matrix *B*.
    weight : string, optional (default="weight")
        Edge attribute name for the projection weight.
    agent_labels, artifact_labels : sequence or None, optional
        Row and column labels of a matrix *B* (default ``0, 1, ...``, or a
        ``DataFrame``'s index and columns).  Only valid with a matrix *B*.

    Returns
    -------
    projection : graph
        Undirected graph on the agent nodes; pairs with a positive weight are
        joined by an edge carrying the *weight* attribute.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.
    ValueError
        If a matrix *B* is not two-dimensional and binary, or its labels do
        not match its shape.

    See Also
    --------
    bipartite_projection, hyper_projection, probs_projection, ycn_projection

    References
    ----------
    .. [1] Coscia, M. & Rossi, L. (2019). The impact of projection and
       backboning on network topologies. *Proc. IEEE/ACM ASONAM 2019*.
       https://doi.org/10.1145/3341161.3342862 (arXiv:1906.09081).

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import simple_projection
    >>> B = nx.Graph([(1, "a"), (1, "b"), (2, "a"), (2, "c"), (3, "b"), (3, "c")])
    >>> P = simple_projection(B, agent_nodes=[1, 2, 3])
    >>> sorted(P.edges(data="weight"))
    [(1, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)]

    An agents x artifacts incidence matrix works directly:

    >>> import numpy as np
    >>> M = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    >>> P = simple_projection(M, agent_labels=[1, 2, 3])
    >>> sorted(P.edges(data="weight"))
    [(1, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)]
    """
    inc = _incidence(B, agent_nodes, agent_labels, artifact_labels)
    return _projection_graph(inc, "simple", weight=weight)


@_graph_or_matrix
def hyper_projection(
    B: GraphInput,
    agent_nodes: Iterable[Hashable] | None = None,
    weight: str = "weight",
    *,
    newman: bool = False,
    agent_labels: Sequence[Hashable] | None = None,
    artifact_labels: Sequence[Hashable] | None = None,
) -> nx.Graph:
    r"""Project a bipartite graph with hyperbolic (degree-discounted) weights.

    Each shared artifact ``z`` contributes the inverse of its degree
    ``d(z) = |\Gamma(z)|``, so sharing a small artifact counts for more than
    sharing a large one:

    .. math::

        w_{uv} = \sum_{z \in \Gamma(u) \cap \Gamma(v)} \frac{1}{d(z)}

    This is the "hyperbolic" projection as defined by Coscia & Rossi [1]_,
    who attribute the idea to Newman [2]_.  Newman's own collaboration
    weight [3]_ discounts by ``d(z) - 1`` instead, the number of *other*
    co-authors, so that each author's ties sum to the number of their
    multi-author papers:

    .. math::

        w_{uv} = \sum_{z \in \Gamma(u) \cap \Gamma(v)} \frac{1}{d(z) - 1}

    Pass ``newman=True`` for that variant; it equals
    :func:`~networkx.algorithms.bipartite.projection.collaboration_weighted_projected_graph`.  The
    two weightings coincide only up to a constant when every artifact has the
    same degree, so they can rank pairs differently.

    Parameters
    ----------
    B : graph or array-like
        An undirected bipartite NetworkX graph, or an agents x artifacts
        binary incidence matrix (NumPy array, SciPy sparse array or matrix,
        or pandas ``DataFrame``).
    agent_nodes : iterable or None, optional (default=None)
        Nodes in the "agent" partition of a graph *B*.  If ``None`` they are
        inferred as in :func:`backbone_from_projection`.  Must be ``None``
        for a matrix *B*.
    weight : string, optional (default="weight")
        Edge attribute name for the projection weight.
    newman : bool, optional (default=False)
        If ``True`` use Newman's ``1 / (d(z) - 1)`` discount.
    agent_labels, artifact_labels : sequence or None, optional
        Row and column labels of a matrix *B*.  Only valid with a matrix *B*.

    Returns
    -------
    projection : graph
        Undirected weighted graph on the agent nodes.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.
    ValueError
        If a matrix *B* is not two-dimensional and binary, or its labels do
        not match its shape.

    See Also
    --------
    bipartite_projection, simple_projection, ycn_projection

    References
    ----------
    .. [1] Coscia, M. & Rossi, L. (2019). The impact of projection and
       backboning on network topologies. *Proc. IEEE/ACM ASONAM 2019*.
       https://doi.org/10.1145/3341161.3342862 (arXiv:1906.09081).
    .. [2] Newman, M. E. J. (2001). Scientific collaboration networks. I.
       Network construction and fundamental results. *Physical Review E*,
       64, 016131.
    .. [3] Newman, M. E. J. (2001). Scientific collaboration networks. II.
       Shortest paths, weighted networks, and centrality. *Physical Review
       E*, 64, 016132.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import hyper_projection
    >>> B = nx.Graph([(1, "a"), (1, "b"), (2, "a"), (2, "c"), (3, "b"), (3, "c")])
    >>> P = hyper_projection(B, agent_nodes=[1, 2, 3])
    >>> sorted(P.edges(data="weight"))
    [(1, 2, 0.5), (1, 3, 0.5), (2, 3, 0.5)]
    >>> P = hyper_projection(B, agent_nodes=[1, 2, 3], newman=True)
    >>> sorted(P.edges(data="weight"))
    [(1, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)]
    """
    inc = _incidence(B, agent_nodes, agent_labels, artifact_labels)
    return _projection_graph(inc, "newman" if newman else "hyper", weight=weight)


@_graph_or_matrix
def probs_projection(
    B: GraphInput,
    agent_nodes: Iterable[Hashable] | None = None,
    weight: str = "weight",
    directed: bool = False,
    *,
    agent_labels: Sequence[Hashable] | None = None,
    artifact_labels: Sequence[Hashable] | None = None,
) -> nx.Graph:
    r"""Project a bipartite graph using ProbS random-walk weights.

    ProbS (probabilistic spreading, or mass diffusion) [1]_ [2]_ sends a unit
    of resource from agent ``u`` evenly to its artifacts, and from each
    artifact evenly to that artifact's agents.  The share of ``u``'s resource
    that reaches ``v`` is the directed weight

    .. math::

        w_{u \to v} = \frac{1}{d(u)}
            \sum_{z \in \Gamma(u) \cap \Gamma(v)} \frac{1}{d(z)},

    the probability that a two-step random walk from ``u`` ends at ``v``.
    Zhou et al. [2]_ write the same matrix with the indices swapped
    (``w_{vu}``).  As in Coscia & Rossi [1]_, the undirected projection
    averages the two directions, ``(w_{u \to v} + w_{v \to u}) / 2``.

    Parameters
    ----------
    B : graph or array-like
        An undirected bipartite NetworkX graph, or an agents x artifacts
        binary incidence matrix (NumPy array, SciPy sparse array or matrix,
        or pandas ``DataFrame``).
    agent_nodes : iterable or None, optional (default=None)
        Nodes in the "agent" partition of a graph *B*.  If ``None`` they are
        inferred as in :func:`backbone_from_projection`.  Must be ``None``
        for a matrix *B*.
    weight : string, optional (default="weight")
        Edge attribute name for the projection weight.
    directed : bool, optional (default=False)
        If ``True`` return the asymmetric scores as a :class:`~networkx.DiGraph`;
        otherwise the two directions are averaged.
    agent_labels, artifact_labels : sequence or None, optional
        Row and column labels of a matrix *B*.  Only valid with a matrix *B*.

    Returns
    -------
    projection : graph
        Weighted :class:`~networkx.Graph` (or :class:`~networkx.DiGraph`) on
        the agent nodes.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.
    ValueError
        If a matrix *B* is not two-dimensional and binary, or its labels do
        not match its shape.

    See Also
    --------
    bipartite_projection, ycn_projection

    References
    ----------
    .. [1] Coscia, M. & Rossi, L. (2019). The impact of projection and
       backboning on network topologies. *Proc. IEEE/ACM ASONAM 2019*.
       https://doi.org/10.1145/3341161.3342862 (arXiv:1906.09081).
    .. [2] Zhou, T., Ren, J., Medo, M., & Zhang, Y.-C. (2007). Bipartite
       network projection and personal recommendation. *Physical Review E*,
       76, 046115.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import probs_projection
    >>> B = nx.Graph([("a", "x"), ("a", "y"), ("b", "x"), ("c", "y")])
    >>> P = probs_projection(B, agent_nodes=["a", "b", "c"], directed=True)
    >>> P["a"]["b"]["weight"], P["b"]["a"]["weight"]
    (0.25, 0.5)
    """
    inc = _incidence(B, agent_nodes, agent_labels, artifact_labels)
    return _projection_graph(inc, "probs", weight=weight, directed=directed)


@_graph_or_matrix
def ycn_projection(
    B: GraphInput,
    agent_nodes: Iterable[Hashable] | None = None,
    weight: str = "weight",
    directed: bool = False,
    max_iter: int = 1000,
    tol: float = 1e-12,
    *,
    agent_labels: Sequence[Hashable] | None = None,
    artifact_labels: Sequence[Hashable] | None = None,
) -> nx.Graph:
    r"""Project a bipartite graph using YCN random-walk stationary flow.

    YCN (Yildirim & Coscia [1]_, who call it BPR) extends ProbS to random
    walks of unbounded length.  With the ProbS transition matrix
    ``T_{uv} = w_{u \to v}`` (including the walk's return to ``u``) and its
    stationary distribution ``pi``, reached by power iteration from the
    uniform vector, the directed weight is the stationary flow

    .. math::

        U_{uv} = \pi_u T_{uv}.

    The undirected projection averages ``U_{uv}`` and ``U_{vu}`` [2]_.  On a
    connected bipartite graph ``pi_u = d(u) / |E|``, so YCN equals the
    hyperbolic weight divided by the number of bipartite edges ``|E|`` and is
    already symmetric.  On a disconnected graph the walk keeps each
    component's initial mass, ``pi_u = (n_c / n) d(u) / |E_c|`` for a
    component with ``n_c`` of the ``n`` agents and ``|E_c|`` edges.  Agents
    without artifacts get no edges.

    Parameters
    ----------
    B : graph or array-like
        An undirected bipartite NetworkX graph, or an agents x artifacts
        binary incidence matrix (NumPy array, SciPy sparse array or matrix,
        or pandas ``DataFrame``).
    agent_nodes : iterable or None, optional (default=None)
        Nodes in the "agent" partition of a graph *B*.  If ``None`` they are
        inferred as in :func:`backbone_from_projection`.  Must be ``None``
        for a matrix *B*.
    weight : string, optional (default="weight")
        Edge attribute name for the projection weight.
    directed : bool, optional (default=False)
        If ``True`` return the asymmetric flows as a :class:`~networkx.DiGraph`;
        otherwise the two directions are averaged.
    max_iter : int, optional (default=1000)
        Maximum power iterations for the stationary distribution.
    tol : float, optional (default=1e-12)
        L1 convergence tolerance for the stationary distribution.
    agent_labels, artifact_labels : sequence or None, optional
        Row and column labels of a matrix *B*.  Only valid with a matrix *B*.

    Returns
    -------
    projection : graph
        Weighted :class:`~networkx.Graph` (or :class:`~networkx.DiGraph`) on
        the agent nodes.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.
    ValueError
        If a matrix *B* is not two-dimensional and binary, or its labels do
        not match its shape.

    See Also
    --------
    bipartite_projection, probs_projection, hyper_projection

    References
    ----------
    .. [1] Yildirim, M. A., & Coscia, M. (2014). Using random walks to
       generate associations between objects. *PLoS ONE*, 9(8), e104813.
    .. [2] Coscia, M. & Rossi, L. (2019). The impact of projection and
       backboning on network topologies. *Proc. IEEE/ACM ASONAM 2019*.
       https://doi.org/10.1145/3341161.3342862 (arXiv:1906.09081).

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import ycn_projection
    >>> B = nx.Graph([(1, "a"), (1, "b"), (2, "a"), (2, "c"), (3, "b"), (3, "c")])
    >>> P = ycn_projection(B, agent_nodes=[1, 2, 3])
    >>> round(P[1][2]["weight"], 4)  # hyperbolic weight 0.5 / 6 edges
    0.0833
    """
    inc = _incidence(B, agent_nodes, agent_labels, artifact_labels)
    return _projection_graph(
        inc, "ycn", weight=weight, directed=directed, max_iter=max_iter, tol=tol
    )


@_graph_or_matrix
def bipartite_projection(
    B: GraphInput,
    agent_nodes: Iterable[Hashable] | None = None,
    method: str = "simple",
    weight: str = "weight",
    directed: bool = False,
    max_iter: int = 1000,
    tol: float = 1e-12,
    *,
    agent_labels: Sequence[Hashable] | None = None,
    artifact_labels: Sequence[Hashable] | None = None,
) -> nx.Graph:
    """Dispatch weighted bipartite projections by method name.

    Parameters
    ----------
    B : graph or array-like
        An undirected bipartite NetworkX graph, or an agents x artifacts
        binary incidence matrix (NumPy array, SciPy sparse array or matrix,
        or pandas ``DataFrame``).
    agent_nodes : iterable or None, optional (default=None)
        Nodes in the "agent" partition of a graph *B*.  If ``None`` they are
        inferred as in :func:`backbone_from_projection`.  Must be ``None``
        for a matrix *B*.
    method : {"simple", "hyper", "newman", "probs", "ycn"}, optional
        Projection weighting (default ``"simple"``): see
        :func:`simple_projection`, :func:`hyper_projection` (``"hyper"``,
        alias ``"hyperbolic"``; ``"newman"``, alias ``"collaboration"``, is
        ``hyper_projection(..., newman=True)``), :func:`probs_projection`
        and :func:`ycn_projection`.
    weight : string, optional (default="weight")
        Edge attribute name for the projection weight.
    directed : bool, optional (default=False)
        Return directed scores; only supported for ``"probs"`` and ``"ycn"``.
    max_iter : int, optional (default=1000)
        Maximum iterations for ``method="ycn"``.
    tol : float, optional (default=1e-12)
        Convergence tolerance for ``method="ycn"``.
    agent_labels, artifact_labels : sequence or None, optional
        Row and column labels of a matrix *B*.  Only valid with a matrix *B*.

    Returns
    -------
    projection : graph
        Weighted projection on the agent nodes.

    Raises
    ------
    ValueError
        If *method* is unknown, ``directed=True`` is requested for a
        symmetric method, or a matrix *B* is invalid.
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.

    See Also
    --------
    simple_projection, hyper_projection, probs_projection, ycn_projection

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import bipartite_projection
    >>> B = nx.davis_southern_women_graph()
    >>> women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    >>> P = bipartite_projection(B, women, method="hyper")
    >>> P.number_of_nodes()
    18
    """
    method_l = _projection_name(method)
    if directed and method_l not in ("probs", "ycn"):
        raise ValueError(f"{method} projection does not support directed=True")
    inc = _incidence(B, agent_nodes, agent_labels, artifact_labels)
    return _projection_graph(
        inc, method_l, weight=weight, directed=directed, max_iter=max_iter, tol=tol
    )


# =====================================================================
# Scoring every agent pair
# =====================================================================


def _score_pairs(
    inc,
    tails,
    attr,
    *,
    signed=False,
    alpha=None,
    mtc="none",
    projection="simple",
    projection_weight="weight",
    projection_directed=False,
    projection_max_iter=1000,
    projection_tol=1e-12,
    missing_as_zero=True,
):
    """Score every agent pair and return the (thresholded) backbone graph.

    ``tails(i, j, k)`` returns the upper and lower tail probabilities of the
    observed co-occurrence counts ``k`` for agent index arrays ``i < j`` (the
    lower tail may be ``None`` when ``signed`` is false); ``tails=None``
    scores no pairs.  Pairs are visited in blocks of agent rows, in the
    order of ``np.triu_indices``.  With ``alpha`` only pairs whose p-value
    (after the correction *mtc*, which needs every p-value) is ``< alpha``
    become edges; each edge stores the unadjusted p-value, the ``"sign"``
    when *signed*, and the *projection* weight (0.0 for non-co-occurring
    pairs).  With ``missing_as_zero=False`` (only used with ``alpha``) pairs
    that never co-occur are not tests: they are left out of the correction
    *mtc* and never become edges.
    """
    import numpy as np

    if projection_directed:
        raise ValueError(
            "projection_directed=True is not supported for undirected backbones"
        )
    weights = _ProjectionWeights(
        inc.R, projection, max_iter=projection_max_iter, tol=projection_tol
    )
    use_mtc = mtc.lower() != "none"
    if use_mtc:
        _adjust_array([], mtc)  # validate the method name early
    agents = inc.agents
    n = len(agents)
    G = nx.Graph()
    G.add_nodes_from(agents)
    if n < 2 or tails is None:
        return G

    def add_edges(start, stop, r, c, pval, sign, counts=None):
        if not len(r):
            return
        w = weights.block(start, stop, counts)[r, c]
        rows = (start + r).tolist()
        cols = c.tolist()
        if signed:
            data = (
                {attr: p, "sign": s, projection_weight: x}
                for p, s, x in zip(pval.tolist(), sign.tolist(), w.tolist())
            )
        else:
            data = (
                {attr: p, projection_weight: x}
                for p, x in zip(pval.tolist(), w.tolist())
            )
        G.add_edges_from((agents[i], agents[j], d) for i, j, d in zip(rows, cols, data))

    def score_block(start, stop):
        r, c = _upper_positions(start, stop, n)
        counts = weights.cooccurrence(start, stop)
        p_hi, p_lo = tails(start + r, c, counts[r, c])
        if signed:
            pval, sign = _signed_pvalues(p_hi, p_lo)
        else:
            pval, sign = np.clip(np.asarray(p_hi, dtype=float), 0.0, 1.0), None
        return r, c, counts, pval, sign

    if not use_mtc:
        for start, stop in _row_blocks(n):
            r, c, counts, pval, sign = score_block(start, stop)
            if alpha is not None:
                keep = pval < alpha
                if not missing_as_zero:
                    keep &= counts[r, c] > 0
                r, c, pval = r[keep], c[keep], pval[keep]
                sign = sign[keep] if signed else None
            add_edges(start, stop, r, c, pval, sign, counts)
        return G

    # Multiple-testing correction needs all p-values before thresholding.
    blocks = list(_row_blocks(n))
    pvals, signs, tested = [], [], []
    for start, stop in blocks:
        r, c, counts, pval, sign = score_block(start, stop)
        pvals.append(pval)
        signs.append(sign)
        tested.append(
            np.ones(len(pval), dtype=bool) if missing_as_zero else counts[r, c] > 0
        )
    tested = np.concatenate(tested)
    keep_all = np.zeros(len(tested), dtype=bool)
    keep_all[tested] = _adjust_array(np.concatenate(pvals)[tested], mtc) < alpha
    offset = 0
    for (start, stop), pval, sign in zip(blocks, pvals, signs):
        keep = keep_all[offset : offset + len(pval)]
        offset += len(pval)
        if keep.any():
            r, c = _upper_positions(start, stop, n)
            add_edges(
                start,
                stop,
                r[keep],
                c[keep],
                pval[keep],
                sign[keep] if signed else None,
            )
    return G


# =====================================================================
# Bipartite Configuration Model
# =====================================================================


def _bicm_fit(R, required=None, prohibited=None, tol=1e-8, max_iter=10000):
    """Fit Bipartite Configuration Model probabilities to ``R``.

    Solves the maximum-entropy (Saracco et al. 2015) fixed point
    ``p_ik = x_i y_k / (1 + x_i y_k)`` with expected row and column sums equal
    to the observed ones.  Cells in the boolean masks ``required`` /
    ``prohibited`` are fixed at probability 1 / 0 and the model is fitted on
    the remaining (free) cells with correspondingly reduced margins
    (SDSM-EC, Neal & Neal 2023).  Rows or columns whose residual degree is 0
    or equal to their number of free cells are fixed at 0 / 1 outside the
    solve (iteratively, since fixing them changes the other margins).
    """
    import numpy as np

    R = (np.asarray(R) > 0).astype(int)
    n_rows, n_cols = R.shape
    P = np.zeros((n_rows, n_cols), dtype=float)
    free = np.ones((n_rows, n_cols), dtype=bool)
    if required is not None:
        P[required] = 1.0
        free &= ~required
    if prohibited is not None:
        free &= ~prohibited
    if n_rows == 0 or n_cols == 0:
        return P

    row_deg = R.sum(axis=1)
    col_deg = R.sum(axis=0)

    # Peel rows/columns whose free cells are all forced to 0 or 1.
    while True:
        fixed_mass = np.where(free, 0.0, P)
        r_res = row_deg - np.rint(fixed_mass.sum(axis=1)).astype(int)
        c_res = col_deg - np.rint(fixed_mass.sum(axis=0)).astype(int)
        r_free = free.sum(axis=1)
        c_free = free.sum(axis=0)
        if (
            (r_res < 0).any()
            or (r_res > r_free).any()
            or (c_res < 0).any()
            or (c_res > c_free).any()
        ):
            raise nx.NetworkXError(
                "Edge constraints are inconsistent with the observed degrees."
            )
        rows0 = (r_free > 0) & (r_res == 0)
        rows1 = (r_free > 0) & (r_res == r_free)
        cols0 = (c_free > 0) & (c_res == 0)
        cols1 = (c_free > 0) & (c_res == c_free)
        if not (rows0.any() or rows1.any() or cols0.any() or cols1.any()):
            break
        for mask_r, val in ((rows0, 0.0), (rows1, 1.0)):
            cells = free & mask_r[:, None]
            P[cells] = val
            free &= ~cells
        for mask_c, val in ((cols0, 0.0), (cols1, 1.0)):
            cells = free & mask_c[None, :]
            P[cells] = val
            free &= ~cells

    rows = np.flatnonzero(free.any(axis=1))
    cols = np.flatnonzero(free.any(axis=0))
    if rows.size == 0:
        return P

    r = r_res[rows].astype(float)
    c = c_res[cols].astype(float)
    F = free[np.ix_(rows, cols)]
    total = r.sum()

    if F.all():
        # No edge constraints: rows (columns) with equal degree share x (y),
        # so solve on the unique degrees weighted by multiplicity.
        ru, rinv, rn = np.unique(r, return_inverse=True, return_counts=True)
        cu, cinv, cn = np.unique(c, return_inverse=True, return_counts=True)
        x = ru / np.sqrt(total)
        y = cu / np.sqrt(total)
        err = np.inf
        for _ in range(max_iter):
            xy = x[:, None] * y[None, :]
            x = ru / ((cn * y)[None, :] / (1.0 + xy)).sum(axis=1)
            xy = x[:, None] * y[None, :]
            y = cu / ((rn * x)[:, None] / (1.0 + xy)).sum(axis=0)
            xy = x[:, None] * y[None, :]
            Pu = xy / (1.0 + xy)
            err = max(
                np.abs((Pu * cn).sum(axis=1) - ru).max(),
                np.abs((Pu * rn[:, None]).sum(axis=0) - cu).max(),
            )
            if err < tol:
                break
        sub = Pu[np.ix_(rinv.ravel(), cinv.ravel())]
    else:
        Ff = F.astype(float)
        x = r / np.sqrt(total)
        y = c / np.sqrt(total)
        err = np.inf
        for _ in range(max_iter):
            xy = x[:, None] * y[None, :]
            x = r / (Ff * y[None, :] / (1.0 + xy)).sum(axis=1)
            xy = x[:, None] * y[None, :]
            y = c / (Ff * x[:, None] / (1.0 + xy)).sum(axis=0)
            xy = x[:, None] * y[None, :]
            sub = Ff * xy / (1.0 + xy)
            err = max(
                np.abs(sub.sum(axis=1) - r).max(), np.abs(sub.sum(axis=0) - c).max()
            )
            if err < tol:
                break

    if not err < tol:
        warnings.warn(
            f"BiCM fit did not converge in {max_iter} iterations "
            f"(max margin error {err:.3g} > tol={tol:g}).",
            RuntimeWarning,
            stacklevel=3,
        )

    block = P[np.ix_(rows, cols)]
    block[F] = sub[F]
    P[np.ix_(rows, cols)] = block
    return P


def _bicm_classes(row_deg, col_deg, tol=1e-8, max_iter=10000):
    """Unconstrained BiCM probabilities grouped by degree.

    Without edge constraints ``P[i, k]`` depends only on the degrees of row
    ``i`` and column ``k``, so the fit is returned compactly as
    ``(Q, row_class, col_class)`` with ``P = Q[row_class][:, col_class]``,
    where the classes index the distinct row and column degrees.  This
    follows :func:`_bicm_fit` step by step (same peeling of empty and full
    rows/columns, same fixed-point iteration on the distinct residual
    degrees), so the expanded matrix is bit-identical to ``_bicm_fit(R)``,
    but needs only the degree sequences: memory is ``O(n_rows + n_cols)``
    instead of ``O(n_rows * n_cols)``.
    """
    import numpy as np

    row_deg = np.asarray(row_deg, dtype=np.int64)
    col_deg = np.asarray(col_deg, dtype=np.int64)
    n_rows, n_cols = len(row_deg), len(col_deg)
    _, r_rep, row_class = np.unique(row_deg, return_index=True, return_inverse=True)
    _, c_rep, col_class = np.unique(col_deg, return_index=True, return_inverse=True)
    Q = np.zeros((len(r_rep), len(c_rep)), dtype=float)
    if n_rows == 0 or n_cols == 0:
        return Q, row_class, col_class

    # Peel rows/columns whose free cells are all forced to 0 or 1.  Cells are
    # fixed a whole row or column at a time, so an unpeeled row's free cells
    # are exactly the unpeeled columns, and its fixed mass is the number of
    # columns peeled at 1 (and vice versa).
    never = np.iinfo(np.int64).max
    r_round = np.full(n_rows, never)
    c_round = np.full(n_cols, never)
    r_val = np.zeros(n_rows)
    c_val = np.zeros(n_cols)
    r_active = np.ones(n_rows, dtype=bool)
    c_active = np.ones(n_cols, dtype=bool)
    ones_r = ones_c = 0
    rnd = 0
    while True:
        r_free = int(c_active.sum())
        c_free = int(r_active.sum())
        r_res = row_deg - ones_c
        c_res = col_deg - ones_r
        rows0 = r_active & (r_free > 0) & (r_res == 0)
        rows1 = r_active & (r_free > 0) & (r_res == r_free)
        cols0 = c_active & (c_free > 0) & (c_res == 0)
        cols1 = c_active & (c_free > 0) & (c_res == c_free)
        if not (rows0.any() or rows1.any() or cols0.any() or cols1.any()):
            break
        r_val[rows1] = 1.0
        c_val[cols1] = 1.0
        r_round[rows0 | rows1] = rnd
        c_round[cols0 | cols1] = rnd
        r_active &= ~(rows0 | rows1)
        c_active &= ~(cols0 | cols1)
        ones_r += int(rows1.sum())
        ones_c += int(cols1.sum())
        rnd += 1

    rows = np.flatnonzero(r_active) if c_active.any() else np.zeros(0, dtype=int)
    cols = np.flatnonzero(c_active) if r_active.any() else np.zeros(0, dtype=int)
    Pu = None
    if rows.size:
        r = (row_deg - ones_c)[rows].astype(float)
        c = (col_deg - ones_r)[cols].astype(float)
        total = r.sum()
        ru, rinv, rn = np.unique(r, return_inverse=True, return_counts=True)
        cu, cinv, cn = np.unique(c, return_inverse=True, return_counts=True)
        x = ru / np.sqrt(total)
        y = cu / np.sqrt(total)
        err = np.inf
        for _ in range(max_iter):
            xy = x[:, None] * y[None, :]
            x = ru / ((cn * y)[None, :] / (1.0 + xy)).sum(axis=1)
            xy = x[:, None] * y[None, :]
            y = cu / ((rn * x)[:, None] / (1.0 + xy)).sum(axis=0)
            xy = x[:, None] * y[None, :]
            Pu = xy / (1.0 + xy)
            err = max(
                np.abs((Pu * cn).sum(axis=1) - ru).max(),
                np.abs((Pu * rn[:, None]).sum(axis=0) - cu).max(),
            )
            if err < tol:
                break
        if not err < tol:
            warnings.warn(
                f"BiCM fit did not converge in {max_iter} iterations "
                f"(max margin error {err:.3g} > tol={tol:g}).",
                RuntimeWarning,
                stacklevel=3,
            )
        r_pos = np.zeros(n_rows, dtype=np.intp)
        c_pos = np.zeros(n_cols, dtype=np.intp)
        r_pos[rows] = rinv.ravel()
        c_pos[cols] = cinv.ravel()

    # A cell takes the value of whichever of its row and column was peeled
    # first (rows first within a round), else the fitted probability.
    rr = r_round[r_rep][:, None]
    cr = c_round[c_rep][None, :]
    free = (rr == never) & (cr == never)
    Q = np.where(rr <= cr, r_val[r_rep][:, None], c_val[c_rep][None, :])
    if Pu is not None and free.any():
        Q[free] = Pu[np.ix_(r_pos[r_rep], c_pos[c_rep])][free]
    return Q, row_class, col_class


@overload
def bicm(
    B: GraphInput,
    agent_nodes: Iterable[Hashable] | None = ...,
    return_labels: Literal[False] = ...,
    *,
    tol: float = ...,
    max_iter: int = ...,
    agent_labels: Sequence[Hashable] | None = ...,
    artifact_labels: Sequence[Hashable] | None = ...,
) -> np.ndarray: ...  # numpydoc ignore=GL08
@overload
def bicm(
    B: GraphInput,
    agent_nodes: Iterable[Hashable] | None = ...,
    *,
    return_labels: Literal[True],
    tol: float = ...,
    max_iter: int = ...,
    agent_labels: Sequence[Hashable] | None = ...,
    artifact_labels: Sequence[Hashable] | None = ...,
) -> tuple[np.ndarray, list[Hashable], list[Hashable]]: ...  # numpydoc ignore=GL08
@overload
def bicm(
    B: GraphInput,
    agent_nodes: Iterable[Hashable] | None = ...,
    return_labels: bool = ...,
    *,
    tol: float = ...,
    max_iter: int = ...,
    agent_labels: Sequence[Hashable] | None = ...,
    artifact_labels: Sequence[Hashable] | None = ...,
) -> (
    np.ndarray | tuple[np.ndarray, list[Hashable], list[Hashable]]
): ...  # numpydoc ignore=GL08
@_graph_or_matrix
def bicm(
    B: GraphInput,
    agent_nodes: Iterable[Hashable] | None = None,
    return_labels: bool = False,
    *,
    tol: float = 1e-8,
    max_iter: int = 10000,
    agent_labels: Sequence[Hashable] | None = None,
    artifact_labels: Sequence[Hashable] | None = None,
) -> np.ndarray | tuple[np.ndarray, list[Hashable], list[Hashable]]:
    """Estimate Bipartite Configuration Model edge probabilities.

    Fits the maximum-entropy BiCM [1]_: ``P[i, k] = x_i y_k / (1 + x_i y_k)``
    where the fitness parameters solve ``sum_k P[i, k] = r_i`` and
    ``sum_i P[i, k] = c_k`` for the observed agent degrees ``r`` and artifact
    degrees ``c``.  The fixed point
    ``x_i = r_i / sum_k y_k / (1 + x_i y_k)``,
    ``y_k = c_k / sum_i x_i / (1 + x_i y_k)`` is iterated to convergence.
    Degree-0 rows/columns get probability 0 and full-degree rows/columns get
    probability 1 (they are fixed before solving).

    Parameters
    ----------
    B : graph or array-like
        An undirected bipartite NetworkX graph, or an agents x artifacts
        binary incidence matrix (NumPy array, SciPy sparse array or matrix,
        or pandas ``DataFrame``).
    agent_nodes : iterable or None, optional (default=None)
        Nodes in the "agent" partition of a graph *B*.  If ``None`` they are
        inferred as in :func:`backbone_from_projection`.  Must be ``None``
        for a matrix *B*.
    return_labels : bool, optional (default=False)
        If ``True``, also return ``(agents, artifacts)`` labels.
    tol : float, optional (default=1e-8)
        Convergence tolerance on the maximum absolute error between expected
        and observed degrees.
    max_iter : int, optional (default=10000)
        Maximum fixed-point iterations.  A :class:`RuntimeWarning` is emitted
        if the fit has not converged.
    agent_labels, artifact_labels : sequence or None, optional
        Row and column labels of a matrix *B*.  Only valid with a matrix *B*.

    Returns
    -------
    P : np.ndarray
        Probability matrix of shape ``(n_agents, n_artifacts)`` where
        ``P[i, k]`` is the estimated probability of an edge between
        agent ``i`` and artifact ``k``.  Rows and columns are ordered by
        ``str`` of their labels.
    (P, agents, artifacts) : tuple
        Returned when ``return_labels=True``.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.
    ValueError
        If a matrix *B* is not two-dimensional and binary, or its labels do
        not match its shape.

    Warns
    -----
    RuntimeWarning
        If the fixed point did not converge within *max_iter* iterations.

    See Also
    --------
    sdsm

    References
    ----------
    .. [1] Saracco, F., Di Clemente, R., Gabrielli, A., & Squartini, T. (2015).
       Randomizing bipartite networks: the case of the World Trade Web.
       *Scientific Reports*, 5, 10595.

    Examples
    --------
    >>> import networkx as nx
    >>> import numpy as np
    >>> from networkx_backbone import bicm
    >>> B = nx.davis_southern_women_graph()
    >>> women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    >>> P, agents, artifacts = bicm(B, women, return_labels=True)
    >>> degrees = np.array([B.degree(a) for a in agents])
    >>> bool(np.allclose(P.sum(axis=1), degrees, atol=1e-6))
    True
    """
    inc = _incidence(B, agent_nodes, agent_labels, artifact_labels)
    Q, row_class, col_class = _bicm_classes(
        *_degrees(inc.R), tol=tol, max_iter=max_iter
    )
    P = Q[row_class][:, col_class]
    if return_labels:
        return P, inc.agents, inc.artifacts
    return P


# =====================================================================
# Curveball / fastball sampling
# =====================================================================


def _curveball_trades(row_sets, n_swaps, rng):
    """Apply ``n_swaps`` curveball trades in place to a list of row sets."""
    n_rows = len(row_sets)
    if n_rows < 2 or n_swaps <= 0:
        return
    first = rng.integers(0, n_rows, size=n_swaps)
    second = rng.integers(0, n_rows - 1, size=n_swaps)
    second = second + (second >= first)
    for r1, r2 in zip(first.tolist(), second.tolist()):
        s1 = row_sets[r1]
        s2 = row_sets[r2]
        only1 = s1 - s2
        if not only1:
            continue
        only2 = s2 - s1
        if not only2:
            continue
        common = s1 & s2
        trade_pool = list(only1) + list(only2)
        rng.shuffle(trade_pool)
        keep1 = len(only1)
        row_sets[r1] = common | set(trade_pool[:keep1])
        row_sets[r2] = common | set(trade_pool[keep1:])


def _rows_to_matrix(row_sets, n_cols):
    import numpy as np

    out = np.zeros((len(row_sets), n_cols), dtype=int)
    for i, cols in enumerate(row_sets):
        if cols:
            out[i, list(cols)] = 1
    return out


def fastball(
    matrix: npt.ArrayLike, n_swaps: int | None = None, seed: Seed = None
) -> np.ndarray:
    """Randomize a binary matrix while preserving row and column sums.

    This is a Curveball-style randomizer commonly used for bipartite
    null-model generation.

    Parameters
    ----------
    matrix : array-like
        Binary matrix with shape ``(n_rows, n_cols)``.
    n_swaps : int or None, optional
        Number of row-pair trades. Defaults to ``5 * n_rows``.
    seed : integer, random_state, or None
        Random seed for reproducibility (int, :class:`numpy.random.Generator`,
        :class:`numpy.random.RandomState` or ``None``).

    Returns
    -------
    randomized : np.ndarray
        Randomized binary matrix with preserved row/column sums.

    Raises
    ------
    ValueError
        If *matrix* is not two-dimensional or *n_swaps* is negative.

    See Also
    --------
    fdsm

    References
    ----------
    .. [1] Godard, K., & Neal, Z. P. (2022). fastball: A fast algorithm to
       sample bipartite graphs with fixed degree sequences. *Journal of Complex
       Networks*, 10(6), cnac049.

    Examples
    --------
    >>> import numpy as np
    >>> from networkx_backbone import fastball
    >>> M = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    >>> S = fastball(M, seed=1)
    >>> bool((S.sum(axis=0) == M.sum(axis=0)).all() and (S.sum(axis=1) == M.sum(axis=1)).all())
    True
    """
    import numpy as np

    M = np.asarray(matrix)
    if M.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    R = (M > 0).astype(int)
    n_rows, n_cols = R.shape
    if n_rows == 0 or n_cols == 0:
        return R.copy()

    if n_swaps is None:
        n_swaps = max(1, n_rows * 5)
    if n_swaps < 0:
        raise ValueError("n_swaps must be >= 0")

    rng = _as_generator(seed)
    row_sets = [set(np.flatnonzero(R[i]).tolist()) for i in range(n_rows)]
    _curveball_trades(row_sets, n_swaps, rng)
    return _rows_to_matrix(row_sets, n_cols)


def _row_sets(R):
    """Row supports of a dense or sparse binary matrix as a list of sets."""
    import numpy as np
    import scipy.sparse as sp

    if sp.issparse(R):
        R = sp.csr_array(R)
        R.sort_indices()
        return [
            set(R.indices[R.indptr[i] : R.indptr[i + 1]].tolist())
            for i in range(R.shape[0])
        ]
    return [set(np.flatnonzero(row).tolist()) for row in np.asarray(R)]


def _row_sets_to_csr(row_sets, n_cols):
    import itertools

    import numpy as np
    import scipy.sparse as sp

    lengths = np.fromiter(
        (len(s) for s in row_sets), dtype=np.int64, count=len(row_sets)
    )
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    indices = np.fromiter(
        itertools.chain.from_iterable(row_sets), dtype=np.int64, count=int(indptr[-1])
    )
    data = np.ones(len(indices), dtype=np.int64)
    return sp.csr_array((data, indices, indptr), shape=(len(row_sets), n_cols))


def _fdsm_row_sets(R, trials, rng, burn_in=None, thin=None):
    """Yield ``trials`` FDSM samples (as row sets) via a curveball Markov chain.

    The chain starts at the observed matrix ``R`` (dense or sparse), runs
    ``burn_in`` trades, then yields a sample every ``thin`` trades.  Every
    sample has exactly the row and column sums of ``R``.  The same list of
    sets is yielded each time, updated in place.
    """
    n_rows = R.shape[0]
    if burn_in is None:
        burn_in = 10 * n_rows
    if thin is None:
        thin = 5 * n_rows
    if burn_in < 0 or thin < 0:
        raise ValueError("burn_in and thin must be >= 0")
    row_sets = _row_sets(R)
    _curveball_trades(row_sets, burn_in, rng)
    for _ in range(trials):
        _curveball_trades(row_sets, thin, rng)
        yield row_sets


def _fdsm_samples(R, trials, rng, burn_in=None, thin=None):
    """Dense-matrix version of :func:`_fdsm_row_sets`."""
    for row_sets in _fdsm_row_sets(R, trials, rng, burn_in=burn_in, thin=thin):
        yield _rows_to_matrix(row_sets, R.shape[1])


def _sorted_upper_keys(C, n):
    """``(keys, values)`` of the strict upper triangle of sparse ``C``, by key."""
    import numpy as np

    C = C.tocoo()
    up = C.row < C.col
    keys = C.row[up].astype(np.int64) * n + C.col[up]
    vals = C.data[up]
    order = np.argsort(keys, kind="stable")
    return keys[order], vals[order]


def _lookup(keys, query):
    """Positions of ``query`` in sorted ``keys`` and a found mask."""
    import numpy as np

    pos = np.searchsorted(keys, query)
    if len(keys) == 0:
        return pos, np.zeros(len(query), dtype=bool)
    pos = np.minimum(pos, len(keys) - 1)
    return pos, keys[pos] == query


# =====================================================================
# Fixed fill / row / column models -- Neal et al. (2021)
# =====================================================================


def _fixedfill_pmf(n_agents, n_artifacts, n_filled):
    """Exact co-occurrence pmf for one agent pair under the fixed-fill model.

    The null is uniform over all ``n_agents x n_artifacts`` binary matrices
    with ``n_filled`` ones.  If the two agents' rows hold ``s`` ones (a
    hypergeometric count), the number of placements with exactly ``x`` shared
    columns is ``C(n, x) C(n - x, s - 2x) 2^(s - 2x)``, so

    ``P(X = x) = sum_s C(n, x) C(n - x, s - 2x) 2^(s - 2x) C(N - 2n, F - s) / C(N, F)``

    with ``n`` artifacts, ``N = n_agents * n`` cells and ``F`` filled cells.
    """
    import numpy as np
    from scipy.special import gammaln, logsumexp

    n = int(n_artifacts)
    N = int(n_agents) * n
    F = int(n_filled)
    K = 2 * n

    def lchoose(a, b):
        return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)

    pmf = np.zeros(n + 1, dtype=float)
    log_total = lchoose(N, F)
    for x in range(n + 1):
        lo = max(2 * x, F - (N - K))
        hi = min(K, F, n + x)
        if lo > hi:
            continue
        s = np.arange(lo, hi + 1, dtype=float)
        log_terms = (
            lchoose(n, x)
            + lchoose(n - x, s - 2 * x)
            + (s - 2 * x) * np.log(2.0)
            + lchoose(N - K, F - s)
            - log_total
        )
        pmf[x] = np.exp(logsumexp(log_terms))
    total = pmf.sum()
    if total > 0:
        pmf /= total
    return pmf


def _per_distinct(func, k, a, b, base):
    """Evaluate ``func(k, a, b)`` once per distinct ``(k, a, b)`` triple.

    ``k``, ``a`` and ``b`` are non-negative integer arrays with ``a, b <
    base``.  The results (a tuple of arrays or ``None``) are broadcast back to
    every pair; element-wise functions give exactly the values they would
    give on the full arrays.
    """
    import numpy as np

    k = np.asarray(k, dtype=np.int64)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    keys = (k * base + a) * base + b
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    out = func(k[first], a[first], b[first])
    return tuple(None if x is None else np.asarray(x)[inverse] for x in out)


def _fixedfill_tails(inc, *, signed=False):
    """Pair tails under the fixed-fill model (identical for every pair)."""
    na, nf = inc.R.shape
    if na < 2 or nf == 0:
        return None
    pmf = _fixedfill_pmf(na, nf, int(inc.R.sum()))
    return lambda i, j, k: _pmf_tails(pmf, k)


def _fixedrow_tails(inc, *, signed=False):
    """Pair tails under the fixed-row model: hypergeometric in the degrees."""
    from scipy import stats as sp_stats

    na, nf = inc.R.shape
    if na < 2 or nf == 0:
        return None
    deg, _ = _degrees(inc.R)

    def hypergeom_tails(k, di, dj):
        p_hi = sp_stats.hypergeom.sf(k - 1, nf, di, dj)
        p_lo = sp_stats.hypergeom.cdf(k, nf, di, dj) if signed else None
        return p_hi, p_lo

    # The tails depend only on (k, d_i, d_j).
    return lambda i, j, k: _per_distinct(hypergeom_tails, k, deg[i], deg[j], nf + 1)


def _fixedcol_tails(inc, *, signed=False):
    """Pair tails under the fixed-column model (identical for every pair)."""
    import numpy as np

    na, nf = inc.R.shape
    if na < 2 or nf == 0:
        return None
    _, col_deg = _degrees(inc.R)
    col_sums = col_deg.astype(float)
    pair_prob = np.clip(col_sums * (col_sums - 1) / (na * (na - 1)), 0.0, 1.0)
    pmf = _poisson_binomial_pmf(pair_prob)
    return lambda i, j, k: _pmf_tails(pmf, k)


@_graph_or_matrix
def fixedfill(
    B: GraphInput,
    agent_nodes: Iterable[Hashable] | None = None,
    alpha: float = 0.05,
    signed: bool = False,
    projection: str = "simple",
    projection_weight: str = "weight",
    projection_directed: bool = False,
    projection_max_iter: int = 1000,
    projection_tol: float = 1e-12,
    *,
    agent_labels: Sequence[Hashable] | None = None,
    artifact_labels: Sequence[Hashable] | None = None,
) -> nx.Graph:
    """Backbone from the fixed-fill null model for bipartite projections.

    Under the fixed fill model (FFM) [1]_ every bipartite matrix with the same
    number of agents, artifacts and edges (filled cells) is equally likely.
    The co-occurrence count of every agent pair then has the same exact
    distribution, obtained by conditioning on how many of the filled cells
    fall in the pair's two rows; the p-value is its upper tail at the
    observed count (and the lower tail too when ``signed=True``).

    Parameters
    ----------
    B : graph or array-like
        An undirected bipartite NetworkX graph, or an agents x artifacts
        binary incidence matrix (NumPy array, SciPy sparse array or matrix,
        or pandas ``DataFrame``).
    agent_nodes : iterable or None, optional (default=None)
        Nodes in the "agent" partition of a graph *B*.  If ``None`` they are
        inferred as in :func:`backbone_from_projection`.  Must be ``None``
        for a matrix *B*.
    alpha : float, optional (default=0.05)
        Significance level; only pairs with p-value ``< alpha`` are kept.
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test: the p-value is two-sided and a
        ``"sign"`` attribute marks significantly strong (``+1``) or weak
        (``-1``) co-occurrence.
    projection : {"simple", "hyper", "newman", "probs", "ycn"}, optional
        Projection weighting assigned to each returned edge (see
        :func:`bipartite_projection`).
    projection_weight : str, optional (default="weight")
        Edge attribute name for the projection weight.
    projection_directed : bool, optional (default=False)
        Not supported for the undirected backbone (raises if ``True``).
    projection_max_iter : int, optional (default=1000)
        Maximum iterations for ``projection="ycn"``.
    projection_tol : float, optional (default=1e-12)
        Convergence tolerance for ``projection="ycn"``.
    agent_labels, artifact_labels : sequence or None, optional
        Row and column labels of a matrix *B*.  Only valid with a matrix *B*.

    Returns
    -------
    backbone : graph
        Graph on the agent nodes whose edges are the significant pairs, each
        with a ``"fixedfill_pvalue"`` attribute and the projection weight.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.
    ValueError
        If *projection* is unknown, ``projection_directed=True``, or a matrix
        *B* is invalid.

    See Also
    --------
    fixedrow, fixedcol, sdsm, fdsm, backbone_from_projection

    References
    ----------
    .. [1] Neal, Z. P., Domagalski, R., & Sagan, B. (2021). Comparing
       alternatives to the fixed degree sequence model. *Scientific
       Reports*, 11, 23929.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fixedfill
    >>> B = nx.davis_southern_women_graph()
    >>> women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    >>> H = fixedfill(B, women, alpha=0.05)
    >>> all(d["fixedfill_pvalue"] < 0.05 for _, _, d in H.edges(data=True))
    True
    """
    inc = _incidence(B, agent_nodes, agent_labels, artifact_labels)
    return _score_pairs(
        inc,
        _fixedfill_tails(inc, signed=signed),
        "fixedfill_pvalue",
        signed=signed,
        alpha=alpha,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
    )


@_graph_or_matrix
def fixedrow(
    B: GraphInput,
    agent_nodes: Iterable[Hashable] | None = None,
    alpha: float = 0.05,
    signed: bool = False,
    projection: str = "simple",
    projection_weight: str = "weight",
    projection_directed: bool = False,
    projection_max_iter: int = 1000,
    projection_tol: float = 1e-12,
    *,
    agent_labels: Sequence[Hashable] | None = None,
    artifact_labels: Sequence[Hashable] | None = None,
) -> nx.Graph:
    """Backbone from the fixed-row null model for bipartite projections.

    Under the fixed row model (FRM) [1]_ each agent keeps its degree ``d_i``
    and its artifacts are a uniformly random ``d_i``-subset of all
    artifacts, so the co-occurrence of agents ``i`` and ``j`` is exactly
    hypergeometric with parameters ``(n_artifacts, d_i, d_j)``.

    Parameters
    ----------
    B : graph or array-like
        An undirected bipartite NetworkX graph, or an agents x artifacts
        binary incidence matrix (NumPy array, SciPy sparse array or matrix,
        or pandas ``DataFrame``).
    agent_nodes : iterable or None, optional (default=None)
        Nodes in the "agent" partition of a graph *B*.  If ``None`` they are
        inferred as in :func:`backbone_from_projection`.  Must be ``None``
        for a matrix *B*.
    alpha : float, optional (default=0.05)
        Significance level; only pairs with p-value ``< alpha`` are kept.
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test and add a ``"sign"`` attribute.
    projection : {"simple", "hyper", "newman", "probs", "ycn"}, optional
        Projection weighting assigned to each returned edge (see
        :func:`bipartite_projection`).
    projection_weight : str, optional (default="weight")
        Edge attribute name for the projection weight.
    projection_directed : bool, optional (default=False)
        Not supported for the undirected backbone (raises if ``True``).
    projection_max_iter : int, optional (default=1000)
        Maximum iterations for ``projection="ycn"``.
    projection_tol : float, optional (default=1e-12)
        Convergence tolerance for ``projection="ycn"``.
    agent_labels, artifact_labels : sequence or None, optional
        Row and column labels of a matrix *B*.  Only valid with a matrix *B*.

    Returns
    -------
    backbone : graph
        Graph on the agent nodes whose edges are the significant pairs, each
        with a ``"fixedrow_pvalue"`` attribute and the projection weight.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.
    ValueError
        If *projection* is unknown, ``projection_directed=True``, or a matrix
        *B* is invalid.

    See Also
    --------
    fixedfill, fixedcol, sdsm, fdsm, backbone_from_projection

    References
    ----------
    .. [1] Neal, Z. P., Domagalski, R., & Sagan, B. (2021). Comparing
       alternatives to the fixed degree sequence model. *Scientific
       Reports*, 11, 23929.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fixedrow
    >>> B = nx.davis_southern_women_graph()
    >>> women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    >>> H = fixedrow(B, women, alpha=0.05)
    >>> all(d["fixedrow_pvalue"] < 0.05 for _, _, d in H.edges(data=True))
    True
    """
    inc = _incidence(B, agent_nodes, agent_labels, artifact_labels)
    return _score_pairs(
        inc,
        _fixedrow_tails(inc, signed=signed),
        "fixedrow_pvalue",
        signed=signed,
        alpha=alpha,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
    )


@_graph_or_matrix
def fixedcol(
    B: GraphInput,
    agent_nodes: Iterable[Hashable] | None = None,
    alpha: float = 0.05,
    signed: bool = False,
    projection: str = "simple",
    projection_weight: str = "weight",
    projection_directed: bool = False,
    projection_max_iter: int = 1000,
    projection_tol: float = 1e-12,
    *,
    agent_labels: Sequence[Hashable] | None = None,
    artifact_labels: Sequence[Hashable] | None = None,
) -> nx.Graph:
    """Backbone from the fixed-column null model for bipartite projections.

    Under the fixed column model (FCM) [1]_ each artifact ``k`` keeps its
    degree ``c_k`` and its agents are a uniformly random ``c_k``-subset of the
    ``m`` agents, independently across artifacts.  A given pair co-occurs on
    artifact ``k`` with probability ``c_k (c_k - 1) / (m (m - 1))``, so the
    co-occurrence count is a Poisson-binomial variable with the same
    distribution for every pair; it is computed exactly once and its tails
    give the p-values.

    Parameters
    ----------
    B : graph or array-like
        An undirected bipartite NetworkX graph, or an agents x artifacts
        binary incidence matrix (NumPy array, SciPy sparse array or matrix,
        or pandas ``DataFrame``).
    agent_nodes : iterable or None, optional (default=None)
        Nodes in the "agent" partition of a graph *B*.  If ``None`` they are
        inferred as in :func:`backbone_from_projection`.  Must be ``None``
        for a matrix *B*.
    alpha : float, optional (default=0.05)
        Significance level; only pairs with p-value ``< alpha`` are kept.
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test and add a ``"sign"`` attribute.
    projection : {"simple", "hyper", "newman", "probs", "ycn"}, optional
        Projection weighting assigned to each returned edge (see
        :func:`bipartite_projection`).
    projection_weight : str, optional (default="weight")
        Edge attribute name for the projection weight.
    projection_directed : bool, optional (default=False)
        Not supported for the undirected backbone (raises if ``True``).
    projection_max_iter : int, optional (default=1000)
        Maximum iterations for ``projection="ycn"``.
    projection_tol : float, optional (default=1e-12)
        Convergence tolerance for ``projection="ycn"``.
    agent_labels, artifact_labels : sequence or None, optional
        Row and column labels of a matrix *B*.  Only valid with a matrix *B*.

    Returns
    -------
    backbone : graph
        Graph on the agent nodes whose edges are the significant pairs, each
        with a ``"fixedcol_pvalue"`` attribute and the projection weight.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.
    ValueError
        If *projection* is unknown, ``projection_directed=True``, or a matrix
        *B* is invalid.

    See Also
    --------
    fixedfill, fixedrow, sdsm, fdsm, backbone_from_projection

    Notes
    -----
    The tail is the exact Poisson-binomial one; R ``backbone::fixedcol`` uses the
    refined normal approximation, which differs by up to 0.008 on Davis
    Southern Women.  ``fixedfill`` and ``fixedrow`` equal R to rounding error.

    References
    ----------
    .. [1] Neal, Z. P., Domagalski, R., & Sagan, B. (2021). Comparing
       alternatives to the fixed degree sequence model. *Scientific
       Reports*, 11, 23929.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fixedcol
    >>> B = nx.davis_southern_women_graph()
    >>> women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    >>> H = fixedcol(B, women, alpha=0.05)
    >>> all(d["fixedcol_pvalue"] < 0.05 for _, _, d in H.edges(data=True))
    True
    """
    inc = _incidence(B, agent_nodes, agent_labels, artifact_labels)
    return _score_pairs(
        inc,
        _fixedcol_tails(inc, signed=signed),
        "fixedcol_pvalue",
        signed=signed,
        alpha=alpha,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
    )


# =====================================================================
# High-level wrappers
# =====================================================================


@_graph_or_matrix
def backbone_from_projection(
    B: GraphInput,
    agent_nodes: Iterable[Hashable] | None = None,
    method: str = "sdsm",
    alpha: float = 0.05,
    projection: str = "simple",
    projection_weight: str = "weight",
    projection_directed: bool = False,
    projection_max_iter: int = 1000,
    projection_tol: float = 1e-12,
    mtc: str = "none",
    *,
    missing_as_zero: bool = True,
    agent_labels: Sequence[Hashable] | None = None,
    artifact_labels: Sequence[Hashable] | None = None,
    **kwargs: Any,
) -> nx.Graph:
    """Extract a bipartite projection backbone using method-name dispatch.

    Mirrors ``backbone::backbone_from_projection`` in R: the selected null
    model scores every agent pair, the optional multiple-testing correction
    *mtc* is applied across all tested pairs (every pair unless
    ``missing_as_zero=False``), and the pairs with (adjusted) p-value
    ``< alpha`` are returned.  The result is thresholded the same way for
    every *method*.  Pairs are scored in blocks of agent rows and only the
    significant ones become edges, so memory stays bounded for large
    projections (with *mtc* the p-values of all pairs are kept, 8 bytes
    each).

    Parameters
    ----------
    B : graph or array-like
        An undirected bipartite NetworkX graph, or an agents x artifacts
        binary incidence matrix (NumPy array, SciPy sparse array or matrix,
        or pandas ``DataFrame``), as accepted by R ``backbone``.
    agent_nodes : iterable or None, optional (default=None)
        Nodes in the "agent" partition of a graph *B*.  If ``None`` they are
        taken from the ``bipartite`` node attribute (value 0) when present,
        or from a two-coloring of a connected *B*.  Must be ``None`` for a
        matrix *B*, whose rows are the agents.
    method : {"sdsm", "fdsm", "fixedfill", "fixedrow", "fixedcol"}, optional
        Null model (default ``"sdsm"``).
    alpha : float, optional (default=0.05)
        Significance level for the returned backbone.
    projection : {"simple", "hyper", "newman", "probs", "ycn"}, optional
        Projection weighting assigned to each returned edge (see
        :func:`bipartite_projection`).
    projection_weight : str, optional (default="weight")
        Edge attribute name for the projection weight.
    projection_directed : bool, optional (default=False)
        Not supported for the undirected backbone (raises if ``True``).
    projection_max_iter : int, optional (default=1000)
        Maximum iterations for ``projection="ycn"``.
    projection_tol : float, optional (default=1e-12)
        Convergence tolerance for ``projection="ycn"``.
    mtc : string, optional (default="none")
        Multiple-testing correction (see
        :func:`~networkx_backbone.adjust_pvalues`) applied to the p-values
        of all tested agent pairs before thresholding.
    missing_as_zero : bool, optional (default=True)
        If ``True``, every agent pair is tested, including pairs that never
        co-occur (co-occurrence 0).  If ``False``, as with R
        ``backbone_from_projection`` (whose default is
        ``missing_as_zero = FALSE``), pairs that never co-occur are not
        tested: they do not count toward the *mtc* correction and are never
        retained, not even as significantly weak pairs with
        ``signed=True``.  The p-values of the tested pairs are the same
        either way.  See Notes.
    agent_labels, artifact_labels : sequence or None, optional
        Row and column labels of a matrix *B* (default ``0, 1, ...``, or a
        ``DataFrame``'s index and columns).  Only valid with a matrix *B*.
    **kwargs
        Passed to the selected model (e.g. ``signed``, ``trials``, ``seed``,
        ``prohibited``, ``required``, ``pvalue_method``).  SDSM-EC
        constraints follow :func:`sdsm`: pairs that contradict *B* are
        ignored with a ``RuntimeWarning``.

    Returns
    -------
    backbone : graph
        Graph on the agent nodes containing the significant pairs, each with
        a ``"<method>_pvalue"`` attribute (unadjusted), the projection weight
        and, for ``signed=True``, a ``"sign"`` attribute.

    Raises
    ------
    ValueError
        If *method* is unknown or a matrix *B* is invalid.
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it, or
        an SDSM-EC constraint names nodes that are not an ``(agent,
        artifact)`` pair of *B* or is both required and prohibited.
    AmbiguousSolution
        If *agent_nodes* is ``None`` and *B* is disconnected without
        ``bipartite`` node attributes.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.

    See Also
    --------
    sdsm, fdsm, fixedfill, fixedrow, fixedcol, backbone

    Notes
    -----
    By default (``missing_as_zero=True``) every agent pair counts as a test
    for *mtc*, including pairs that never co-occur; R
    ``backbone_from_projection`` with its default ``missing_as_zero = FALSE``
    counts only co-occurring pairs, so its correction is less strict when
    some pairs never meet.  Likewise, with ``signed=True`` a pair that never
    co-occurs can be significantly weak here but is never tested by R.
    ``missing_as_zero=False`` reproduces R's default; ``missing_as_zero=True``
    reproduces R's ``missing_as_zero = TRUE``.  Without *mtc* and with
    ``signed=False`` the two settings give the same backbone, because a pair
    that never co-occurs has upper-tail p-value 1.  When both tails are
    exactly equal R signs the pair ``-1`` and networkx-backbone ``+1``.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import backbone_from_projection
    >>> B = nx.davis_southern_women_graph()
    >>> H = backbone_from_projection(B, method="sdsm", alpha=0.05)
    >>> all(d["sdsm_pvalue"] < 0.05 for _, _, d in H.edges(data=True))
    True

    An incidence matrix gives the same backbone:

    >>> women = sorted((n for n, d in B.nodes(data=True) if d["bipartite"] == 0), key=str)
    >>> events = sorted((n for n, d in B.nodes(data=True) if d["bipartite"] == 1), key=str)
    >>> M = nx.bipartite.biadjacency_matrix(B, row_order=women, column_order=events)
    >>> H2 = backbone_from_projection(
    ...     M, method="sdsm", alpha=0.05, agent_labels=women, artifact_labels=events
    ... )
    >>> sorted(H2.edges()) == sorted(H.edges())
    True

    With ``missing_as_zero=False`` only the pairs that co-occur count toward
    the Bonferroni correction, as in R:

    >>> all_pairs = backbone_from_projection(B, method="fixedfill", alpha=0.06, mtc="bonferroni")
    >>> cooccurring = backbone_from_projection(
    ...     B, method="fixedfill", alpha=0.06, mtc="bonferroni", missing_as_zero=False
    ... )
    >>> all_pairs.number_of_edges(), cooccurring.number_of_edges()
    (0, 1)
    """
    builders = {
        "sdsm": _sdsm_tails,
        "fdsm": _fdsm_tails,
        "fixedfill": _fixedfill_tails,
        "fixedrow": _fixedrow_tails,
        "fixedcol": _fixedcol_tails,
    }
    method_l = method.lower()
    if method_l not in builders:
        raise ValueError(
            "Unknown projection method. Choose one of: "
            "'sdsm', 'fdsm', 'fixedfill', 'fixedrow', 'fixedcol'."
        )
    inc = _incidence(B, agent_nodes, agent_labels, artifact_labels)
    # Score every pair, then threshold uniformly so ``mtc`` sees every test.
    return _score_pairs(
        inc,
        builders[method_l](inc, **kwargs),
        f"{method_l}_pvalue",
        signed=kwargs.get("signed", False),
        alpha=alpha,
        mtc=mtc,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
        missing_as_zero=missing_as_zero,
    )


def backbone_from_weighted(
    G: GraphInput,
    method: str = "disparity",
    weight: str = "weight",
    alpha: float = 0.05,
    signed: bool = False,
    mtc: str = "none",
    collapse_multiedges: bool = True,
    edge_type_attr: str | None = None,
    missing_as_zero: bool = False,
    *,
    nodelist: Sequence[Hashable] | None = None,
    directed: bool = False,
    source: str = "source",
    target: str = "target",
    **kwargs: Any,
) -> nx.Graph:
    """Extract a weighted-network backbone using method-name dispatch.

    Multi-edge inputs can be collapsed automatically before scoring.

    Parameters
    ----------
    G : graph, array-like, sparse array or DataFrame
        A weighted NetworkX graph, a weighted adjacency matrix (NumPy array
        or SciPy sparse array or matrix), or a pandas edge-list
        ``DataFrame``; non-graph input is converted with
        :func:`~networkx_backbone.to_graph`.
    method : {"disparity", "mlf", "lans", "global"}, optional
        Backbone method (aliases such as ``"disparity_filter"`` accepted).
    weight : string, optional (default="weight")
        Edge weight attribute (and the edge-list column or attribute that
        holds the matrix entries for non-graph *G*).
    alpha : float, optional (default=0.05)
        Significance level for the statistical methods.
    signed : bool, optional (default=False)
        If ``True``, use a two-tailed test (for ``disparity``/``mlf``/``lans``)
        so the backbone retains significantly strong and significantly weak
        edges, each annotated with a ``"sign"`` attribute.
    mtc : string, optional (default="none")
        Multiple-testing correction applied before thresholding (see
        :func:`~networkx_backbone.adjust_pvalues`).
    collapse_multiedges : bool, optional (default=True)
        If ``True`` and ``G`` is a ``MultiGraph`` or ``MultiDiGraph``,
        collapse parallel edges using
        :func:`~networkx_backbone.multigraph_to_weighted`.
    edge_type_attr : string or None, optional (default=None)
        Edge attribute used when collapsing multi-edges. If provided, edge
        weights count distinct edge types per node pair; otherwise they count
        parallel edges.
    missing_as_zero : bool, optional (default=False)
        If ``True``, every pair of distinct nodes without an edge is tested as
        an edge of weight 0, as with ``missing_as_zero=TRUE`` in the R
        backbone package (version 3.0), and the significant ones are returned
        as edges with weight 0.  With ``signed=True`` this keeps pairs that
        interact significantly less than expected (``sign`` -1).  The tested
        pairs also count toward the ``mtc`` correction.  As in R, the
        ``"disparity"`` and ``"lans"`` nulls give a weight of 0 a lower-tail
        p-value of 0, so they keep every absent pair they can test (for
        ``"disparity"``, except pairs of two degree-1 nodes); ``"mlf"`` gives
        a graded binomial test.  See the Notes of
        :func:`~networkx_backbone.disparity_filter`,
        :func:`~networkx_backbone.marginal_likelihood_filter` and
        :func:`~networkx_backbone.lans_filter` for how each model scores
        them.  Not supported for ``method="global"``.
    nodelist : sequence or None, optional (default=None)
        Node labels for the rows and columns of an adjacency matrix *G*.
    directed : bool, optional (default=False)
        Read an adjacency matrix or edge list *G* as a directed graph; if
        ``False`` an adjacency matrix must be symmetric.
    source, target : str, optional (default="source", "target")
        Columns of an edge-list *G* holding the endpoints of each edge.
    **kwargs
        ``threshold=...`` for ``method="global"``.

    Returns
    -------
    backbone : graph
        The filtered graph.

    Raises
    ------
    ValueError
        If *method* is unknown, ``method="global"`` lacks ``threshold``,
        ``missing_as_zero=True`` is combined with ``method="global"``, or
        a non-graph *G* is invalid (see :func:`~networkx_backbone.to_graph`).

    Warns
    -----
    RuntimeWarning
        If ``missing_as_zero=True`` would test more than one million absent
        node pairs.

    See Also
    --------
    backbone, backbone_from_projection, backbone_from_unweighted, to_graph

    Notes
    -----
    *nodelist*, *directed*, *source* and *target* only apply to non-graph
    input; a NetworkX graph is used as given.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import backbone_from_weighted
    >>> G = nx.les_miserables_graph()
    >>> H = backbone_from_weighted(G, method="disparity", alpha=0.05)
    >>> H.number_of_edges() <= G.number_of_edges()
    True

    A SciPy sparse adjacency matrix gives the same backbone:

    >>> nodes = list(G)
    >>> A = nx.to_scipy_sparse_array(G, nodelist=nodes)
    >>> H2 = backbone_from_weighted(A, method="disparity", nodelist=nodes)
    >>> dict(H2.edges) == dict(H.edges)
    True

    With ``missing_as_zero=True`` and ``signed=True``, pairs of strong nodes
    that never interact are kept as significantly weak, with weight 0:

    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([("a", "b", 20), ("c", "d", 20), ("a", "c", 1)])
    >>> S = backbone_from_weighted(G, method="mlf", signed=True, missing_as_zero=True)
    >>> sorted((u, v, d["sign"]) for u, v, d in S.edges(data=True) if d["weight"] == 0)
    [('a', 'd', -1), ('b', 'c', -1), ('b', 'd', -1)]
    """
    from networkx_backbone.convert import to_graph
    from networkx_backbone.filters import (
        boolean_filter,
        multigraph_to_weighted,
        threshold_filter,
    )
    from networkx_backbone.statistical import (
        disparity_filter,
        lans_filter,
        marginal_likelihood_filter,
    )
    from networkx_backbone.structural import global_threshold_filter

    G = to_graph(
        G,
        nodelist=nodelist,
        directed=directed,
        source=source,
        target=target,
        weight=weight,
    )
    if collapse_multiedges and G.is_multigraph():
        G = multigraph_to_weighted(G, weight=weight, edge_type_attr=edge_type_attr)

    method_l = method.lower()

    if missing_as_zero and method_l in (
        "global",
        "global_threshold",
        "global_threshold_filter",
    ):
        raise ValueError(
            "missing_as_zero=True is only supported for the statistical "
            "methods 'disparity', 'mlf' and 'lans'."
        )
    score_kw: dict[str, Any] = {
        "weight": weight,
        "signed": signed,
        "missing_as_zero": missing_as_zero,
    }

    if method_l in ("disparity", "disparity_filter"):
        scored = disparity_filter(G, **score_kw)
        return threshold_filter(
            scored, "disparity_pvalue", alpha, mode="below", mtc=mtc
        )

    if method_l in ("mlf", "marginal_likelihood", "marginal_likelihood_filter"):
        scored = marginal_likelihood_filter(G, **score_kw)
        return threshold_filter(scored, "ml_pvalue", alpha, mode="below", mtc=mtc)

    if method_l in ("lans", "lans_filter"):
        scored = lans_filter(G, **score_kw)
        return threshold_filter(scored, "lans_pvalue", alpha, mode="below", mtc=mtc)

    if method_l in ("global", "global_threshold", "global_threshold_filter"):
        threshold = kwargs.get("threshold")
        if threshold is None:
            raise ValueError("global threshold method requires `threshold=...`")
        scored = global_threshold_filter(G, threshold=threshold, weight=weight)
        return boolean_filter(scored, "global_threshold_keep")

    raise ValueError(
        "Unknown weighted method. Choose one of: 'disparity', 'mlf', 'lans', 'global'."
    )


def backbone_from_unweighted(
    G: GraphInput,
    method: str = "sparsify",
    *,
    nodelist: Sequence[Hashable] | None = None,
    directed: bool = False,
    source: str = "source",
    target: str = "target",
    **kwargs: Any,
) -> nx.Graph:
    """Extract an unweighted-network backbone using method-name dispatch.

    Parameters
    ----------
    G : graph, array-like, sparse array or DataFrame
        An unweighted NetworkX graph, an adjacency matrix (NumPy array or
        SciPy sparse array or matrix; every nonzero entry is an edge), or a
        pandas edge-list ``DataFrame``; non-graph input is converted with
        :func:`~networkx_backbone.to_graph`, without edge weights.
    method : {"sparsify", "lspar", "local_degree"}, optional
        Sparsification method (default ``"sparsify"``).
    nodelist : sequence or None, optional (default=None)
        Node labels for the rows and columns of an adjacency matrix *G*.
    directed : bool, optional (default=False)
        Read an adjacency matrix or edge list *G* as a directed graph; if
        ``False`` an adjacency matrix must be symmetric.
    source, target : str, optional (default="source", "target")
        Columns of an edge-list *G* holding the endpoints of each edge.
    **kwargs
        Passed to the selected method (e.g. ``s`` for ``"lspar"``).

    Returns
    -------
    backbone : graph
        Graph containing the retained edges.

    Raises
    ------
    ValueError
        If *method* is unknown or a non-graph *G* is invalid (see
        :func:`~networkx_backbone.to_graph`).

    See Also
    --------
    backbone, backbone_from_weighted, to_graph

    Notes
    -----
    *nodelist*, *directed*, *source* and *target* only apply to non-graph
    input; a NetworkX graph is used as given.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import backbone_from_unweighted
    >>> G = nx.karate_club_graph()
    >>> H = backbone_from_unweighted(G, method="lspar", s=0.5)
    >>> H.number_of_nodes() == G.number_of_nodes()
    True

    The same backbone from a NumPy adjacency matrix:

    >>> A = nx.to_numpy_array(G, weight=None)
    >>> H2 = backbone_from_unweighted(A, method="lspar", s=0.5, nodelist=list(G))
    >>> sorted(H2.edges()) == sorted(H.edges())
    True
    """
    from networkx_backbone.convert import to_graph
    from networkx_backbone.filters import boolean_filter
    from networkx_backbone.unweighted import local_degree, lspar, sparsify

    G = to_graph(
        G,
        nodelist=nodelist,
        directed=directed,
        source=source,
        target=target,
        weight=None,
    )
    method_l = method.lower()
    if method_l == "sparsify":
        scored = sparsify(G, **kwargs)
        return boolean_filter(scored, "sparsify_keep")
    if method_l == "lspar":
        scored = lspar(G, **kwargs)
        return boolean_filter(scored, "sparsify_keep")
    if method_l in ("local_degree", "localdegree"):
        scored = local_degree(G, **kwargs)
        return boolean_filter(scored, "sparsify_keep")

    raise ValueError(
        "Unknown unweighted method. Choose one of: 'sparsify', 'lspar', 'local_degree'."
    )


def backbone(G: GraphInput, method: str | None = None, **kwargs: Any) -> nx.Graph:
    """Unified dispatch wrapper across projection, weighted, and unweighted methods.

    Parameters
    ----------
    G : graph, array-like, sparse array or DataFrame
        A NetworkX graph; a matrix (NumPy array, SciPy sparse array or
        matrix, or pandas ``DataFrame``); or a pandas edge-list
        ``DataFrame``.  A matrix is read as an agents x artifacts incidence
        matrix for the projection methods (and when *method* is ``None``),
        and as an adjacency matrix for the weighted and unweighted methods.
        A ``DataFrame`` with the *source* and *target* columns (default
        ``"source"`` and ``"target"``) is an edge list.  Non-graph input to
        the weighted and unweighted methods is converted with
        :func:`~networkx_backbone.to_graph`.
    method : string or None, optional (default=None)
        Any method name accepted by :func:`backbone_from_projection`,
        :func:`backbone_from_weighted` or :func:`backbone_from_unweighted`.
        If ``None``, the method family is chosen from *G*: incidence
        matrices and undirected simple graphs with ``bipartite`` node
        attributes use the SDSM projection backbone, graphs (and edge
        lists) with *weight* edge attributes (default ``"weight"``) use the
        disparity filter, and other graphs use ``sparsify``.
    **kwargs
        Passed to the selected wrapper (e.g. ``alpha``, ``agent_nodes``,
        ``mtc``, ``signed``, the SDSM-EC constraints ``prohibited`` and
        ``required`` (contradictory pairs are ignored with a warning, see
        :func:`sdsm`), or the conversion options ``nodelist``,
        ``directed``, ``source`` and ``target``).

    Returns
    -------
    backbone : graph
        The extracted backbone.

    Raises
    ------
    ValueError
        If *method* is not recognized, or non-graph *G* is invalid.

    See Also
    --------
    backbone_from_projection, backbone_from_weighted, backbone_from_unweighted,
    to_graph

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import backbone
    >>> B = nx.davis_southern_women_graph()
    >>> H = backbone(B, alpha=0.05)  # bipartite attributes -> SDSM
    >>> all(d["sdsm_pvalue"] < 0.05 for _, _, d in H.edges(data=True))
    True

    Naming a weighted method reads a matrix as a weighted adjacency matrix:

    >>> G = nx.les_miserables_graph()
    >>> A = nx.to_scipy_sparse_array(G, nodelist=list(G))
    >>> H = backbone(A, method="disparity", nodelist=list(G))
    >>> dict(H.edges) == dict(backbone(G, method="disparity").edges)
    True
    """
    from networkx_backbone.convert import _is_dataframe, to_graph

    method_l = method.lower() if isinstance(method, str) else None
    weight = kwargs.get("weight", "weight")
    if method_l is None and _is_dataframe(G):
        source = kwargs.get("source", "source")
        target = kwargs.get("target", "target")
        if source in G.columns and target in G.columns:
            # An edge list: convert it, then choose the method from the graph.
            G = to_graph(
                G,
                directed=kwargs.pop("directed", False),
                source=kwargs.pop("source", "source"),
                target=kwargs.pop("target", "target"),
                weight=weight,
            )
    if not isinstance(G, nx.Graph) and method_l is None:
        # Any other matrix is an incidence matrix for a bipartite projection.
        return backbone_from_projection(G, **kwargs)

    projection_methods = {"sdsm", "fdsm", "fixedfill", "fixedrow", "fixedcol"}
    weighted_methods = {
        "disparity",
        "disparity_filter",
        "mlf",
        "marginal_likelihood",
        "marginal_likelihood_filter",
        "lans",
        "lans_filter",
        "global",
        "global_threshold",
        "global_threshold_filter",
    }
    unweighted_methods = {"sparsify", "lspar", "local_degree", "localdegree"}

    if method_l in projection_methods:
        return backbone_from_projection(G, method=method_l, **kwargs)
    if method_l in weighted_methods:
        return backbone_from_weighted(G, method=method_l, **kwargs)
    if method_l in unweighted_methods:
        return backbone_from_unweighted(G, method=method_l, **kwargs)
    if method is not None:
        raise ValueError(f"Unknown method {method!r}")

    # Only graphs reach this point (matrices returned above); tell mypy so.
    G = cast("nx.Graph", G)
    has_bipartite_attr = any("bipartite" in data for _, data in G.nodes(data=True))
    if (
        has_bipartite_attr
        and not G.is_directed()
        and not G.is_multigraph()
        and nx.is_bipartite(G)
    ):
        return backbone_from_projection(G, **kwargs)

    weighted = any(weight in data for _, _, data in G.edges(data=True))
    if weighted:
        return backbone_from_weighted(G, **kwargs)
    return backbone_from_unweighted(G, **kwargs)


# =====================================================================
# 1. SDSM -- Neal (2014)
# =====================================================================


def _constraint_mask(pairs, name, a_idx, f_idx, shape):
    """Validate ``(agent, artifact)`` constraint pairs into a boolean mask."""
    import numpy as np

    mask = np.zeros(shape, dtype=bool)
    if pairs is None:
        return mask
    for pair in pairs:
        try:
            a, f = pair
        except (TypeError, ValueError) as err:
            raise nx.NetworkXError(
                f"{name} must contain (agent, artifact) pairs; got {pair!r}."
            ) from err
        try:
            if a in a_idx and f in f_idx:
                mask[a_idx[a], f_idx[f]] = True
                continue
            reversed_pair = f in a_idx and a in f_idx
        except TypeError as err:
            raise nx.NetworkXError(
                f"{name} must contain hashable (agent, artifact) pairs; got {pair!r}."
            ) from err
        if reversed_pair:
            raise nx.NetworkXError(
                f"{name} pairs must be ordered (agent, artifact); got "
                f"(artifact, agent) pair {pair!r}."
            )
        raise nx.NetworkXError(
            f"{name} pair {pair!r} is not an (agent, artifact) pair of nodes in B."
        )
    return mask


_MAX_LISTED_CONSTRAINTS = 5


def _drop_contradictory_constraints(R, req_mask, pro_mask, agents, artifacts):
    """Drop constraints that contradict the observed incidence matrix *R*.

    ``required`` cells that are not edges of *R* and ``prohibited`` cells that
    are edges of *R* are removed from the masks, with one ``RuntimeWarning``
    that counts and lists (up to ``_MAX_LISTED_CONSTRAINTS``) the dropped
    pairs of each kind.  A cell in both masks raises ``NetworkXError``.
    Returns the cleaned ``(req_mask, pro_mask)``.
    """
    import numpy as np

    def listed(mask):
        rows, cols = np.nonzero(mask)
        pairs = [
            (agents[i], artifacts[k])
            for i, k in zip(
                rows[:_MAX_LISTED_CONSTRAINTS], cols[:_MAX_LISTED_CONSTRAINTS]
            )
        ]
        text = ", ".join(repr(p) for p in pairs)
        if len(rows) > _MAX_LISTED_CONSTRAINTS:
            text += ", ..."
        return text

    both = req_mask & pro_mask
    if both.any():
        raise nx.NetworkXError(
            f"sdsm: {int(both.sum())} pair(s) are both required and prohibited: "
            f"{listed(both)}."
        )
    bad_req = req_mask & (R == 0)
    bad_pro = pro_mask & (R != 0)
    parts = []
    if bad_req.any():
        parts.append(
            f"{int(bad_req.sum())} required pair(s) that are not edges of B "
            f"[{listed(bad_req)}]"
        )
    if bad_pro.any():
        parts.append(
            f"{int(bad_pro.sum())} prohibited pair(s) that are edges of B "
            f"[{listed(bad_pro)}]"
        )
    if parts:
        from networkx_backbone.statistical import _external_stacklevel

        warnings.warn(
            "sdsm: ignoring SDSM-EC constraints that contradict the data: "
            f"{'; '.join(parts)}.",
            RuntimeWarning,
            stacklevel=_external_stacklevel(),
        )
    return req_mask & ~bad_req, pro_mask & ~bad_pro


def _sdsm_tails(
    inc,
    *,
    signed=False,
    prohibited=None,
    required=None,
    pvalue_method="auto",
    weight=None,
):
    """Pair tails under the SDSM (``None`` when no pair can be scored).

    Without edge constraints the BiCM probabilities depend only on the
    degrees (:func:`_bicm_classes`), so the Poisson-binomial tails are
    computed once per pair of agent-degree classes and looked up per pair.
    With constraints the full probability matrix is fitted; constraints that
    contradict *B* are dropped with a warning
    (:func:`_drop_contradictory_constraints`), and if none remain the
    unconstrained path is used.
    """
    import numpy as np

    if pvalue_method not in ("auto", "exact", "rna"):
        raise ValueError("pvalue_method must be one of 'auto', 'exact', 'rna'.")
    agents, artifacts, R = inc
    na, nf = R.shape

    constrained = required is not None or prohibited is not None
    if constrained:
        a_idx = {v: idx for idx, v in enumerate(agents)}
        f_idx = {v: idx for idx, v in enumerate(artifacts)}
        Rd = R.toarray()
        req_mask = _constraint_mask(required, "required", a_idx, f_idx, Rd.shape)
        pro_mask = _constraint_mask(prohibited, "prohibited", a_idx, f_idx, Rd.shape)
        req_mask, pro_mask = _drop_contradictory_constraints(
            Rd, req_mask, pro_mask, agents, artifacts
        )
        constrained = bool(req_mask.any() or pro_mask.any())

    if na < 2 or R.nnz == 0:
        return None
    method = _resolve_pvalue_method(pvalue_method, na * (na - 1) // 2, nf)

    if constrained:
        P = _bicm_fit(Rd, required=req_mask, prohibited=pro_mask)
        return lambda i, j, k: _pairwise_pb_tails(
            P, k, method=method, need_lower=signed, pairs=(i, j)
        )

    row_deg, col_deg = _degrees(R)
    Q, row_class, col_class = _bicm_classes(row_deg, col_deg)
    Prow = Q[:, col_class]  # the P row of each agent-degree class
    if method == "rna":
        mu = Prow @ Prow.T
        P2 = Prow * Prow
        s2 = P2 @ P2.T
        P3 = P2 * Prow
        s3 = P3 @ P3.T

        def rna_tails(k, a, b):
            upper, lower = _rna_tails(mu[a, b], s2[a, b], s3[a, b], k)
            return upper, (lower if signed else None)

        # The tails depend only on (k, class_i, class_j).
        return lambda i, j, k: _per_distinct(
            rna_tails, k, row_class[i], row_class[j], len(mu)
        )

    # Exact: one Poisson-binomial pmf per unordered pair of degree classes
    # (P[i] * P[j] is the same product for both orders), tabulated up to the
    # largest possible co-occurrence.
    n_cls = Prow.shape[0]
    ca, cb = np.triu_indices(n_cls)
    pair_id = np.zeros((n_cls, n_cls), dtype=np.intp)
    pair_id[ca, cb] = pair_id[cb, ca] = np.arange(len(ca))
    k_max = min(nf, int(row_deg.max()))
    sf_tab = np.empty((len(ca), k_max + 1))
    cdf_tab = np.empty((len(ca), k_max + 1)) if signed else None
    chunk = max(1, int(2_000_000 // (nf + 1)))
    for start in range(0, len(ca), chunk):
        stop = min(start + chunk, len(ca))
        pmf = _poisson_binomial_pmf(Prow[ca[start:stop]] * Prow[cb[start:stop]])
        sf_tab[start:stop] = np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1][:, : k_max + 1]
        if signed:
            cdf_tab[start:stop] = np.cumsum(pmf, axis=1)[:, : k_max + 1]

    def exact_tails(i, j, k):
        pid = pair_id[row_class[i], row_class[j]]
        kk = np.clip(k, 0, k_max)
        upper = np.clip(sf_tab[pid, kk], 0.0, 1.0)
        lower = np.clip(cdf_tab[pid, kk], 0.0, 1.0) if signed else None
        return upper, lower

    return exact_tails


def _fdsm_tails(inc, *, trials=1000, seed=None, signed=False, burn_in=None, thin=None):
    """Monte Carlo pair tails under the FDSM.

    Only pairs that co-occur in the data need per-sample counts: a pair with
    observed count 0 always satisfies ``random >= 0``, and for the lower tail
    it only needs the number of samples in which it co-occurs, which is
    accumulated sparsely.
    """
    import numpy as np
    import scipy.sparse as sp

    if trials < 1:
        raise ValueError("trials must be >= 1")
    R = inc.R
    na, nf = R.shape
    rng = _as_generator(seed)
    obs_keys, obs_vals = _sorted_upper_keys(R @ R.T, na)
    exceed = np.zeros(len(obs_keys), dtype=np.int64)
    below = np.zeros(len(obs_keys), dtype=np.int64)
    zero_hits = sp.csr_array((na, na), dtype=np.int64)
    for row_sets in _fdsm_row_sets(R, trials, rng, burn_in=burn_in, thin=thin):
        Rs = _row_sets_to_csr(row_sets, nf)
        keys, vals = _sorted_upper_keys(Rs @ Rs.T, na)
        pos, hit = _lookup(obs_keys, keys)
        rand = np.zeros(len(obs_keys), dtype=np.int64)
        rand[pos[hit]] = vals[hit]
        exceed += rand >= obs_vals
        if signed:
            below += rand <= obs_vals
            miss = keys[~hit]
            zero_hits = zero_hits + sp.csr_array(
                (np.ones(len(miss), dtype=np.int64), (miss // na, miss % na)),
                shape=(na, na),
            )
    zero_keys, zero_counts = _sorted_upper_keys(zero_hits, na)

    def tails(i, j, k):
        keys = i.astype(np.int64) * na + j
        exceed_ij = np.full(len(keys), trials, dtype=np.int64)
        pos, hit = _lookup(obs_keys, keys)
        exceed_ij[hit] = exceed[pos[hit]]
        p_hi = (exceed_ij + 1) / (trials + 1)
        if not signed:
            return p_hi, None
        below_ij = np.full(len(keys), trials, dtype=np.int64)
        below_ij[hit] = below[pos[hit]]
        zpos, zhit = _lookup(zero_keys, keys)
        zhit &= ~hit
        below_ij[zhit] -= zero_counts[zpos[zhit]]
        return p_hi, (below_ij + 1) / (trials + 1)

    return tails


@_graph_or_matrix
def sdsm(
    B: GraphInput,
    agent_nodes: Iterable[Hashable] | None = None,
    alpha: float = 0.05,
    weight: str | None = None,
    signed: bool = False,
    prohibited: Iterable[tuple[Hashable, Hashable]] | None = None,
    required: Iterable[tuple[Hashable, Hashable]] | None = None,
    projection: str = "simple",
    projection_weight: str = "weight",
    projection_directed: bool = False,
    projection_max_iter: int = 1000,
    projection_tol: float = 1e-12,
    pvalue_method: str = "auto",
    *,
    agent_labels: Sequence[Hashable] | None = None,
    artifact_labels: Sequence[Hashable] | None = None,
) -> nx.Graph:
    """Score a bipartite projection using the SDSM.

    The Stochastic Degree Sequence Model [1]_ treats each bipartite edge as
    an independent Bernoulli variable whose probability ``P[i, k]`` comes from
    the Bipartite Configuration Model (BiCM, see :func:`bicm`), the
    maximum-entropy model whose *expected* row and column sums equal the
    observed agent and artifact degrees.  The co-occurrence count of agents
    ``i`` and ``j`` is then a Poisson-binomial variable with success
    probabilities ``P[i, k] * P[j, k]``, and the p-value is its upper tail at
    the observed count.  As in R ``backbone``, the tail is evaluated with the
    refined normal approximation (RNA) of Hong (2013) [3]_; when the exact
    distribution is cheap (at most 200 artifacts), it is computed exactly by
    convolution instead (see *pvalue_method*).

    Without edge constraints the BiCM probabilities depend only on the
    agent and artifact degrees, so the tails are computed once per pair of
    distinct agent degrees and the incidence matrix stays sparse.

    Parameters
    ----------
    B : graph or array-like
        An undirected bipartite NetworkX graph, or an agents x artifacts
        binary incidence matrix (NumPy array, SciPy sparse array or matrix,
        or pandas ``DataFrame``).
    agent_nodes : iterable or None, optional (default=None)
        Nodes in the "agent" partition of a graph *B*.  If ``None`` they are
        inferred as in :func:`backbone_from_projection`.  Must be ``None``
        for a matrix *B*.
    alpha : float, optional (default=0.05)
        Retained for API compatibility. This function returns the full
        projected graph with p-values on all edges; apply
        :func:`~networkx_backbone.threshold_filter` (or use
        :func:`backbone_from_projection`) to select significant edges.
    weight : None or string, optional (default=None)
        Not used for SDSM (bipartite is unweighted); reserved for API
        consistency.
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test: the stored p-value becomes two-sided
        and a ``"sign"`` edge attribute marks significantly strong (``+1``)
        versus significantly weak (``-1``) co-occurrences.
    prohibited : iterable of (agent, artifact) pairs or None, optional
        SDSM-EC edge constraints (Neal & Neal, 2023) [2]_: cells fixed to
        probability 0 in the null model (edges that cannot occur).  Pairs
        that are edges of *B* contradict the data and are ignored with a
        ``RuntimeWarning``.
    required : iterable of (agent, artifact) pairs or None, optional
        SDSM-EC edge constraints: cells fixed to probability 1 in the null
        model (edges that must occur).  Pairs that are not edges of *B*
        contradict the data and are ignored with a ``RuntimeWarning``.
        With constraints the BiCM is refitted on the unconstrained cells only,
        so expected degrees still match the observed ones.  Constraints use a
        dense ``n_agents x n_artifacts`` probability matrix.  If no
        constraint remains (none given, or all ignored), the result equals
        the unconstrained SDSM.
    projection : {"simple", "hyper", "newman", "probs", "ycn"}, optional
        Projection weighting assigned to each returned edge (see
        :func:`bipartite_projection`).
    projection_weight : str, optional (default="weight")
        Edge attribute name for the projection weight.
    projection_directed : bool, optional (default=False)
        Whether to use directed projection scores. Not supported for the
        undirected backbone output.
    projection_max_iter : int, optional (default=1000)
        Maximum iterations for ``projection="ycn"``.
    projection_tol : float, optional (default=1e-12)
        Convergence tolerance for ``projection="ycn"``.
    pvalue_method : {"auto", "exact", "rna"}, optional (default="auto")
        How to evaluate Poisson-binomial tails: ``"exact"`` (convolution,
        ``O(n_artifacts^2)`` per pair of degree classes), ``"rna"`` (refined
        normal approximation, as in R ``backbone``) or ``"auto"`` (exact
        when there are at most 200 artifacts and
        ``n_pairs * (n_artifacts + 1)^2 <= 2e8``, else RNA).
    agent_labels, artifact_labels : sequence or None, optional
        Row and column labels of a matrix *B*.  Only valid with a matrix *B*.

    Returns
    -------
    backbone : graph
        Full unipartite projection among agent nodes.  Each edge has a
        ``"sdsm_pvalue"`` attribute and projection weights.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite, *agent_nodes* is not one side of it, a
        constraint is not an ``(agent, artifact)`` pair of nodes of *B*
        (unknown node, two nodes of the same side, or reversed order), or a
        pair is both required and prohibited.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.
    ValueError
        If *pvalue_method* or *projection* is unknown, or a matrix *B* is
        invalid.

    Warns
    -----
    RuntimeWarning
        If the BiCM fit does not converge, or if constraints that contradict
        *B* (required non-edges, prohibited edges) are ignored.  One warning
        per call gives the number of ignored pairs of each kind and lists up
        to five of them.

    See Also
    --------
    bicm, fdsm, backbone_from_projection

    Notes
    -----
    **Contradictory constraints.**  A ``required`` pair that is not an edge
    of *B*, or a ``prohibited`` pair that is, contradicts the observed data:
    fixing that cell would make the null model's expected degrees differ
    from the observed ones.  Such pairs are dropped from the constraints
    with a warning, and the others are kept.  A pair listed as both required
    and prohibited is an error in the constraints themselves and raises.
    R ``backbone`` 3.0.4 has no such case: it encodes constraints in the
    incidence matrix itself (in its SDSM-EC code a cell value of 10 is a
    prohibited edge and 11 a required edge), so each cell is either an
    observed 0/1 or a constraint.  R counts a 10 cell as absent and an 11
    cell as present in the observed co-occurrences, so marking an observed
    edge 10 (or an absent edge 11) silently overrides the data rather than
    being ignored or flagged.


    **Comparison with R backbone.**  With ``pvalue_method="rna"`` the p-values
    equal R ``backbone::sdsm`` up to the BiCM fit (``1e-8``), except that a
    pair that never co-occurs gets the exact value 1 (R evaluates the
    approximation there, but only when ``missing_as_zero = TRUE``).  The
    default uses the exact tail when it is cheap, as on Davis Southern Women
    (14 events), where it differs from R's approximation by up to 0.017: the
    smallest p-value is 0.088 (R: 0.097), and neither finds a significant
    pair at alpha = 0.05.  With edge constraints R estimates the
    probabilities of the unconstrained cells by a logistic regression on the
    row and column sums instead of refitting the BiCM, so its expected
    degrees are not exactly the observed ones; on Davis with a few
    constraints the p-values differ by up to 0.09.

    References
    ----------
    .. [1] Neal, Z. P. (2014). The backbone of bipartite projections:
       Inferring relationships from co-authorship, co-sponsorship,
       co-attendance and other co-behaviors. *Social Networks*, 39, 84-97.
    .. [2] Neal, Z. P., & Neal, J. W. (2023). Stochastic Degree Sequence Model
       with Edge Constraints (SDSM-EC) for Backbone Extraction. *Complex
       Networks 12*, 127-136.
    .. [3] Hong, Y. (2013). On computing the distribution function for the
       Poisson binomial distribution. *Computational Statistics & Data
       Analysis*, 59, 41-51.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import sdsm
    >>> B = nx.davis_southern_women_graph()
    >>> women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    >>> scored = sdsm(B, agent_nodes=women)
    >>> from networkx_backbone import threshold_filter
    >>> backbone = threshold_filter(scored, "sdsm_pvalue", 0.05, mode="below")
    >>> all("sdsm_pvalue" in data for _, _, data in backbone.edges(data=True))
    True
    """
    if pvalue_method not in ("auto", "exact", "rna"):
        raise ValueError("pvalue_method must be one of 'auto', 'exact', 'rna'.")
    inc = _incidence(B, agent_nodes, agent_labels, artifact_labels)
    tails = _sdsm_tails(
        inc,
        signed=signed,
        prohibited=prohibited,
        required=required,
        pvalue_method=pvalue_method,
    )
    return _score_pairs(
        inc,
        tails,
        "sdsm_pvalue",
        signed=signed,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
    )


# =====================================================================
# 2. FDSM -- Neal et al. (2021)
# =====================================================================


@_graph_or_matrix
def fdsm(
    B: GraphInput,
    agent_nodes: Iterable[Hashable] | None = None,
    alpha: float = 0.05,
    trials: int = 1000,
    seed: Seed = None,
    signed: bool = False,
    projection: str = "simple",
    projection_weight: str = "weight",
    projection_directed: bool = False,
    projection_max_iter: int = 1000,
    projection_tol: float = 1e-12,
    burn_in: int | None = None,
    thin: int | None = None,
    *,
    agent_labels: Sequence[Hashable] | None = None,
    artifact_labels: Sequence[Hashable] | None = None,
) -> nx.Graph:
    """Score a bipartite projection using the FDSM.

    The Fixed Degree Sequence Model [1]_ uses Monte Carlo simulation to
    estimate p-values.  Random bipartite graphs that *exactly* preserve both
    the row sums (agent degrees) and column sums (artifact degrees) are drawn
    with a curveball Markov chain (:func:`fastball`) started at the observed
    graph: after ``burn_in`` trades, one sample is taken every ``thin``
    trades.  The p-value is ``(count + 1) / (trials + 1)``, where ``count``
    is the number of samples whose co-occurrence is at least (for the lower
    tail: at most) the observed one, so it is never exactly zero.  Samples
    are kept as sparse matrices.

    Parameters
    ----------
    B : graph or array-like
        An undirected bipartite NetworkX graph, or an agents x artifacts
        binary incidence matrix (NumPy array, SciPy sparse array or matrix,
        or pandas ``DataFrame``).
    agent_nodes : iterable or None, optional (default=None)
        Nodes in the "agent" partition of a graph *B*.  If ``None`` they are
        inferred as in :func:`backbone_from_projection`.  Must be ``None``
        for a matrix *B*.
    alpha : float, optional (default=0.05)
        Retained for API compatibility. This function returns the full
        projected graph with p-values on all edges; apply
        :func:`~networkx_backbone.threshold_filter` (or use
        :func:`backbone_from_projection`) to select significant edges.
    trials : int, optional (default=1000)
        Number of Monte Carlo samples.
    seed : integer, random_state, or None (default)
        Random seed for reproducibility (int, :class:`numpy.random.Generator`,
        :class:`numpy.random.RandomState` or ``None``).
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test: the stored p-value becomes two-sided
        and a ``"sign"`` edge attribute marks significantly strong (``+1``)
        versus significantly weak (``-1``) co-occurrences.
    projection : {"simple", "hyper", "newman", "probs", "ycn"}, optional
        Projection weighting assigned to each returned edge (see
        :func:`bipartite_projection`).
    projection_weight : str, optional (default="weight")
        Edge attribute name for the projection weight.
    projection_directed : bool, optional (default=False)
        Whether to use directed projection scores. Not supported for the
        undirected backbone output.
    projection_max_iter : int, optional (default=1000)
        Maximum iterations for ``projection="ycn"``.
    projection_tol : float, optional (default=1e-12)
        Convergence tolerance for ``projection="ycn"``.
    burn_in : int or None, optional (default=None)
        Curveball trades before the first sample (default ``10 * n_agents``).
    thin : int or None, optional (default=None)
        Curveball trades between consecutive samples (default
        ``5 * n_agents``).
    agent_labels, artifact_labels : sequence or None, optional
        Row and column labels of a matrix *B*.  Only valid with a matrix *B*.

    Returns
    -------
    backbone : graph
        Full unipartite projection among agent nodes.  Each edge has an
        ``"fdsm_pvalue"`` attribute and projection weights.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.
    ValueError
        If ``trials < 1``, *burn_in* / *thin* is negative, or a matrix *B*
        is invalid.

    See Also
    --------
    fastball, sdsm, backbone_from_projection

    Notes
    -----
    **Comparison with R backbone.**  R ``backbone::fdsm`` draws every sample
    independently with ``fastball()`` from the observed matrix, using 5 trades
    per row of the smaller side, and reports ``count / trials``.  On Davis
    Southern Women that is 70 trades per sample, too few to forget the
    observed matrix, and its p-values are biased upward by up to about 0.03;
    with 100 trades per row R's p-values match these within Monte Carlo
    error.  Both find the same 5 significant pairs at alpha = 0.05.

    References
    ----------
    .. [1] Neal, Z. P., Domagalski, R., & Sagan, B. (2021). Comparing
       alternatives to the fixed degree sequence model. *Scientific
       Reports*, 11, 23929.
    .. [2] Godard, K., & Neal, Z. P. (2022). fastball: A fast algorithm to
       sample bipartite graphs with fixed degree sequences. *Journal of
       Complex Networks*, 10(6), cnac049.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fdsm
    >>> B = nx.davis_southern_women_graph()
    >>> women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    >>> scored = fdsm(B, agent_nodes=women, trials=100, seed=42)
    >>> from networkx_backbone import threshold_filter
    >>> backbone = threshold_filter(scored, "fdsm_pvalue", 0.05, mode="below")
    >>> all("fdsm_pvalue" in data for _, _, data in backbone.edges(data=True))
    True
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    inc = _incidence(B, agent_nodes, agent_labels, artifact_labels)
    tails = _fdsm_tails(
        inc, trials=trials, seed=seed, signed=signed, burn_in=burn_in, thin=thin
    )
    return _score_pairs(
        inc,
        tails,
        "fdsm_pvalue",
        signed=signed,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
    )


_COMPLEXITY = {
    "simple_projection": {
        "time": "O(e_b + n_a^2 + sum_z d(z)^2)",
        "space": "O(e_b + B)",
        "notes": (
            "n_a=agents, n_f=artifacts, e_b=bipartite edges, d(z)=artifact "
            "degree; sparse products over blocks of B = 2^22 agent-pair cells."
        ),
    },
    "hyper_projection": {
        "time": "O(e_b + n_a^2 + sum_z d(z)^2)",
        "space": "O(e_b + B)",
    },
    "probs_projection": {
        "time": "O(e_b + n_a^2 + sum_z d(z)^2)",
        "space": "O(e_b + B)",
    },
    "ycn_projection": {
        "time": "O(I e_b + n_a^2 + sum_z d(z)^2)",
        "space": "O(e_b + B)",
        "notes": "I=max_iter for stationary-flow convergence.",
    },
    "bipartite_projection": {
        "time": "O(I e_b + n_a^2 + sum_z d(z)^2)",
        "space": "O(e_b + B)",
        "notes": "Worst-case over supported projection methods.",
    },
    "sdsm": {
        "time": "O(I (n_a + n_f) + c^2 n_f^2 + n_a^2 + sum_z d(z)^2)",
        "space": "O(e_b + c^2 n_f + B) plus the n_a^2 output edges",
        "notes": (
            "c=distinct agent degrees, I=BiCM iterations; the c^2 n_f^2 term "
            "is the exact Poisson-binomial path (RNA: c^2 n_f).  Edge "
            "constraints use a dense n_a x n_f probability matrix."
        ),
    },
    "fdsm": {
        "time": "O(T (S d + sum_z d(z)^2) + n_a^2)",
        "space": "O(e_b + nnz(R R^T) + B) plus the n_a^2 output edges",
        "notes": "T=trials, S=curveball trades per sample, d=max agent degree.",
    },
    "fixedfill": {
        "time": "O(n_f^2 + n_a^2 + sum_z d(z)^2)",
        "space": "O(e_b + n_f + B)",
    },
    "fixedrow": {
        "time": "O(n_a^2 + sum_z d(z)^2)",
        "space": "O(e_b + B)",
    },
    "fixedcol": {
        "time": "O(n_f^2 + n_a^2 + sum_z d(z)^2)",
        "space": "O(e_b + n_f + B)",
    },
    "bicm": {
        "time": "O(I c_a c_f + n_a n_f + e_b)",
        "space": "O(n_a n_f)",
        "notes": "I=fixed-point iterations, c_a/c_f=distinct degrees; output is dense.",
    },
    "fastball": {
        "time": "O(r c + S c)",
        "space": "O(r c)",
        "notes": "r=rows, c=cols, S=n_swaps.",
    },
    "backbone_from_projection": {
        "time": "Cost of the selected model",
        "space": "O(e_b + B + kept edges); O(n_a^2) floats with mtc",
        "notes": "Pairs are thresholded block by block; only significant ones become edges.",
    },
    "backbone_from_weighted": {
        "time": "O(m log n)",
        "space": "O(n + m)",
        "notes": "Worst-case over {disparity, mlf, lans, global_threshold}.",
    },
    "backbone_from_unweighted": {
        "time": "O(mn)",
        "space": "O(n + m)",
        "notes": "Worst-case over {sparsify, lspar, local_degree}.",
    },
    "backbone": {
        "time": "O(n + m) dispatch checks + selected method cost",
        "space": "O(1) additional beyond selected method",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
