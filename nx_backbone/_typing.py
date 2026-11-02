"""Type aliases and typed decorator shims for the public API (private).

Everything here is cheap to import: numpy, scipy and pandas are imported only
under :data:`typing.TYPE_CHECKING`, and the aliases are :pep:`695` ``type``
statements, whose values are evaluated lazily, so the core package still
imports with only NetworkX installed (``tests/test_minimal_deps.py``).

NetworkX's decorators (``not_implemented_for``, ``py_random_state`` and
``argmap`` from :mod:`networkx.utils`) are typed by ``types-networkx`` as
returning ``Callable[..., Any]``, which would erase the signature of every
function they decorate.  The ``not_implemented_for`` and ``py_random_state``
below are NetworkX's own functions at runtime; only the type checker sees the
signature-preserving :class:`IdentityDecorator` types.  (``bipartite.py``
casts its ``argmap`` instance to :class:`IdentityDecorator` the same way.)
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, overload

if TYPE_CHECKING:
    import datetime
    import random

    import networkx as nx
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    import scipy.sparse as sp

__all__ = [
    "Frequency",
    "GraphInput",
    "Hyperedge",
    "Hyperedges",
    "Dispatchable",
    "DispatchDecorator",
    "IdentityDecorator",
    "PySeed",
    "Seed",
    "Snapshots",
    "TemporalNetwork",
    "not_implemented_for",
    "py_random_state",
]

#: A non-empty hyperedge as returned by the hypergraph functions.
type Hyperedge = frozenset[Hashable]

#: A hypergraph as accepted by the hypergraph functions: any iterable of
#: iterables of node labels, e.g. ``[(1, 2, 3), (2, 4)]``.
type Hyperedges = Iterable[Iterable[Hashable]]

#: Seeds accepted by functions that draw from :func:`numpy.random.default_rng`
#: (or NetworkX's ``np_random_state``).
type Seed = int | np.random.Generator | np.random.RandomState | None

#: Seeds accepted by functions decorated with NetworkX's ``py_random_state``.
type PySeed = int | random.Random | np.random.Generator | np.random.RandomState | None

#: A graph, or a matrix (NumPy array-like, SciPy sparse array or matrix, pandas
#: ``DataFrame``): an adjacency matrix or edge list for :func:`to_graph`, an
#: agents x artifacts incidence matrix for the bipartite methods.
type GraphInput = nx.Graph | npt.ArrayLike | sp.sparray | sp.spmatrix | pd.DataFrame


#: Snapshots of a temporal network: ``{time: graph}`` with mutually comparable
#: times, or a list of graphs read as times ``0, ..., T - 1``.
type Snapshots = Mapping[Any, nx.Graph] | Sequence[nx.Graph]

#: A temporal network in any of the three forms of
#: :mod:`nx_backbone.temporal`: snapshots, or a multigraph of events
#: (``time=``) or spans (``start=``/``end=``).
type TemporalNetwork = Snapshots | nx.MultiGraph

#: A time step (``freq=``, ``duration=``, ``gap=``): a number for numeric
#: times, a :class:`datetime.timedelta` for datetimes, or a pandas frequency
#: string or offset (pandas only).
type Frequency = float | datetime.timedelta | str | pd.DateOffset


class IdentityDecorator(Protocol):
    """A decorator that returns a function of the same type."""

    def __call__[F: Callable[..., Any]](self, func: F, /) -> F: ...


class Dispatchable[**P, R](Protocol):
    """A function made dispatchable to NetworkX backends.

    Called with its own arguments, or with ``backend="..."`` (plus any
    backend-specific keyword arguments), as ``nx._dispatchable`` allows.
    """

    @overload
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    @overload
    def __call__(self, *args: Any, backend: str, **backend_kwargs: Any) -> R: ...


class DispatchDecorator(Protocol):
    """The decorator ``_dispatchable(edge_attrs=...)`` returns."""

    def __call__[**P, R](self, func: Callable[P, R], /) -> Dispatchable[P, R]: ...


if TYPE_CHECKING:

    def not_implemented_for(*graph_types: str) -> IdentityDecorator: ...

    def py_random_state(random_state_argument: str | int) -> IdentityDecorator: ...

else:
    from networkx.utils import not_implemented_for, py_random_state
