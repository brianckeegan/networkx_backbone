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

from collections.abc import Callable, Hashable, Iterable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import random

    import networkx as nx
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    import scipy.sparse as sp

__all__ = [
    "GraphInput",
    "Hyperedge",
    "Hyperedges",
    "IdentityDecorator",
    "PySeed",
    "Seed",
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


class IdentityDecorator(Protocol):
    """A decorator that returns a function of the same type."""

    def __call__[F: Callable[..., Any]](self, func: F, /) -> F: ...


if TYPE_CHECKING:

    def not_implemented_for(*graph_types: str) -> IdentityDecorator: ...

    def py_random_state(random_state_argument: str | int) -> IdentityDecorator: ...

else:
    from networkx.utils import not_implemented_for, py_random_state
