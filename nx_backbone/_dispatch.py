"""NetworkX backend dispatching for the graph-in, graph-out functions.

``nx._dispatchable`` lets a NetworkX backend (nx-parallel, nx-cugraph, ...)
supply its own implementation of a function, or lets the caller pick one with
``backend="..."``.  Every backbone scorer and filter takes a graph and returns
a new graph: a copy of the input, with all of its node, edge and graph
attributes, plus the new score attributes.  So the decorator here always passes
``returns_graph=True`` and ``preserve_all_attrs=True``.

Dispatch names share one global registry with NetworkX itself, where a
duplicate name raises ``KeyError`` at import time.  Two of our function names
(``adamic_adar_index`` and ``resource_allocation_index``) are already NetworkX
link-prediction functions, and NetworkX may add more backbone methods later, so
every dispatch name carries the ``backbone_`` prefix:
``disparity_filter`` dispatches as ``"backbone_disparity_filter"``.  A backend
that implements one of our functions registers it under that name.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, overload

import networkx as nx

if TYPE_CHECKING:
    from collections.abc import Callable

    from nx_backbone._typing import Dispatchable, DispatchDecorator

__all__: list[str] = []

#: Prefix of every dispatch name registered by this package.
DISPATCH_PREFIX = "backbone_"


# The overloads tell type checkers that the decorated function keeps its
# signature and also accepts ``backend=...`` (see ``_typing.Dispatchable``).
@overload
def _dispatchable[**P, R](
    func: Callable[P, R],
    *,
    edge_attrs: str | dict[str, Any] | None = ...,
    node_attrs: str | dict[str, Any] | None = ...,
) -> Dispatchable[P, R]: ...
@overload
def _dispatchable(
    func: None = ...,
    *,
    edge_attrs: str | dict[str, Any] | None = ...,
    node_attrs: str | dict[str, Any] | None = ...,
) -> DispatchDecorator: ...
def _dispatchable(
    func: Callable[..., Any] | None = None,
    *,
    edge_attrs: str | dict[str, Any] | None = None,
    node_attrs: str | dict[str, Any] | None = None,
) -> Any:
    """Make *func* dispatchable to NetworkX backends.

    Parameters
    ----------
    func : callable, optional
        The function to decorate.  When omitted, return a decorator.
    edge_attrs : str or dict, optional
        Passed to ``nx._dispatchable``: the parameter(s) naming the edge
        attribute(s) the function reads.
    node_attrs : str or dict, optional
        Passed to ``nx._dispatchable``: the parameter(s) naming the node
        attribute(s) the function reads.

    Returns
    -------
    callable
        The dispatchable function, registered as
        ``DISPATCH_PREFIX + func.__name__``.
    """
    if func is None:
        return partial(_dispatchable, edge_attrs=edge_attrs, node_attrs=node_attrs)
    return nx._dispatchable(
        func,
        name=DISPATCH_PREFIX + func.__name__,
        edge_attrs=edge_attrs,
        node_attrs=node_attrs,
        preserve_all_attrs=True,
        returns_graph=True,
    )
