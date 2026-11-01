"""
Temporal networks: input forms and conversions between them.

NetworkX has no temporal graph class, so ``networkx_backbone.temporal``
accepts a network observed over time in any of three forms, all built from
ordinary NetworkX graphs:

- **Snapshots**: a dict ``{t: G_t}`` of :class:`~networkx.Graph` or
  :class:`~networkx.DiGraph` objects whose keys are mutually comparable times,
  or a list ``[G_0, ..., G_{T-1}]``, read as times ``0, ..., T - 1``.  All
  snapshots are directed or all are undirected; multigraph snapshots are
  rejected.  A node is present at time ``t`` if it is in ``G_t``, isolated or
  not.
- **Events**: one :class:`~networkx.MultiGraph` or
  :class:`~networkx.MultiDiGraph` with one edge per event.  Each edge carries
  its time in the ``time`` attribute, plus any other attributes such as
  ``weight``.
- **Spans**: one :class:`~networkx.MultiGraph` or
  :class:`~networkx.MultiDiGraph` with one edge per contact that lasts.  Each
  edge carries ``start`` and ``end`` attributes.  A span covers the
  **half-open interval** ``[start, end)``: it includes its start and excludes
  its end, so a span that ends at 5 and one that starts at 5 touch without
  overlapping.  A span with ``start == end`` is an instantaneous contact at
  ``start``.

Every function takes the attribute names as keywords (``time=``, ``start=``,
``end=``), and :func:`~networkx_backbone.temporal_form` tells the three forms apart.

**Binning.** Events and spans become snapshots by binning time into steps of
width ``freq=``, starting at ``origin=`` (default: the earliest time).  Bin
``k`` is the half-open interval ``[origin + k * freq, origin + (k + 1) *
freq)`` and its snapshot is keyed by its start.  An event belongs to the bin
that contains its time; a span belongs to every bin it overlaps (its interval
and the bin's intersect), so a span from 0 to 10 is observed in each of five
bins of width 2.  Snapshots run from the first to the last occupied bin;
empty bins in between are kept as empty graphs.  The parallel edges of a node
pair within one bin collapse into one edge with
:func:`~networkx_backbone.multigraph_to_weighted` (``aggregate="sum"`` by
default): its ``weight`` combines their weights (a missing weight counts as
1) and ``edge_count`` counts them.  Attributes on which all of them agree are
kept.  A node is present in a bin only if it has an event or span there.

**Times.** Plain numbers work with NetworkX alone, as do
:class:`datetime.datetime` times with a :class:`datetime.timedelta` step.
pandas is needed only for frequency strings such as ``freq="1D"`` (or pandas
offsets such as ``"MS"``, which step by calendar months) and for the
``DataFrame`` functions :func:`~networkx_backbone.from_event_table` and :func:`~networkx_backbone.to_event_table`;
it is imported lazily, only on those paths.  pandas ``Timestamp`` and
``Timedelta`` values work wherever datetimes do.

**Round trips.** Converting snapshots to events or spans loses what an edge
list cannot hold: empty snapshots, isolated nodes, and per-snapshot node and
graph attributes.  :func:`~networkx_backbone.events_from_snapshots` and
:func:`~networkx_backbone.spans_from_snapshots` therefore store that information in the graph
attribute ``M.graph["temporal"]``, a dict with keys ``"form"`` (``"events"``
or ``"spans"``), ``"times"`` (the snapshot times in order), ``"snapshots"``
(for each time, the snapshot's graph attributes, ``"graph"``, and its nodes
with their attributes, ``"nodes"``) and, for spans, ``"end_time"`` (when the
last snapshot ends).  :func:`~networkx_backbone.snapshots_from_events` and
:func:`~networkx_backbone.snapshots_from_spans` called without ``freq`` and ``origin`` use this
record to return the original snapshots exactly (as a dict, up to the order
of edges): the same times, nodes, edges, and node, edge and graph attributes.
The record describes the snapshots as they were converted.  If the events or
spans no longer fit it (a time that is not a recorded snapshot time, or two
edges of one node pair in one snapshot), it is ignored and the input is
binned as usual.  The other directions aggregate and are lossy; each
function's Notes say what is kept.
"""

from __future__ import annotations

import sys
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from datetime import timedelta
from itertools import pairwise
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx

from networkx_backbone._docstrings import append_complexity_docstrings
from networkx_backbone.convert import _is_dataframe
from networkx_backbone.filters import multigraph_to_weighted

if TYPE_CHECKING:
    import pandas as pd

    from networkx_backbone._typing import Frequency, Snapshots, TemporalNetwork

__all__ = [
    "temporal_form",
    "convert_temporal",
    "snapshots_from_events",
    "snapshots_from_spans",
    "events_from_snapshots",
    "spans_from_snapshots",
    "events_from_spans",
    "spans_from_events",
    "from_event_table",
    "to_event_table",
]

#: Graph attribute holding the snapshot record of a converted multigraph.
RECORD = "temporal"

_FORMS = ("snapshots", "events", "spans")
_AGGREGATES = ("count", "sum", "max", "min", "mean")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _pandas(why):
    """Import pandas, or explain why it is needed."""
    try:
        import pandas as pd
    except ImportError as err:
        raise ImportError(
            f"pandas is required for {why}; install it with `pip install pandas`."
        ) from err
    return pd


def _missing(x):
    """True for None, NaN, NaT and pandas NA."""
    if x is None:
        return True
    try:
        return bool(x != x)
    except (TypeError, ValueError):
        pd = sys.modules.get("pandas")
        return pd is not None and x is pd.NA


def _same(a, b):
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _pair(u, v, directed):
    return (u, v) if directed else frozenset((u, v))


def _sorted(values, what):
    try:
        return sorted(values)
    except TypeError as err:
        raise TypeError(f"The {what} must be mutually comparable times.") from err


def _check_aggregate(aggregate):
    if aggregate not in _AGGREGATES:
        raise ValueError(
            f"aggregate must be one of {', '.join(map(repr, _AGGREGATES))}; "
            f"got {aggregate!r}."
        )


def _width(freq, name="freq", allow_zero=False):
    """Return a step as a fixed width (number or timedelta) or a pandas offset."""
    pd = sys.modules.get("pandas")
    if isinstance(freq, str) or (pd is not None and isinstance(freq, pd.DateOffset)):
        pd = _pandas(f"a frequency string or offset such as {name}={freq!r}")
        try:
            offset = pd.tseries.frequencies.to_offset(freq)
        except ValueError as err:
            raise ValueError(f"{name}={freq!r} is not a pandas frequency.") from err
        try:
            nanos = offset.nanos
        except (ValueError, TypeError, AttributeError):
            return offset  # a calendar offset: months, anchored weeks, ...
        if nanos < 0 or (nanos == 0 and not allow_zero):
            raise ValueError(f"{name} must be positive; got {freq!r}.")
        if nanos % 1000:
            return offset
        return timedelta(microseconds=nanos // 1000)
    if isinstance(freq, bool) or not isinstance(freq, (Real, timedelta)):
        raise TypeError(
            f"{name} must be a number, a datetime.timedelta, or a pandas "
            f"frequency string such as '1D'; got {freq!r}."
        )
    zero = timedelta(0) if isinstance(freq, timedelta) else 0
    if freq < zero or (freq == zero and not allow_zero):
        raise ValueError(f"{name} must be positive; got {freq!r}.")
    return freq


def _is_fixed(width):
    return isinstance(width, (Real, timedelta))


def _shift(t, width):
    """``t + width``, treating a zero width as no shift (any time type)."""
    if _is_fixed(width) and not width:
        return t
    try:
        return t + width
    except TypeError as err:
        raise TypeError(
            f"Cannot add the step {width!r} to the time {t!r}: use numeric "
            "steps with numeric times and timedeltas (or frequency strings) "
            "with datetimes."
        ) from err


def _bin_starts(lo, hi, width, origin):
    """Starts of the bins of width *width* from the one holding *lo* to *hi*."""
    if origin is None:
        origin = lo
    try:
        if _is_fixed(width):
            k0 = int((lo - origin) // width)
            k1 = int((hi - origin) // width)
            starts = [origin + k * width for k in range(k0, k1 + 1)]
            while starts[0] > lo:  # floating-point rounding
                k0 -= 1
                starts.insert(0, origin + k0 * width)
            return starts
        if origin > lo:
            raise ValueError(
                "origin must not be later than the earliest time when freq is "
                "a calendar offset."
            )
        starts = [origin]
        while True:
            nxt = starts[-1] + width
            if not nxt > starts[-1]:
                raise ValueError(f"freq={width!r} does not move time forward.")
            if nxt > hi:
                break
            starts.append(nxt)
        return starts[bisect_right(starts, lo) - 1 :]
    except TypeError as err:
        raise TypeError(
            f"Cannot bin times such as {lo!r} with freq={width!r} and "
            f"origin={origin!r}: use numeric steps with numeric times and "
            "timedeltas (or frequency strings) with datetimes."
        ) from err


def _check_multigraph(M, form):
    if not isinstance(M, nx.Graph):
        raise TypeError(
            f"{form.capitalize()} input must be a MultiGraph or MultiDiGraph; "
            f"got {type(M).__name__}."
        )
    if not M.is_multigraph():
        raise nx.NetworkXNotImplemented(
            f"not implemented for graph type: {form} input is a MultiGraph or "
            "MultiDiGraph with one edge per "
            f"{'event' if form == 'events' else 'span'} (nx.MultiGraph(G) "
            "converts a simple graph)."
        )


def _read_edges(M, names, form):
    """``[(u, v, data, value, ...)]`` for the attributes *names* (attr, keyword)."""
    rows = []
    for u, v, d in M.edges(data=True):
        values = []
        for attr, keyword in names:
            x = d.get(attr)
            if _missing(x):
                raise ValueError(
                    f"Edge ({u!r}, {v!r}) has no {attr!r} attribute; every edge "
                    f"of {form} input needs one (name it with {keyword}=)."
                )
            values.append(x)
        rows.append((u, v, d, *values))
    if form == "spans":
        for u, v, _, s, e in rows:
            try:
                backwards = e < s
            except TypeError as err:
                raise TypeError(
                    f"Edge ({u!r}, {v!r}) has start {s!r} and end {e!r}, which "
                    "cannot be compared."
                ) from err
            if backwards:
                raise ValueError(
                    f"Edge ({u!r}, {v!r}) ends ({e!r}) before it starts ({s!r})."
                )
    return rows


def _check_clash(d, names, u, v):
    for attr, keyword in names:
        if attr in d:
            raise ValueError(
                f"Edge ({u!r}, {v!r}) already has an attribute {attr!r}; pass "
                f"{keyword}= another name for the {keyword} attribute."
            )


def _snapshots(S):
    """Validated ``(sorted [(t, G_t)], is_list)`` for snapshot input."""
    if isinstance(S, Mapping):
        items, is_list = list(S.items()), False
    elif isinstance(S, Sequence) and not isinstance(S, (str, bytes)):
        items, is_list = list(enumerate(S)), True
    else:
        raise TypeError(
            "Expected snapshots (a dict {time: graph} or a list of graphs), or "
            f"a MultiGraph of events or spans; got {type(S).__name__}."
        )
    if not items:
        raise ValueError("There are no snapshots.")
    for t, G in items:
        if not isinstance(G, nx.Graph):
            raise TypeError(
                f"The snapshot at time {t!r} is a {type(G).__name__}, not a "
                "NetworkX graph."
            )
        if G.is_multigraph():
            raise nx.NetworkXNotImplemented(
                "not implemented for multigraph type: snapshots must be Graph "
                f"or DiGraph objects (the snapshot at time {t!r} is a "
                f"{type(G).__name__}); collapse parallel edges with "
                "multigraph_to_weighted."
            )
        if _missing(t):
            raise ValueError("Snapshot times must not be missing (None or NaN).")
    if len({G.is_directed() for _, G in items}) > 1:
        raise nx.NetworkXError("Snapshots must be all directed or all undirected.")
    order = _sorted([t for t, _ in items], "snapshot keys")
    lookup = dict(items)
    return [(t, lookup[t]) for t in order], is_list


def _record(M, form):
    """The snapshot record of *M* if it was made from snapshots in *form*."""
    rec = M.graph.get(RECORD)
    if (
        isinstance(rec, Mapping)
        and rec.get("form") == form
        and isinstance(rec.get("times"), Sequence)
        and isinstance(rec.get("snapshots"), Sequence)
        and len(rec["times"]) == len(rec["snapshots"])
    ):
        return rec
    return None


def _make_record(form, items):
    return {
        "form": form,
        "times": [t for t, _ in items],
        "snapshots": [
            {
                "graph": dict(G.graph),
                "nodes": {n: dict(a) for n, a in G.nodes(data=True)},
            }
            for _, G in items
        ],
    }


def _multigraph_from_snapshots(items):
    """Empty multigraph with every snapshot node (latest attributes)."""
    directed = items[0][1].is_directed()
    M = nx.MultiDiGraph() if directed else nx.MultiGraph()
    latest = {}
    for _, G in items:
        for n, a in G.nodes(data=True):
            latest[n] = a
    M.add_nodes_from((n, dict(a)) for n, a in latest.items())
    M.graph.update(items[-1][1].graph)
    return M


def _multigraph_like(M):
    """Empty multigraph of M's class with its nodes and its graph attributes."""
    H = M.__class__()
    H.graph.update((k, v) for k, v in M.graph.items() if k != RECORD)
    H.add_nodes_from((n, dict(a)) for n, a in M.nodes(data=True))
    return H


def _without(d, drop):
    return {a: x for a, x in d.items() if a not in drop}


def _common(datas, drop):
    first = datas[0]
    return {
        a: x
        for a, x in first.items()
        if a not in drop and all(a in d and _same(d[a], x) for d in datas[1:])
    }


def _aggregate(groups, drop, weight, aggregate):
    """Attributes of each group of parallel edges, via multigraph_to_weighted.

    Each group is a list of edge-data dicts.  Group ``i`` becomes the parallel
    edges between the stand-in nodes ``2i`` and ``2i + 1`` of one multigraph,
    so a single call to multigraph_to_weighted collapses them all.
    """
    Q = nx.MultiGraph()
    for i, datas in enumerate(groups):
        Q.add_edges_from((2 * i, 2 * i + 1, _without(d, drop)) for d in datas)
    W = multigraph_to_weighted(Q, weight=weight, aggregate=aggregate)
    keep = set(drop) | {weight, "edge_count"}
    out = []
    for i, datas in enumerate(groups):
        attrs = _common(datas, keep)
        attrs.update(W[2 * i][2 * i + 1])
        out.append(attrs)
    return out


def _restore(M, rec, placed, drop):
    """The recorded snapshots with *placed* edges, or None if they do not fit."""
    directed = M.is_directed()
    times = list(rec["times"])
    seen = set()
    for u, v, _, k0, k1 in placed:
        if k0 is None or k1 is None or k1 < k0:
            return None
        for k in range(k0, k1 + 1):
            key = (k, _pair(u, v, directed))
            if key in seen:
                return None
            seen.add(key)
    cls = nx.DiGraph if directed else nx.Graph
    out = {}
    for t, snap in zip(times, rec["snapshots"]):
        G = cls()
        G.graph.update(snap["graph"])
        G.add_nodes_from((n, dict(a)) for n, a in snap["nodes"].items())
        out[t] = G
    for u, v, d, k0, k1 in placed:
        attrs = _without(d, drop)
        for k in range(k0, k1 + 1):
            G = out[times[k]]
            for n in (u, v):
                if n not in G:
                    G.add_node(n, **M.nodes[n])
            G.add_edge(u, v, **attrs)
    return out


def _binned(M, placed, starts, drop, weight, aggregate):
    """Snapshots of the bins in *starts* from ``(u, v, data, k0, k1)`` edges."""
    directed = M.is_directed()
    groups = {}
    present = {}
    for u, v, d, k0, k1 in placed:
        for k in range(k0, k1 + 1):
            groups.setdefault((k, _pair(u, v, directed)), (u, v, []))[2].append(d)
            nodes = present.setdefault(k, {})
            nodes[u] = nodes[v] = None
    if not groups:
        return {}
    keys = list(groups)
    attrs = _aggregate([groups[key][2] for key in keys], drop, weight, aggregate)
    cls = nx.DiGraph if directed else nx.Graph
    base = {a: x for a, x in M.graph.items() if a != RECORD}
    out = {}
    for k in range(min(present), max(present) + 1):
        G = cls()
        G.graph.update(base)
        G.add_nodes_from((n, dict(M.nodes[n])) for n in present.get(k, ()))
        out[starts[k]] = G
    for key, a in zip(keys, attrs):
        u, v, _ = groups[key]
        out[starts[key[0]]].add_edge(u, v, **a)
    return out


def _final_end(times, freq, end_time, is_list):
    """When the last snapshot ends."""
    last = times[-1]
    if end_time is not None:
        if freq is not None:
            raise ValueError("Pass freq= or end_time=, not both.")
        try:
            later = end_time > last
        except TypeError as err:
            raise TypeError(
                f"end_time={end_time!r} cannot be compared with the snapshot "
                f"time {last!r}."
            ) from err
        if not later:
            raise ValueError(
                f"end_time={end_time!r} must be later than the last snapshot "
                f"time {last!r}."
            )
        return end_time
    if freq is not None:
        return _shift(last, _width(freq))
    if is_list:
        return last + 1
    steps = {b - a for a, b in pairwise(times)}
    if len(steps) == 1:
        return last + steps.pop()
    raise ValueError(
        "Cannot tell when the last snapshot ends: "
        + (
            "there is only one snapshot"
            if len(times) == 1
            else "the snapshot times are not evenly spaced"
        )
        + ". Pass freq= (the last snapshot lasts one step) or end_time=."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def temporal_form(
    T: TemporalNetwork,
    time: str | None = "time",
    start: str | None = "start",
    end: str | None = "end",
) -> Literal["snapshots", "events", "spans"]:
    """Return the form of a temporal network: snapshots, events, or spans.

    Parameters
    ----------
    T : dict, list, MultiGraph or MultiDiGraph
        Snapshots (a dict ``{time: graph}`` or a list of graphs), or a
        multigraph of events or spans.
    time : str or None, optional (default="time")
        Edge attribute holding an event's time.  ``None`` never reads *T* as
        events.
    start, end : str or None, optional (default="start", "end")
        Edge attributes holding a span's start and end.  ``None`` never reads
        *T* as spans.

    Returns
    -------
    form : {"snapshots", "events", "spans"}
        ``"events"`` when every edge has a *time*, ``"spans"`` when every
        edge has a *start* and an *end*.  A multigraph without edges is read
        from its ``graph["temporal"]`` record when it has one.

    Raises
    ------
    TypeError
        If *T* is neither a graph nor a dict or list of graphs, or a snapshot
        is not a graph.
    ValueError
        If there are no snapshots, or a snapshot time is missing.
    NetworkXError
        If *T* is a simple graph; if its edges have both the event and the
        span attributes, or neither on every edge, or (no edges) its form is
        unknown; or if snapshots mix directed and undirected graphs.
    NetworkXNotImplemented
        If a snapshot is a multigraph.

    See Also
    --------
    convert_temporal

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import temporal_form
    >>> temporal_form([nx.path_graph(3), nx.path_graph(4)])
    'snapshots'
    >>> M = nx.MultiGraph([(0, 1, {"time": 3}), (0, 1, {"time": 5})])
    >>> temporal_form(M)
    'events'
    >>> M = nx.MultiGraph([(0, 1, {"t0": 3, "t1": 7})])
    >>> temporal_form(M, start="t0", end="t1")
    'spans'
    """
    if not isinstance(T, nx.Graph):
        _snapshots(T)
        return "snapshots"
    if not T.is_multigraph():
        raise nx.NetworkXError(
            "A single simple graph is not a temporal network. Pass a list or "
            "dict of snapshots ([G] for one snapshot), or a MultiGraph of "
            "events or spans."
        )
    m = T.number_of_edges()
    if m == 0:
        rec = T.graph.get(RECORD)
        form = rec.get("form") if isinstance(rec, Mapping) else None
        if form in ("events", "spans"):
            return form
        raise nx.NetworkXError(
            "The multigraph has no edges, so it cannot be read as events or spans."
        )
    n_time = n_span = 0
    for _, _, d in T.edges(data=True):
        n_time += time is not None and not _missing(d.get(time))
        n_span += (
            start is not None
            and end is not None
            and not _missing(d.get(start))
            and not _missing(d.get(end))
        )
    if n_time == m and n_span == m:
        raise nx.NetworkXError(
            f"Every edge has both a {time!r} attribute and {start!r}/{end!r} "
            "attributes, so it could be events or spans. Pass time=None to "
            "read spans, or start=None to read events."
        )
    if n_time == m:
        return "events"
    if n_span == m:
        return "spans"
    raise nx.NetworkXError(
        f"Neither events nor spans: {n_time} of {m} edges have a {time!r} "
        f"attribute and {n_span} of {m} have {start!r} and {end!r}. Name the "
        "attributes with time=, start= and end=."
    )


def snapshots_from_events(
    M: nx.MultiGraph,
    time: str = "time",
    freq: Frequency | None = None,
    origin: Any = None,
    weight: str = "weight",
    aggregate: str = "sum",
) -> dict[Any, nx.Graph]:
    """Bin timestamped events into snapshots.

    Parameters
    ----------
    M : MultiGraph or MultiDiGraph
        One edge per event, with its time in the *time* attribute.
    time : str, optional (default="time")
        Edge attribute holding each event's time.
    freq : number, timedelta, str or None, optional (default=None)
        Width of each bin: a number for numeric times, a
        :class:`datetime.timedelta` or a pandas frequency string such as
        ``"1D"`` for datetimes.  ``None`` makes one snapshot per distinct
        time.
    origin : time or None, optional (default=None)
        Start of bin 0 (default: the earliest time).  Only used with *freq*.
    weight : str, optional (default="weight")
        Edge attribute combined across the events of a node pair in one bin.
    aggregate : {"sum", "count", "max", "min", "mean"}, optional (default="sum")
        How those weights combine, as in
        :func:`~networkx_backbone.multigraph_to_weighted`.

    Returns
    -------
    snapshots : dict
        ``{bin start: Graph or DiGraph}`` in time order, from the first to the
        last bin with an event, empty bins included.  Each graph has the
        nodes with an event in its bin (with their attributes from *M*) and
        *M*'s graph attributes.  Each edge has ``weight``, ``edge_count``
        (the number of events) and the attributes its events share.  If *M*
        carries a ``graph["temporal"]`` record from
        :func:`events_from_snapshots` and *freq* and *origin* are ``None``,
        the original snapshots are returned instead (see Notes).

    Raises
    ------
    NetworkXNotImplemented
        If *M* is not a multigraph.
    ValueError
        If an event has no time, *freq* is not positive, *origin* is given
        without *freq*, or *aggregate* is unknown.
    TypeError
        If the times and *freq* cannot be combined.
    ImportError
        If *freq* is a string and pandas is not installed.

    See Also
    --------
    events_from_snapshots, snapshots_from_spans, convert_temporal

    Notes
    -----
    Bin ``k`` is the half-open interval ``[origin + k * freq, origin + (k +
    1) * freq)``.  For a calendar frequency such as ``"MS"`` (pandas only),
    bins step by calendar periods from *origin*.

    Lossless round trip: ``snapshots_from_events(events_from_snapshots(S))``
    returns *S* as a dict (up to the order of edges), with its empty
    snapshots, isolated nodes, and node, edge and graph attributes, by
    reading the ``graph["temporal"]`` record.  The record is ignored when an
    event's time is not a recorded snapshot time or a node pair has two
    events at one time.

    Lossy direction: binning drops the event-level detail (each pair keeps
    one edge per bin), nodes without an event in a bin, and the attributes
    the events of a bin do not share; ``weight`` and ``edge_count`` are
    added.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import snapshots_from_events
    >>> M = nx.MultiGraph()
    >>> M.add_edges_from([("a", "b", {"time": 0}), ("a", "b", {"time": 1}),
    ...                   ("b", "c", {"time": 5})])
    [0, 1, 0]
    >>> S = snapshots_from_events(M, freq=2)
    >>> {t: sorted(G.edges(data="weight")) for t, G in S.items()}
    {0: [('a', 'b', 2)], 2: [], 4: [('b', 'c', 1)]}

    Without *freq*, every distinct time is a snapshot:

    >>> sorted(snapshots_from_events(M))
    [0, 1, 5]
    """
    _check_multigraph(M, "events")
    _check_aggregate(aggregate)
    events = _read_edges(M, [(time, "time")], "events")
    if freq is None and origin is not None:
        raise ValueError("origin= is only used with freq=.")
    rec = _record(M, "events") if freq is None else None
    if rec is not None:
        index = {t: i for i, t in enumerate(rec["times"])}
        placed = [(u, v, d, index.get(t), index.get(t)) for u, v, d, t in events]
        restored = _restore(M, rec, placed, {time})
        if restored is not None:
            return restored
    if not events:
        return {}
    times = [t for *_, t in events]
    if freq is None:
        starts = _sorted(set(times), "event times")
    else:
        lo, hi = min(times), max(times)
        starts = _bin_starts(lo, hi, _width(freq), origin)
    placed = []
    for u, v, d, t in events:
        k = bisect_right(starts, t) - 1
        placed.append((u, v, d, k, k))
    return _binned(M, placed, starts, {time}, weight, aggregate)


def snapshots_from_spans(
    M: nx.MultiGraph,
    start: str = "start",
    end: str = "end",
    freq: Frequency | None = None,
    origin: Any = None,
    weight: str = "weight",
    aggregate: str = "sum",
) -> dict[Any, nx.Graph]:
    """Bin spans (contacts with a start and an end) into snapshots.

    Parameters
    ----------
    M : MultiGraph or MultiDiGraph
        One edge per span, covering ``[start, end)``.
    start, end : str, optional (default="start", "end")
        Edge attributes holding each span's start and end.
    freq : number, timedelta, str or None, optional (default=None)
        Width of each bin (see :func:`snapshots_from_events`).  ``None``
        makes one snapshot per interval between consecutive distinct start
        or end times, so each snapshot holds exactly the spans active
        throughout it.
    origin : time or None, optional (default=None)
        Start of bin 0 (default: the earliest start).  Only used with *freq*.
    weight : str, optional (default="weight")
        Edge attribute combined across the spans of a node pair in one bin.
    aggregate : {"sum", "count", "max", "min", "mean"}, optional (default="sum")
        How those weights combine, as in
        :func:`~networkx_backbone.multigraph_to_weighted`.

    Returns
    -------
    snapshots : dict
        ``{bin start: Graph or DiGraph}`` in time order, from the first to the
        last bin a span overlaps, empty bins included.  A span is observed in
        every bin it overlaps; a span with ``start == end`` only in the bin
        holding its start.  Nodes, edges and attributes are as in
        :func:`snapshots_from_events`.  With a ``graph["temporal"]`` record
        from :func:`spans_from_snapshots` and no *freq* or *origin*, the
        original snapshots are returned.

    Raises
    ------
    NetworkXNotImplemented
        If *M* is not a multigraph.
    ValueError
        If a span has no start or end or ends before it starts, *freq* is not
        positive, *origin* is given without *freq*, or *aggregate* is
        unknown.
    TypeError
        If the times and *freq* cannot be combined.
    ImportError
        If *freq* is a string and pandas is not installed.

    See Also
    --------
    spans_from_snapshots, snapshots_from_events, convert_temporal

    Notes
    -----
    A span's weight is not split across the bins it overlaps: a span of
    weight 3 adds 3 to each of them.

    Lossless round trip: ``snapshots_from_spans(spans_from_snapshots(S))``
    returns *S* as a dict (up to the order of edges), using the
    ``graph["temporal"]`` record.  Lossy direction: as for
    :func:`snapshots_from_events`.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import snapshots_from_spans
    >>> M = nx.MultiGraph()
    >>> M.add_edges_from([("a", "b", {"start": 0, "end": 5}),
    ...                   ("b", "c", {"start": 3, "end": 4})])
    [0, 0]
    >>> S = snapshots_from_spans(M, freq=2)
    >>> {t: sorted(G.edges) for t, G in S.items()}
    {0: [('a', 'b')], 2: [('a', 'b'), ('b', 'c')], 4: [('a', 'b')]}

    Without *freq*, snapshots start wherever a span starts or ends:

    >>> {t: sorted(G.edges) for t, G in snapshots_from_spans(M).items()}
    {0: [('a', 'b')], 3: [('a', 'b'), ('b', 'c')], 4: [('a', 'b')]}
    """
    _check_multigraph(M, "spans")
    _check_aggregate(aggregate)
    spans = _read_edges(M, [(start, "start"), (end, "end")], "spans")
    if freq is None and origin is not None:
        raise ValueError("origin= is only used with freq=.")
    rec = _record(M, "spans") if freq is None else None
    if rec is not None:
        index = {t: i for i, t in enumerate(rec["times"])}
        ends = {**index, rec.get("end_time"): len(index)}
        placed = []
        for u, v, d, s, e in spans:
            k1 = ends.get(e)
            placed.append((u, v, d, index.get(s), None if k1 is None else k1 - 1))
        restored = _restore(M, rec, placed, {start, end})
        if restored is not None:
            return restored
    if not spans:
        return {}
    if freq is None:
        starts = _sorted({s for *_, s, _ in spans} | {e for *_, e in spans}, "times")
    else:
        lo = min(s for *_, s, _ in spans)
        hi = max(e for *_, e in spans)
        starts = _bin_starts(lo, hi, _width(freq), origin)
    placed = []
    for u, v, d, s, e in spans:
        k0 = bisect_right(starts, s) - 1
        k1 = bisect_left(starts, e) - 1 if e > s else k0
        placed.append((u, v, d, k0, k1))
    return _binned(M, placed, starts, {start, end}, weight, aggregate)


def events_from_snapshots(S: Snapshots, time: str = "time") -> nx.MultiGraph:
    """Return one event per edge per snapshot.

    Parameters
    ----------
    S : dict or list of graphs
        Snapshots: ``{time: Graph or DiGraph}``, or a list read as times
        ``0, ..., T - 1``.
    time : str, optional (default="time")
        Edge attribute that stores each event's time (its snapshot's key).

    Returns
    -------
    M : MultiGraph or MultiDiGraph
        One edge per snapshot edge, with that edge's attributes and its
        *time*.  *M* has every node of every snapshot (isolated ones too),
        with its attributes from the latest snapshot that holds it, and the
        last snapshot's graph attributes, plus the ``M.graph["temporal"]``
        record that makes the conversion reversible.

    Raises
    ------
    TypeError
        If *S* is not a dict or list of graphs, or its times are not
        comparable.
    ValueError
        If *S* is empty, a time is missing, or an edge already has a *time*
        attribute.
    NetworkXError
        If the snapshots mix directed and undirected graphs.
    NetworkXNotImplemented
        If a snapshot is a multigraph.

    See Also
    --------
    snapshots_from_events, spans_from_snapshots, convert_temporal

    Notes
    -----
    Lossless: :func:`snapshots_from_events` with no *freq* returns *S* (as a
    dict) from *M*'s ``graph["temporal"]`` record, including empty snapshots,
    isolated nodes and per-snapshot node and graph attributes.  Without the
    record (for example after :func:`to_event_table`), the events alone keep
    the edges, their attributes and times, and the nodes, but not empty
    snapshots, isolated nodes per snapshot, or per-snapshot attributes.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import events_from_snapshots, snapshots_from_events
    >>> S = {2020: nx.Graph([("a", "b")]), 2021: nx.Graph([("a", "b"), ("b", "c")])}
    >>> M = events_from_snapshots(S)
    >>> sorted((u, v, t) for u, v, t in M.edges(data="time"))
    [('a', 'b', 2020), ('a', 'b', 2021), ('b', 'c', 2021)]
    >>> R = snapshots_from_events(M)
    >>> all(nx.utils.graphs_equal(R[t], S[t]) for t in S)
    True
    """
    items, _ = _snapshots(S)
    M = _multigraph_from_snapshots(items)
    for t, G in items:
        for u, v, d in G.edges(data=True):
            _check_clash(d, [(time, "time")], u, v)
            M.add_edge(u, v, **d, **{time: t})
    M.graph[RECORD] = _make_record("events", items)
    return M


def spans_from_snapshots(
    S: Snapshots,
    start: str = "start",
    end: str = "end",
    freq: Frequency | None = None,
    end_time: Any = None,
) -> nx.MultiGraph:
    """Merge each edge's runs of consecutive snapshots into spans.

    Parameters
    ----------
    S : dict or list of graphs
        Snapshots: ``{time: Graph or DiGraph}``, or a list read as times
        ``0, ..., T - 1``.
    start, end : str, optional (default="start", "end")
        Edge attributes that store each span's start and end.
    freq : number, timedelta, str or None, optional (default=None)
        Length of the last snapshot: it ends at its time plus *freq*.
    end_time : time or None, optional (default=None)
        When the last snapshot ends, instead of *freq*.

    Returns
    -------
    M : MultiGraph or MultiDiGraph
        One edge per run: a maximal sequence of consecutive snapshots in which
        the edge is present with the same attributes.  The span starts at the
        run's first snapshot time and ends at the time of the snapshot after
        the run (half-open, so it covers the run's snapshots and no other);
        a run reaching the last snapshot ends when that snapshot ends.  Nodes,
        graph attributes and the ``M.graph["temporal"]`` record are as in
        :func:`events_from_snapshots`; the record also stores
        ``"end_time"``.

    Raises
    ------
    TypeError
        If *S* is not a dict or list of graphs, or its times are not
        comparable.
    ValueError
        If *S* is empty, a time is missing, an edge already has a *start* or
        *end* attribute, or the end of the last snapshot cannot be determined.
    NetworkXError
        If the snapshots mix directed and undirected graphs.
    NetworkXNotImplemented
        If a snapshot is a multigraph.
    ImportError
        If *freq* is a string and pandas is not installed.

    See Also
    --------
    snapshots_from_spans, events_from_snapshots, convert_temporal

    Notes
    -----
    The last snapshot ends at *end_time* if given, else at its time plus
    *freq*, else one step later when *S* is a list (times ``0, ..., T - 1``)
    or its times are evenly spaced.  Otherwise (uneven times, or a single
    snapshot in a dict) a ``ValueError`` asks for *freq* or *end_time*.

    Lossless: :func:`snapshots_from_spans` with no *freq* returns *S* from
    the record.  Splitting runs where an edge's attributes change keeps each
    snapshot's edge attributes, so the spans alone keep the edges, their
    attributes and when they were present.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import spans_from_snapshots, snapshots_from_spans
    >>> S = [nx.Graph([(0, 1)]), nx.Graph([(0, 1), (1, 2)]), nx.Graph([(1, 2)])]
    >>> M = spans_from_snapshots(S)
    >>> sorted((u, v, d["start"], d["end"]) for u, v, d in M.edges(data=True))
    [(0, 1, 0, 2), (1, 2, 1, 3)]
    >>> R = snapshots_from_spans(M)
    >>> all(nx.utils.graphs_equal(R[t], S[t]) for t in range(3))
    True
    """
    items, is_list = _snapshots(S)
    times = [t for t, _ in items]
    stop = _final_end(times, freq, end_time, is_list)
    directed = items[0][1].is_directed()
    names = [(start, "start"), (end, "end")]
    M = _multigraph_from_snapshots(items)

    def close(run, until):
        u, v, i, d = run
        M.add_edge(u, v, **d, **{start: times[i], end: until})

    runs: dict[Any, Any] = {}
    for i, (t, G) in enumerate(items):
        current: dict[Any, Any] = {}
        for u, v, d in G.edges(data=True):
            _check_clash(d, names, u, v)
            key = _pair(u, v, directed)
            run = runs.pop(key, None)
            if run is not None and _same(run[3], d):
                current[key] = run
            else:
                if run is not None:
                    close(run, t)
                current[key] = (u, v, i, d)
        for run in runs.values():
            close(run, t)
        runs = current
    for run in runs.values():
        close(run, stop)
    M.graph[RECORD] = {**_make_record("spans", items), "end_time": stop}
    return M


def events_from_spans(
    M: nx.MultiGraph,
    start: str = "start",
    end: str = "end",
    time: str = "time",
    freq: Frequency | None = None,
    origin: Any = None,
) -> nx.MultiGraph:
    """Turn spans into events: one at each span's start, or one per bin.

    Parameters
    ----------
    M : MultiGraph or MultiDiGraph
        One edge per span, covering ``[start, end)``.
    start, end : str, optional (default="start", "end")
        Edge attributes holding each span's start and end.
    time : str, optional (default="time")
        Edge attribute that stores each event's time.
    freq : number, timedelta, str or None, optional (default=None)
        ``None`` makes one event per span, at its start.  Otherwise, one
        event per bin the span overlaps, at the bin's start, with the bins of
        :func:`snapshots_from_spans`.
    origin : time or None, optional (default=None)
        Start of bin 0 (default: the earliest start).  Only used with *freq*.

    Returns
    -------
    events : MultiGraph or MultiDiGraph
        Same class, nodes and graph attributes as *M* (without a
        ``graph["temporal"]`` record).  Each event has its span's attributes
        except *start* and *end*.

    Raises
    ------
    NetworkXNotImplemented
        If *M* is not a multigraph.
    ValueError
        If a span has no start or end or ends before it starts, a span
        already has a *time* attribute, *freq* is not positive, or *origin*
        is given without *freq*.
    TypeError
        If the times and *freq* cannot be combined.
    ImportError
        If *freq* is a string and pandas is not installed.

    See Also
    --------
    spans_from_events, snapshots_from_spans, convert_temporal

    Notes
    -----
    An event is instantaneous and a span is not, so no choice keeps
    everything.  The two choices here each keep one well-defined thing:

    - ``freq=None`` keeps when each contact began (onset events), the usual
      reading of contact data as events.  Durations are lost.
    - With *freq*, binning the events with the same *freq* and *origin*
      gives the same snapshots, weights included, as binning the spans:
      ``snapshots_from_events(events_from_spans(M, freq=f), freq=f)`` equals
      ``snapshots_from_spans(M, freq=f)``.  Times within a bin are lost.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import events_from_spans
    >>> M = nx.MultiGraph([("a", "b", {"start": 1, "end": 6})])
    >>> [t for *_, t in events_from_spans(M).edges(data="time")]
    [1]
    >>> [t for *_, t in events_from_spans(M, freq=2, origin=0).edges(data="time")]
    [0, 2, 4]
    """
    _check_multigraph(M, "spans")
    spans = _read_edges(M, [(start, "start"), (end, "end")], "spans")
    if freq is None and origin is not None:
        raise ValueError("origin= is only used with freq=.")
    H = _multigraph_like(M)
    drop = {start, end}
    if freq is not None and spans:
        lo = min(s for *_, s, _ in spans)
        hi = max(e for *_, e in spans)
        starts = _bin_starts(lo, hi, _width(freq), origin)
    for u, v, d, s, e in spans:
        attrs = _without(d, drop)
        _check_clash(attrs, [(time, "time")], u, v)
        if freq is None:
            H.add_edge(u, v, **attrs, **{time: s})
            continue
        k0 = bisect_right(starts, s) - 1
        k1 = bisect_left(starts, e) - 1 if e > s else k0
        for k in range(k0, k1 + 1):
            H.add_edge(u, v, **attrs, **{time: starts[k]})
    return H


def spans_from_events(
    M: nx.MultiGraph,
    time: str = "time",
    start: str = "start",
    end: str = "end",
    *,
    duration: Frequency,
    gap: Frequency | None = 0,
    weight: str = "weight",
    aggregate: str = "sum",
) -> nx.MultiGraph:
    """Turn events into spans: each lasts *duration*, and close ones merge.

    Parameters
    ----------
    M : MultiGraph or MultiDiGraph
        One edge per event, with its time in the *time* attribute.
    time : str, optional (default="time")
        Edge attribute holding each event's time.
    start, end : str, optional (default="start", "end")
        Edge attributes that store each span's start and end.
    duration : number, timedelta or str
        How long each event lasts: an event at ``t`` covers ``[t, t +
        duration)``.  ``0`` makes instantaneous spans.
    gap : number, timedelta, str or None, optional (default=0)
        Events of the same node pair merge into one span when the next one
        starts no later than *gap* after the current span ends; ``0`` merges
        overlapping and touching spans.  ``None`` never merges.
    weight : str, optional (default="weight")
        Edge attribute combined across the events of a span.
    aggregate : {"sum", "count", "max", "min", "mean"}, optional (default="sum")
        How those weights combine, as in
        :func:`~networkx_backbone.multigraph_to_weighted`.

    Returns
    -------
    spans : MultiGraph or MultiDiGraph
        Same class, nodes and graph attributes as *M* (without a
        ``graph["temporal"]`` record).  Each span runs from its first event's
        time to its last event's time plus *duration*, and has ``weight``,
        ``edge_count`` (its number of events) and the attributes its events
        share.

    Raises
    ------
    NetworkXNotImplemented
        If *M* is not a multigraph.
    ValueError
        If an event has no time, *duration* or *gap* is negative, or
        *aggregate* is unknown.
    TypeError
        If the times and *duration* or *gap* cannot be combined.
    ImportError
        If *duration* or *gap* is a string and pandas is not installed.

    See Also
    --------
    events_from_spans, snapshots_from_events, convert_temporal

    Notes
    -----
    Lossy: event times inside a merged span, and the attributes its events do
    not share, are lost.  With ``gap=None`` each event becomes its own span,
    and :func:`events_from_spans` gives back the event times.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import spans_from_events
    >>> M = nx.MultiGraph()
    >>> M.add_edges_from([("a", "b", {"time": 0}), ("a", "b", {"time": 1}),
    ...                   ("a", "b", {"time": 9})])
    [0, 1, 2]
    >>> S = spans_from_events(M, duration=2)
    >>> sorted((d["start"], d["end"], d["weight"]) for *_, d in S.edges(data=True))
    [(0, 3, 2), (9, 11, 1)]
    """
    _check_multigraph(M, "events")
    _check_aggregate(aggregate)
    events = _read_edges(M, [(time, "time")], "events")
    span = _width(duration, "duration", allow_zero=True)
    slack = None if gap is None else _width(gap, "gap", allow_zero=True)
    directed = M.is_directed()
    by_pair: dict[Any, list[Any]] = {}
    for u, v, d, t in events:
        by_pair.setdefault(_pair(u, v, directed), []).append((u, v, d, t))
    groups = []
    for evs in by_pair.values():
        try:
            evs.sort(key=lambda r: r[3])
        except TypeError as err:
            raise TypeError("The event times must be mutually comparable.") from err
        current = None
        for u, v, d, t in evs:
            stop = _shift(t, span)
            if (
                current is not None
                and slack is not None
                and t <= _shift(current[4], slack)
            ):
                current[2].append(d)
                current[4] = max(current[4], stop)
                continue
            if current is not None:
                groups.append(current)
            current = [u, v, [d], t, stop]
        if current is not None:
            groups.append(current)
    attrs = _aggregate([g[2] for g in groups], {time, start, end}, weight, aggregate)
    H = _multigraph_like(M)
    for (u, v, _, s, e), a in zip(groups, attrs):
        H.add_edge(u, v, **a, **{start: s, end: e})
    return H


_CONVERTERS: dict[tuple[str, str], Any] = {
    ("events", "snapshots"): snapshots_from_events,
    ("spans", "snapshots"): snapshots_from_spans,
    ("snapshots", "events"): events_from_snapshots,
    ("snapshots", "spans"): spans_from_snapshots,
    ("spans", "events"): events_from_spans,
    ("events", "spans"): spans_from_events,
}


def convert_temporal(
    T: TemporalNetwork,
    to: Literal["snapshots", "events", "spans"],
    **kwargs: Any,
) -> dict[Any, nx.Graph] | nx.MultiGraph:
    """Convert a temporal network to snapshots, events, or spans.

    Detects the form of *T* with :func:`temporal_form` and calls the matching
    converter.

    Parameters
    ----------
    T : dict, list, MultiGraph or MultiDiGraph
        Snapshots, events, or spans.
    to : {"snapshots", "events", "spans"}
        The form to return.
    **kwargs
        Passed to the converter: :func:`snapshots_from_events`,
        :func:`snapshots_from_spans`, :func:`events_from_snapshots`,
        :func:`spans_from_snapshots`, :func:`events_from_spans` or
        :func:`spans_from_events`.  ``time``, ``start`` and ``end`` also name
        the attributes used to detect the form.

    Returns
    -------
    converted : dict, MultiGraph or MultiDiGraph
        *T* in the requested form.  If *T* is already in that form, a copy
        (snapshots as a dict of graph copies).

    Raises
    ------
    ValueError
        If *to* is not a known form.
    TypeError
        If *kwargs* are given but *T* is already in the requested form, or
        the converter does not take them.

    See Also
    --------
    temporal_form

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import convert_temporal
    >>> M = nx.MultiGraph([(0, 1, {"time": 0}), (0, 1, {"time": 3})])
    >>> S = convert_temporal(M, to="snapshots", freq=2)
    >>> {t: G.number_of_edges() for t, G in S.items()}
    {0: 1, 2: 1}
    >>> spans = convert_temporal(M, to="spans", duration=1)
    >>> sorted((d["start"], d["end"]) for *_, d in spans.edges(data=True))
    [(0, 1), (3, 4)]
    """
    if to not in _FORMS:
        raise ValueError(f"to must be one of {_FORMS}; got {to!r}.")
    names = {k: kwargs[k] for k in ("time", "start", "end") if k in kwargs}
    form = temporal_form(T, **names)
    if form == to:
        extra = sorted(set(kwargs) - set(names))
        if extra:
            raise TypeError(
                f"T is already {form}; there is nothing to convert, so "
                f"{extra} cannot be used."
            )
        if isinstance(T, nx.Graph):
            return T.copy()
        items, _ = _snapshots(T)
        return {t: G.copy() for t, G in items}
    result: dict[Any, nx.Graph] | nx.MultiGraph = _CONVERTERS[form, to](T, **kwargs)
    return result


def from_event_table(
    df: pd.DataFrame,
    source: str = "source",
    target: str = "target",
    time: str = "time",
    end: str | None = None,
    weight: str | None = None,
    directed: bool = False,
    edge_attrs: bool | Sequence[str] | None = None,
) -> nx.MultiGraph:
    """Return events (or spans) from a pandas table with one row per event.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per event (or span).
    source, target : str, optional (default="source", "target")
        Columns holding the two endpoints.
    time : str, optional (default="time")
        Column holding each event's time, or each span's start when *end*
        is given.  It becomes the edge attribute of the same name.
    end : str or None, optional (default=None)
        Column holding each span's end.  If given, the result is spans with
        ``start=time`` and ``end=end``.
    weight : str or None, optional (default=None)
        Column copied to the edge attribute of the same name.
    directed : bool, optional (default=False)
        Return a :class:`~networkx.MultiDiGraph` (``source`` to ``target``)
        instead of a :class:`~networkx.MultiGraph`.
    edge_attrs : bool, sequence of str or None, optional (default=None)
        Further columns to copy as edge attributes; ``True`` copies every
        other column.

    Returns
    -------
    M : MultiGraph or MultiDiGraph
        One edge per row, in row order, with the *time* (and *end*) attribute
        and the copied columns.  A missing value (``NaN``, ``None``, ``NaT``)
        in a copied column leaves that attribute off the edge.

    Raises
    ------
    TypeError
        If *df* is not a pandas ``DataFrame``.
    ValueError
        If a named column is missing, an endpoint, time or end is missing, or
        a span ends before it starts.

    See Also
    --------
    to_event_table, snapshots_from_events, networkx_backbone.to_graph

    Notes
    -----
    Duplicate rows are kept as parallel edges: two identical rows are two
    events, which :func:`snapshots_from_events` adds up when it bins them.
    (:func:`~networkx_backbone.to_graph` refuses duplicates instead, because a
    simple graph would silently drop them.)  Values come from
    ``Series.tolist()``, so datetime columns give pandas ``Timestamp`` times.

    Examples
    --------
    >>> import pandas as pd  # doctest: +SKIP
    >>> from networkx_backbone import from_event_table
    >>> df = pd.DataFrame({"source": ["a", "a", "b"], "target": ["b", "b", "c"],
    ...                    "time": [0, 1, 5], "w": [2, 3, 1]})  # doctest: +SKIP
    >>> M = from_event_table(df, weight="w")  # doctest: +SKIP
    >>> sorted(M.edges(data=True))  # doctest: +SKIP
    [('a', 'b', {'time': 0, 'w': 2}), ('a', 'b', {'time': 1, 'w': 3}), ('b', 'c', {'time': 5, 'w': 1})]
    """
    if not _is_dataframe(df):
        raise TypeError(
            f"from_event_table expects a pandas DataFrame; got {type(df).__name__}."
        )
    required = [source, target, time] + ([end] if end is not None else [])
    if edge_attrs is True:
        extra = [c for c in df.columns if c not in required and c != weight]
    elif not edge_attrs:
        extra = []
    else:
        extra = list(edge_attrs)
    copied = ([weight] if weight is not None else []) + extra
    missing = [c for c in required + copied if c not in df.columns]
    if missing:
        raise ValueError(
            f"The event table has no column(s) {missing}; name the columns "
            "with source=, target=, time=, end=, weight= and edge_attrs=."
        )
    copied = [c for c in dict.fromkeys(copied) if c not in required]
    columns = {c: df[c].tolist() for c in dict.fromkeys(required + copied)}
    labels = df.index.tolist()
    M: nx.MultiGraph = nx.MultiDiGraph() if directed else nx.MultiGraph()
    for i, label in enumerate(labels):
        for c in required:
            if _missing(columns[c][i]):
                raise ValueError(f"Row {label!r} has no value in column {c!r}.")
        data = {time: columns[time][i]}
        if end is not None:
            s, e = columns[time][i], columns[end][i]
            if e < s:
                raise ValueError(
                    f"Row {label!r} ends ({e!r}) before it starts ({s!r})."
                )
            data[end] = e
        for c in copied:
            x = columns[c][i]
            if not _missing(x):
                data[c] = x
        M.add_edge(columns[source][i], columns[target][i], **data)
    return M


def to_event_table(
    T: TemporalNetwork,
    source: str = "source",
    target: str = "target",
    time: str = "time",
    start: str = "start",
    end: str = "end",
    edge_attrs: bool | Sequence[str] = True,
) -> pd.DataFrame:
    """Return events or spans as a pandas table with one row per edge.

    Parameters
    ----------
    T : dict, list, MultiGraph or MultiDiGraph
        Events, spans, or snapshots (written as the events of
        :func:`events_from_snapshots`).
    source, target : str, optional (default="source", "target")
        Columns for the two endpoints.
    time : str, optional (default="time")
        Edge attribute (and column) holding an event's time.
    start, end : str, optional (default="start", "end")
        Edge attributes (and columns) holding a span's start and end.
    edge_attrs : bool or sequence of str, optional (default=True)
        Edge attributes to write as further columns: ``True`` for all,
        ``False`` for none, or a list of names.

    Returns
    -------
    df : pandas.DataFrame
        Columns *source*, *target*, then *time* (events) or *start* and *end*
        (spans), then the edge attributes in order of first appearance.  Rows
        are sorted by time (stable).  An attribute an edge lacks is ``NaN``.

    Raises
    ------
    ImportError
        If pandas is not installed.
    ValueError
        If an edge attribute is named like the *source* or *target* column.

    See Also
    --------
    from_event_table, events_from_snapshots

    Notes
    -----
    ``from_event_table(to_event_table(M), weight=..., edge_attrs=True)``
    gives back the events or spans of *M*: endpoints, times, ends and edge
    attributes (an integer column with a missing value becomes float).  Node
    attributes, isolated nodes, graph attributes and the
    ``graph["temporal"]`` record are not written.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import to_event_table
    >>> M = nx.MultiGraph([("b", "c", {"time": 5}), ("a", "b", {"time": 0, "w": 2})])
    >>> to_event_table(M)  # doctest: +SKIP
      source target  time    w
    0      a      b     0  2.0
    1      b      c     5  NaN
    """
    pd = _pandas("to_event_table")
    form = temporal_form(T, time=time, start=start, end=end)
    M = T if isinstance(T, nx.Graph) else events_from_snapshots(T, time=time)
    fixed = [start, end] if form == "spans" else [time]
    rows = []
    order: dict[str, None] = {}
    for u, v, d in M.edges(data=True):
        row = {source: u, target: v}
        for a, x in d.items():
            if a in fixed:
                row[a] = x
            elif edge_attrs is True or (edge_attrs and a in edge_attrs):
                if a in (source, target):
                    raise ValueError(
                        f"The edge attribute {a!r} clashes with the {a!r} column; "
                        "rename the columns with source= and target=."
                    )
                row[a] = x
                order[a] = None
        rows.append(row)
    rows.sort(key=lambda r: r[fixed[0]])
    return pd.DataFrame(rows, columns=[source, target, *fixed, *order])


_COMPLEXITY = {
    "temporal_form": {
        "time": "O(m + S log S)",
        "space": "O(S)",
        "notes": "m multigraph edges, S snapshots.",
    },
    "convert_temporal": {
        "time": "that of the converter",
        "space": "that of the converter",
    },
    "snapshots_from_events": {
        "time": "O(n + m log m + B)",
        "space": "O(n + m + B)",
        "notes": "B bins.",
    },
    "snapshots_from_spans": {
        "time": "O(n + m log m + m_B + B)",
        "space": "O(n + m_B + B)",
        "notes": "m_B span-bin overlaps.",
    },
    "events_from_snapshots": {
        "time": "O(S log S + n_S + m_S)",
        "space": "O(n_S + m_S)",
        "notes": "n_S, m_S nodes and edges summed over snapshots.",
    },
    "spans_from_snapshots": {
        "time": "O(S log S + n_S + m_S)",
        "space": "O(n_S + m_S)",
        "notes": "n_S, m_S nodes and edges summed over snapshots.",
    },
    "events_from_spans": {
        "time": "O(n + m log B + m_B)",
        "space": "O(n + m_B)",
        "notes": "m_B span-bin overlaps (m with freq=None).",
    },
    "spans_from_events": {"time": "O(n + m log m)", "space": "O(n + m)"},
    "from_event_table": {"time": "O(r)", "space": "O(r)", "notes": "r rows."},
    "to_event_table": {
        "time": "O(m log m)",
        "space": "O(m)",
        "notes": "Snapshots are first converted to events.",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
