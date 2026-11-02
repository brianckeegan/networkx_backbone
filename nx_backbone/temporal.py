"""
Temporal networks: input forms and conversions between them.

NetworkX has no temporal graph class, so ``nx_backbone.temporal``
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
``end=``), and :func:`~nx_backbone.temporal_form` tells the three forms apart.

**Binning.** Events and spans become snapshots by binning time into steps of
width ``freq=``, starting at ``origin=`` (default: the earliest time).  Bin
``k`` is the half-open interval ``[origin + k * freq, origin + (k + 1) *
freq)`` and its snapshot is keyed by its start.  An event belongs to the bin
that contains its time; a span belongs to every bin it overlaps (its interval
and the bin's intersect), so a span from 0 to 10 is observed in each of five
bins of width 2.  Snapshots run from the first to the last occupied bin;
empty bins in between are kept as empty graphs.  The parallel edges of a node
pair within one bin collapse into one edge with
:func:`~nx_backbone.multigraph_to_weighted` (``aggregate="sum"`` by
default): its ``weight`` combines their weights (a missing weight counts as
1) and ``edge_count`` counts them.  Attributes on which all of them agree are
kept.  A node is present in a bin only if it has an event or span there.

**Times.** Plain numbers work with NetworkX alone, as do
:class:`datetime.datetime` times with a :class:`datetime.timedelta` step.
pandas is needed only for frequency strings such as ``freq="1D"`` (or pandas
offsets such as ``"MS"``, which step by calendar months) and for the
``DataFrame`` functions :func:`~nx_backbone.from_event_table` and :func:`~nx_backbone.to_event_table`;
it is imported lazily, only on those paths.  pandas ``Timestamp`` and
``Timedelta`` values work wherever datetimes do.

**Round trips.** Converting snapshots to events or spans loses what an edge
list cannot hold: empty snapshots, isolated nodes, and per-snapshot node and
graph attributes.  :func:`~nx_backbone.events_from_snapshots` and
:func:`~nx_backbone.spans_from_snapshots` therefore store that information in the graph
attribute ``M.graph["temporal"]``, a dict with keys ``"form"`` (``"events"``
or ``"spans"``), ``"times"`` (the snapshot times in order), ``"snapshots"``
(for each time, the snapshot's graph attributes, ``"graph"``, and its nodes
with their attributes, ``"nodes"``) and, for spans, ``"end_time"`` (when the
last snapshot ends).  :func:`~nx_backbone.snapshots_from_events` and
:func:`~nx_backbone.snapshots_from_spans` called without ``freq`` and ``origin`` use this
record to return the original snapshots exactly (as a dict, up to the order
of edges): the same times, nodes, edges, and node, edge and graph attributes.
The record describes the snapshots as they were converted.  If the events or
spans no longer fit it (a time that is not a recorded snapshot time, or two
edges of one node pair in one snapshot), it is ignored and the input is
binned as usual.  The other directions aggregate and are lossy; each
function's Notes say what is kept.

**Temporal backbone scores.** :func:`~nx_backbone.observation_fraction`,
:func:`~nx_backbone.persistence`,
:func:`~nx_backbone.backbone_stability` and
:func:`~nx_backbone.weight_trend` take any of the three forms (events
and spans are binned first, with ``freq=``) and score every edge over the
resulting steps.  An edge's **eligible** steps are those in which both of its
endpoints are present.  In an eligible step an absent edge is an observed
zero (weight 0); in any other step the edge is unknown and is not counted, so
an edge whose endpoint joins late is scored only from the step it joins.
Each scorer returns one static graph: the
:func:`~nx_backbone.flatten_temporal` of the steps (a
:class:`~networkx.Graph`, or :class:`~networkx.DiGraph` for directed input,
with every node of every step and its attributes from the latest step it is
in), restricted to the observed edges, which carry their score attributes.
``H.graph["temporal"]`` records the method, the
input form, the step ``times``, ``n_steps``, ``freq``, ``origin`` and the
parameters.  The usual filters then apply, as in
``threshold_filter(H, "obs_fraction", 0.5, mode="above")``.  The scores are
descriptive: they come without p-values.  All but
:func:`~nx_backbone.weight_trend` work with NetworkX alone (numpy, when
installed, vectorizes them).

**Model-based scores.** :func:`~nx_backbone.edge_persistence_hmm`
(numpy) follows the same input and output conventions.  It treats each edge
as a two-state hidden Markov chain (inactive or active) observed with error
in its eligible steps (the other steps carry no evidence), and adds
posterior activity scores plus a per-step ``hmm_posterior`` dict; the fitted
model is in ``H.graph["temporal"]["hmm"]``.
:func:`~nx_backbone.kalman_backbone` (numpy and SciPy) tests every edge
in every step with a static significance test and smooths each edge's
evidence over time with a Kalman filter and smoother.  It adds an edge-level
``kalman_pvalue`` and per-step series as dict attributes keyed by time;
:func:`~nx_backbone.kalman_snapshots` expands those into one graph per
step, and the fitted model is in ``H.graph["temporal"]["kalman"]``.
"""

from __future__ import annotations

import sys
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from datetime import timedelta
from itertools import pairwise
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, cast

import networkx as nx

from nx_backbone._docstrings import append_complexity_docstrings
from nx_backbone.convert import _is_dataframe
from nx_backbone.filters import multigraph_to_weighted

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from nx_backbone._typing import Frequency, Snapshots, TemporalNetwork

__all__ = [
    "temporal_form",
    "convert_temporal",
    "flatten_temporal",
    "snapshots_from_events",
    "snapshots_from_spans",
    "events_from_snapshots",
    "spans_from_snapshots",
    "events_from_spans",
    "spans_from_events",
    "from_event_table",
    "to_event_table",
    "observation_fraction",
    "persistence",
    "backbone_stability",
    "weight_trend",
    "edge_persistence_hmm",
    "kalman_backbone",
    "kalman_snapshots",
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
    >>> from nx_backbone import temporal_form
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
        :func:`~nx_backbone.multigraph_to_weighted`.

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
    >>> from nx_backbone import snapshots_from_events
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
        :func:`~nx_backbone.multigraph_to_weighted`.

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
    >>> from nx_backbone import snapshots_from_spans
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
    >>> from nx_backbone import events_from_snapshots, snapshots_from_events
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
    >>> from nx_backbone import spans_from_snapshots, snapshots_from_spans
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
    >>> from nx_backbone import events_from_spans
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
        :func:`~nx_backbone.multigraph_to_weighted`.

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
    >>> from nx_backbone import spans_from_events
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
    >>> from nx_backbone import convert_temporal
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


def _window_bounds(window):
    """``(t0, t1)`` of a half-open window; either end may be None (open)."""
    if window is None:
        return None, None
    if (
        isinstance(window, (str, bytes))
        or not isinstance(window, Sequence)
        or len(window) != 2
    ):
        raise TypeError(f"window must be a pair (t0, t1); got {window!r}.")
    t0, t1 = window
    if t0 is not None and t1 is not None:
        try:
            empty = not t0 < t1
        except TypeError as err:
            raise TypeError(
                f"The window bounds {t0!r} and {t1!r} cannot be compared."
            ) from err
        if empty:
            raise ValueError(f"window=({t0!r}, {t1!r}) is empty: t0 must be < t1.")
    return t0, t1


def _in_window(t, t0, t1):
    try:
        return (t0 is None or t >= t0) and (t1 is None or t < t1)
    except TypeError as err:
        raise TypeError(
            f"The time {t!r} cannot be compared with the window bounds."
        ) from err


def _union_length(intervals):
    """Total length of the union of ``[s, e)`` intervals (any time type)."""
    total = None
    cur_s = cur_e = None
    for s, e in sorted(intervals, key=lambda iv: iv[0]):
        if cur_e is not None and s <= cur_e:
            if e > cur_e:
                cur_e = e
            continue
        if cur_e is not None:
            total = cur_e - cur_s if total is None else total + (cur_e - cur_s)
        cur_s, cur_e = s, e
    if cur_e is not None:
        total = cur_e - cur_s if total is None else total + (cur_e - cur_s)
    return total


def flatten_temporal(
    T: TemporalNetwork,
    weight: str = "weight",
    aggregate: str = "sum",
    window: tuple[Any, Any] | None = None,
    time: str = "time",
    start: str = "start",
    end: str = "end",
) -> nx.Graph:
    """Collapse a temporal network into one static weighted graph.

    Every observation of a node pair (an edge in a snapshot, an event, or a
    span) is combined into a single edge whose ``weight`` aggregates the
    observations' weights, and which records how often and when the pair
    was observed.  The result is an ordinary :class:`~networkx.Graph` or
    :class:`~networkx.DiGraph`, so any static backbone method applies to it.

    Parameters
    ----------
    T : dict, list, MultiGraph or MultiDiGraph
        Snapshots, events, or spans (see :func:`temporal_form`).
    weight : str, optional (default="weight")
        Edge attribute holding each observation's weight; a missing weight
        counts as 1.  The aggregated value is stored under the same name.
    aggregate : {"sum", "count", "max", "min", "mean", "duration"}, optional
        How the observations' weights combine, as in
        :func:`~nx_backbone.multigraph_to_weighted`; ``"count"`` gives
        the number of observations.  ``"duration"`` (spans only) gives the
        total time the pair is in contact: the length of the union of its
        spans within *window*, so overlapping spans are not counted twice.
        Default ``"sum"``.
    window : tuple (t0, t1) or None, optional (default=None)
        Keep only what happens in the half-open interval ``[t0, t1)``; either
        bound may be ``None`` for an open end.  Snapshots and events are kept
        when their time falls in the window; spans are clipped to it (a span
        with ``start == end`` is kept when its start falls in the window).
    time : str, optional (default="time")
        Edge attribute holding an event's time.
    start, end : str, optional (default="start", "end")
        Edge attributes holding a span's start and end.

    Returns
    -------
    H : Graph or DiGraph
        One edge per node pair observed in the window, with:

        - *weight*: the aggregated weight;
        - ``n_obs``: the number of observations (snapshots containing the
          edge, events, or spans);
        - ``first`` and ``last``: the earliest and latest observation times
          (for spans, the earliest clipped start and the latest clipped end);
        - ``active_time`` (spans only): the length of the union of the pair's
          clipped spans;
        - the attributes that all of the pair's observations share.

        Nodes are those present in the window, with their attributes
        (from the latest snapshot that holds them, or from *T*); without a
        window every node of *T* is kept, including isolated nodes.  The
        graph attributes come from the latest snapshot in the window (or from
        *T*), and ``H.graph["temporal"]`` records the input form, *aggregate*,
        *window*, and the first and last observation times.

    Raises
    ------
    ValueError
        If *aggregate* is unknown, ``"duration"`` is used with snapshots or
        events, or *window* is empty.
    TypeError
        If *window* is not a pair, or times cannot be compared with it.
    NetworkXError
        If *T* is not a temporal network (see :func:`temporal_form`).

    See Also
    --------
    convert_temporal, multigraph_to_weighted, snapshots_from_events

    Notes
    -----
    To collapse at a coarser resolution, bin first and then flatten: for
    example ``flatten_temporal(snapshots_from_events(M, freq=7))`` counts the
    weeks in which a pair interacts, where ``flatten_temporal(M,
    aggregate="count")`` counts its events.

    The result drops the order and timing of the observations beyond
    ``first``, ``last`` and ``active_time``; use the temporal filters to score
    edges by how they behave over time.

    Examples
    --------
    >>> import networkx as nx
    >>> from nx_backbone import flatten_temporal
    >>> G1 = nx.Graph([("a", "b", {"weight": 2}), ("b", "c", {"weight": 1})])
    >>> G2 = nx.Graph([("a", "b", {"weight": 3})])
    >>> H = flatten_temporal({0: G1, 1: G2})
    >>> H["a"]["b"]
    {'weight': 5, 'n_obs': 2, 'first': 0, 'last': 1}
    >>> flatten_temporal({0: G1, 1: G2}, aggregate="mean")["a"]["b"]["weight"]
    2.5

    Spans can be weighted by how long the pair is in contact:

    >>> M = nx.MultiGraph()
    >>> M.add_edges_from([("a", "b", {"start": 0, "end": 4.5}),
    ...                   ("a", "b", {"start": 9, "end": 9.5})])
    [0, 1]
    >>> H = flatten_temporal(M, aggregate="duration")
    >>> H["a"]["b"]["weight"], H["a"]["b"]["n_obs"]
    (5.0, 2)
    >>> flatten_temporal(M, aggregate="duration", window=(2, 10))["a"]["b"]["weight"]
    3.0
    """
    if aggregate not in (*_AGGREGATES, "duration"):
        raise ValueError(
            f"aggregate must be one of "
            f"{', '.join(map(repr, (*_AGGREGATES, 'duration')))}; got {aggregate!r}."
        )
    t0, t1 = _window_bounds(window)
    form = temporal_form(T, time=time, start=start, end=end)
    if aggregate == "duration" and form != "spans":
        raise ValueError(
            f"aggregate='duration' needs spans (edges with {start!r} and "
            f"{end!r}); the input is {form}."
        )
    windowed = window is not None

    # (u, v, data, first, last, interval or None) per observation in the window.
    obs: list[tuple[Any, Any, dict, Any, Any, tuple[Any, Any] | None]] = []
    nodes: dict[Any, dict] = {}
    if form == "snapshots":
        items, _ = _snapshots(T)
        items = [(t, G) for t, G in items if _in_window(t, t0, t1)]
        directed = (
            items[0][1].is_directed() if items else _snapshots(T)[0][0][1].is_directed()
        )
        for t, G in items:
            for n, a in G.nodes(data=True):
                nodes[n] = a
            obs.extend((u, v, d, t, t, None) for u, v, d in G.edges(data=True))
        graph_attrs = dict(items[-1][1].graph) if items else {}
        drop: set[str] = set()
    else:
        M = cast(nx.MultiGraph, T)
        directed = M.is_directed()
        graph_attrs = {k: x for k, x in M.graph.items() if k != RECORD}
        if form == "events":
            drop = {time}
            for u, v, d, t in _read_edges(M, [(time, "time")], "events"):
                if _in_window(t, t0, t1):
                    obs.append((u, v, d, t, t, None))
        else:
            drop = {start, end}
            for u, v, d, s, e in _read_edges(
                M, [(start, "start"), (end, "end")], "spans"
            ):
                cs = s if t0 is None or s >= t0 else t0
                ce = e if t1 is None or e <= t1 else t1
                if s == e:
                    keep = _in_window(s, t0, t1)
                    cs = ce = s
                else:
                    keep = cs < ce
                if keep:
                    obs.append((u, v, d, cs, ce, (cs, ce)))
        if windowed:
            for u, v, *_ in obs:
                nodes[u] = M.nodes[u]
                nodes[v] = M.nodes[v]
        else:
            nodes = dict(M.nodes(data=True))

    groups: dict[Any, tuple[Any, Any, list]] = {}
    for row in obs:
        u, v = row[0], row[1]
        groups.setdefault(_pair(u, v, directed), (u, v, []))[2].append(row)
    keys = list(groups)
    attrs = _aggregate(
        [[row[2] for row in groups[k][2]] for k in keys],
        drop | {"n_obs", "first", "last", "active_time"},
        weight,
        "sum" if aggregate == "duration" else aggregate,
    )

    H: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    H.graph.update(graph_attrs)
    H.add_nodes_from((n, dict(a)) for n, a in nodes.items())
    first_all = last_all = None
    for k, a in zip(keys, attrs):
        u, v, rows = groups[k]
        a["n_obs"] = a.pop("edge_count")
        a["first"] = min(row[3] for row in rows)
        a["last"] = max(row[4] for row in rows)
        if form == "spans":
            a["active_time"] = _union_length([row[5] for row in rows])
            if aggregate == "duration":
                a[weight] = a["active_time"]
        H.add_edge(u, v, **a)
        if first_all is None or a["first"] < first_all:
            first_all = a["first"]
        if last_all is None or a["last"] > last_all:
            last_all = a["last"]
    H.graph[RECORD] = {
        "flattened_from": form,
        "aggregate": aggregate,
        "window": window,
        "first": first_all,
        "last": last_all,
    }
    return H


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
    to_event_table, snapshots_from_events, nx_backbone.to_graph

    Notes
    -----
    Duplicate rows are kept as parallel edges: two identical rows are two
    events, which :func:`snapshots_from_events` adds up when it bins them.
    (:func:`~nx_backbone.to_graph` refuses duplicates instead, because a
    simple graph would silently drop them.)  Values come from
    ``Series.tolist()``, so datetime columns give pandas ``Timestamp`` times.

    Examples
    --------
    >>> import pandas as pd  # doctest: +SKIP
    >>> from nx_backbone import from_event_table
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
    >>> from nx_backbone import to_event_table
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


# ---------------------------------------------------------------------------
# Temporal backbone scores: shared machinery
# ---------------------------------------------------------------------------
#
# Every scorer below works in three stages:
#
# 1. ``_steps`` normalizes any input form into T snapshots in time order (the
#    conversions above do the binning), so presence is simply "the node is in
#    the step's graph".
# 2. ``_Table`` indexes the edges ever present and builds, over the T steps,
#    a per-edge *present* mask, an *eligible* mask (both endpoints present)
#    and, when asked, a weight series (0 where eligible but absent).  The masks
#    are numpy boolean arrays when numpy is installed and lists of lists of
#    bools otherwise; each scorer that has a pure-Python path implements both.
# 3. ``_scored_graph`` builds the output from ``flatten_temporal`` of the
#    steps: the union of observed edges with the score attributes, and
#    ``H.graph["temporal"]``.


def _numpy():
    """numpy, or None when it is not installed."""
    try:
        import numpy as np
    except ImportError:
        return None
    return np


def _require_numpy(what):
    np = _numpy()
    if np is None:
        raise ImportError(
            f"{what} requires numpy; install it with `pip install numpy`."
        )
    return np


class _Steps:
    """A temporal network as T snapshots in time order.

    ``times[k]`` is the time of step *k*, ``graphs[k]`` its snapshot and
    ``presence[k]`` the nodes present in it: the snapshot's nodes, isolated
    or not (after binning, the nodes with an event or span in the step).
    """

    def __init__(self, form, times, graphs, directed, freq, origin):
        self.form = form
        self.times = times
        self.graphs = graphs
        self.directed = directed
        self.freq = freq
        self.origin = origin

    @property
    def presence(self):
        return [set(G) for G in self.graphs]


def _steps(T, *, freq, origin, time, start, end, weight, aggregate):
    """Normalize snapshots, events or spans into a :class:`_Steps`."""
    form = temporal_form(T, time=time, start=start, end=end)
    if form == "snapshots":
        if freq is not None or origin is not None:
            raise ValueError(
                "freq= and origin= bin events and spans; snapshots are already "
                "steps (pass a dict {time: graph} to give them times)."
            )
        items, _ = _snapshots(T)
        directed = items[0][1].is_directed()
    else:
        convert = snapshots_from_events if form == "events" else snapshots_from_spans
        names = {"time": time} if form == "events" else {"start": start, "end": end}
        S = convert(
            T, **names, freq=freq, origin=origin, weight=weight, aggregate=aggregate
        )
        items = list(S.items())
        directed = T.is_directed()
        if not items:
            raise ValueError(f"The {form} input has no edges, so it has no steps.")
    return _Steps(
        form, [t for t, _ in items], [G for _, G in items], directed, freq, origin
    )


def _weight_value(d, weight, u, v, t):
    """A numeric edge weight; a missing weight counts as 1."""
    x = d.get(weight, 1)
    if isinstance(x, bool) or not isinstance(x, Real) or x != x:
        raise nx.NetworkXError(
            f"Edge ({u!r}, {v!r}) at time {t!r} has a non-numeric or NaN "
            f"{weight!r} attribute: {x!r}."
        )
    return float(x)


class _Table:
    """Per-edge masks and weight series over the T steps.

    ``ends[j]`` is edge *j* as first seen; ``at[k]`` lists the edges present
    in step *k*.  ``present`` and ``eligible`` are ``m x T`` boolean masks
    (numpy arrays, or lists of lists without numpy); ``weights`` (only when
    *weight* is given) holds each edge's weight in the steps it is present
    and 0 elsewhere.
    """

    def __init__(
        self, steps: _Steps, weight: str | None = None, use_numpy: bool = True
    ) -> None:
        self.steps = steps
        self.np = _numpy() if use_numpy else None
        graphs = steps.graphs
        n_steps = len(graphs)
        nodes: dict[Any, int] = {}
        presence = [[nodes.setdefault(n, len(nodes)) for n in G] for G in graphs]
        self.index: dict[Any, int] = {}
        self.ends: list[tuple[Any, Any]] = []
        self.at: list[list[int]] = []
        wts: list[list[float]] = []
        for k, G in enumerate(graphs):
            js = []
            ws = []
            for u, v, d in G.edges(data=True):
                key = _pair(u, v, steps.directed)
                j = self.index.get(key)
                if j is None:
                    j = self.index[key] = len(self.ends)
                    self.ends.append((u, v))
                js.append(j)
                if weight is not None:
                    ws.append(_weight_value(d, weight, u, v, steps.times[k]))
            self.at.append(js)
            wts.append(ws)
        m = len(self.ends)
        ends = [(nodes[u], nodes[v]) for u, v in self.ends]
        np = self.np
        if np is not None:
            P = np.zeros((len(nodes), n_steps), dtype=bool)
            for k, idx in enumerate(presence):
                P[idx, k] = True
            uv = np.array(ends, dtype=np.intp).reshape(m, 2)
            self.eligible = P[uv[:, 0]] & P[uv[:, 1]]
            self.present = self.mask(self.at)
            if weight is not None:
                self.weights = np.zeros((m, n_steps))
                for k, (js, ws) in enumerate(zip(self.at, wts)):
                    self.weights[js, k] = ws
        else:
            sets = [set(idx) for idx in presence]
            self.eligible = [[u in s and v in s for s in sets] for u, v in ends]
            self.present = self.mask(self.at)
            if weight is not None:
                self.weights = [[0.0] * n_steps for _ in range(m)]
                for k, (js, ws) in enumerate(zip(self.at, wts)):
                    for j, w in zip(js, ws):
                        self.weights[j][k] = w

    def mask(self, at):
        """An ``m x T`` mask that is True for the edges ``at[k]`` of step k."""
        m, n_steps = len(self.ends), len(self.at)
        if self.np is not None:
            M = self.np.zeros((m, n_steps), dtype=bool)
            for k, js in enumerate(at):
                M[list(js), k] = True
            return M
        rows = [[False] * n_steps for _ in range(m)]
        for k, js in enumerate(at):
            for j in js:
                rows[j][k] = True
        return rows

    def above(self, min_weight):
        """The present mask restricted to weights above *min_weight*."""
        if self.np is not None:
            return self.present & (self.weights > min_weight)
        return [
            [p and w > min_weight for p, w in zip(prow, wrow)]
            for prow, wrow in zip(self.present, self.weights)
        ]


def _counts(table, mask):
    """Per edge: steps in *mask*, eligible steps, first and last mask step."""
    np = table.np
    if np is not None:
        n = mask.sum(axis=1)
        n_steps = mask.shape[1]
        first = np.where(n > 0, mask.argmax(axis=1), -1)
        last = np.where(n > 0, n_steps - 1 - mask[:, ::-1].argmax(axis=1), -1)
        n_elig = table.eligible.sum(axis=1)
        return n.tolist(), n_elig.tolist(), first.tolist(), last.tolist()
    n, n_elig, first, last = [], [], [], []
    for row, erow in zip(mask, table.eligible):
        ks = [k for k, x in enumerate(row) if x]
        n.append(len(ks))
        n_elig.append(sum(erow))
        first.append(ks[0] if ks else -1)
        last.append(ks[-1] if ks else -1)
    return n, n_elig, first, last


def _runs(table, mask, skip):
    """Per edge: number of runs, longest run, and the stay-active share.

    A run is a maximal sequence of consecutive steps in *mask*.  With *skip*,
    consecutive means consecutive among the edge's eligible steps (ineligible
    steps are passed over); otherwise among all steps.
    """
    np = table.np
    if np is not None:
        m, n_steps = mask.shape
        ks = np.arange(n_steps)
        elig = table.eligible if skip else np.ones_like(mask)
        # Index of each step's previous and next eligible step (-1 / T: none).
        last = np.maximum.accumulate(np.where(elig, ks, -1), axis=1)
        prev = np.concatenate([np.full((m, 1), -1), last[:, :-1]], axis=1)
        later = np.where(elig, ks, n_steps)[:, ::-1]
        first_after = np.minimum.accumulate(later, axis=1)[:, ::-1]
        nxt = np.concatenate([first_after[:, 1:], np.full((m, 1), n_steps)], axis=1)
        # Column T is always False, so index -1 and T both read "not in mask".
        padded = np.concatenate([mask, np.zeros((m, 1), dtype=bool)], axis=1)
        rows = np.arange(m)[:, None]
        starts = mask & ~padded[rows, prev]
        n_runs = starts.sum(axis=1)
        stays = (mask & padded[rows, nxt]).sum(axis=1)
        moves = (mask & (nxt < n_steps)).sum(axis=1)
        run_of_cell = np.cumsum(starts.ravel()) - 1
        lengths = np.bincount(run_of_cell[mask.ravel()], minlength=int(n_runs.sum()))
        max_run = np.zeros(m, dtype=np.intp)
        np.maximum.at(max_run, np.nonzero(starts)[0], lengths)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(moves > 0, stays / np.maximum(moves, 1), np.nan)
        return n_runs.tolist(), max_run.tolist(), share.tolist()
    n_runs, max_run, share = [], [], []
    for row, erow in zip(mask, table.eligible):
        seq = [x for x, e in zip(row, erow) if e or not skip]
        runs, cur = [], 0
        for x in seq:
            if x:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        moves = sum(1 for x in seq[:-1] if x)
        stays = sum(1 for a, b in pairwise(seq) if a and b)
        n_runs.append(len(runs))
        max_run.append(max(runs, default=0))
        share.append(stays / moves if moves else float("nan"))
    return n_runs, max_run, share


def _scored_graph(table, keep, attrs, method, parameters, weight):
    """The union of the edges *keep*, with their score attributes.

    The union comes from :func:`flatten_temporal` of the steps, so nodes,
    node and graph attributes, and each edge's summed *weight*, ``n_obs``,
    ``first`` and ``last`` follow its conventions; edge *j* (``keep[j]``
    true) then gets ``attrs[j]``, and the other edges are removed.
    """
    steps = table.steps
    H = flatten_temporal(dict(zip(steps.times, steps.graphs)), weight=weight)
    for j, (u, v) in enumerate(table.ends):
        if keep[j]:
            H.edges[u, v].update(attrs[j])
        else:
            H.remove_edge(u, v)
    H.graph[RECORD] = {
        "method": method,
        "input_form": steps.form,
        "times": list(steps.times),
        "n_steps": len(steps.times),
        "freq": steps.freq,
        "origin": steps.origin,
        "parameters": parameters,
    }
    return H


def _time(steps, k):
    return steps.times[k] if k >= 0 else None


# ---------------------------------------------------------------------------
# Temporal backbone scores: public API
# ---------------------------------------------------------------------------


def observation_fraction(
    T: TemporalNetwork,
    *,
    weight: str | None = None,
    min_weight: float = 0,
    freq: Frequency | None = None,
    origin: Any = None,
    time: str = "time",
    start: str = "start",
    end: str = "end",
    aggregate: str = "sum",
) -> nx.Graph:
    """Score each edge by the share of its eligible steps in which it is observed.

    An edge's *eligible* steps are those in which both of its endpoints are
    present.  In an eligible step the edge is either observed or an observed
    zero; in the other steps it is unknown and not counted.

    Parameters
    ----------
    T : dict, list, MultiGraph or MultiDiGraph
        A temporal network: snapshots (a dict ``{time: graph}`` or a list of
        graphs), or a multigraph of events or spans (see
        :mod:`nx_backbone.temporal`).
    weight : str or None, optional (default=None)
        If given, an edge counts as observed only when its *weight* is above
        *min_weight* (a missing weight counts as 1).  ``None`` counts every
        edge present in a step.
    min_weight : float, optional (default=0)
        The weight an observed edge must exceed when *weight* is given.
    freq : number, timedelta, str or None, optional (default=None)
        Step width for binning events or spans into snapshots, as in
        :func:`snapshots_from_events`: a number for numeric times, a
        :class:`datetime.timedelta` or a pandas frequency string such as
        ``"1D"`` (pandas only) for datetimes.  ``None`` makes one step per
        distinct time.  Snapshots are already steps, so *freq* must be
        ``None`` for them.
    origin : time or None, optional (default=None)
        Start of the first bin (default: the earliest time); with *freq*
        only.
    time : str, optional (default="time")
        Edge attribute holding an event's time.
    start, end : str, optional (default="start", "end")
        Edge attributes holding a span's start and end.
    aggregate : {"sum", "count", "max", "min", "mean"}, optional (default="sum")
        How the weights of a node pair's events or spans in one bin combine
        (see :func:`~nx_backbone.multigraph_to_weighted`).

    Returns
    -------
    H : Graph or DiGraph
        The union of the edges observed in any step, with the attributes
        :func:`flatten_temporal` gives them over the steps (the summed
        *weight*, ``n_obs``, ``first``, ``last``, and the attributes all
        their observations share), plus

        - ``obs_fraction``: ``n_obs / n_eligible``;
        - ``n_obs``: the number of steps in which the edge is observed
          (replacing the count of :func:`flatten_temporal`, which also
          counts the steps where the weight is at most *min_weight*);
        - ``n_eligible``: the number of steps in which both endpoints are
          present;
        - ``first_obs``, ``last_obs``: the times of the first and last
          observation.

        Nodes are every node present in any step, with their attributes from
        the latest step they are in, and graph attributes come from the
        latest step.  ``H.graph["temporal"]`` records the
        method, the input form, the step times, ``n_steps``, *freq*,
        *origin* and the parameters.

    Raises
    ------
    NetworkXNotImplemented
        If a snapshot is a multigraph.
    NetworkXError
        If *weight* is given and a weight is not numeric, or *T* is not a
        temporal network.
    ValueError
        If *freq* or *origin* is given with snapshots, or the input has no
        steps.

    See Also
    --------
    persistence : Tell steady edges from intermittent ones.
    threshold_filter : Keep the edges above a chosen fraction.

    Notes
    -----
    A node is present in a snapshot if it is in that graph, isolated or not;
    after binning events or spans, a node is present in a step only if it
    has an event or span there.  So an edge whose endpoint joins late is
    scored only on the steps after it joins.

    Examples
    --------
    Node ``"c"`` joins at time 3, so the edge ``("b", "c")`` has two eligible
    steps and is observed in both; ``("a", "b")`` is absent at time 4 while
    both endpoints are present, an observed zero:

    >>> import networkx as nx
    >>> import nx_backbone as nb
    >>> S = {
    ...     1: nx.Graph([("a", "b")]),
    ...     2: nx.Graph([("a", "b")]),
    ...     3: nx.Graph([("a", "b"), ("b", "c")]),
    ...     4: nx.Graph([("b", "c"), ("a", "z")]),
    ... }
    >>> H = nb.observation_fraction(S)
    >>> H["a"]["b"]  # doctest: +NORMALIZE_WHITESPACE
    {'weight': 3, 'n_obs': 3, 'first': 1, 'last': 3, 'obs_fraction': 0.75,
     'n_eligible': 4, 'first_obs': 1, 'last_obs': 3}
    >>> H["b"]["c"]["obs_fraction"]
    1.0
    >>> B = nb.threshold_filter(H, "obs_fraction", 0.8, mode="above")
    >>> sorted(tuple(sorted(e)) for e in B.edges)
    [('a', 'z'), ('b', 'c')]
    """
    steps = _steps(
        T,
        freq=freq,
        origin=origin,
        time=time,
        start=start,
        end=end,
        weight=weight or "weight",
        aggregate=aggregate,
    )
    table = _Table(steps, weight)
    observed = table.present if weight is None else table.above(min_weight)
    n, n_elig, first, last = _counts(table, observed)
    attrs = [
        {
            "obs_fraction": a / b,
            "n_obs": a,
            "n_eligible": b,
            "first_obs": _time(steps, f),
            "last_obs": _time(steps, z),
        }
        if a
        else {}
        for a, b, f, z in zip(n, n_elig, first, last)
    ]
    parameters = {"weight": weight, "min_weight": min_weight, "aggregate": aggregate}
    return _scored_graph(
        table, n, attrs, "observation_fraction", parameters, weight or "weight"
    )


def persistence(
    T: TemporalNetwork,
    *,
    weight: str | None = None,
    min_weight: float = 0,
    ineligible: Literal["skip", "break"] = "skip",
    freq: Frequency | None = None,
    origin: Any = None,
    time: str = "time",
    start: str = "start",
    end: str = "end",
    aggregate: str = "sum",
) -> nx.Graph:
    """Score each edge by its runs of consecutive observed steps.

    A run is a maximal sequence of consecutive eligible steps (both
    endpoints present) in which the edge is observed.  An edge seen in steps
    1-10 has one run of 10; an edge seen in 10 scattered steps has many short
    runs, although both have the same :func:`observation_fraction`.

    Parameters
    ----------
    T : dict, list, MultiGraph or MultiDiGraph
        A temporal network: snapshots (a dict ``{time: graph}`` or a list of
        graphs), or a multigraph of events or spans (see
        :mod:`nx_backbone.temporal`).
    weight : str or None, optional (default=None)
        If given, an edge counts as observed only when its *weight* is above
        *min_weight* (a missing weight counts as 1).
    min_weight : float, optional (default=0)
        The weight an observed edge must exceed when *weight* is given.
    ineligible : {"skip", "break"}, optional (default="skip")
        How steps in which an endpoint is absent (unknown for the edge)
        count.  ``"skip"`` passes over them: they neither break nor extend a
        run, and a step's *next* step is the next eligible one.
        ``"break"`` treats them as unobserved, so they end a run, and a
        step's next step is simply the following step.
    freq : number, timedelta, str or None, optional (default=None)
        Step width for binning events or spans into snapshots, as in
        :func:`snapshots_from_events`: a number for numeric times, a
        :class:`datetime.timedelta` or a pandas frequency string such as
        ``"1D"`` (pandas only) for datetimes.  ``None`` makes one step per
        distinct time.  Snapshots are already steps, so *freq* must be
        ``None`` for them.
    origin : time or None, optional (default=None)
        Start of the first bin (default: the earliest time); with *freq*
        only.
    time : str, optional (default="time")
        Edge attribute holding an event's time.
    start, end : str, optional (default="start", "end")
        Edge attributes holding a span's start and end.
    aggregate : {"sum", "count", "max", "min", "mean"}, optional (default="sum")
        How the weights of a node pair's events or spans in one bin combine
        (see :func:`~nx_backbone.multigraph_to_weighted`).

    Returns
    -------
    H : Graph or DiGraph
        The union of the edges observed in any step, with the attributes
        :func:`flatten_temporal` gives them over the steps (the summed
        *weight*, ``n_obs``, ``first``, ``last``, and the attributes all
        their observations share), plus

        - ``max_run``: the length of the longest run, in steps;
        - ``mean_run``: the mean run length;
        - ``n_runs``: the number of runs;
        - ``persistence``: the share of the edge's observed steps whose next
          step is also observed, an empirical "stays active" probability.
          Observed steps without a next step (the edge's last eligible step,
          or the last step) are left out; ``persistence`` is NaN when no
          observed step has a next step.

        Nodes and ``H.graph["temporal"]`` are as in
        :func:`observation_fraction`.

    Raises
    ------
    NetworkXNotImplemented
        If a snapshot is a multigraph.
    NetworkXError
        If *weight* is given and a weight is not numeric, or *T* is not a
        temporal network.
    ValueError
        If *ineligible* is unknown, *freq* or *origin* is given with
        snapshots, or the input has no steps.

    See Also
    --------
    observation_fraction : How often, rather than how steadily.

    Examples
    --------
    ``("a", "b")`` is observed, absent, and observed again; ``"c"`` is absent
    at time 2, so ``("b", "c")`` is unknown there and its run is not broken:

    >>> import networkx as nx
    >>> import nx_backbone as nb
    >>> S = {
    ...     1: nx.Graph([("a", "b"), ("b", "c")]),
    ...     2: nx.Graph([("a", "b")]),
    ...     3: nx.Graph([("b", "c"), ("a", "z")]),
    ...     4: nx.Graph([("a", "b"), ("b", "c")]),
    ... }
    >>> H = nb.persistence(S)
    >>> scores = ("max_run", "mean_run", "n_runs", "persistence")
    >>> {a: H["a"]["b"][a] for a in scores}
    {'max_run': 2, 'mean_run': 1.5, 'n_runs': 2, 'persistence': 0.5}
    >>> {a: H["b"]["c"][a] for a in scores}
    {'max_run': 3, 'mean_run': 3.0, 'n_runs': 1, 'persistence': 1.0}

    With ``ineligible="break"`` the unknown step ends the run:

    >>> nb.persistence(S, ineligible="break")["b"]["c"]["max_run"]
    2
    """
    if ineligible not in ("skip", "break"):
        raise ValueError(f"ineligible must be 'skip' or 'break'; got {ineligible!r}.")
    steps = _steps(
        T,
        freq=freq,
        origin=origin,
        time=time,
        start=start,
        end=end,
        weight=weight or "weight",
        aggregate=aggregate,
    )
    table = _Table(steps, weight)
    observed = table.present if weight is None else table.above(min_weight)
    n = _counts(table, observed)[0]
    n_runs, max_run, share = _runs(table, observed, ineligible == "skip")
    attrs = [
        {"max_run": r, "mean_run": a / k, "n_runs": k, "persistence": p} if a else {}
        for a, k, r, p in zip(n, n_runs, max_run, share)
    ]
    parameters = {
        "weight": weight,
        "min_weight": min_weight,
        "ineligible": ineligible,
        "aggregate": aggregate,
    }
    return _scored_graph(table, n, attrs, "persistence", parameters, weight or "weight")


def _resolve_scorer(method, key, caller="backbone_stability"):
    """``(name, function, score key)`` for a static backbone method."""
    import nx_backbone as nb
    from nx_backbone.describe import METHODS, SCORE_KEYS, _resolve

    if isinstance(method, str):
        name, entry = _resolve(method)
        func = getattr(nb, name)
    elif callable(method):
        func = method
        base = getattr(method, "func", method)  # look through functools.partial
        name = getattr(base, "__name__", repr(method))
        try:
            name, entry = _resolve(name)
        except ValueError:
            entry = None
    else:
        raise TypeError(f"method must be a function or its name; got {method!r}.")
    if entry is not None and entry.kind not in ("weighted", "unweighted"):
        raise ValueError(
            f"{caller} runs a static backbone method on each "
            f"snapshot; {name!r} is a {entry.kind} method."
        )
    if key is None:
        key = SCORE_KEYS.get(name)
        if key is None:
            known = "is not in" if name not in METHODS else "has no score key in"
            raise ValueError(
                f"{name!r} {known} the describe registry; pass key=, the edge "
                "attribute to cut on (a p-value, a score or a *_keep flag)."
            )
    return name, func, key


def _cut_settings(key, threshold, mode, mtc, sign):
    """Validate the per-step cut on *key*; return ``(flag, cut, side)``."""
    flag = key.endswith("_keep")
    if flag:
        if threshold is not None or mode is not None or mtc not in (None, "none"):
            raise ValueError(
                f"{key!r} is a keep flag; threshold=, mode= and mtc= do not apply."
            )
    pvalue = key.endswith("_pvalue")
    if threshold is None and not flag and not pvalue:
        raise ValueError(f"{key!r} is not a p-value; pass threshold= (and mode=).")
    cut = 0.05 if threshold is None else threshold
    side = mode or ("below" if pvalue else "above")
    if sign not in (1, -1, None):
        raise ValueError(f"sign must be 1, -1 or None; got {sign!r}.")
    return flag, cut, side


def _scored_steps(steps, name, func, key, method_kwargs):
    """Run *func* on every snapshot with edges; yield ``(k, G_k, scored)``.

    Raises if a snapshot's scored graph has no *key* or a NaN *key*, and
    notes the snapshot on an exception raised by *func*.
    """
    for k, (t, G) in enumerate(zip(steps.times, steps.graphs)):
        if G.number_of_edges() == 0:
            continue
        try:
            scored = func(G, **method_kwargs)
        except Exception as err:
            err.add_note(
                f"Raised by {name} on the snapshot at time {t!r} "
                f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)."
            )
            raise
        values = [d.get(key) for _, _, d in scored.edges(data=True)]
        if not any(x is not None for x in values):
            raise nx.NetworkXError(
                f"{name} gave no edge a {key!r} attribute in the snapshot at "
                f"time {t!r}; pass key=, the attribute to cut on."
            )
        if any(x != x for x in values if x is not None):
            raise nx.NetworkXError(
                f"{name} gave a NaN {key!r} in the snapshot at time {t!r} "
                f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges); "
                "the snapshot may be too small for the method."
            )
        yield k, G, scored


def _kept_at(table, name, func, key, cut, method_kwargs):
    """Per step, the edges a static backbone keeps (``backbone_stability``).

    *cut* is ``(flag, threshold, mode, mtc, sign)``: the first three from
    :func:`_cut_settings`.  Only edges present in a snapshot can be kept.
    """
    from nx_backbone.filters import boolean_filter, threshold_filter

    flag, threshold, mode, mtc, sign = cut
    steps = table.steps
    kept_at = [[] for _ in steps.times]  # edge indices kept in each step
    for k, G, scored in _scored_steps(steps, name, func, key, method_kwargs):
        signs = [d.get("sign") for _, _, d in scored.edges(data=True)]
        if sign == -1 and all(s is None for s in signs):
            raise ValueError(
                f"sign=-1 keeps significantly weak edges, but {name} added no "
                "sign attribute; pass signed=True if the method has it."
            )
        if flag:
            kept = boolean_filter(scored, key)
        else:
            kept = threshold_filter(
                scored, key, threshold, mode=mode, mtc=mtc or "none"
            )
        for u, v, d in kept.edges(data=True):
            if sign is not None and d.get("sign", sign) != sign:
                continue
            if G.has_edge(u, v):
                kept_at[k].append(table.index[_pair(u, v, steps.directed)])
    return kept_at


def backbone_stability(
    T: TemporalNetwork,
    method: str | Callable[..., nx.Graph],
    *,
    key: str | None = None,
    threshold: float | None = None,
    mode: Literal["below", "above"] | None = None,
    mtc: str | None = None,
    sign: Literal[1, -1] | None = 1,
    freq: Frequency | None = None,
    origin: Any = None,
    time: str = "time",
    start: str = "start",
    end: str = "end",
    aggregate: str = "sum",
    **method_kwargs: Any,
) -> nx.Graph:
    """Score each edge by how often a static backbone keeps it across steps.

    Runs a backbone method (any static scorer, such as
    :func:`~nx_backbone.disparity_filter`) on every snapshot, applies
    the same cut in every step, and counts the steps in which each edge is
    kept: a temporal consensus backbone.  Where
    :func:`~nx_backbone.consensus_backbone` intersects methods, this
    aggregates one method over time.

    Parameters
    ----------
    T : dict, list, MultiGraph or MultiDiGraph
        A temporal network: snapshots (a dict ``{time: graph}`` or a list of
        graphs), or a multigraph of events or spans (see
        :mod:`nx_backbone.temporal`).
    method : str or callable
        The backbone method, as a function or by name (``"disparity_filter"``
        or an alias such as ``"disparity"``).  It is called as
        ``method(G_t, **method_kwargs)`` on each snapshot.
    key : str or None, optional (default=None)
        The edge attribute to cut on.  ``None`` looks the method up in
        ``nx_backbone.describe.SCORE_KEYS`` (``"disparity_pvalue"``
        for the disparity filter, ``"mst_keep"`` for the maximum spanning
        tree, ...); a function outside the registry needs *key*.
    threshold : float or None, optional (default=None)
        The cut for a numeric *key*, applied with
        :func:`~nx_backbone.threshold_filter` in every step.  ``None``
        means 0.05 for a p-value (a key ending in ``_pvalue``) and is an
        error for other scores.  Not used with a ``*_keep`` flag, which is
        applied with :func:`~nx_backbone.boolean_filter`.
    mode : {"below", "above"} or None, optional (default=None)
        As in :func:`~nx_backbone.threshold_filter`.  ``None`` means
        ``"below"`` for a p-value and ``"above"`` for other scores.
    mtc : str or None, optional (default=None)
        Multiple-testing correction applied within each step (see
        :func:`~nx_backbone.adjust_pvalues`); p-values only.
    sign : {1, -1} or None, optional (default=1)
        For signed scorers (``signed=True``, which add a ``sign``
        attribute): keep only the significantly strong (``1``) or weak
        (``-1``) edges, or both (``None``).  Ignored for edges without a
        ``sign`` attribute.
    freq : number, timedelta, str or None, optional (default=None)
        Step width for binning events or spans into snapshots, as in
        :func:`snapshots_from_events`: a number for numeric times, a
        :class:`datetime.timedelta` or a pandas frequency string such as
        ``"1D"`` (pandas only) for datetimes.  ``None`` makes one step per
        distinct time.  Snapshots are already steps, so *freq* must be
        ``None`` for them.
    origin : time or None, optional (default=None)
        Start of the first bin (default: the earliest time); with *freq*
        only.
    time : str, optional (default="time")
        Edge attribute holding an event's time.
    start, end : str, optional (default="start", "end")
        Edge attributes holding a span's start and end.
    aggregate : {"sum", "count", "max", "min", "mean"}, optional (default="sum")
        How the weights of a node pair's events or spans in one bin combine
        (see :func:`~nx_backbone.multigraph_to_weighted`).
    **method_kwargs
        Passed to *method* in every step.  A ``weight`` entry also names the
        attribute that binning combines.

    Returns
    -------
    H : Graph or DiGraph
        The union of the edges present in any step, with the attributes
        :func:`flatten_temporal` gives them over the steps (the summed
        *weight*, ``n_obs``, ``first``, ``last``, and the attributes all
        their observations share), plus

        - ``stability``: ``n_kept / n_eligible``;
        - ``n_kept``: the number of steps in which the backbone keeps the
          edge;
        - ``n_eligible``: the number of steps in which both endpoints are
          present (an absent edge in such a step counts as not kept);
        - ``first_kept``, ``last_kept``: the times of the first and last
          step that keeps it (``None`` if none does).

        Nodes and ``H.graph["temporal"]`` are as in
        :func:`observation_fraction`.

    Raises
    ------
    ValueError
        If *method* is not a static weighted or unweighted method, *key* is
        needed but not given, *threshold*, *mode* or *mtc* do not fit *key*,
        *sign* is unknown or needs a signed scorer, *freq* or *origin* is
        given with snapshots, or the input has no steps.
    NetworkXError
        If *method* gives an edge of a snapshot a NaN *key* (common on very
        small snapshots), or gives no edge a *key*.
    NetworkXNotImplemented
        If a snapshot is a multigraph, or *method* does not support the
        snapshots' graph type.

    See Also
    --------
    consensus_backbone : Intersect the backbones of several methods.
    observation_fraction : The same share for presence instead of the
        backbone.

    Notes
    -----
    Snapshots without edges are not scored.  Only edges present in a
    snapshot can be kept in it, even if *method* scores absent pairs too.
    An exception raised by *method* propagates with a note naming the
    snapshot.  To pass *method* an argument named like one of this
    function's own (``threshold`` of
    :func:`~nx_backbone.global_threshold_filter`, for example), bind
    it with :func:`functools.partial`; the score key is still looked up.

    Examples
    --------
    Keep the edges that the disparity filter (at ``alpha = 0.05``) keeps in
    at least half of their eligible steps:

    >>> import networkx as nx
    >>> import nx_backbone as nb
    >>> G = nx.les_miserables_graph()
    >>> S = {}
    >>> for t in range(4):
    ...     S[t] = G.copy()
    ...     for i, (u, v, d) in enumerate(S[t].edges(data=True)):
    ...         d["weight"] += (i + t) % 3  # a little noise in every step
    >>> H = nb.backbone_stability(S, nb.disparity_filter, threshold=0.05)
    >>> H["Valjean"]["Javert"]["n_eligible"]
    4
    >>> B = nb.threshold_filter(H, "stability", 0.5, mode="above")
    >>> B.number_of_edges() < G.number_of_edges()
    True
    """
    name, func, key = _resolve_scorer(method, key)
    flag, cut, side = _cut_settings(key, threshold, mode, mtc, sign)
    steps = _steps(
        T,
        freq=freq,
        origin=origin,
        time=time,
        start=start,
        end=end,
        weight=method_kwargs.get("weight", "weight"),
        aggregate=aggregate,
    )
    table = _Table(steps)
    kept_at = _kept_at(
        table, name, func, key, (flag, cut, side, mtc, sign), method_kwargs
    )
    n, n_elig, first, last = _counts(table, table.mask(kept_at))
    attrs = [
        {
            "stability": a / b,
            "n_kept": a,
            "n_eligible": b,
            "first_kept": _time(steps, f),
            "last_kept": _time(steps, z),
        }
        for a, b, f, z in zip(n, n_elig, first, last)
    ]
    parameters = {
        "method": name,
        "key": key,
        "threshold": None if flag else cut,
        "mode": None if flag else side,
        "mtc": mtc,
        "sign": sign,
        "aggregate": aggregate,
        **method_kwargs,
    }
    keep = [True] * len(table.ends)
    weight = method_kwargs.get("weight", "weight")
    return _scored_graph(table, keep, attrs, "backbone_stability", parameters, weight)


def weight_trend(
    T: TemporalNetwork,
    *,
    weight: str = "weight",
    pvalue: bool = False,
    freq: Frequency | None = None,
    origin: Any = None,
    time: str = "time",
    start: str = "start",
    end: str = "end",
    aggregate: str = "sum",
) -> nx.Graph:
    """Summarize each edge's weight series: level, variability and trend.

    Each edge's series runs over its eligible steps (both endpoints present),
    with weight 0 in the eligible steps in which the edge is absent.

    Parameters
    ----------
    T : dict, list, MultiGraph or MultiDiGraph
        A temporal network: snapshots (a dict ``{time: graph}`` or a list of
        graphs), or a multigraph of events or spans (see
        :mod:`nx_backbone.temporal`).
    weight : str, optional (default="weight")
        Edge attribute holding the weight; a missing weight counts as 1.
    pvalue : bool, optional (default=False)
        Also report ``weight_slope_pvalue``, the two-sided p-value of the
        ordinary least-squares t-test of the slope (needs SciPy).  It is
        descriptive only: weights over time are autocorrelated, which the
        test ignores, so the p-value is usually too small.
    freq : number, timedelta, str or None, optional (default=None)
        Step width for binning events or spans into snapshots, as in
        :func:`snapshots_from_events`: a number for numeric times, a
        :class:`datetime.timedelta` or a pandas frequency string such as
        ``"1D"`` (pandas only) for datetimes.  ``None`` makes one step per
        distinct time.  Snapshots are already steps, so *freq* must be
        ``None`` for them.
    origin : time or None, optional (default=None)
        Start of the first bin (default: the earliest time); with *freq*
        only.
    time : str, optional (default="time")
        Edge attribute holding an event's time.
    start, end : str, optional (default="start", "end")
        Edge attributes holding a span's start and end.
    aggregate : {"sum", "count", "max", "min", "mean"}, optional (default="sum")
        How the weights of a node pair's events or spans in one bin combine
        (see :func:`~nx_backbone.multigraph_to_weighted`).

    Returns
    -------
    H : Graph or DiGraph
        The union of the edges present in any step, with the attributes
        :func:`flatten_temporal` gives them over the steps (the summed
        *weight*, ``n_obs``, ``first``, ``last``, and the attributes all
        their observations share), plus

        - ``weight_mean``, ``weight_std``: the mean and the population
          standard deviation (``ddof=0``, as :func:`numpy.std`) of the
          series;
        - ``weight_cv``: ``weight_std / weight_mean``, NaN when the mean is
          0;
        - ``weight_slope``: the least-squares slope of the series against
          the step number (0, 1, ..., in step units), NaN with fewer than 2
          eligible steps;
        - ``weight_slope_time`` (numeric times only): the slope against the
          times themselves, in weight per unit of time;
        - ``weight_slope_pvalue`` (with *pvalue*): NaN with fewer than 3
          eligible steps.

        Nodes and ``H.graph["temporal"]`` are as in
        :func:`observation_fraction`.

    Raises
    ------
    ImportError
        If numpy (or, with *pvalue*, SciPy) is not installed.
    NetworkXError
        If a weight is not numeric or NaN, or *T* is not a temporal network.
    NetworkXNotImplemented
        If a snapshot is a multigraph.
    ValueError
        If *freq* or *origin* is given with snapshots, or the input has no
        steps.

    See Also
    --------
    observation_fraction : How often each edge is observed.

    Examples
    --------
    >>> import networkx as nx
    >>> import nx_backbone as nb
    >>> S = [nx.Graph() for _ in range(4)]
    >>> for t, w in enumerate([1, 2, 3, 6]):
    ...     S[t].add_edge("a", "b", weight=w)
    >>> S[2].add_edge("b", "c", weight=4)
    >>> H = nb.weight_trend(S)
    >>> H["a"]["b"]["weight_mean"], H["a"]["b"]["weight_slope"]
    (3.0, 1.6)

    ``"c"`` is present only at step 2, so its edge has one eligible step and
    no slope:

    >>> H["b"]["c"]["weight_std"], H["b"]["c"]["weight_slope"]
    (0.0, nan)
    """
    np = _require_numpy("weight_trend")
    steps = _steps(
        T,
        freq=freq,
        origin=origin,
        time=time,
        start=start,
        end=end,
        weight=weight,
        aggregate=aggregate,
    )
    table = _Table(steps, weight)
    elig = table.eligible
    n = elig.sum(axis=1)
    W = np.where(elig, table.weights, 0.0)
    mean = W.sum(axis=1) / n
    dev = np.where(elig, table.weights - mean[:, None], 0.0)
    std = np.sqrt((dev**2).sum(axis=1) / n)

    def slope(x):
        xm = (elig * x).sum(axis=1) / n
        dx = np.where(elig, x - xm[:, None], 0.0)
        sxx = (dx**2).sum(axis=1)
        sxy = (dx * dev).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(sxx > 0, sxy / sxx, np.nan), sxx, sxy

    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.where(mean != 0, std / mean, np.nan)
    b, sxx, sxy = slope(np.arange(len(steps.times), dtype=float))
    columns = {
        "weight_mean": mean,
        "weight_std": std,
        "weight_cv": cv,
        "weight_slope": b,
    }
    if all(isinstance(t, Real) and not isinstance(t, bool) for t in steps.times):
        columns["weight_slope_time"] = slope(np.asarray(steps.times, dtype=float))[0]
    if pvalue:
        try:
            from scipy import stats
        except ImportError as err:
            raise ImportError(
                "weight_trend(pvalue=True) requires SciPy; install it with "
                "`pip install scipy`."
            ) from err
        df = n - 2
        rss = np.maximum((dev**2).sum(axis=1) - np.nan_to_num(b) * sxy, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            se = np.sqrt(rss / df / sxx)
            tstat = b / se
        p = 2 * stats.t.sf(np.abs(tstat), np.maximum(df, 1))
        p = np.where(se == 0, np.where(b == 0, 1.0, 0.0), p)
        columns["weight_slope_pvalue"] = np.where(df > 0, p, np.nan)
    lists = {a: c.tolist() for a, c in columns.items()}
    m = len(table.ends)
    attrs = [{a: lists[a][j] for a in lists} for j in range(m)]
    parameters = {"weight": weight, "pvalue": pvalue, "aggregate": aggregate}
    keep = [True] * m
    return _scored_graph(table, keep, attrs, "weight_trend", parameters, weight)


_OBSERVE = ("presence", "keep", "pvalue")


def edge_persistence_hmm(
    T: TemporalNetwork,
    *,
    observe: Literal["presence", "keep", "pvalue"] = "presence",
    method: str | Callable[..., nx.Graph] | None = None,
    key: str | None = None,
    threshold: float | None = None,
    mode: Literal["below", "above"] | None = None,
    mtc: str | None = None,
    sign: Literal[1, -1] | None = 1,
    weight: str | None = None,
    min_weight: float = 0,
    params: Mapping[str, float] | None = None,
    initial: Literal["stationary", "fit"] = "stationary",
    min_steps: int = 1,
    max_iter: int = 500,
    tol: float = 1e-6,
    n_init: int = 1,
    seed: Any = None,
    freq: Frequency | None = None,
    origin: Any = None,
    time: str = "time",
    start: str = "start",
    end: str = "end",
    aggregate: str = "sum",
    **method_kwargs: Any,
) -> nx.Graph:
    """Score each edge by a two-state hidden Markov model of its activity.

    Each edge is either inactive (state 0) or active (state 1) in every
    step, and switches between the two as a Markov chain.  What is seen in a
    step is a noisy flag of the state: an active edge can be missed and an
    inactive one can be seen.  The model therefore fills short dropouts
    inside an active spell and discounts isolated blips, which
    :func:`observation_fraction` and :func:`persistence` count at face value.
    One set of parameters is shared by all edges and fitted by Baum-Welch
    (EM) pooled over the edges, or fixed with *params*.

    Parameters
    ----------
    T : dict, list, MultiGraph or MultiDiGraph
        A temporal network: snapshots (a dict ``{time: graph}`` or a list of
        graphs), or a multigraph of events or spans (see
        :mod:`nx_backbone.temporal`).
    observe : {"presence", "keep", "pvalue"}, optional (default="presence")
        What the model observes in each eligible step (both endpoints
        present):

        - ``"presence"``: whether the edge is present (with *weight*, whether
          its weight is above *min_weight*), as in
          :func:`observation_fraction`;
        - ``"keep"``: whether the static backbone *method* keeps the edge in
          that step, with the same per-step cut as
          :func:`backbone_stability` (*key*, *threshold*, *mode*, *mtc*,
          *sign*);
        - ``"pvalue"``: the p-value *key* that *method* gives the edge in that
          step (1 where the edge is absent, is not scored, or has the other
          *sign*).
    method : str, callable or None, optional (default=None)
        The static backbone method for ``observe="keep"`` or ``"pvalue"``,
        as in :func:`backbone_stability`; called as
        ``method(G_t, **method_kwargs)`` on each snapshot.
    key : str or None, optional (default=None)
        The edge attribute of *method* to cut on (``"keep"``) or read as a
        p-value (``"pvalue"``, where it must end in ``_pvalue``).  ``None``
        looks it up as in :func:`backbone_stability`.
    threshold, mode, mtc : optional (default=None)
        The per-step cut for ``observe="keep"``, as in
        :func:`backbone_stability`.
    sign : {1, -1} or None, optional (default=1)
        For signed scorers with ``observe="keep"`` or ``"pvalue"``: count
        only the significantly strong (``1``) or weak (``-1``) edges, or both
        (``None``), as in :func:`backbone_stability`.
    weight : str or None, optional (default=None)
        With ``observe="presence"``, an edge counts as observed only when
        its *weight* is above *min_weight* (a missing weight counts as 1).
        With the other modes it is passed to *method*.  It also names the
        attribute that binning combines.
    min_weight : float, optional (default=0)
        The weight an observed edge must exceed (``observe="presence"`` with
        *weight* only).
    params : dict or None, optional (default=None)
        Parameters to hold fixed instead of fitting; any subset of ``a01``
        (inactive to active), ``a10`` (active to inactive), the emission
        parameters (``alpha``, the false-positive rate, and ``beta``, the
        miss rate, for flags; ``a`` for p-values) and ``pi1``.  Fixed
        ``alpha`` and ``beta`` must satisfy ``alpha < 1 - beta``.
    initial : {"stationary", "fit"}, optional (default="stationary")
        The probability ``pi1`` that an edge is active in the first step: the
        stationary probability of the chain, ``a01 / (a01 + a10)``, or a
        parameter fitted by EM.
    min_steps : int, optional (default=1)
        ``hmm_keep`` is true when the most likely state path has the edge
        active in at least this many eligible steps.
    max_iter : int, optional (default=500)
        Maximum number of EM iterations.
    tol : float, optional (default=1e-6)
        EM stops when no fitted parameter changes by more than *tol*.
    n_init : int, optional (default=1)
        Number of EM starts; the first uses fixed default starting values,
        the others random ones drawn with *seed*.  The fit with the highest
        likelihood wins.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state for the extra starts.
        See :ref:`Randomness<networkx:randomness>`.
    freq : number, timedelta, str or None, optional (default=None)
        Step width for binning events or spans into snapshots, as in
        :func:`snapshots_from_events`.  Snapshots are already steps, so
        *freq* must be ``None`` for them.
    origin : time or None, optional (default=None)
        Start of the first bin (default: the earliest time); with *freq*
        only.
    time : str, optional (default="time")
        Edge attribute holding an event's time.
    start, end : str, optional (default="start", "end")
        Edge attributes holding a span's start and end.
    aggregate : {"sum", "count", "max", "min", "mean"}, optional (default="sum")
        How the weights of a node pair's events or spans in one bin combine
        (see :func:`~nx_backbone.multigraph_to_weighted`).
    **method_kwargs
        Passed to *method* in every step.

    Returns
    -------
    H : Graph or DiGraph
        The union of the edges present in any step, with the attributes
        :func:`flatten_temporal` gives them over the steps (the summed
        *weight*, ``n_obs``, ``first``, ``last``, and the attributes all
        their observations share), plus

        - ``hmm_posterior``: a dict ``{time: P(active)}`` over the edge's
          eligible steps, the posterior probability that the edge is active
          in that step given all of its observations;
        - ``hmm_active_fraction``: the mean of ``hmm_posterior``, the
          expected share of eligible steps in the active state;
        - ``hmm_score``: the largest value of ``hmm_posterior``, the
          posterior probability that the edge is active in its most active
          step;
        - ``hmm_n_active``: the number of eligible steps in which the most
          likely (Viterbi) state path has the edge active;
        - ``hmm_keep``: ``hmm_n_active >= min_steps``.

        Nodes are as in :func:`observation_fraction`.
        ``H.graph["temporal"]`` records the method, the input form, the step
        times, ``n_steps``, *freq*, *origin* and the parameters, and its
        ``"hmm"`` entry the model: ``a01``, ``a10``, the emission parameters,
        ``pi1``, ``observation`` (``"binary"`` or ``"pvalue"``), the total
        ``loglik``, ``n_iter``, ``converged``, ``label_swapped``,
        ``constrained`` and ``fixed`` (the names of the fixed parameters),
        and two diagnostics: ``n_active_steps``, the number of eligible
        steps that the most likely state paths mark active, summed over the
        edges, and ``unobserved_active_share``, the share of those steps in
        which the edge was not observed (its flag was 0; ``None`` for
        p-value observations or when no step is active).  See Notes.

    Raises
    ------
    ImportError
        If numpy is not installed.
    ValueError
        If *observe* is unknown, *method* is missing or given with
        ``observe="presence"``, an option does not fit *observe* or *key*,
        *min_steps* is not a positive integer, *freq* or *origin* is given
        with snapshots, or the input has no steps.
    NetworkXError
        If a fixed parameter is out of range or unknown, the fixed
        parameters make the observations impossible, a weight is not numeric,
        *method* gives a NaN or no *key*, or *T* is not a temporal network.
    NetworkXNotImplemented
        If a snapshot is a multigraph.

    Warns
    -----
    RuntimeWarning
        If EM does not converge in *max_iter* iterations, or if fixed
        parameters force a fitted emission parameter onto the
        identifiability constraint.

    See Also
    --------
    persistence : Runs of consecutive observed steps, without a model.
    backbone_stability : The share of steps a static backbone keeps an edge.

    Notes
    -----
    The hidden state of edge *e* in step *t* is ``s_t`` in {0, 1}, with
    ``P(s_t = 1 | s_{t-1} = 0) = a01`` and ``P(s_t = 0 | s_{t-1} = 1) =
    a10``.  A flag ``o_t`` (presence or keep) has ``P(o_t = 1 | s_t = 0) =
    alpha`` and ``P(o_t = 1 | s_t = 1) = 1 - beta``.  A p-value is uniform on
    [0, 1] when the edge is inactive and Beta(``a``, 1) (density ``a p^(a -
    1)``, ``0 < a < 1``) when it is active; this beta-uniform emission is a
    modeling choice of this library.  Steps in which an endpoint is absent
    are unknown: the chain runs through them, but they carry no evidence.

    Forward-backward recursions give the exact posteriors, the Viterbi
    algorithm the most likely path (ties go to the inactive state), and EM
    the parameters, pooling the expected counts over all edges.  Swapping
    the two states' labels leaves the likelihood unchanged, so
    ``alpha < 1 - beta`` (an active edge is seen more often than an
    inactive one) is imposed: a fit that violates it is relabeled, which is
    recorded as ``label_swapped``, or, when fixed parameters forbid
    relabeling, clipped to the constraint, recorded as ``constrained``.

    ``hmm_score`` is a persistence-smoothed membership score, **not** a
    p-value: it is a posterior probability under a model fitted to the same
    data, and it assumes every edge follows the same chain.  Applied to keep
    flags, ``alpha`` and ``beta`` are the rates at which the per-step
    backbone disagrees with the smoothed state, not physical measurement
    errors.

    **Miss rate against switching rate.**  Because all edges share one set
    of parameters, a pooled fit can explain a gap in an edge's record in two
    ways: the edge switched off (a transition, ``a10``) or it stayed active
    and was missed (an emission, ``beta``).  When the edges are a mixture,
    for example a block of persistent edges plus edges that come and go at
    random, no single chain fits both, and the likelihood can prefer a large
    miss rate with slow switching.  An edge seen in about half of its steps
    then reads as one long active spell that was missed often, and can get
    more active steps (``hmm_n_active``) than an edge that was truly active
    in one unbroken run.  More EM starts (*n_init*) need not help: the
    large miss rate can be the maximum of the likelihood, not a poor local
    optimum, and more starts can even move a fit from a small-miss-rate
    local optimum to it.  The data alone cannot tell this case from a
    genuinely high miss rate (in simulations the two give overlapping
    fitted values), so no warning is raised.  Instead ``H.graph["temporal"]["hmm"]`` records the
    fitted ``beta`` and ``unobserved_active_share``, the share of decoded
    active steps in which the edge was not seen.  When either is large,
    compare the edges' ``hmm_n_active`` with their observed counts
    (:func:`observation_fraction`) and runs (:func:`persistence`).

    What counts as "active" is a modeling choice.  If you know the
    measurement process, for example that an active tie is recorded in
    nearly every step, fix the miss rate (``params={"beta": 0.05}``, say)
    and let the gaps be read as inactivity.  Fix ``alpha`` likewise if you
    know the rate of stray observations.  If you do not know them, report
    the fitted values with the backbone.  Fixing ``beta`` is not a cure-all:
    the transitions are still fitted and shared, and when most of the edges
    in the fit are persistent they come out slow enough that a flickering
    edge can still read as active.  The model assumes one chain for all
    edges; when the edges are of different kinds, read its scores next to
    :func:`persistence`.  A weak Beta prior on the error rates (MAP-EM) was
    evaluated for issue #112 and not adopted: a prior weak enough not to
    bias fits with a genuinely high miss rate barely moved this case, and
    its pull fades as the number of edges and steps grows.

    Examples
    --------
    Edge ``("a", "b")`` is present in steps 0-7 except step 4; ``("c",
    "d")`` appears only in step 2.  With fixed parameters (rare switches,
    10% observation errors) the dropout is filled in and the blip is
    discounted:

    >>> import networkx as nx
    >>> import nx_backbone as nb
    >>> S = [nx.Graph() for _ in range(8)]
    >>> for t, G in enumerate(S):
    ...     G.add_nodes_from("abcd")
    ...     if t != 4:
    ...         G.add_edge("a", "b")
    >>> S[2].add_edge("c", "d")
    >>> p = {"a01": 0.1, "a10": 0.1, "alpha": 0.1, "beta": 0.1}
    >>> H = nb.edge_persistence_hmm(S, params=p)
    >>> round(H["a"]["b"]["hmm_posterior"][4], 3)
    0.876
    >>> H["a"]["b"]["hmm_keep"], H["a"]["b"]["hmm_n_active"]
    (True, 8)
    >>> round(H["c"]["d"]["hmm_score"], 3), H["c"]["d"]["hmm_keep"]
    (0.132, False)

    The record says how much of the decoded activity was seen: of the 8
    active steps, one (the dropout) was not observed.

    >>> fit = H.graph["temporal"]["hmm"]
    >>> fit["n_active_steps"], fit["unobserved_active_share"]
    (8, 0.125)
    >>> B = nb.boolean_filter(H, "hmm_keep")
    >>> sorted(B.edges)
    [('a', 'b')]
    """
    np = _require_numpy("edge_persistence_hmm")
    from nx_backbone import _hmm

    if observe not in _OBSERVE:
        raise ValueError(f"observe must be one of {_OBSERVE}; got {observe!r}.")
    if isinstance(min_steps, bool) or not isinstance(min_steps, int) or min_steps < 1:
        raise ValueError(f"min_steps must be a positive integer; got {min_steps!r}.")
    if sign not in (1, -1, None):
        raise ValueError(f"sign must be 1, -1 or None; got {sign!r}.")
    parameters: dict[str, Any] = {"observe": observe}
    if observe == "presence":
        options = {"method": method, "key": key, "threshold": threshold}
        options.update(mode=mode, mtc=mtc)
        given = [k for k, v in options.items() if v is not None]
        given += sorted(method_kwargs)
        if given:
            raise ValueError(
                f"{', '.join(given)} only apply with observe='keep' or 'pvalue'."
            )
        parameters.update(weight=weight, min_weight=min_weight)
        bin_weight = weight or "weight"
    else:
        if method is None:
            raise ValueError(f"observe={observe!r} needs method=.")
        if min_weight != 0:
            raise ValueError("min_weight only applies with observe='presence'.")
        if weight is not None:
            method_kwargs["weight"] = weight
        name, func, key = _resolve_scorer(method, key)
        if observe == "keep":
            flag, cut, side = _cut_settings(key, threshold, mode, mtc, sign)
            parameters.update(
                method=name,
                key=key,
                threshold=None if flag else cut,
                mode=None if flag else side,
                mtc=mtc,
                sign=sign,
            )
        else:
            if not key.endswith("_pvalue"):
                raise ValueError(f"observe='pvalue' needs a p-value key; got {key!r}.")
            if threshold is not None or mode is not None or mtc is not None:
                raise ValueError(
                    "threshold=, mode= and mtc= do not apply with observe='pvalue'."
                )
            parameters.update(method=name, key=key, sign=sign)
        parameters.update(method_kwargs)
        bin_weight = method_kwargs.get("weight", "weight")
    steps = _steps(
        T,
        freq=freq,
        origin=origin,
        time=time,
        start=start,
        end=end,
        weight=bin_weight,
        aggregate=aggregate,
    )
    table = _Table(steps, weight if observe == "presence" else None)
    observation = "binary"
    if observe == "presence":
        obs = table.present if weight is None else table.above(min_weight)
    elif observe == "keep":
        cuts = (flag, cut, side, mtc, sign)
        obs = table.mask(_kept_at(table, name, func, key, cuts, method_kwargs))
    else:
        observation = "pvalue"
        obs = np.ones(table.eligible.shape)
        for k, G, scored in _scored_steps(steps, name, func, key, method_kwargs):
            signed = False
            for u, v, d in scored.edges(data=True):
                signed = signed or "sign" in d
                pv = d.get(key)
                if pv is None or not G.has_edge(u, v):
                    continue
                if sign is not None and d.get("sign", sign) != sign:
                    continue
                obs[table.index[_pair(u, v, steps.directed)], k] = pv
            if sign == -1 and not signed:
                raise ValueError(
                    f"sign=-1 keeps significantly weak edges, but {name} added "
                    "no sign attribute; pass signed=True if the method has it."
                )
    obs = np.asarray(obs, dtype=float)
    fit = _hmm.fit_hmm(
        obs,
        table.eligible,
        observation=observation,
        fixed=params,
        initial=initial,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        seed=seed,
    )
    elig = table.eligible
    post = fit["posterior"]
    n_elig = elig.sum(axis=1)
    frac = np.where(elig, post, 0.0).sum(axis=1) / np.maximum(n_elig, 1)
    score = np.where(elig, post, -np.inf).max(axis=1, initial=-np.inf)
    n_active = ((fit["path"] == 1) & elig).sum(axis=1)
    times = steps.times
    attrs = [
        {
            "hmm_posterior": {
                times[k]: float(post[j, k]) for k in np.flatnonzero(elig[j])
            },
            "hmm_active_fraction": float(frac[j]),
            "hmm_score": float(score[j]),
            "hmm_n_active": int(n_active[j]),
            "hmm_keep": bool(n_active[j] >= min_steps),
        }
        for j in range(len(table.ends))
    ]
    parameters.update(
        params=None if params is None else dict(params),
        initial=initial,
        min_steps=min_steps,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        seed=seed if seed is None or isinstance(seed, int) else repr(seed),
        aggregate=aggregate,
    )
    keep = [True] * len(table.ends)
    H = _scored_graph(
        table, keep, attrs, "edge_persistence_hmm", parameters, bin_weight
    )
    # Diagnostic (#112): how much of the decoded activity was never seen.
    active = (fit["path"] == 1) & elig
    n_active_steps = int(active.sum())
    unobserved_share = None
    if observation == "binary" and n_active_steps:
        unobserved_share = float((active & (obs == 0)).sum() / n_active_steps)
    model: dict[str, Any] = dict(fit["params"])
    model.update(
        observation=observation,
        loglik=fit["loglik"],
        n_iter=fit["n_iter"],
        converged=fit["converged"],
        label_swapped=fit["label_swapped"],
        constrained=fit["constrained"],
        fixed=sorted(params or ()),
        n_active_steps=n_active_steps,
        unobserved_active_share=unobserved_share,
    )
    H.graph[RECORD]["hmm"] = model
    return H


#: Static scorers whose p-values are continuous under their null model, so
#: their probit scores are N(0, 1) per endpoint test (see kalman_backbone).
_CONTINUOUS_PVALUES = frozenset({"disparity_filter", "glab_filter"})
#: Static scorers that give an edge the smaller of its two endpoints'
#: p-values (the "OR rule"), which is not uniform under the null.
_MIN_OF_ENDPOINTS = frozenset({"disparity_filter", "glab_filter"})
#: Static scorers whose p-values are discrete (counts, empirical or posterior
#: distributions over weights).  Their mid-p or randomized transforms need
#: P(X > x), which none of them reports.  noise_corrected_filter is discrete
#: with its default exact binomial p-values (see _discrete_pvalues).
_DISCRETE_PVALUES = frozenset(
    {
        "noise_corrected_filter",
        "marginal_likelihood_filter",
        "ecm_filter",
        "lans_filter",
        "polya_filter",
        "gloss_filter",
    }
)


def _discrete_pvalues(name, kwargs):
    """Whether *name*, called with *kwargs*, gives discrete p-values.

    noise_corrected_filter's default p-values are exact binomial tails; its
    ``pvalue_method="delta"`` p-values are continuous, though not calibrated
    (see its Notes).
    """
    if name == "noise_corrected_filter":
        return kwargs.get("pvalue_method", "binomial") == "binomial"
    return name in _DISCRETE_PVALUES


def _accepts(func, name):
    """Whether *func* (or the function a partial wraps) names parameter *name*."""
    import inspect

    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return name in params


def _step_positions(steps):
    """Step positions for the process variance, and the unit of one step.

    Binned steps (``freq`` given) are equally spaced by construction.
    Otherwise the gaps between consecutive times are measured in units of the
    smallest gap when the times can be subtracted (numbers, datetimes);
    anything else is equally spaced.
    """
    n = len(steps.times)
    equal = [float(k) for k in range(n)]
    if steps.freq is not None or n < 2:
        return equal, None
    try:
        gaps = [b - a for a, b in pairwise(steps.times)]
        unit = min(gaps)
        lengths = [g / unit for g in gaps]
    except (TypeError, ValueError, ZeroDivisionError):
        return equal, None
    positions = [0.0]
    for x in lengths:
        positions.append(positions[-1] + float(x))
    if all(x == 1.0 for x in lengths):
        return equal, unit
    return positions, unit


def kalman_backbone(
    T: TemporalNetwork,
    method: str | Callable[..., nx.Graph],
    *,
    key: str | None = None,
    mode: Literal["smoother", "filter"] = "smoother",
    q: float | None = None,
    half_life: float | None = None,
    n_sim: int = 9999,
    seed: Any = None,
    sidak: bool = True,
    discrete: Literal["raise", "raw"] = "raise",
    freq: Frequency | None = None,
    origin: Any = None,
    time: str = "time",
    start: str = "start",
    end: str = "end",
    aggregate: str = "sum",
    **method_kwargs: Any,
) -> nx.Graph:
    """Smooth each edge's per-step significance with a Kalman filter.

    Runs a static significance test (such as
    :func:`~nx_backbone.disparity_filter`) on every snapshot, turns
    each edge's p-value into a probit score ``y_t = Φ⁻¹(1 − p_t)``, which is
    N(0, 1) under that test's null, and smooths each edge's series with a
    local-level state-space model::

        x_t = x_{t−1} + η_t,   η_t ~ N(0, q·Δt)
        y_t = x_t + ε_t,       ε_t ~ N(0, 1)

    using a Kalman filter [1]_ and Rauch–Tung–Striebel smoother [2]_.  For a
    fixed process variance *q* the smoothed level is linear in the scores, so
    its null distribution (no signal in any step) is known exactly and gives
    calibrated per-step p-values.  For fixed *q* the result is a
    time-localized, weighted Stouffer combination of the per-step evidence:
    ``q → 0`` pools all observed steps and ``q → ∞`` returns the per-step
    p-values.  This is a novel adaptation of standard state-space tools to
    backboning; it was proposed in ``docs/design/temporal-backbones.md``.

    Parameters
    ----------
    T : dict, list, MultiGraph or MultiDiGraph
        A temporal network: snapshots (a dict ``{time: graph}`` or a list of
        graphs), or a multigraph of events or spans (see
        :mod:`nx_backbone.temporal`).
    method : str or callable
        A static weighted significance test, as a function or by name
        (``"disparity_filter"`` or an alias such as ``"disparity"``), called
        as ``method(G_t, **method_kwargs)`` on each snapshot.  Supported:
        :func:`~nx_backbone.disparity_filter` and
        :func:`~nx_backbone.glab_filter`, whose p-values are
        continuous (see *sidak*); see *discrete* for the others.
    key : str or None, optional (default=None)
        The p-value attribute.  ``None`` looks the method up in
        ``nx_backbone.describe.SCORE_KEYS``; it must end in
        ``_pvalue``.
    mode : {"smoother", "filter"}, optional (default="smoother")
        ``"smoother"`` uses every step (past and future); ``"filter"`` uses
        only the steps up to each step, as an online method would.
    q : float or None, optional (default=None)
        Process variance per step, ``q ≥ 0``: how fast an edge's
        significance may drift.  ``None`` uses *half_life* or, if that is
        ``None`` too, the maximum pooled likelihood estimate over all edges
        (see Notes).
    half_life : float or None, optional (default=None)
        Set *q* by the half-life, in steps, of the smoother's steady-state
        kernel: the weight of a step halves every *half_life* steps away.
        Not with *q*.
    n_sim : int, optional (default=9999)
        Null series simulated per missingness pattern for the edge-level
        ``kalman_pvalue``; it cannot go below ``1 / (n_sim + 1)``.
    seed : integer, random_state, or None (default)
        Random seed for that simulation.
    sidak : bool, optional (default=True)
        For a method that gives an edge the smaller of its two endpoints'
        p-values (:func:`~nx_backbone.disparity_filter` and
        :func:`~nx_backbone.glab_filter`), replace each per-step
        p-value ``p`` by ``1 − (1 − p)²``, the Šidák correction for the
        minimum of two independent tests (see Notes).  Ignored for other
        methods.
    discrete : {"raise", "raw"}, optional (default="raise")
        What to do for a method whose p-values are discrete
        (:func:`~nx_backbone.noise_corrected_filter` with its default
        exact binomial p-values,
        :func:`~nx_backbone.marginal_likelihood_filter`,
        :func:`~nx_backbone.ecm_filter`,
        :func:`~nx_backbone.lans_filter`,
        :func:`~nx_backbone.polya_filter`,
        :func:`~nx_backbone.gloss_filter`).  ``"raise"`` refuses;
        ``"raw"`` uses the p-values as they are, which keeps the tests valid
        but conservative (see Notes).
    freq : number, timedelta, str or None, optional (default=None)
        Step width for binning events or spans into snapshots, as in
        :func:`snapshots_from_events`: a number for numeric times, a
        :class:`datetime.timedelta` or a pandas frequency string such as
        ``"1D"`` (pandas only) for datetimes.  ``None`` makes one step per
        distinct time.  Snapshots are already steps, so *freq* must be
        ``None`` for them.
    origin : time or None, optional (default=None)
        Start of the first bin (default: the earliest time); with *freq*
        only.
    time : str, optional (default="time")
        Edge attribute holding an event's time.
    start, end : str, optional (default="start", "end")
        Edge attributes holding a span's start and end.
    aggregate : {"sum", "count", "max", "min", "mean"}, optional (default="sum")
        How the weights of a node pair's events or spans in one bin combine
        (see :func:`~nx_backbone.multigraph_to_weighted`).
    **method_kwargs
        Passed to *method* in every step.  A ``weight`` entry also names the
        attribute that binning combines.  ``missing_as_zero`` defaults to
        ``True`` for a method that has it (see Notes); ``signed=True`` is
        supported for the continuous methods.

    Returns
    -------
    H : Graph or DiGraph
        The union of the edges present in any step, with the attributes
        :func:`flatten_temporal` gives them over the steps, plus

        - ``kalman_pvalue``: the edge-level p-value, a test across all
          steps of ``kalman_score`` (see Notes); cut it with
          :func:`~nx_backbone.threshold_filter`, with ``mtc=`` across
          edges;
        - ``kalman_score``: the largest per-step z-score
          ``m_t / sd₀,t``;
        - ``kalman_n_obs``: the number of steps with a score (see Notes);
        - ``kalman_pvalues``, ``kalman_levels``: dicts ``{time: value}``
          over the edge's eligible steps, with the per-step p-value
          ``1 − Φ(m_t / sd₀,t)`` and the smoothed (or filtered) level
          ``m_t``;
        - ``kalman_innovations``: a dict ``{time: value}`` of standardized
          one-step prediction errors over the scored steps after the
          first, a burst or change indicator calibrated only under the
          fitted model.

        Nodes and ``H.graph["temporal"]`` are as in
        :func:`observation_fraction`; ``H.graph["temporal"]["kalman"]``
        also records the fitted ``q``, its ``half_life``, where it came from
        (``"q_source"``: ``"q"``, ``"half_life"`` or ``"likelihood"``), the
        pooled log-likelihood, how absent edges were treated (``"absent"``:
        ``"zero"`` or ``"unknown"``), the step unit, and the number of
        distinct missingness patterns.  :func:`kalman_snapshots` expands
        the per-step series into one graph per step.

    Raises
    ------
    ImportError
        If numpy or SciPy is not installed.
    ValueError
        If *method* is not a static weighted or unweighted method, *key* is
        not a p-value, *method* has discrete p-values and *discrete* is
        ``"raise"``, both *q* and *half_life* are given, *q* or *half_life*
        cannot be used, *q* must be estimated but no edge has two scored
        steps, *freq* or *origin* is given with snapshots, or the input has
        no steps.
    NetworkXError
        If *method* gives an edge of a snapshot a NaN p-value (common on very
        small snapshots), or gives no edge a *key*, as in
        :func:`backbone_stability`.
    NetworkXNotImplemented
        If a snapshot is a multigraph, or *method* does not support the
        snapshots' graph type.

    Warns
    -----
    UserWarning
        If *method* cannot score absent edges (it has no
        ``missing_as_zero``), so the eligible steps in which an edge is
        absent are treated as unknown.

    See Also
    --------
    backbone_stability : Count the steps in which a static backbone keeps
        each edge.
    kalman_snapshots : Expand the per-step series into per-step graphs.

    Notes
    -----
    **Steps and absent edges.**  An edge is scored in a step when it is
    present, and also when it is absent but eligible (both endpoints
    present), as an observed zero: *method* is then called with
    ``missing_as_zero=True`` (see :func:`~nx_backbone.disparity_filter`),
    which tests the absent pairs as weight-0 edges.  This costs O(n²) per
    snapshot; pass ``missing_as_zero=False`` to treat such steps as unknown
    instead.  A method without ``missing_as_zero`` is run as is, with the
    same effect and a warning.  Steps in which an endpoint is absent, or in
    which *method* leaves a pair untested, are unknown: the filter predicts
    through them without an update, so
    missing steps are handled exactly.  An observed zero is the weakest
    possible outcome; under a continuous test its p-value is 1, whose probit
    score is clipped to about −8.22 (``Φ⁻¹(1e-16)``), as is a p-value of 0
    to +8.22, so a single step cannot dominate.  Unequally spaced snapshot
    times scale the process variance by the gap, in units of the smallest
    gap; binned steps are equally spaced.

    **The exact null.**  The smoothed level ``m_t = Σ_s a_ts y_s`` is linear
    in the scores, with weights that depend only on *q*, the step times and
    the edge's scored steps.  If the per-step p-values are uniform and
    independent (no signal at any step), ``m_t`` is normal with variance
    ``sd₀,t² = Σ_s a_ts²``, computed by an O(T) recursion, which gives
    ``kalman_pvalues``.  The posterior standard deviation of the level would
    give miscalibrated p-values.  The edge-level ``kalman_pvalue`` tests the
    maximum over steps of ``m_t / sd₀,t``; its null distribution is
    simulated once per missingness pattern (*n_sim* series, shared by the
    edges with that pattern), and the Monte Carlo p-value
    ``(1 + #{null ≥ observed}) / (1 + n_sim)`` is valid.  It is one test per
    edge, so correct it across edges only.  Each of the E × T per-step
    p-values is a separate test: correct them within a step (for example
    with :func:`kalman_snapshots` and ``threshold_filter(..., mtc="bh")``)
    and do not read them as one family.

    **Choosing q.**  With neither *q* nor *half_life*, *q* maximizes the
    pooled Gaussian log-likelihood of the one-step prediction errors over
    all edges, by a bounded search over ``log q`` in ``[1e-6, 1e3]``.  Most
    edges of a network carry no signal, so the estimate is often near the
    lower bound, which pools all steps (a Stouffer combination over time).
    *half_life* is usually the better choice for a backbone: with unit
    steps the steady-state filter gain ``K`` satisfies
    ``(1 − K)^half_life = 1/2`` and ``q = K² / (1 − K)``.

    **Validity of the per-step p-values.**  The calibration above holds when
    each step's p-value is uniform under the null.  Four cases differ:

    - Discrete p-values are stochastically larger than uniform.  The
      smoothing weights are nonnegative, so with ``discrete="raw"`` the
      tests stay valid but are conservative.  Mid-p and randomized
      transforms would remove that, but they need ``P(X > x)``, which the
      library's discrete tests do not report.
    - :func:`~nx_backbone.disparity_filter` and
      :func:`~nx_backbone.glab_filter` give an edge the smaller of
      its two endpoints' p-values, which is not uniform under the null:
      it is too small.  Smoothing adds up such a bias over steps, so the
      default ``sidak=True`` replaces ``p`` by ``1 − (1 − p)²``, which is
      exact for two independent endpoint tests.  The two tests share the
      edge's weight, though, so the corrected p-value is too large and the
      smoothed tests are conservative.  In one simulation (260 edges among
      60 nodes, 20 steps, every node's weights iid exponential, which is
      the disparity filter's null; ``half_life=3``), the share of per-step
      p-values below 0.05 was 0.005 with ``sidak=True`` and 0.17 with
      ``sidak=False``, against 0.076 for the disparity filter on single
      snapshots.
    - :func:`~nx_backbone.noise_corrected_filter` tests each edge
      once, with exact binomial p-values that are discrete (use
      ``discrete="raw"``).  Its ``pvalue_method="delta"`` p-values are
      continuous but far too small on low-count graphs (see its Notes),
      and smoothing adds up such a bias over steps.
    - Observed zeros only lower the scores, which is conservative.

    **Risks.**  The p-values are exact for the edge-global null (no signal
    at any step), not for a time-local one: a signal in some steps raises
    the smoothed level in nearby steps too (temporal leakage, which
    ``mode="filter"`` limits to later steps).  In the design prototype the
    rejection rate ten steps after a signal window was 0.067 at a nominal
    0.05.  Dyad-level confounders that persist over time make the scores
    positively autocorrelated under the null, which makes the tests
    anti-conservative; check that ``kalman_innovations`` look like white
    noise.

    References
    ----------
    .. [1] Kalman, R. E. (1960). A new approach to linear filtering and
       prediction problems. *Journal of Basic Engineering*, 82(1), 35-45.
       https://doi.org/10.1115/1.3662552
    .. [2] Rauch, H. E., Tung, F., & Striebel, C. T. (1965). Maximum
       likelihood estimates of linear dynamic systems. *AIAA Journal*,
       3(8), 1445-1450. https://doi.org/10.2514/3.3166

    Examples
    --------
    Twelve noisy copies of Les Misérables (each weight drawn from an
    exponential distribution with that mean) in which the light edge
    Valjean-Gavroche is 50 times heavier in steps 4 to 7 only:

    >>> import random
    >>> import networkx as nx
    >>> import nx_backbone as nb
    >>> rng = random.Random(1)
    >>> G = nx.les_miserables_graph()
    >>> S = {}
    >>> for t in range(12):
    ...     S[t] = G.copy()
    ...     for u, v, d in S[t].edges(data=True):
    ...         d["weight"] = rng.expovariate(1 / d["weight"])
    ...     if 4 <= t < 8:
    ...         S[t]["Valjean"]["Gavroche"]["weight"] *= 50
    >>> H = nb.kalman_backbone(S, "disparity", half_life=2, n_sim=999, seed=0)

    The edge-level test finds the edge, and the per-step p-values place the
    signal in time:

    >>> d = H["Valjean"]["Gavroche"]
    >>> d["kalman_pvalue"] < 0.01
    True
    >>> d["kalman_pvalues"][5] < 0.05 < d["kalman_pvalues"][0]
    True
    >>> round(H.graph["temporal"]["kalman"]["q"], 3)
    0.121
    >>> B = nb.threshold_filter(H, "kalman_pvalue", 0.05, mtc="bh")
    >>> B.has_edge("Valjean", "Gavroche")
    True
    """
    np = _require_numpy("kalman_backbone")
    try:
        import scipy  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "kalman_backbone requires SciPy; install it with `pip install scipy`."
        ) from err
    from nx_backbone import _kalman

    name, func, key = _resolve_scorer(method, key, "kalman_backbone")
    if not key.endswith("_pvalue"):
        raise ValueError(
            f"kalman_backbone smooths p-values; {key!r} is not one (pass key=)."
        )
    if discrete not in ("raise", "raw"):
        raise ValueError(f"discrete must be 'raise' or 'raw'; got {discrete!r}.")
    discrete_pvalues = _discrete_pvalues(name, method_kwargs)
    if discrete_pvalues and discrete == "raise":
        raise ValueError(
            f"{name} gives discrete p-values, whose probit scores are not "
            "N(0, 1) under the null, and it does not report P(X > x) for a "
            "mid-p transform.  Pass discrete='raw' to use them as they are "
            "(valid but conservative), or use a continuous test such as "
            "disparity_filter."
        )
    if mode not in ("smoother", "filter"):
        raise ValueError(f"mode must be 'smoother' or 'filter'; got {mode!r}.")
    if q is not None and half_life is not None:
        raise ValueError("Pass q= or half_life=, not both.")
    if q is not None:
        _kalman._process_variance(q, None, 1)  # validates q
    if half_life is not None:
        _kalman.half_life_to_q(half_life)  # validates half_life
    if int(n_sim) < 1:
        raise ValueError(f"n_sim must be a positive integer; got {n_sim!r}.")

    weight = method_kwargs.get("weight", "weight")
    steps = _steps(
        T,
        freq=freq,
        origin=origin,
        time=time,
        start=start,
        end=end,
        weight=weight,
        aggregate=aggregate,
    )
    table = _Table(steps)
    kwargs = dict(method_kwargs)
    if "missing_as_zero" not in kwargs and _accepts(func, "missing_as_zero"):
        kwargs["missing_as_zero"] = True
    absent = "zero" if kwargs.get("missing_as_zero") else "unknown"
    if absent == "unknown" and "missing_as_zero" not in method_kwargs:
        import warnings

        warnings.warn(
            f"kalman_backbone: {name} cannot score absent edges (it has no "
            "missing_as_zero), so the eligible steps in which an edge is "
            "absent are treated as unknown, not as observed zeros.",
            UserWarning,
            stacklevel=2,
        )

    m, n_steps = len(table.ends), len(steps.times)
    P = np.full((m, n_steps), np.nan)
    signs = np.full((m, n_steps), np.nan)
    for k, _, scored in _scored_steps(steps, name, func, key, kwargs):
        # Every eligible pair the method scored: present, or absent and
        # tested as weight 0 (missing_as_zero).
        adj = scored.adj
        for j in np.flatnonzero(table.eligible[:, k]):
            u, v = table.ends[j]
            d = adj[u].get(v)
            if d is None or d.get(key) is None:
                continue
            P[j, k] = d[key]
            if "sign" in d:
                signs[j, k] = d["sign"]
    scored_steps = ~np.isnan(P)
    has_sign = ~np.isnan(signs)
    endpoint_min = name in _MIN_OF_ENDPOINTS
    if has_sign.any():
        if discrete_pvalues or endpoint_min:
            why = "combine two endpoints" if endpoint_min else "are discrete"
            raise ValueError(
                f"signed (two-sided) p-values of {name} {why} and cannot be "
                "turned into one-sided scores; drop signed=True."
            )
        if (scored_steps & ~has_sign).any():
            raise nx.NetworkXError(
                f"{name} gave a sign to some scored edges but not others."
            )
        Y = _kalman.probit_scores(P, sign=np.where(has_sign, signs, 1.0))
    else:
        if endpoint_min and sidak:
            P = P * (2.0 - P)  # 1 − (1 − p)², without cancellation
        Y = _kalman.probit_scores(P)

    positions, unit = _step_positions(steps)
    loglik = None
    if half_life is not None:
        q_used, source = _kalman.half_life_to_q(half_life), "half_life"
    elif q is not None:
        q_used, source = float(q), "q"
    elif m == 0:
        q_used, source = None, "likelihood"
    else:
        if not (scored_steps.sum(axis=1) >= 2).any():
            raise ValueError(
                "Estimating q needs an edge scored in at least two steps; "
                "pass q= or half_life=."
            )
        q_used, loglik = _kalman.estimate_q(Y, times=positions)
        source = "likelihood"
    if q_used is not None and loglik is None and m:
        loglik = _kalman.pooled_loglik(Y, q_used, times=positions)

    attrs: list[dict[str, Any]] = [{} for _ in range(m)]
    n_patterns = 0
    if m:
        res = _kalman.kalman_scores(Y, q_used, times=positions, mode=mode)
        stat, pvalue = _kalman.max_score_test(
            Y, q_used, times=positions, mode=mode, n_sim=n_sim, seed=seed
        )
        n_patterns = len(np.unique(scored_steps, axis=0))
        times = steps.times
        rows = zip(
            table.eligible.tolist(),
            res.pvalue.tolist(),
            res.level.tolist(),
            res.innovation.tolist(),
        )
        n_obs = scored_steps.sum(axis=1).tolist()
        for j, (elig, ps, ms, nus) in enumerate(rows):
            series_p, series_m, series_nu = {}, {}, {}
            for t, e, p, lvl, nu in zip(times, elig, ps, ms, nus):
                if e and p == p:
                    series_p[t] = p
                    series_m[t] = lvl
                if nu == nu:
                    series_nu[t] = nu
            attrs[j] = {
                "kalman_pvalue": float(pvalue[j]),
                "kalman_score": float(stat[j]),
                "kalman_n_obs": n_obs[j],
                "kalman_pvalues": series_p,
                "kalman_levels": series_m,
                "kalman_innovations": series_nu,
            }
    parameters = {
        "method": name,
        "key": key,
        "mode": mode,
        "q": q,
        "half_life": half_life,
        "n_sim": n_sim,
        "seed": seed,
        "sidak": sidak,
        "discrete": discrete,
        "aggregate": aggregate,
        **method_kwargs,
    }
    H = _scored_graph(table, [True] * m, attrs, "kalman_backbone", parameters, weight)
    H.graph[RECORD]["kalman"] = {
        "q": q_used,
        "half_life": None if q_used is None else _kalman.q_to_half_life(q_used),
        "q_source": source,
        "loglik": loglik,
        "mode": mode,
        "absent": absent,
        "step_unit": unit,
        "n_sim": n_sim,
        "n_patterns": n_patterns,
    }
    return H


def kalman_snapshots(H: nx.Graph) -> dict[Any, nx.Graph]:
    """Expand the per-step series of :func:`kalman_backbone` into snapshots.

    Parameters
    ----------
    H : Graph or DiGraph
        The scored graph returned by :func:`kalman_backbone`.

    Returns
    -------
    dict
        ``{time: G_t}`` for every step, in time order.  ``G_t`` (the same
        graph type as *H*) holds the edges with a value at that step, with
        the attributes ``kalman_pvalue`` and ``kalman_level`` (the values of
        ``kalman_pvalues`` and ``kalman_levels`` at that step) and, where
        defined, ``kalman_innovation``.  Its nodes are those edges'
        endpoints, with their attributes from *H*.  Filter each with
        :func:`~nx_backbone.threshold_filter`, for example with
        ``mtc="bh"`` to correct the per-step tests within each step.

    Raises
    ------
    NetworkXError
        If *H* was not returned by :func:`kalman_backbone`.

    Examples
    --------
    >>> import networkx as nx
    >>> import nx_backbone as nb
    >>> G = nx.les_miserables_graph()
    >>> H = nb.kalman_backbone([G, G, G], "disparity", q=0.5, seed=0)
    >>> steps = nb.kalman_snapshots(H)
    >>> list(steps)
    [0, 1, 2]
    >>> B = nb.threshold_filter(steps[1], "kalman_pvalue", 0.05)
    >>> B.number_of_edges() < G.number_of_edges()
    True
    """
    meta = H.graph.get(RECORD)
    if not isinstance(meta, dict) or meta.get("method") != "kalman_backbone":
        raise nx.NetworkXError("H is not the result of kalman_backbone.")
    out: dict[Any, nx.Graph] = {t: H.__class__() for t in meta["times"]}
    series = (
        ("kalman_pvalues", "kalman_pvalue"),
        ("kalman_levels", "kalman_level"),
        ("kalman_innovations", "kalman_innovation"),
    )
    for u, v, d in H.edges(data=True):
        for t in d["kalman_pvalues"]:
            G = out[t]
            for n in (u, v):
                if n not in G:
                    G.add_node(n, **H.nodes[n])
            G.add_edge(
                u, v, **{one: d[many][t] for many, one in series if t in d[many]}
            )
    return out


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
    "flatten_temporal": {
        "time": "O(n + m + S log S)",
        "space": "O(n + m)",
        "notes": "m observations (snapshot edges, events or spans), S snapshots.",
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
    "observation_fraction": {
        "time": "O(n_S + m_S + m T)",
        "space": "O(n T + m T)",
        "notes": "T steps, m distinct edges; plus binning for events and spans.",
    },
    "persistence": {
        "time": "O(n_S + m_S + m T)",
        "space": "O(n T + m T)",
        "notes": "T steps, m distinct edges; plus binning for events and spans.",
    },
    "backbone_stability": {
        "time": "O(n_S + m_S + m T) plus T runs of the method",
        "space": "O(n T + m T)",
        "notes": "T steps, m distinct edges; plus binning for events and spans.",
    },
    "weight_trend": {
        "time": "O(n_S + m_S + m T)",
        "space": "O(n T + m T)",
        "notes": "T steps, m distinct edges; plus binning for events and spans.",
    },
    "edge_persistence_hmm": {
        "time": "O(n_S + m_S + n_init I m T) plus T runs of the method",
        "space": "O(n T + m T)",
        "notes": (
            "T steps, m distinct edges, I EM iterations (at most max_iter); "
            "plus binning for events and spans.  The method runs only with "
            'observe="keep" or "pvalue".'
        ),
    },
    "kalman_backbone": {
        "time": "O(n_S + m_S + I m T + P n_sim T) plus T runs of the method",
        "space": "O(n T + m T + n_sim T)",
        "notes": (
            "T steps, m distinct edges, P distinct missingness patterns, I "
            "likelihood evaluations when q is estimated; with "
            "missing_as_zero the method also tests the O(n^2) absent pairs "
            "of each snapshot."
        ),
    },
    "kalman_snapshots": {"time": "O(m T)", "space": "O(m T)"},
}

append_complexity_docstrings(globals(), _COMPLEXITY)
