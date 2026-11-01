Temporal networks: formats and conversions
==========================================

Networks observed over time come in three shapes, and
:mod:`networkx_backbone.temporal` accepts all three, built from ordinary
NetworkX graphs. This page shows each form and every conversion between
them. The temporal backbone methods planned in the :doc:`../roadmap` will take
any of the three forms.

.. contents:: On this page
   :local:
   :depth: 1

The three forms
---------------

**Snapshots** are a dict ``{time: graph}`` (or a list, read as times
``0, 1, ...``). A node is present at a time if it is in that snapshot, even
when it has no edges:

>>> import networkx as nx
>>> import networkx_backbone as nb
>>> S = {
...     0: nx.Graph([("ann", "bob")]),
...     1: nx.Graph([("ann", "bob"), ("bob", "cat")]),
...     2: nx.Graph(),
...     3: nx.Graph([("bob", "cat")]),
... }
>>> S[2].add_node("dan")  # present at time 2, but with no contacts

**Events** are one :class:`~networkx.MultiGraph` (or
:class:`~networkx.MultiDiGraph`) with one edge per event and its time in the
``time`` attribute:

>>> E = nx.MultiGraph()
>>> E.add_edges_from([
...     ("ann", "bob", {"time": 0.5}),
...     ("ann", "bob", {"time": 1.2, "weight": 3}),
...     ("bob", "cat", {"time": 3.7}),
... ])
[0, 1, 0]

**Spans** are a multigraph whose edges last from ``start`` to ``end``. A span
covers the half-open interval ``[start, end)``: it includes its start and
excludes its end, so a span ending at 2 and one starting at 2 do not overlap.

>>> P = nx.MultiGraph()
>>> P.add_edges_from([
...     ("ann", "bob", {"start": 0, "end": 2}),
...     ("bob", "cat", {"start": 1, "end": 4}),
... ])
[0, 0]

:func:`~networkx_backbone.temporal_form` tells them apart. Pass ``time=``,
``start=`` or ``end=`` when your attributes have other names:

>>> nb.temporal_form(S), nb.temporal_form(E), nb.temporal_form(P)
('snapshots', 'events', 'spans')

Snapshots to events, and back
-----------------------------

:func:`~networkx_backbone.events_from_snapshots` writes one event per edge per
snapshot, timed by the snapshot's key:

>>> M = nb.events_from_snapshots(S)
>>> sorted((u, v, t) for u, v, t in M.edges(data="time"))
[('ann', 'bob', 0), ('ann', 'bob', 1), ('bob', 'cat', 1), ('bob', 'cat', 3)]

An edge list cannot hold the empty snapshot at time 2 or the isolated
``dan``, so the converter records them in ``M.graph["temporal"]``.
:func:`~networkx_backbone.snapshots_from_events` uses that record to give the
snapshots back exactly:

>>> R = nb.snapshots_from_events(M)
>>> list(R), list(R[2])
([0, 1, 2, 3], ['dan'])
>>> all(nx.utils.graphs_equal(R[t], S[t]) for t in S)
True

Snapshots to spans, and back
----------------------------

:func:`~networkx_backbone.spans_from_snapshots` merges each edge's run of
consecutive snapshots into one span. A run ends at the time of the next
snapshot; the last snapshot lasts one step here because the times are evenly
spaced (otherwise pass ``freq=`` or ``end_time=``):

>>> M = nb.spans_from_snapshots(S)
>>> sorted((u, v, d["start"], d["end"]) for u, v, d in M.edges(data=True))
[('ann', 'bob', 0, 2), ('bob', 'cat', 1, 2), ('bob', 'cat', 3, 4)]
>>> R = nb.snapshots_from_spans(M)
>>> all(nx.utils.graphs_equal(R[t], S[t]) for t in S)
True

Events to snapshots: binning
----------------------------

Binning with a step width, ``freq=``, groups the events into snapshots. Bin
``k`` covers ``[origin + k * freq, origin + (k + 1) * freq)`` and is keyed by
its start; ``origin`` defaults to the earliest time. Events of the same pair
in one bin collapse into one edge whose ``weight`` is their summed weight (a
missing weight counts as 1) and whose ``edge_count`` is their number:

>>> B = nb.snapshots_from_events(E, freq=1, origin=0)
>>> {t: sorted(G.edges(data="weight")) for t, G in B.items()}
{0: [('ann', 'bob', 1)], 1: [('ann', 'bob', 3)], 2: [], 3: [('bob', 'cat', 1)]}
>>> B = nb.snapshots_from_events(E, freq=2, origin=0)
>>> B[0]["ann"]["bob"]
{'edge_count': 2, 'weight': 4}

Empty bins between the first and the last event are kept (time 2 above). A
node is present in a bin only if it has an event there, so ``cat`` is absent
from bin 0. ``aggregate="count"``, ``"max"``, ``"min"`` or ``"mean"`` combine
the weights as :func:`~networkx_backbone.multigraph_to_weighted` does. Without
``freq``, every distinct time is its own snapshot.

Spans to snapshots
------------------

A span is observed in every bin it overlaps:

>>> B = nb.snapshots_from_spans(P, freq=1)
>>> {t: sorted(G.edges) for t, G in B.items()}
{0: [('ann', 'bob')], 1: [('ann', 'bob'), ('bob', 'cat')], 2: [('bob', 'cat')], 3: [('bob', 'cat')]}

Without ``freq``, a snapshot starts wherever a span starts or ends:

>>> {t: sorted(G.edges) for t, G in nb.snapshots_from_spans(P).items()}
{0: [('ann', 'bob')], 1: [('ann', 'bob'), ('bob', 'cat')], 2: [('bob', 'cat')]}

Events to spans
---------------

:func:`~networkx_backbone.spans_from_events` gives each event a ``duration``
and merges the events of a pair that overlap, touch, or come within ``gap``
of each other:

>>> M = nb.spans_from_events(E, duration=1)
>>> sorted((u, v, d["start"], d["end"]) for u, v, d in M.edges(data=True))
[('ann', 'bob', 0.5, 2.2), ('bob', 'cat', 3.7, 4.7)]

Spans to events
---------------

:func:`~networkx_backbone.events_from_spans` keeps each contact's onset by
default. With ``freq``, it writes one event per bin the span overlaps, so the
events bin to the same snapshots as the spans:

>>> sorted(t for *_, t in nb.events_from_spans(P).edges(data="time"))
[0, 1]
>>> sorted(t for *_, t in nb.events_from_spans(P, freq=1).edges(data="time"))
[0, 1, 1, 2, 3]

One function for all directions
-------------------------------

:func:`~networkx_backbone.convert_temporal` detects the input's form and calls
the matching converter with your keyword arguments:

>>> B = nb.convert_temporal(E, to="snapshots", freq=2, origin=0)
>>> list(B)
[0, 2]

Collapsing to a static weighted graph
-------------------------------------

:func:`~networkx_backbone.flatten_temporal` collapses any of the three forms
into one ordinary graph, with one edge per node pair, so every static
backbone method applies to it. Each edge's ``weight`` combines the weights of
the pair's observations (a missing weight counts as 1), ``n_obs`` counts the
observations, and ``first`` and ``last`` record when the pair was first and
last seen. Isolated nodes in the snapshots are kept:

>>> H = nb.flatten_temporal(S)
>>> sorted(H.edges(data=True))
[('ann', 'bob', {'weight': 2, 'n_obs': 2, 'first': 0, 'last': 1}), ('bob', 'cat', {'weight': 2, 'n_obs': 2, 'first': 1, 'last': 3})]
>>> sorted(H)
['ann', 'bob', 'cat', 'dan']

``aggregate=`` chooses how the weights combine (``"sum"``, ``"count"``,
``"mean"``, ``"max"`` or ``"min"``):

>>> sorted(nb.flatten_temporal(E).edges(data="weight"))
[('ann', 'bob', 4), ('bob', 'cat', 1)]
>>> sorted(nb.flatten_temporal(E, aggregate="count").edges(data="weight"))
[('ann', 'bob', 2), ('bob', 'cat', 1)]

For spans, ``aggregate="duration"`` weights each pair by how long it is in
contact (overlapping spans are not counted twice), and every span edge also
gets ``active_time``. ``window=(t0, t1)`` keeps only the half-open interval
``[t0, t1)``, clipping spans to it:

>>> sorted(nb.flatten_temporal(P, aggregate="duration").edges(data="weight"))
[('ann', 'bob', 2), ('bob', 'cat', 3)]
>>> sorted(nb.flatten_temporal(P, aggregate="duration", window=(1, 3)).edges(data="weight"))
[('ann', 'bob', 1), ('bob', 'cat', 2)]

To count the steps in which a pair interacts rather than its events, bin
first and then flatten:

>>> sorted(nb.flatten_temporal(nb.snapshots_from_events(E, freq=1, origin=0)).edges(data="n_obs"))
[('ann', 'bob', 2), ('bob', 'cat', 1)]

The flattened graph keeps no order beyond ``first``, ``last`` and
``active_time``; the temporal filters on the :doc:`../roadmap` score edges by
how they behave over time instead.

Datetimes and pandas
--------------------

Numeric times need only NetworkX, and so do :class:`datetime.datetime` times
with a :class:`datetime.timedelta` step. pandas is needed for frequency
strings such as ``freq="1D"`` or ``"MS"`` (calendar months), and for event
tables. :func:`~networkx_backbone.from_event_table` reads a ``DataFrame`` with
one row per event (or per span, with ``end=``), and
:func:`~networkx_backbone.to_event_table` writes one back:

>>> import pandas as pd
>>> df = pd.DataFrame({
...     "source": ["ann", "ann", "bob"],
...     "target": ["bob", "bob", "cat"],
...     "time": pd.to_datetime(["2024-01-01 09:00", "2024-01-01 17:00", "2024-01-03 08:00"]),
... })
>>> M = nb.from_event_table(df)
>>> daily = nb.snapshots_from_events(M, freq="1D", origin=pd.Timestamp("2024-01-01"))
>>> {str(t.date()): G.number_of_edges() for t, G in daily.items()}
{'2024-01-01': 1, '2024-01-02': 0, '2024-01-03': 1}
>>> nb.to_event_table(M).equals(df)
True

Bins start at ``origin``, here midnight; by default they would start at the
earliest event, 9:00. Duplicate rows are kept as separate events.

What each conversion keeps
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Conversion
     - Keeps
     - Loses
   * - ``events_from_snapshots``, then ``snapshots_from_events``
     - Everything: times (empty snapshots too), nodes (isolated too), and
       node, edge and graph attributes, through ``M.graph["temporal"]``
     - Nothing (a list comes back as a dict; edge order may change)
   * - ``spans_from_snapshots``, then ``snapshots_from_spans``
     - Everything, as above; runs split where an edge's attributes change
     - Nothing (as above)
   * - ``snapshots_from_events`` / ``snapshots_from_spans`` (binning)
     - Which pairs interact in each bin, the combined ``weight``, the
       ``edge_count``, attributes shared within the bin, and node attributes
     - Times within a bin, unshared attributes, nodes with no event in a
       bin
   * - ``spans_from_events``
     - Each merged span's extent, combined ``weight``, ``edge_count`` and
       shared attributes; nodes and graph attributes
     - Event times inside a merged span (none merge with ``gap=None``)
   * - ``events_from_spans``
     - Onsets (default), or the bins each span overlaps (``freq=``); other
       attributes, nodes and graph attributes
     - Durations (default) or times within a bin (``freq=``)
   * - ``flatten_temporal``
     - One edge per pair: combined ``weight``, ``n_obs``, ``first``,
       ``last`` (and ``active_time`` for spans), shared attributes; nodes
       (isolated too, without a window) and graph attributes
     - The order and timing of observations, unshared attributes
   * - ``from_event_table`` / ``to_event_table``
     - Endpoints, times, ends, weights and edge attributes
     - Node and graph attributes, isolated nodes, the
       ``graph["temporal"]`` record
