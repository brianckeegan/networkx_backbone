"""Tests for networkx_backbone.temporal: input forms and conversions."""

import copy
import doctest
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx
import pytest

import networkx_backbone as nb
from tests.test_minimal_deps import _run

ROOT = Path(__file__).resolve().parents[1]


def _edges(G):
    """Edges with data, keyed so undirected orientation does not matter."""
    if G.is_directed():
        return {(u, v): d for u, v, d in G.edges(data=True)}
    return {frozenset((u, v)): d for u, v, d in G.edges(data=True)}


def _canon(S):
    """Everything a snapshot dict holds, in comparable form."""
    return {
        t: (
            type(G),
            dict(G.graph),
            list(G.nodes(data=True)),
            _edges(G),
        )
        for t, G in dict(S).items()
    }


def _multiset(M, *drop):
    """The edges of a multigraph as a multiset, ignoring attributes *drop*."""
    return Counter(
        (
            (u, v) if M.is_directed() else frozenset((u, v)),
            tuple(sorted((k, x) for k, x in d.items() if k not in drop)),
        )
        for u, v, d in M.edges(data=True)
    )


def _rich_snapshots(directed):
    """Empty snapshots (first and inside), isolated nodes and attributes."""
    cls = nx.DiGraph if directed else nx.Graph
    S = {t: cls(day=t) for t in (10, 20, 30, 40, 50)}
    S[20].add_edge("a", "b", weight=2, kind="call")
    S[20].add_node("iso", role="lurker")
    S[20].add_node("a", age=1)
    S[30].add_edge("a", "b", weight=2, kind="call")
    S[30].add_edge(("t", 1), 3, weight=0.5)
    S[30].add_node("a", age=2)
    S[50].add_edge("b", "a", weight=7)  # same pair, new attributes
    S[50].add_edge("a", "a")  # self-loop, no attributes
    return S


# ---------------------------------------------------------------------------
# Forms
# ---------------------------------------------------------------------------


def test_temporal_form_detects_each_form():
    G = nx.path_graph(3)
    assert nb.temporal_form([G, G]) == "snapshots"
    assert nb.temporal_form({"2020": G}) == "snapshots"
    ev = nx.MultiDiGraph([(0, 1, {"time": 1})])
    assert nb.temporal_form(ev) == "events"
    sp = nx.MultiGraph([(0, 1, {"s": 1, "e": 2})])
    assert nb.temporal_form(sp, start="s", end="e") == "spans"
    both = nx.MultiGraph([(0, 1, {"time": 1, "start": 1, "end": 2})])
    assert nb.temporal_form(both, time=None) == "spans"
    assert nb.temporal_form(both, start=None) == "events"
    # An edgeless multigraph is read from its record.
    empty = nb.events_from_snapshots({0: nx.Graph()})
    assert nb.temporal_form(empty) == "events"


def test_temporal_form_rejects_invalid_input():
    G = nx.path_graph(3)
    cases = [
        (G, nx.NetworkXError, "simple graph"),
        (nx.MultiGraph([(0, 1, {"time": 1, "start": 1, "end": 2})]), nx.NetworkXError, "time=None"),
        (nx.MultiGraph([(0, 1, {"time": 1}), (1, 2)]), nx.NetworkXError, "1 of 2"),
        (nx.MultiGraph([(0, 1)]), nx.NetworkXError, "Name the"),
        (nx.MultiGraph(), nx.NetworkXError, "no edges"),
        ([], ValueError, "no snapshots"),
        ([G, "x"], TypeError, "not a NetworkX graph"),
        ("abc", TypeError, "Expected snapshots"),
        ({0: G, "a": G}, TypeError, "comparable"),
        ({None: G}, ValueError, "missing"),
        ([nx.MultiGraph()], nx.NetworkXNotImplemented, "multigraph"),
        ([G, nx.DiGraph()], nx.NetworkXError, "all directed or all undirected"),
    ]  # fmt: skip
    for T, exc, match in cases:
        with pytest.raises(exc, match=match):
            nb.temporal_form(T)


# ---------------------------------------------------------------------------
# Lossless round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("directed", [False, True])
def test_snapshots_events_snapshots_is_lossless(directed):
    S = _rich_snapshots(directed)
    before = copy.deepcopy(_canon(S))
    M = nb.events_from_snapshots(S)
    assert _canon(S) == before  # input untouched
    assert M.is_directed() is directed and M.is_multigraph()
    assert M.number_of_edges() == sum(G.number_of_edges() for G in S.values())
    # Node attributes from the latest snapshot holding the node.
    assert M.nodes["iso"] == {"role": "lurker"} and M.nodes["a"] == {}
    assert M.graph["temporal"]["times"] == [10, 20, 30, 40, 50]
    R = nb.snapshots_from_events(M)
    assert list(R) == list(S)
    assert _canon(R) == before
    R[20].nodes["a"]["age"] = 99  # the result shares no dicts with M
    assert nb.snapshots_from_events(M)[20].nodes["a"] == {"age": 1}
    # A list comes back as a dict keyed 0..T-1.
    L = list(S.values())
    assert _canon(nb.snapshots_from_events(nb.events_from_snapshots(L))) == _canon(
        dict(enumerate(L))
    )


@pytest.mark.parametrize("directed", [False, True])
def test_snapshots_spans_snapshots_is_lossless(directed):
    S = _rich_snapshots(directed)
    before = copy.deepcopy(_canon(S))
    M = nb.spans_from_snapshots(S)  # evenly spaced: the last one lasts 10
    assert _canon(S) == before
    spans = sorted(
        ("-".join(sorted(map(str, (u, v)))), d["start"], d["end"])
        for u, v, d in M.edges(data=True)
    )
    # a-b runs 20-40 with one set of attributes, then 50-60 with another.
    assert spans == [
        ("('t', 1)-3", 30, 40),
        ("a-a", 50, 60),
        ("a-b", 20, 40),
        ("a-b", 50, 60),
    ]
    assert M.graph["temporal"]["end_time"] == 60
    R = nb.snapshots_from_spans(M)
    assert list(R) == list(S)
    assert _canon(R) == before


def test_record_is_ignored_when_the_events_no_longer_fit():
    S = {0: nx.Graph([(0, 1)]), 1: nx.Graph([(1, 2)])}
    M = nb.events_from_snapshots(S)
    M.add_edge(0, 1, time=0.5)  # not a snapshot time
    R = nb.snapshots_from_events(M)
    assert list(R) == [0, 0.5, 1]
    assert R[0][0][1] == {"weight": 1, "edge_count": 1}
    M = nb.events_from_snapshots(S)
    M.add_edge(0, 1, time=0)  # a second event of a pair in one snapshot
    assert nb.snapshots_from_events(M)[0][0][1]["edge_count"] == 2
    # freq= always bins.
    M = nb.events_from_snapshots(S)
    assert list(nb.snapshots_from_events(M, freq=2)) == [0]


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def test_snapshots_from_events_bins_and_aggregates():
    M = nx.MultiDiGraph(name="calls")
    M.add_node("z", color="blue")
    M.add_edges_from(
        [
            ("a", "b", {"time": 1, "weight": 2, "kind": "call"}),
            ("a", "b", {"time": 2, "weight": 3, "kind": "call"}),
            ("b", "a", {"time": 2, "weight": 4, "kind": "sms"}),
            ("a", "z", {"time": 9, "kind": "call"}),
        ]
    )
    S = nb.snapshots_from_events(M, freq=3, origin=0)
    assert list(S) == [0, 3, 6, 9]  # empty bins between are kept
    assert all(G.is_directed() and G.graph == {"name": "calls"} for G in S.values())
    assert S[0]["a"]["b"] == {"kind": "call", "weight": 5, "edge_count": 2}
    assert S[0]["b"]["a"] == {"kind": "sms", "weight": 4, "edge_count": 1}
    # Presence: only nodes with an event in the bin.
    assert set(S[0]) == {"a", "b"} and len(S[3]) == 0 and set(S[9]) == {"a", "z"}
    assert S[9].nodes["z"] == {"color": "blue"}
    assert S[9]["a"]["z"]["weight"] == 1  # a missing weight counts as 1
    for aggregate, expected in [("count", 2), ("max", 3), ("mean", 2.5)]:
        G = nb.snapshots_from_events(M, freq=3, origin=0, aggregate=aggregate)[0]
        assert G["a"]["b"]["weight"] == expected
    # Default origin is the earliest time; freq=None is one snapshot per time.
    assert list(nb.snapshots_from_events(M, freq=3)) == [1, 4, 7]
    assert list(nb.snapshots_from_events(M)) == [1, 2, 9]
    assert nb.snapshots_from_events(nx.MultiGraph()) == {}
    # Floats, and an origin later than some events.
    F = nx.MultiGraph([(0, 1, {"time": 0.25}), (0, 1, {"time": 0.75})])
    assert list(nb.snapshots_from_events(F, freq=0.5)) == [0.25, 0.75]
    assert list(nb.snapshots_from_events(F, freq=0.5, origin=0.5)) == [0.0, 0.5]


def test_snapshots_from_spans_uses_half_open_overlap():
    M = nx.MultiGraph()
    M.add_edges_from(
        [
            ("a", "b", {"start": 0, "end": 4, "weight": 3}),
            ("b", "c", {"start": 4, "end": 6}),  # touches a-b, no overlap
            ("c", "d", {"start": 5, "end": 5}),  # instantaneous
        ]
    )
    S = nb.snapshots_from_spans(M, freq=2)
    assert {t: sorted(map(sorted, G.edges)) for t, G in S.items()} == {
        0: [["a", "b"]],
        2: [["a", "b"]],
        4: [["b", "c"], ["c", "d"]],
    }
    assert S[2]["a"]["b"]["weight"] == 3  # not split across bins
    # freq=None: elementary intervals between the distinct boundaries.
    S = nb.snapshots_from_spans(M)
    assert {t: G.number_of_edges() for t, G in S.items()} == {0: 1, 4: 1, 5: 2}
    assert nb.snapshots_from_spans(nx.MultiGraph()) == {}


def test_three_forms_give_the_same_snapshots():
    S = {0: nx.Graph([(0, 1), (1, 2)]), 1: nx.Graph([(0, 1)]), 2: nx.Graph([(1, 2)])}
    events = nb.events_from_snapshots(S)
    spans = nb.spans_from_snapshots(S)
    del events.graph["temporal"], spans.graph["temporal"]
    expected = {t: set(map(frozenset, G.edges)) for t, G in S.items()}
    for R in (
        nb.snapshots_from_events(events, freq=1),
        nb.snapshots_from_events(events),
        nb.snapshots_from_spans(spans, freq=1),
    ):
        assert {t: set(map(frozenset, G.edges)) for t, G in R.items()} == expected


# ---------------------------------------------------------------------------
# Snapshots to spans, spans to events, events to spans
# ---------------------------------------------------------------------------


def test_spans_from_snapshots_end_of_last_snapshot():
    def last_end(S, **kw):
        M = nb.spans_from_snapshots(S, **kw)
        return max(d["end"] for *_, d in M.edges(data=True))

    G = nx.Graph([(0, 1)])
    assert last_end([G, G]) == 2  # a list steps by 1
    assert last_end([G]) == 1
    assert last_end({0: G, 5: G}) == 10  # evenly spaced
    assert last_end({0: G, 5: G, 6: G}, freq=2) == 8
    assert last_end({0: G, 5: G, 6: G}, end_time=7.5) == 7.5
    t0 = datetime(2024, 1, 1)
    assert last_end({t0: G, t0 + timedelta(days=1): G}) == t0 + timedelta(days=2)
    for S, kw, match in [
        ({0: G, 5: G, 6: G}, {}, "not evenly spaced"),
        ({0: G}, {}, "only one snapshot"),
        ({0: G}, {"freq": 1, "end_time": 3}, "not both"),
        ({0: G}, {"end_time": 0}, "later than"),
    ]:
        with pytest.raises(ValueError, match=match):
            nb.spans_from_snapshots(S, **kw)


def test_events_from_spans_onsets_and_bins():
    M = nx.MultiGraph(name="m")
    M.add_node("iso")
    M.add_edges_from(
        [
            ("a", "b", {"start": 1, "end": 6, "weight": 2}),
            ("b", "c", {"start": 4, "end": 4}),
            ("a", "b", {"start": 6, "end": 7}),
        ]
    )
    E = nb.events_from_spans(M)
    assert _multiset(E) == _multiset(
        nx.MultiGraph(
            [
                ("a", "b", {"time": 1, "weight": 2}),
                ("b", "c", {"time": 4}),
                ("a", "b", {"time": 6}),
            ]
        )
    )
    assert "iso" in E and E.graph == {"name": "m"}
    E = nb.events_from_spans(M, freq=2, origin=0)
    assert sorted(t for *_, t in E.edges(data="time")) == [0, 2, 4, 4, 6]
    # Binning the events gives the snapshots of binning the spans.
    for freq, origin in [(2, 0), (3, None), (1, None)]:
        E = nb.events_from_spans(M, freq=freq, origin=origin)
        assert _canon(nb.snapshots_from_events(E, freq=freq, origin=origin)) == _canon(
            nb.snapshots_from_spans(M, freq=freq, origin=origin)
        )


def test_spans_from_events_duration_and_gap():
    M = nx.MultiGraph()
    M.add_edges_from(
        [
            ("a", "b", {"time": 0, "weight": 1, "kind": "x"}),
            ("b", "a", {"time": 2, "weight": 2, "kind": "x"}),
            ("a", "b", {"time": 5, "weight": 4, "kind": "y"}),
            ("c", "d", {"time": 1}),
        ]
    )

    def spans(**kw):
        H = nb.spans_from_events(M, **kw)
        return sorted(
            (min(map(str, (u, v))), d["start"], d["end"], d["weight"], d["edge_count"])
            for u, v, d in H.edges(data=True)
        )

    assert spans(duration=2) == [
        ("a", 0, 4, 3, 2),
        ("a", 5, 7, 4, 1),
        ("c", 1, 3, 1, 1),
    ]
    assert spans(duration=2, gap=1) == [("a", 0, 7, 7, 3), ("c", 1, 3, 1, 1)]
    assert spans(duration=1, gap=None) == [
        ("a", 0, 1, 1, 1), ("a", 2, 3, 2, 1), ("a", 5, 6, 4, 1), ("c", 1, 2, 1, 1),
    ]  # fmt: skip
    assert spans(duration=0)[0][1:3] == (0, 0)
    H = nb.spans_from_events(M, duration=2)
    kinds = {d["start"]: d.get("kind") for *_, d in H.edges(data=True)}
    assert kinds == {0: "x", 5: "y", 1: None}  # only shared attributes kept
    # Datetimes with a timedelta need no pandas.
    t0 = datetime(2024, 1, 1)
    D = nx.MultiGraph([(0, 1, {"time": t0}), (0, 1, {"time": t0 + timedelta(hours=3)})])
    (d,) = [d for *_, d in nb.spans_from_events(D, duration=timedelta(hours=1), gap=timedelta(hours=2)).edges(data=True)]  # fmt: skip
    assert (d["start"], d["end"]) == (t0, t0 + timedelta(hours=4))


def test_lossy_directions():
    # events -> snapshots -> events: per-bin aggregation adds weight and
    # edge_count and merges parallel events at one time.
    M = nx.MultiGraph(
        [(0, 1, {"time": 0}), (0, 1, {"time": 0}), (1, 2, {"time": 1, "c": 5})]
    )
    back = nb.events_from_snapshots(nb.snapshots_from_events(M))
    assert _multiset(back) == Counter(
        {
            (frozenset((0, 1)), (("edge_count", 2), ("time", 0), ("weight", 2))): 1,
            (frozenset((1, 2)), (("c", 5), ("edge_count", 1), ("time", 1), ("weight", 1))): 1,
        }
    )  # fmt: skip
    # spans -> events -> spans: onsets lose durations.
    sp = nx.MultiGraph([(0, 1, {"start": 0, "end": 5})])
    again = nb.spans_from_events(nb.events_from_spans(sp), duration=1)
    assert [(d["start"], d["end"]) for *_, d in again.edges(data=True)] == [(0, 1)]


# ---------------------------------------------------------------------------
# convert_temporal and validation
# ---------------------------------------------------------------------------


def test_convert_temporal_routes_every_direction():
    S = {0: nx.Graph([(0, 1)]), 2: nx.Graph([(1, 2)])}
    ev = nb.events_from_snapshots(S)
    sp = nb.spans_from_snapshots(S)
    routes = [
        (S, "events", {}, nb.events_from_snapshots(S)),
        (S, "spans", {"freq": 2}, nb.spans_from_snapshots(S, freq=2)),
        (ev, "snapshots", {"freq": 1}, nb.snapshots_from_events(ev, freq=1)),
        (sp, "snapshots", {}, nb.snapshots_from_spans(sp)),
        (ev, "spans", {"duration": 1}, nb.spans_from_events(ev, duration=1)),
        (sp, "events", {}, nb.events_from_spans(sp)),
    ]
    for T, to, kw, expected in routes:
        got = nb.convert_temporal(T, to=to, **kw)
        if to == "snapshots":
            assert _canon(got) == _canon(expected)
        else:
            assert _multiset(got) == _multiset(expected)
    # Already in the requested form: a copy.
    copied = nb.convert_temporal([nx.Graph([(0, 1)])], to="snapshots")
    assert list(copied) == [0] and list(copied[0].edges) == [(0, 1)]
    assert nb.convert_temporal(ev, to="events") is not ev
    with pytest.raises(TypeError, match="already"):
        nb.convert_temporal(ev, to="events", freq=1)
    with pytest.raises(ValueError, match="to must be"):
        nb.convert_temporal(ev, to="graph")


def test_conversion_errors():
    ev = nx.MultiGraph([(0, 1, {"time": 1})])
    sp = nx.MultiGraph([(0, 1, {"start": 3, "end": 1})])
    cases = [
        (nb.snapshots_from_events, (nx.Graph([(0, 1)]),), {}, nx.NetworkXNotImplemented),
        (nb.snapshots_from_events, ([],), {}, TypeError),
        (nb.snapshots_from_events, (nx.MultiGraph([(0, 1)]),), {}, ValueError),
        (nb.snapshots_from_events, (ev,), {"freq": 0}, ValueError),
        (nb.snapshots_from_events, (ev,), {"freq": True}, TypeError),
        (nb.snapshots_from_events, (ev,), {"origin": 0}, ValueError),
        (nb.snapshots_from_events, (ev,), {"aggregate": "median"}, ValueError),
        (nb.snapshots_from_events, (ev,), {"freq": timedelta(1)}, TypeError),
        (nb.snapshots_from_spans, (sp,), {}, ValueError),
        (nb.events_from_spans, (sp,), {}, ValueError),
        (nb.spans_from_events, (ev,), {"duration": -1}, ValueError),
        (nb.events_from_snapshots, ([nx.Graph([(0, 1, {"time": 1})])],), {}, ValueError),
        (nb.spans_from_snapshots, ([nx.Graph([(0, 1, {"end": 1})])],), {}, ValueError),
    ]  # fmt: skip
    for func, args, kw, exc in cases:
        with pytest.raises(exc):
            func(*args, **kw)


def test_numeric_and_datetime_paths_need_no_pandas():
    _run(
        """
        from datetime import datetime, timedelta
        import networkx as nx
        import networkx_backbone as nb
        S = [nx.Graph([(0, 1)]), nx.Graph(), nx.Graph([(0, 1)])]
        for to in ("events", "spans"):
            back = nb.convert_temporal(nb.convert_temporal(S, to=to), to="snapshots")
            assert [G.number_of_edges() for G in back.values()] == [1, 0, 1]
        t0 = datetime(2024, 1, 1)
        M = nx.MultiGraph([(0, 1, {"time": t0}), (0, 1, {"time": t0 + timedelta(hours=30)})])
        assert len(nb.snapshots_from_events(M, freq=timedelta(days=1))) == 2
        for call in (lambda: nb.snapshots_from_events(M, freq="1D"),
                     lambda: nb.to_event_table(M)):
            try:
                call()
            except ImportError as exc:
                assert "pandas" in str(exc)
            else:
                raise AssertionError("expected ImportError")
        """
    )


# ---------------------------------------------------------------------------
# pandas: frequency strings and event tables
# ---------------------------------------------------------------------------


def test_pandas_frequencies():
    pd = pytest.importorskip("pandas")
    ts = pd.Timestamp
    M = nx.MultiGraph(
        [
            ("a", "b", {"time": ts("2024-01-01 05:00")}),
            ("a", "b", {"time": ts("2024-01-03 01:00"), "weight": 3}),
            ("b", "c", {"time": ts("2024-02-10")}),
        ]
    )
    days = nb.snapshots_from_events(M, freq="1D", origin=ts("2024-01-01"))
    same = nb.snapshots_from_events(M, freq=timedelta(days=1), origin=ts("2024-01-01"))
    assert _canon(days) == _canon(same)
    assert list(days)[:3] == [ts("2024-01-01"), ts("2024-01-02"), ts("2024-01-03")]
    months = nb.snapshots_from_events(M, freq="MS", origin=ts("2024-01-01"))
    assert list(months) == [ts("2024-01-01"), ts("2024-02-01")]
    assert months[ts("2024-01-01")]["a"]["b"]["weight"] == 4
    with pytest.raises(ValueError, match="origin"):
        nb.snapshots_from_events(M, freq="MS", origin=ts("2024-01-02"))
    with pytest.raises(ValueError, match="not a pandas frequency"):
        nb.snapshots_from_events(M, freq="fortnightly")
    sp = nb.spans_from_events(M, duration="1D", gap="2D")
    assert sorted(d["edge_count"] for *_, d in sp.edges(data=True)) == [1, 2]
    S = {ts("2024-01-01"): nx.Graph([(0, 1)]), ts("2024-03-01"): nx.Graph([(0, 1)])}
    ends = nb.spans_from_snapshots(S, freq="MS").edges(data="end")
    assert [e for *_, e in ends] == [ts("2024-04-01")]


def test_event_table_round_trips():
    pd = pytest.importorskip("pandas")
    M = nx.MultiDiGraph()
    M.add_edges_from(
        [
            ("b", "c", {"time": pd.Timestamp("2024-01-02"), "weight": 2, "kind": "x"}),
            ("a", "b", {"time": pd.Timestamp("2024-01-01"), "weight": 1}),
            ("a", "b", {"time": pd.Timestamp("2024-01-01"), "weight": 1}),
        ]
    )
    df = nb.to_event_table(M)
    assert list(df.columns) == ["source", "target", "time", "weight", "kind"]
    assert df["time"].is_monotonic_increasing and len(df) == 3
    back = nb.from_event_table(df, weight="weight", directed=True, edge_attrs=True)
    assert back.is_directed() and _multiset(back) == _multiset(M)  # duplicates kept
    assert _multiset(nb.from_event_table(df, directed=True, edge_attrs=["kind"]), "weight") == _multiset(M, "weight")  # fmt: skip
    # Spans: start=time, end=end.
    sp = nx.MultiGraph([(0, 1, {"start": 0, "end": 2}), (1, 2, {"start": 1, "end": 1})])
    df = nb.to_event_table(sp)
    assert list(df.columns) == ["source", "target", "start", "end"]
    assert _multiset(nb.from_event_table(df, time="start", end="end")) == _multiset(sp)
    # Snapshots are written as their events.
    df = nb.to_event_table({5: nx.Graph([(0, 1)])}, edge_attrs=False)
    assert df.to_dict("list") == {"source": [0], "target": [1], "time": [5]}


def test_event_table_validation():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"u": [0, 1], "v": [1, 2], "time": [0, None], "end": [1, 0]})
    with pytest.raises(TypeError, match="DataFrame"):
        nb.from_event_table(df.to_dict())
    with pytest.raises(ValueError, match="no column"):
        nb.from_event_table(df)
    with pytest.raises(ValueError, match="no value in column 'time'"):
        nb.from_event_table(df, source="u", target="v")
    with pytest.raises(ValueError, match="before it starts"):
        nb.from_event_table(df.fillna(5), source="u", target="v", end="end")
    with pytest.raises(ValueError, match="clashes"):
        nb.to_event_table(nx.MultiGraph([(0, 1, {"time": 0, "source": 1})]))


def test_user_guide_examples_run():
    pytest.importorskip("pandas")
    path = ROOT / "docs" / "user_guide" / "temporal.rst"
    if not path.exists():
        pytest.skip("docs not available")
    result = doctest.testfile(
        str(path),
        module_relative=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        encoding="utf-8",
    )
    assert result.attempted > 0
    assert result.failed == 0
