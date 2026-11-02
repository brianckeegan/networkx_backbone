"""Tests for the temporal backbone scores of nx_backbone.temporal.

observation_fraction and persistence (#68), backbone_stability and
weight_trend (#69), and the shared machinery they build on (#67).  The API
contract (no mutation, order invariance, mixed labels, directed input,
multigraph rejection) is checked in tests/test_api_contract.py.
"""

import doctest
import math
import random
import warnings
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import networkx as nx
import pytest

import nx_backbone as nb
from nx_backbone import temporal
from nx_backbone.describe import METHODS, SCORE_KEYS
from tests.test_minimal_deps import _run

ROOT = Path(__file__).resolve().parents[1]
NAN = float("nan")

# Steps 0-5.  "c" joins at step 2; "d" is present at steps 0, 2 and 3 only.
STEP_EDGES = [
    [("a", "b"), ("a", "d")],
    [("a", "b")],
    [("b", "c"), ("a", "d")],
    [("a", "b"), ("b", "c"), ("c", "d")],
    [("a", "b"), ("a", "c")],
    [("b", "c"), ("a", "c")],
]


def _weight(k, u, v):
    return 1 + (k + ord(u) + ord(v)) % 3


def _hand_snapshots(cls=nx.Graph, step=10, t0=0):
    """The hand example as snapshots ``{t0 + k * step: G_k}``, weighted."""
    return {
        t0 + k * step: cls([(u, v, {"weight": _weight(k, u, v)}) for u, v in edges])
        for k, edges in enumerate(STEP_EDGES)
    }


def _key(G, u, v):
    return (u, v) if G.is_directed() else frozenset((u, v))


def _attrs(H, names):
    return {_key(H, u, v): tuple(d[a] for a in names) for u, v, d in H.edges(data=True)}


def _same(a, b):
    """Equal, treating NaN as equal to NaN and floats approximately."""
    if isinstance(a, float) or isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == pytest.approx(b)
    return a == b


def _assert_same(x, y):
    assert x.keys() == y.keys()
    for e in x:
        assert all(_same(a, b) for a, b in zip(x[e], y[e], strict=True)), (
            e,
            x[e],
            y[e],
        )


OBS = ("obs_fraction", "n_obs", "n_eligible", "first_obs", "last_obs")
RUNS = ("max_run", "mean_run", "n_runs", "persistence")
STAB = ("stability", "n_kept", "n_eligible", "first_kept", "last_kept")
TREND = ("weight_mean", "weight_std", "weight_cv", "weight_slope")


def test_hand_computed_scores_with_late_and_leaving_nodes():
    S = _hand_snapshots()
    ab, ad, bc, cd, ac = (frozenset(e) for e in ("ab", "ad", "bc", "cd", "ac"))
    expected_obs = {
        ab: (4 / 6, 4, 6, 0, 40),
        ad: (2 / 3, 2, 3, 0, 20),  # "d" absent at steps 1, 4, 5
        bc: (3 / 4, 3, 4, 20, 50),  # "c" joins at step 2
        cd: (1 / 2, 1, 2, 30, 30),
        ac: (2 / 4, 2, 4, 40, 50),
    }
    expected_runs = {
        "skip": {
            ab: (2, 2.0, 2, 0.5),
            ad: (2, 2.0, 1, 0.5),  # steps 0 and 2: step 1 is unknown
            bc: (2, 1.5, 2, 0.5),
            cd: (1, 1.0, 1, NAN),  # no eligible step after step 3
            ac: (2, 2.0, 1, 1.0),
        },
        "break": {
            ab: (2, 2.0, 2, 0.5),
            ad: (1, 1.0, 2, 0.0),
            bc: (2, 1.5, 2, 0.5),
            cd: (1, 1.0, 1, 0.0),
            ac: (2, 2.0, 1, 1.0),
        },
    }
    for use_numpy in (True, False):
        with pytest.MonkeyPatch.context() as mp:
            if not use_numpy:
                mp.setattr(temporal, "_numpy", lambda: None)
            for T in (S, list(S.values())):
                H = nb.observation_fraction(T)
                exp = expected_obs
                if isinstance(T, list):  # times are 0..5 instead of 0..50
                    exp = {e: (*v[:3], v[3] // 10, v[4] // 10) for e, v in exp.items()}
                _assert_same(_attrs(H, OBS), exp)
            for mode, exp in expected_runs.items():
                _assert_same(_attrs(nb.persistence(S, ineligible=mode), RUNS), exp)
            # Weights above min_weight=1: only the weight-2 and -3 steps count.
            S[30]["c"]["d"]["weight"] = 1
            H = nb.observation_fraction(S, weight="weight", min_weight=1)
            for u, v, d in H.edges(data=True):
                ks = [k for k, (t, G) in enumerate(S.items()) if G.has_edge(u, v)]
                heavy = [k for k in ks if _weight(k, u, v) > 1]
                assert d["n_obs"] == len(heavy) > 0
                assert d["first_obs"] == 10 * heavy[0]
            assert not H.has_edge("c", "d")  # its only weight is 1


def test_output_graph_and_metadata():
    S = _hand_snapshots()
    S[0].graph["name"] = "old"
    S[50].graph["name"] = "new"
    S[0].nodes["a"]["role"] = "first"
    S[50].nodes["a"]["role"] = "last"
    S[30].add_node("iso", role="lurker")  # an isolated node is present
    S[20]["b"]["c"]["kind"] = "call"
    S[30]["b"]["c"]["kind"] = "call"
    S[50]["b"]["c"]["kind"] = "call"
    H = nb.persistence(S)
    assert type(H) is nx.Graph
    assert set(H) == {"a", "b", "c", "d", "iso"}
    assert H.nodes["a"] == {"role": "last"} and H.nodes["iso"] == {"role": "lurker"}
    assert H.graph["name"] == "new"
    bc = H["b"]["c"]
    assert (
        bc["kind"] == "call"
        and bc["n_obs"] == 3
        and (bc["first"], bc["last"]) == (20, 50)
    )
    assert bc["weight"] == sum(_weight(k, "b", "c") for k in (2, 3, 5))
    assert H.graph["temporal"] == {
        "method": "persistence",
        "input_form": "snapshots",
        "times": [0, 10, 20, 30, 40, 50],
        "n_steps": 6,
        "freq": None,
        "origin": None,
        "parameters": {
            "weight": None,
            "min_weight": 0,
            "ineligible": "skip",
            "aggregate": "sum",
        },
    }
    B = nb.threshold_filter(H, "persistence", 0.75, mode="above")
    assert {frozenset(e) for e in B.edges} == {frozenset("ac")}


def _events_and_spans(cls, times, freq):
    """The hand example as events (two per observation) and as spans."""
    E, P = cls(), cls()
    for k, edges in enumerate(STEP_EDGES):
        t = times[k]
        for u, v in edges:
            w = _weight(k, u, v)
            E.add_edge(u, v, time=t, weight=w - 0.5)
            E.add_edge(u, v, time=t + freq / 3, weight=0.5)
            P.add_edge(u, v, start=t + freq / 4, end=t + freq / 2, weight=w)
    return E, P


def test_three_input_forms_agree():
    stability = partial(nb.global_threshold_filter, threshold=2)
    day = datetime(2024, 1, 1)
    cases = [  # (snapshot times, freq for binning, freq for steps)
        ([10 * k for k in range(6)], 10, 10),
        (
            [day + timedelta(days=k) for k in range(6)],
            timedelta(days=1),
            timedelta(days=1),
        ),
    ]
    try:
        import pandas  # noqa: F401
    except ImportError:
        pass
    else:
        cases.append(
            ([day + timedelta(days=k) for k in range(6)], "1D", timedelta(days=1))
        )
    for directed in (False, True):
        cls = nx.DiGraph if directed else nx.Graph
        mcls = nx.MultiDiGraph if directed else nx.MultiGraph
        for times, freq, step in cases:
            S = {times[k]: G for k, G in enumerate(_hand_snapshots(cls).values())}
            E, P = _events_and_spans(mcls, times, step)
            # The normalizer gives identical steps: times, presence, edges.
            norm = {"time": "time", "start": "start", "end": "end"}
            norm.update(weight="weight", aggregate="sum")
            ref = temporal._steps(S, freq=None, origin=None, **norm)
            for T in (E, P):
                got = temporal._steps(T, freq=freq, origin=times[0], **norm)
                assert got.times == ref.times and got.directed == directed
                assert got.presence == ref.presence
                for G, R in zip(got.graphs, ref.graphs, strict=True):
                    weights = [
                        {_key(X, u, v): w for u, v, w in X.edges(data="weight")}
                        for X in (G, R)
                    ]
                    assert weights[0] == weights[1]
            for func, names, kw in (
                (nb.observation_fraction, OBS, {}),
                (nb.persistence, RUNS, {}),
                (nb.backbone_stability, STAB, {"method": stability}),
                (nb.weight_trend, TREND, {}),
            ):
                expected = _attrs(func(S, **kw), names)
                assert len(expected) == 5
                for T in (E, P):
                    H = func(T, freq=freq, origin=times[0], **kw)
                    assert H.is_directed() == directed
                    assert H.graph["temporal"]["times"] == times
                    _assert_same(_attrs(H, names), expected)


def test_mixed_labels_and_order_invariance():
    labels = {"a": 0, "b": "b", "c": ("c", 1), "d": 3.5}
    S = _hand_snapshots(nx.DiGraph)
    M = {t: nx.relabel_nodes(G, labels) for t, G in S.items()}
    rev = {}
    for t in reversed(list(M)):
        R = nx.DiGraph()
        R.add_nodes_from(reversed(list(M[t].nodes)))
        R.add_edges_from(reversed(list(M[t].edges(data=True))))
        rev[t] = R
    for func, names in (
        (nb.observation_fraction, OBS),
        (nb.persistence, RUNS),
        (partial(nb.backbone_stability, method="disparity_filter"), STAB),
        (nb.weight_trend, TREND),
    ):
        H = func(S)
        expected = {(labels[u], labels[v]): x for (u, v), x in _attrs(H, names).items()}
        _assert_same(_attrs(func(M), names), expected)
        _assert_same(_attrs(func(rev), names), expected)


def _perturbed_les_mis():
    """Four perturbed Les Misérables snapshots; Marius and Gavroche join late."""
    G = nx.les_miserables_graph()
    S = {}
    for t in range(4):
        H = G.copy()
        for i, (u, v, d) in enumerate(list(H.edges(data=True))):
            d["weight"] += (i * (t + 1)) % 4
            if (i + t) % 5 == 0:
                H.remove_edge(u, v)
        if t < 2:
            H.remove_nodes_from(["Marius", "Gavroche"])
        S[t] = H
    return S


def test_backbone_stability_matches_running_the_scorer_by_hand():
    S = _perturbed_les_mis()
    cases = [  # (method, backbone_stability kwargs, per-step cut)
        (
            nb.disparity_filter,
            {},
            lambda H: nb.threshold_filter(H, "disparity_pvalue", 0.05),
        ),
        (
            "disparity",
            {"threshold": 0.2, "mtc": "bh"},
            lambda H: nb.threshold_filter(H, "disparity_pvalue", 0.2, mtc="bh"),
        ),
        (
            "maximum_spanning_tree_backbone",
            {},
            lambda H: nb.boolean_filter(H, "mst_keep"),
        ),
        (
            nb.jaccard_backbone,
            {"threshold": 0.3},
            lambda H: nb.threshold_filter(H, "jaccard", 0.3, mode="above"),
        ),
        (
            lambda G: nb.disparity_filter(G, signed=True),
            {"key": "disparity_pvalue", "sign": None, "threshold": 0.1},
            lambda H: nb.threshold_filter(H, "disparity_pvalue", 0.1),
        ),
    ]
    for method, kwargs, cut in cases:
        func = getattr(nb, method) if isinstance(method, str) else method
        if method == "disparity":
            func = nb.disparity_filter
        H = nb.backbone_stability(S, method, **kwargs)
        kept = {}
        for t, G in S.items():
            for u, v in cut(func(G)).edges:
                if G.has_edge(u, v):
                    kept.setdefault(frozenset((u, v)), []).append(t)
        assert kept, method
        for u, v, d in H.edges(data=True):
            ts = kept.get(frozenset((u, v)), [])
            elig = sum(u in G and v in G for G in S.values())
            assert (d["n_kept"], d["n_eligible"]) == (len(ts), elig)
            assert d["stability"] == len(ts) / elig
            assert (d["first_kept"], d["last_kept"]) == (
                (ts[0], ts[-1]) if ts else (None, None)
            )
        mg = H["Marius"]["Gavroche"] if H.has_edge("Marius", "Gavroche") else None
        assert mg is None or mg["n_eligible"] == 2


def test_backbone_stability_signs_and_errors():
    S = _perturbed_les_mis()
    both = nb.backbone_stability(
        S, "disparity_filter", sign=None, signed=True, threshold=0.5
    )
    strong = nb.backbone_stability(S, "disparity_filter", signed=True, threshold=0.5)
    weak = nb.backbone_stability(
        S, "disparity_filter", sign=-1, signed=True, threshold=0.5
    )
    for u, v, d in both.edges(data=True):
        assert d["n_kept"] == strong[u][v]["n_kept"] + weak[u][v]["n_kept"]
    assert sum(d["n_kept"] for *_, d in weak.edges(data=True)) > 0
    assert strong.graph["temporal"]["parameters"]["signed"] is True

    tiny = [nx.Graph([(0, 1, {"weight": 1.0})])] * 2
    errors = [
        (ValueError, "Unknown method", {"method": "no_such_method"}),
        (ValueError, "temporal method", {"method": "persistence"}),
        (ValueError, "bipartite method", {"method": "sdsm"}),
        (ValueError, "pass key=", {"method": lambda G: G}),
        (ValueError, "not a p-value", {"method": "jaccard_backbone"}),
        (ValueError, "keep flag", {"method": "lspar", "threshold": 0.5}),
        (ValueError, "sign must be", {"method": "disparity_filter", "sign": 0}),
        (ValueError, "no sign attribute", {"method": "disparity_filter", "sign": -1}),
        (TypeError, "function or its name", {"method": 3}),
        (
            nx.NetworkXError,
            "no edge a 'nope'",
            {"method": "disparity_filter", "key": "nope", "threshold": 0.1},
        ),
        (
            nx.NetworkXError,
            "NaN",
            {
                "method": lambda G: nx.Graph([(u, v, {"p": NAN}) for u, v in G.edges]),
                "key": "p",
                "threshold": 0.1,
            },
        ),
        (ValueError, "freq= and origin=", {"method": "disparity_filter", "freq": 1}),
    ]
    for exc, match, kwargs in errors:
        with pytest.raises(exc, match=match):
            nb.backbone_stability(tiny, **kwargs)

    def broken(G):
        raise RuntimeError("too small")

    with pytest.raises(RuntimeError) as info:
        nb.backbone_stability(tiny, broken, key="p", threshold=0.5)
    assert "snapshot at time 0" in info.value.__notes__[0]


def test_weight_trend_matches_numpy_and_scipy():
    np = pytest.importorskip("numpy")
    stats = pytest.importorskip("scipy.stats")
    times = [0, 1, 2, 4, 7, 11]  # unequally spaced
    S = dict(zip(times, _hand_snapshots().values()))
    H = nb.weight_trend(S, pvalue=True)
    for u, v, d in H.edges(data=True):
        ks = [k for k, G in enumerate(S.values()) if u in G and v in G]
        y = np.array(
            [
                S[times[k]][u][v]["weight"] if S[times[k]].has_edge(u, v) else 0.0
                for k in ks
            ]
        )
        assert d["weight_mean"] == pytest.approx(y.mean())
        assert d["weight_std"] == pytest.approx(y.std())
        assert d["weight_cv"] == pytest.approx(y.std() / y.mean())
        assert d["weight_slope"] == pytest.approx(np.polyfit(ks, y, 1)[0])
        tx = [times[k] for k in ks]
        assert d["weight_slope_time"] == pytest.approx(np.polyfit(tx, y, 1)[0])
        if len(ks) >= 3:
            assert d["weight_slope_pvalue"] == pytest.approx(
                stats.linregress(ks, y).pvalue
            )
        else:
            assert math.isnan(d["weight_slope_pvalue"])
    # One eligible step: no slope; mean 0 gives a NaN coefficient of variation.
    S = [nx.Graph([(0, 1, {"weight": 2})]), nx.Graph([(1, 2, {"weight": 0})])]
    S[1].add_node(0)
    H = nb.weight_trend(S)
    assert math.isnan(H[1][2]["weight_slope"]) and math.isnan(H[1][2]["weight_cv"])
    assert H[0][1]["weight_slope"] == -2.0
    # Datetime steps: slopes per step only.
    day = datetime(2024, 1, 1)
    H = nb.weight_trend({day + timedelta(days=k): G for k, G in enumerate(S)})
    assert "weight_slope_time" not in H[0][1] and H[0][1]["weight_slope"] == -2.0


def test_input_errors():
    S = _hand_snapshots()
    with pytest.raises(nx.NetworkXNotImplemented):
        nb.observation_fraction({0: S[0], 1: nx.MultiGraph(S[10])})
    with pytest.raises(ValueError, match="freq= and origin="):
        nb.persistence(S, freq=10)
    with pytest.raises(ValueError, match="ineligible"):
        nb.persistence(S, ineligible="ignore")
    S[10]["a"]["b"]["weight"] = "heavy"
    for func in (nb.weight_trend, partial(nb.observation_fraction, weight="weight")):
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            func(S)
    with pytest.raises(nx.NetworkXError, match="simple graph"):
        nb.observation_fraction(nx.Graph([(0, 1)]))


def test_score_keys_and_describe():
    # Every registered score key is produced by its method.
    G = nx.les_miserables_graph()
    U = nx.Graph(G.edges)
    for name, key in SCORE_KEYS.items():
        kind = METHODS[name].kind
        assert kind in ("weighted", "unweighted"), name
        kwargs = {
            "global_threshold_filter": {"threshold": 2},
            "modularity_backbone": {"seed": 1},
        }.get(name, {})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            H = getattr(nb, name)(G if kind == "weighted" else U, **kwargs)
        assert all(key in d for *_, d in H.edges(data=True)), name

    S = _hand_snapshots()
    E, _ = _events_and_spans(nx.MultiGraph, [10 * k for k in range(6)], 10)
    H = nb.backbone_stability(E, "disparity_filter", freq=10, threshold=0.5)
    B = nb.threshold_filter(H, "stability", 0.5, mode="above")
    text = nb.describe_backbone("backbone_stability", original=H, backbone=B)
    assert "6 time steps, given as timestamped events binned with freq=10" in text
    assert "scored with the disparity filter (Serrano et al., 2009)" in text
    assert "Serrano, M. A." in text  # the inner method's reference is listed
    assert f"retains {B.number_of_edges()} of 5 edges" in text
    assert "total edge weight" not in text
    text = nb.describe_backbone(
        "observation_fraction", backbone=nb.observation_fraction(S)
    )
    assert "6 time steps, given as snapshots" in text and "eligible steps" in text
    with pytest.raises(TypeError, match="scored graph"):
        nb.describe_backbone("persistence", original=S, backbone=nb.persistence(S))


def test_scores_need_only_networkx():
    _run(
        """
        import networkx as nx
        import nx_backbone as nb
        S = [nx.Graph([(0, 1), (1, 2)]), nx.Graph([(0, 1)]), nx.Graph([(0, 1), (1, 2)])]
        assert nb.observation_fraction(S)[1][2]["n_eligible"] == 2  # 2 is absent at 1
        assert nb.persistence(S)[0][1]["persistence"] == 1.0
        W = [G.copy() for G in S]
        for G in W:
            nx.set_edge_attributes(G, 2.0, "weight")
        H = nb.backbone_stability(W, "global_sparsification", s=0.5)
        assert H[0][1]["n_eligible"] == 3
        for func in (nb.weight_trend, nb.edge_persistence_hmm):
            try:
                func(S)
            except ImportError as err:
                assert "numpy" in str(err)
            else:
                raise AssertionError("expected ImportError")
        """
    )


def test_numpy_and_pure_python_paths_agree():
    pytest.importorskip("numpy")
    rng = random.Random(7)
    for trial in range(30):
        cls = nx.DiGraph if trial % 2 else nx.Graph
        S = []
        for _ in range(rng.randint(1, 8)):
            G = cls()
            G.add_nodes_from(n for n in range(7) if rng.random() < 0.7)
            nodes = list(G)
            for _ in range(rng.randint(0, 12) if nodes else 0):
                G.add_edge(
                    rng.choice(nodes), rng.choice(nodes), weight=rng.randint(0, 3)
                )
            S.append(G)
        if not any(G.number_of_edges() for G in S):
            continue
        calls = [
            (nb.observation_fraction, OBS, {"weight": "weight"}),
            (nb.persistence, RUNS, {}),
            (nb.persistence, RUNS, {"ineligible": "break"}),
            (nb.backbone_stability, STAB, {"method": "strongest_n_ties", "n": 1}),
        ]
        for func, names, kw in calls:
            fast = _attrs(func(S, **kw), names)
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(temporal, "_numpy", lambda: None)
                _assert_same(_attrs(func(S, **kw), names), fast)


def test_tutorial_and_concepts_examples_run():
    pytest.importorskip("numpy")
    for path in ("tutorials/temporal_backbones.rst", "concepts.rst"):
        result = doctest.testfile(
            str(ROOT / "docs" / path),
            module_relative=False,
            optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        )
        assert result.attempted > 0 and result.failed == 0, path
