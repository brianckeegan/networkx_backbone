"""Round-trip tests of the hypergraph adapters against the real libraries.

Each test needs one of xgi, HyperNetX, HypergraphX, or HAT and is skipped when
it is missing.  Install them all with ``pip install -e ".[interop,test]"`` and
run just these tests with ``pytest -m interop`` (the "Interop" CI workflow does
this on pull requests and weekly).  ``tests/test_hypergraph_io.py`` covers the
adapter logic with stand-in modules, so it runs without the libraries.

Set ``NETWORKX_BACKBONE_REQUIRE_INTEROP=1`` (as CI does) to make a library
that fails to import an error instead of a skip.
"""

import importlib
import json
import os
import warnings

import pytest

import networkx_backbone as nb

pytestmark = [
    pytest.mark.interop,
    # The libraries' own deprecation and future warnings are not ours to fix.
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
]


def _import(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if os.environ.get("NETWORKX_BACKBONE_REQUIRE_INTEROP"):
            return importlib.import_module(name)
        return pytest.importorskip(name)


# Hypergraphs with integer, string, tuple, and mixed-type node labels; the
# first repeats a hyperedge and the second has a singleton hyperedge.
CASES = {
    "int": ([(1, 2, 3), (2, 3, 4), (5, 6), (1, 2, 3)], [2.5, 1, 4, 0.5]),
    "str": ([("a", "b"), ("b", "c", "d"), ("d",)], [1, 2, 3]),
    "tuple": ([((0, 1), (0, 2)), ((0, 2), (1, 1))], [1.5, 2]),
    "mixed": ([(1, "a"), ("a", 2)], [3, 4]),
}


def _expected(name):
    edges, weights = CASES[name]
    return [frozenset(e) for e in edges], weights


def _merged(edges, weights):
    """Hyperedges and summed weights after merging repeats (HypergraphX)."""
    out = {}
    for e, w in zip(edges, weights):
        out[e] = out.get(e, 0) + w
    return list(out), list(out.values())


def _adapters():
    return {
        "xgi": ("xgi", nb.to_xgi, nb.from_xgi),
        "hypernetx": ("hypernetx", nb.to_hypernetx, nb.from_hypernetx),
        "hypergraphx": ("hypergraphx", nb.to_hypergraphx, nb.from_hypergraphx),
        "HAT": ("HAT", nb.to_hat, nb.from_hat),
    }


# ---------------------------------------------------------------------------
# from_X(to_X(h)): hyperedges, weights, and node labels survive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("lib", list(_adapters()))
def test_to_from_roundtrip(lib, case):
    module, to_fn, from_fn = _adapters()[lib]
    _import(module)
    edges, weights = _expected(case)
    if lib == "hypergraphx" and case == "mixed":
        # HypergraphX sorts each hyperedge's nodes.
        with pytest.raises(TypeError, match="mutually comparable"):
            to_fn(edges)
        return
    if lib == "hypergraphx":
        # No multi-edges: repeats merge and their weights add up.
        edges, weights = _merged(edges, weights)

    got_edges, got_weights = from_fn(to_fn(edges, weights=weights), return_weights=True)
    assert got_edges == edges
    assert got_weights == pytest.approx(weights)
    # Unweighted: every weight reads back as 1, and labels are unchanged.
    plain_edges, plain_weights = from_fn(to_fn(edges), return_weights=True)
    assert plain_edges == edges
    assert plain_weights == [1] * len(edges)
    assert from_fn(to_fn(edges)) == edges


def test_hypergraphx_repeated_hyperedges_become_weight():
    _import("hypergraphx")
    H = nb.to_hypergraphx([(1, 2), (2, 3), (2, 1)])
    assert H.is_weighted()
    assert nb.from_hypergraphx(H, return_weights=True) == (
        [frozenset({1, 2}), frozenset({2, 3})],
        [2, 1],
    )
    assert not nb.to_hypergraphx([(1, 2), (2, 3)]).is_weighted()


def test_to_adapters_drop_empty_hyperedges_with_their_weights():
    for module, to_fn, from_fn in _adapters().values():
        _import(module)
        got = from_fn(
            to_fn([(1, 2), (), (2, 3)], weights=[5, 9, 7]), return_weights=True
        )
        assert got == ([frozenset({1, 2}), frozenset({2, 3})], [5, 7]), module


# ---------------------------------------------------------------------------
# to_X(from_X(native)): a hypergraph built with the library's own API
# ---------------------------------------------------------------------------


def test_xgi_native_roundtrip():
    xgi = _import("xgi")
    H = xgi.Hypergraph()
    H.add_edges_from({"e1": ["a", "b"], "e2": ["b", "c", "d"], "e3": ["a", "b"]})
    H.set_edge_attributes({"e1": 2.0, "e2": 0.5, "e3": 1.0}, name="weight")
    edges, weights = nb.from_xgi(H, return_weights=True)
    assert edges == [frozenset("ab"), frozenset("bcd"), frozenset("ab")]
    assert weights == [2.0, 0.5, 1.0]
    H2 = nb.to_xgi(edges, weights=weights)
    assert list(H2.edges.members()) == list(H.edges.members())
    assert set(H2.nodes) == set(H.nodes)
    assert [H2.edges[e]["weight"] for e in H2.edges] == weights


def test_hypernetx_native_roundtrip():
    hnx = _import("hypernetx")
    H = hnx.Hypergraph(
        {"e1": ["a", "b"], "e2": ["b", "c", "d"]},
        edge_properties={"e1": {"weight": 2.0}, "e2": {"weight": 0.5}},
    )
    edges, weights = nb.from_hypernetx(H, return_weights=True)
    assert edges == [frozenset("ab"), frozenset("bcd")]
    assert weights == [2.0, 0.5]
    H2 = nb.to_hypernetx(edges, weights=weights)
    assert [set(m) for m in H2.incidence_dict.values()] == [
        set(m) for m in H.incidence_dict.values()
    ]
    assert set(H2.nodes) == set(H.nodes)
    assert nb.from_hypernetx(H2, return_weights=True) == (edges, weights)


def test_hypergraphx_native_roundtrip():
    hgx = _import("hypergraphx")
    H = hgx.Hypergraph(
        edge_list=[("a", "b"), ("b", "c", "d")], weighted=True, weights=[2.0, 0.5]
    )
    edges, weights = nb.from_hypergraphx(H, return_weights=True)
    assert edges == [frozenset("ab"), frozenset("bcd")]
    assert weights == [2.0, 0.5]
    H2 = nb.to_hypergraphx(edges, weights=weights)
    assert list(H2.get_edges()) == list(H.get_edges())
    assert list(H2.get_weights()) == list(H.get_weights())
    assert set(H2.get_nodes()) == set(H.get_nodes())


def test_hat_native_roundtrip():
    HAT = _import("HAT")
    # Built from labelled edges, HAT keeps the labels in nodes["Names"].
    H = HAT.Hypergraph(edge_list=[["a", "b"], ["b", "c", "d"]])
    edges = nb.from_hat(H)
    assert edges == [frozenset("ab"), frozenset("bcd")]
    H2 = nb.to_hat(edges, weights=[2.0, 0.5])
    assert H2.incidence_matrix.shape == (4, 2)
    assert list(H2.edge_weights) == [2.0, 0.5]
    assert nb.from_hat(H2, return_weights=True) == (edges, [2.0, 0.5])
    # An unlabelled HAT hypergraph reads back with positional labels.
    H3 = HAT.Hypergraph(edge_list=[[0, 1], [1, 2, 3]])
    assert nb.from_hat(H3) == [frozenset({0, 1}), frozenset({1, 2, 3})]


def test_hat_keeps_repeated_hyperedges():
    # Regression: HAT drops duplicate columns unless compress=False.
    _import("HAT")
    H = nb.to_hat([(1, 2), (2, 3), (1, 2)])
    assert H.nedges == 3
    assert nb.from_hat(H) == [frozenset({1, 2}), frozenset({2, 3}), frozenset({1, 2})]


def test_xgi_tuple_node_labels():
    # Regression: xgi read a hyperedge of two tuple labels as (members, edge_id).
    _import("xgi")
    edges = [((0, 1), (0, 2)), ((0, 2), (1, 1))]
    H = nb.to_xgi(edges)
    assert H.num_edges == 2
    assert set(H.nodes) == {(0, 1), (0, 2), (1, 1)}


# ---------------------------------------------------------------------------
# HIF interchange with each library's own reader and writer
# ---------------------------------------------------------------------------

HIF_EDGES = [("a", "b", "c"), ("c", "d"), ("a", "b", "c")]
HIF_WEIGHTS = [2.5, 1, 4]
HIF_EXPECTED = [frozenset(e) for e in HIF_EDGES]


@pytest.fixture
def hif_path(tmp_path):
    path = tmp_path / "ours.hif.json"
    nb.write_hif(HIF_EDGES, path, weights=HIF_WEIGHTS)
    return path


def test_hif_xgi(hif_path, tmp_path):
    xgi = _import("xgi")
    # xgi reads our HIF.  Its reader keeps only edge "attrs", not the spec's
    # "weight" field, so the weights read back as the default.
    H = xgi.read_hif(hif_path)
    assert nb.from_xgi(H) == HIF_EXPECTED
    # We read xgi's HIF, including weights (xgi writes them under "attrs").
    out = tmp_path / "xgi.hif.json"
    xgi.write_hif(nb.to_xgi(HIF_EDGES, weights=HIF_WEIGHTS), out)
    assert nb.read_hif(out, return_weights=True) == (HIF_EXPECTED, HIF_WEIGHTS)


def test_hif_hypernetx(hif_path, tmp_path):
    hnx = _import("hypernetx")
    H = hnx.from_hif(filename=str(hif_path))
    assert nb.from_hypernetx(H, return_weights=True) == (HIF_EXPECTED, HIF_WEIGHTS)
    out = tmp_path / "hnx.hif.json"
    hnx.to_hif(nb.to_hypernetx(HIF_EDGES, weights=HIF_WEIGHTS), filename=str(out))
    assert nb.read_hif(out, return_weights=True) == (HIF_EXPECTED, HIF_WEIGHTS)


def test_hif_hypergraphx(hif_path):
    _import("hypergraphx")
    from hypergraphx.readwrite import read_hif

    # Regression: HypergraphX's reader needs "nodes" and "edges" keys, which
    # write_hif used to omit.  It relabels nodes 0..n-1 (the HIF IDs are kept in
    # the node metadata), merges the repeated hyperedge, and ignores weights.
    H = read_hif(str(hif_path))
    label = {n: H.get_node_metadata(n)["node"] for n in H.get_nodes()}
    assert {frozenset(label[v] for v in e) for e in H.get_edges()} == set(HIF_EXPECTED)
    # HypergraphX's own writer (1.8) does not write the incidences, so there is
    # nothing to read back from it.


def test_hif_hat(hif_path):
    HAT = _import("HAT")
    with open(hif_path) as handle:
        H = HAT.Hypergraph.from_hif(json.load(handle))
    assert nb.from_hat(H, return_weights=True) == (HIF_EXPECTED, HIF_WEIGHTS)
    # HAT writes positional node IDs and keeps the labels in attrs["Names"].
    hif = json.loads(
        json.dumps(
            nb.to_hat(HIF_EDGES, weights=HIF_WEIGHTS).to_hif(), default=_json_default
        )
    )
    names = {rec["node"]: rec["attrs"]["Names"] for rec in hif["nodes"]}
    edges, weights = nb.read_hif(hif, return_weights=True)
    assert [frozenset(names[v] for v in e) for e in edges] == HIF_EXPECTED
    assert weights == HIF_WEIGHTS


def _json_default(value):
    # HAT.to_hif returns numpy scalars and arrays.
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value))
