"""Tests for hypergraph ingestion / interoperability adapters."""

import io
import json
import sys
import types

import networkx as nx
import numpy as np
import pytest

import networkx_backbone as nb

HYPEREDGES = [(1, 2, 3), (2, 3, 4), (5, 6)]
EXPECTED = {frozenset(e) for e in HYPEREDGES}


# ---------------------------------------------------------------------------
# Incidence bipartite graph
# ---------------------------------------------------------------------------


def test_hypergraph_to_bipartite_structure():
    B, nodes = nb.hypergraph_to_bipartite(HYPEREDGES)
    assert nx.is_bipartite(B)
    assert nodes == [1, 2, 3, 4, 5, 6]
    node_part = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
    edge_part = {n for n, d in B.nodes(data=True) if d["bipartite"] == 1}
    assert node_part == set(nodes)
    assert len(edge_part) == 3
    # Each hyperedge node records its members.
    assert any(B.nodes[e]["members"] == frozenset({1, 2, 3}) for e in edge_part)


def test_hypergraph_to_bipartite_tuple_edge_labels():
    B, _ = nb.hypergraph_to_bipartite(HYPEREDGES)
    assert B.nodes[("he", 0)]["members"] == frozenset({1, 2, 3})
    assert B.nodes[("he", 2)]["bipartite"] == 1
    B2, _ = nb.hypergraph_to_bipartite(HYPEREDGES, edge_prefix="edge")
    assert ("edge", 1) in B2


def test_hypergraph_to_bipartite_string_label_no_collision():
    # Regression: a node called "he0" used to merge with hyperedge 0's node,
    # creating a self-loop and a non-bipartite graph.
    B, nodes = nb.hypergraph_to_bipartite([("he0", "b"), ("b", "c")])
    assert nx.number_of_selfloops(B) == 0
    assert nx.is_bipartite(B)
    assert nodes == ["b", "c", "he0"]
    assert B.nodes["he0"]["bipartite"] == 0
    assert set(B[("he", 0)]) == {"he0", "b"}
    assert B.number_of_edges() == 4


def test_hypergraph_to_bipartite_tuple_label_collision_raises():
    with pytest.raises(ValueError, match="edge_prefix"):
        nb.hypergraph_to_bipartite([(("he", 0), "b")])
    B, _ = nb.hypergraph_to_bipartite([(("he", 0), "b")], edge_prefix="x")
    assert nx.is_bipartite(B) and nx.number_of_selfloops(B) == 0


def test_hypergraph_to_bipartite_mixed_labels_and_empty_edges():
    B, nodes = nb.hypergraph_to_bipartite([(1, "a"), (), ("a", (2, 3))])
    assert set(nodes) == {1, "a", (2, 3)}
    assert nodes == sorted(nodes, key=repr)
    assert sum(1 for _, d in B.nodes(data=True) if d["bipartite"] == 1) == 2


def test_hypergraph_to_bipartite_feeds_sdsm():
    B, nodes = nb.hypergraph_to_bipartite(HYPEREDGES)
    scored = nb.sdsm(B, agent_nodes=nodes)
    assert all("sdsm_pvalue" in d for _, _, d in scored.edges(data=True))


# ---------------------------------------------------------------------------
# HIF
# ---------------------------------------------------------------------------


def test_hif_roundtrip_dict_file_and_file_like(tmp_path):
    hif = nb.write_hif(HYPEREDGES)
    assert hif["network-type"] == "undirected"
    assert set(nb.read_hif(hif)) == EXPECTED
    # Weights.
    hif = nb.write_hif([(1, 2), (3, 4)], weights=[5, 2])
    edges, weights = nb.read_hif(hif, return_weights=True)
    assert dict(zip((tuple(sorted(e)) for e in edges), weights)) == {
        (1, 2): 5,
        (3, 4): 2,
    }
    # A path.
    path = tmp_path / "h.json"
    nb.write_hif(HYPEREDGES, path)
    assert set(nb.read_hif(path)) == EXPECTED
    # A file-like object, which write_hif returns.
    buf = io.StringIO()
    assert nb.write_hif(HYPEREDGES, buf, weights=[1, 2, 3]) is buf
    buf.seek(0)
    edges, weights = nb.read_hif(buf, return_weights=True)
    assert set(edges) == EXPECTED and weights == [1, 2, 3]


def test_write_hif_skips_empty_hyperedges():
    edges, weights = nb.read_hif(
        nb.write_hif([(1, 2), (), (2, 3)], weights=[4, 9, 5]), return_weights=True
    )
    assert edges == [frozenset({1, 2}), frozenset({2, 3})]
    assert weights == [4, 5]


def test_write_hif_weight_length_checked():
    with pytest.raises(ValueError):
        nb.write_hif([(1, 2), (3, 4)], weights=[1])


def test_read_hif_bad_source_raises():
    with pytest.raises(TypeError):
        nb.read_hif(12345)


def test_hif_numpy_weights_and_ids_to_file(tmp_path):
    # Regression: numpy scalars are not JSON-serialisable.
    path = tmp_path / "np.json"
    edges = [np.array([1, 2]), np.array([2, 3, 4])]
    nb.write_hif(edges, path, weights=np.array([3, 1]))
    got, weights = nb.read_hif(path, return_weights=True)
    assert got == [frozenset({1, 2}), frozenset({2, 3, 4})]
    assert weights == [3, 1]
    assert all(type(w) is int for w in weights)
    assert all(type(v) is int for e in got for v in e)

    hif = nb.write_hif(edges, weights=np.array([1.5, 2.0]))
    json.dumps(hif)  # the returned dict is JSON-serialisable too
    assert [rec["weight"] for rec in hif["edges"]] == [1.5, 2.0]


def test_hif_tuple_node_ids_roundtrip(tmp_path):
    # Regression: tuple labels came back as (unhashable) lists after JSON.
    H = [((0, 1), (0, 2)), ((0, 2), ("a", (1, 2)))]
    path = tmp_path / "tuples.hif"
    nb.write_hif(H, str(path))
    assert nb.read_hif(path) == [frozenset(e) for e in H]
    # JSON-decoded dicts behave the same as files.
    assert nb.read_hif(json.loads(json.dumps(nb.write_hif(H)))) == [
        frozenset(e) for e in H
    ]


def test_hif_mixed_int_and_str_labels_stay_distinct():
    hif = json.loads(json.dumps(nb.write_hif([(1, "1")])))
    assert nb.read_hif(hif) == [frozenset({1, "1"})]


def test_read_hif_directed_raises():
    hif = {
        "network-type": "directed",
        "incidences": [
            {"edge": 0, "node": "a", "direction": "head"},
            {"edge": 0, "node": "b", "direction": "tail"},
        ],
    }
    with pytest.raises(nx.NetworkXNotImplemented):
        nb.read_hif(hif)


def test_write_hif_directed_or_unknown_type_raises():
    with pytest.raises(nx.NetworkXNotImplemented):
        nb.write_hif(HYPEREDGES, network_type="directed")
    with pytest.raises(ValueError, match="network-type"):
        nb.write_hif(HYPEREDGES, network_type="hyper")
    with pytest.raises(ValueError, match="network-type"):
        nb.read_hif({"network-type": "bogus", "incidences": []})


def test_read_hif_spec_variants():
    spec = {
        "network-type": "asc",
        "incidences": [
            {"edge": "e1", "node": "a"},
            {"edge": "e1", "node": "b"},
            {"edge": "e2", "node": "b"},
            {"edge": "e3", "node": "c"},
        ],
        "edges": [
            {"edge": "e1", "attrs": {"weight": 4}},
            {"edge": "e2", "weight": 2.5},
            {"edge": "e3", "attrs": {}},
        ],
    }
    edges, weights = nb.read_hif(spec, return_weights=True)
    assert edges == [frozenset("ab"), frozenset("b"), frozenset("c")]
    assert weights == [4, 2.5, 1]
    # Missing network-type defaults to undirected.
    assert nb.read_hif({"incidences": [{"edge": 0, "node": 1}]}) == [frozenset({1})]
    assert nb.write_hif([(1, 2)], network_type="asc")["network-type"] == "asc"


def test_write_hif_always_writes_nodes_and_edges():
    # Regression: the HAT and HypergraphX HIF readers raise KeyError without
    # "nodes" and "edges", which were omitted (edges unless weighted).
    hif = nb.write_hif([("b", "a"), (), ("a", "c")])
    assert hif["nodes"] == [{"node": "a"}, {"node": "b"}, {"node": "c"}]
    assert hif["edges"] == [{"edge": 0}, {"edge": 1}]
    assert nb.read_hif(hif, return_weights=True) == (
        [frozenset("ab"), frozenset("ac")],
        [1, 1],
    )


# ---------------------------------------------------------------------------
# Adapter logic, exercised with lightweight stand-in modules.  The real
# libraries are tested in test_hypergraph_interop.py (``pytest -m interop``).
# ---------------------------------------------------------------------------


def _install(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


INPUT = [(1, 2, 3), (), (3, 4), (1, 2, 3, 3)]  # empty edge dropped, dup kept
INPUT_WEIGHTS = [2, 99, 1.5, 3]
CLEAN = [frozenset({1, 2, 3}), frozenset({3, 4}), frozenset({1, 2, 3})]
CLEAN_WEIGHTS = [2, 1.5, 3]


def test_xgi_adapter_with_stub(monkeypatch):
    class Hypergraph:
        def __init__(self):
            self._members = {}
            self._attrs = {}
            outer = self

            class _Edges:
                def members(self, dtype=None):
                    assert dtype is dict
                    return dict(outer._members)

                def __getitem__(self, edge):
                    return outer._attrs[edge]

            self.edges = _Edges()

        def add_edges_from(self, ebunch):
            # Regression: a list of edges made of two tuple labels was read by
            # xgi as (members, edge_id); the {id: members} form is unambiguous.
            assert isinstance(ebunch, dict)
            for edge, members in ebunch.items():
                self._members[edge] = set(members)
                self._attrs[edge] = {}

        def set_edge_attributes(self, values, name=None):
            for edge, value in values.items():
                self._attrs[edge][name] = value

    _install(monkeypatch, "xgi", Hypergraph=Hypergraph)
    H = nb.to_xgi(INPUT)
    assert H._members == {i: set(e) for i, e in enumerate(CLEAN)}
    assert nb.from_xgi(H) == CLEAN
    assert nb.from_xgi(H, return_weights=True) == (CLEAN, [1, 1, 1])
    H = nb.to_xgi(INPUT, weights=INPUT_WEIGHTS)
    assert nb.from_xgi(H, return_weights=True) == (CLEAN, CLEAN_WEIGHTS)
    with pytest.raises(ValueError, match="same length"):
        nb.to_xgi(INPUT, weights=[1])


def test_hypernetx_adapter_with_stub(monkeypatch):
    class Hypergraph:
        def __init__(self, setsystem, edge_properties=None):
            self.incidence_dict = dict(setsystem)
            props = edge_properties or {}
            # HyperNetX stores weights as floats, defaulting to 1.0.
            self.edges = {
                e: types.SimpleNamespace(
                    weight=float(props.get(e, {}).get("weight", 1))
                )
                for e in setsystem
            }

    _install(monkeypatch, "hypernetx", Hypergraph=Hypergraph)
    H = nb.to_hypernetx(INPUT)
    assert list(H.incidence_dict) == [0, 1, 2]
    assert nb.from_hypernetx(H) == CLEAN
    assert nb.from_hypernetx(H, return_weights=True) == (CLEAN, [1, 1, 1])
    H = nb.to_hypernetx(INPUT, weights=INPUT_WEIGHTS)
    assert nb.from_hypernetx(H, return_weights=True) == (CLEAN, CLEAN_WEIGHTS)


def test_hypergraphx_adapter_with_stub(monkeypatch):
    class Hypergraph:
        def __init__(self, edge_list=None, weighted=False, weights=None):
            # Like HypergraphX: edges are sorted tuples and must be unique.
            self._edges = [tuple(sorted(e)) for e in edge_list]
            assert len(set(self._edges)) == len(self._edges)
            self.weighted = weighted
            self._weights = dict(zip(self._edges, weights or [1] * len(edge_list)))

        def get_edges(self):
            return list(self._edges)

        def get_weight(self, edge):
            return self._weights[tuple(sorted(edge))]

    _install(monkeypatch, "hypergraphx", Hypergraph=Hypergraph)
    # Regression: repeated hyperedges were passed through, and HypergraphX
    # silently merged them; now they merge here, with weights summed.
    H = nb.to_hypergraphx(INPUT)
    assert H.weighted
    assert nb.from_hypergraphx(H) == CLEAN[:2]
    assert nb.from_hypergraphx(H, return_weights=True) == (CLEAN[:2], [2, 1])
    H = nb.to_hypergraphx(INPUT, weights=INPUT_WEIGHTS)
    assert nb.from_hypergraphx(H, return_weights=True) == (CLEAN[:2], [5, 1.5])
    H = nb.to_hypergraphx([(1, 2), (2, 3)])
    assert not H.weighted
    assert nb.from_hypergraphx(H, return_weights=True) == (
        [frozenset({1, 2}), frozenset({2, 3})],
        [1, 1],
    )
    with pytest.raises(TypeError, match="mutually comparable"):
        nb.to_hypergraphx([(1, "a")])


class _Frame:
    """Just enough of a pandas DataFrame for HAT's nodes/edges tables."""

    def __init__(self, n):
        self._n = n
        self._data = {}

    @property
    def columns(self):
        return list(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        assert len(value) == self._n
        self._data[key] = list(value)


def test_hat_adapter_with_stub(monkeypatch):
    class Hypergraph:
        def __init__(self, incidence_matrix=None, compress=True):
            # Regression: HAT's default compress=True drops repeated columns.
            assert compress is False
            self.incidence_matrix = incidence_matrix
            self.nodes = _Frame(incidence_matrix.shape[0])
            self.edges = _Frame(incidence_matrix.shape[1])

    _install(monkeypatch, "HAT", Hypergraph=Hypergraph)
    H = nb.to_hat([("b", "a"), (), ("c", "b", "d"), ("a", "d"), ("a", "b")])
    M = H.incidence_matrix
    # Rows follow sorted labels a, b, c, d; empty edge dropped, repeat kept.
    assert M.tolist() == [[1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 0, 0], [0, 1, 1, 0]]
    assert H.nodes["Names"] == ["a", "b", "c", "d"]
    assert "weight" not in H.edges.columns
    # Regression: labels used to come back as row indices.
    expected = [frozenset("ab"), frozenset("bcd"), frozenset("ad"), frozenset("ab")]
    assert nb.from_hat(H) == expected
    assert nb.from_hat(H, return_weights=True) == (expected, [1, 1, 1, 1])

    H = nb.to_hat([((0, 1), (0, 2)), ((0, 2), (1, 1))], weights=[2, 0.5])
    assert H.nodes["Names"] == [(0, 1), (0, 2), (1, 1)]  # tuples stay tuples
    assert H.edges["weight"] == [2, 0.5]
    assert nb.from_hat(H, return_weights=True) == (
        [frozenset({(0, 1), (0, 2)}), frozenset({(0, 2), (1, 1)})],
        [2, 0.5],
    )


def test_from_hat_nodes_column_and_dropped_columns():
    # HAT.Hypergraph.from_hif keeps the HIF node IDs in nodes["Nodes"].
    class H:
        incidence_matrix = np.array([[1, 0, 1], [1, 0, 0], [0, 0, 1]])
        nodes = _Frame(3)
        edges = _Frame(3)

    H.nodes["Nodes"] = ["x", "y", "z"]
    H.edges["weight"] = [4, 5, 6]
    # The all-zero column is dropped together with its weight.
    assert nb.from_hat(H(), return_weights=True) == (
        [frozenset("xy"), frozenset("xz")],
        [4, 6],
    )


def test_from_hat_incidence_matrix_variants():
    M = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]])
    expected = [frozenset({0, 1}), frozenset({1, 2})]  # all-zero column dropped
    for attr in ("IM", "incidence_matrix", "incidenceMatrix"):
        plain = types.SimpleNamespace(**{attr: M})
        method = types.SimpleNamespace(**{attr: lambda: M})
        assert nb.from_hat(plain) == expected, attr
        assert nb.from_hat(method) == expected, attr


def test_from_hat_sparse_and_missing_matrix():
    sparse = pytest.importorskip("scipy.sparse")

    class H:
        IM = sparse.csr_matrix(np.array([[1, 0], [1, 1], [0, 1]]))

    assert nb.from_hat(H()) == [frozenset({0, 1}), frozenset({1, 2})]
    with pytest.raises(AttributeError, match="incidence matrix"):
        nb.from_hat(object())


def test_adapter_missing_library_importerror(monkeypatch):
    for module, to_fn, pip_name in [
        ("xgi", nb.to_xgi, "xgi"),
        ("hypernetx", nb.to_hypernetx, "hypernetx"),
        ("hypergraphx", nb.to_hypergraphx, "hypergraphx"),
        ("HAT", nb.to_hat, "HypergraphAnalysisToolbox"),
    ]:
        monkeypatch.setitem(sys.modules, module, None)  # forces ImportError
        with pytest.raises(ImportError, match=f"pip install {pip_name}"):
            to_fn(HYPEREDGES)
