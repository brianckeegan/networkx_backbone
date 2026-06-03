"""Tests for hypergraph ingestion / interoperability adapters."""

import importlib

import networkx as nx
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


def test_hypergraph_to_bipartite_feeds_sdsm():
    B, nodes = nb.hypergraph_to_bipartite(HYPEREDGES)
    scored = nb.sdsm(B, agent_nodes=nodes)
    assert all("sdsm_pvalue" in d for _, _, d in scored.edges(data=True))


# ---------------------------------------------------------------------------
# HIF
# ---------------------------------------------------------------------------


def test_hif_roundtrip_dict():
    hif = nb.write_hif(HYPEREDGES)
    assert hif["network-type"] == "undirected"
    assert set(nb.read_hif(hif)) == EXPECTED


def test_hif_roundtrip_with_weights():
    hif = nb.write_hif([(1, 2), (3, 4)], weights=[5, 2])
    edges, weights = nb.read_hif(hif, return_weights=True)
    assert dict(zip((tuple(sorted(e)) for e in edges), weights)) == {
        (1, 2): 5,
        (3, 4): 2,
    }


def test_hif_roundtrip_file(tmp_path):
    path = tmp_path / "h.json"
    nb.write_hif(HYPEREDGES, path)
    assert set(nb.read_hif(path)) == EXPECTED


def test_write_hif_weight_length_checked():
    with pytest.raises(ValueError):
        nb.write_hif([(1, 2), (3, 4)], weights=[1])


def test_read_hif_bad_source_raises():
    with pytest.raises(TypeError):
        nb.read_hif(12345)


# ---------------------------------------------------------------------------
# Cross-library interoperability (run when a library is available)
# ---------------------------------------------------------------------------


def test_hif_crosscheck_with_xgi(tmp_path):
    xgi = pytest.importorskip("xgi")
    path = tmp_path / "h.json"
    # xgi reads HIF we wrote.
    nb.write_hif(HYPEREDGES, path)
    Hx = xgi.read_hif(path)
    assert {frozenset(m) for m in Hx.edges.members()} == EXPECTED
    # We read HIF xgi wrote.
    xgi.write_hif(xgi.Hypergraph([list(e) for e in HYPEREDGES]), path)
    assert set(nb.read_hif(path)) == EXPECTED


ADAPTERS = [
    ("xgi", nb.to_xgi, nb.from_xgi),
    ("hypernetx", nb.to_hypernetx, nb.from_hypernetx),
    ("hypergraphx", nb.to_hypergraphx, nb.from_hypergraphx),
    ("HAT", nb.to_hat, nb.from_hat),
]


@pytest.mark.parametrize("module,to_fn,from_fn", ADAPTERS, ids=[a[0] for a in ADAPTERS])
def test_adapter_roundtrip_or_importerror(module, to_fn, from_fn):
    try:
        importlib.import_module(module)
    except ImportError:
        # Absent (or broken) optional dependency raises a helpful ImportError.
        with pytest.raises(ImportError, match=module):
            to_fn(HYPEREDGES)
        return

    obj = to_fn(HYPEREDGES)
    recovered = from_fn(obj)
    if module == "HAT":
        # HAT is positional: labels become indices, so compare the order multiset.
        assert sorted(len(e) for e in recovered) == sorted(len(set(e)) for e in HYPEREDGES)
    else:
        assert set(recovered) == EXPECTED
