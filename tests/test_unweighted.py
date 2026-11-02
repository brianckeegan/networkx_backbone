"""Reference-implementation and regression tests for networkx_backbone.unweighted.

Shared behavior (attributes, directed/multigraph rejection, empty graphs,
mixed labels, no input mutation) is covered by ``test_api_contract.py``.
"""

import itertools
import math
import os
import random

import networkx as nx
import pytest

from networkx_backbone import (
    boolean_filter,
    local_degree,
    lspar,
    sparsify,
    threshold_filter,
)

# ---------------------------------------------------------------------
# Straightforward reference implementations of the per-node top-k rules
# ---------------------------------------------------------------------


def _jaccard(G):
    nbrs = {v: set(G[v]) - {v} for v in G}

    def score(u, v):
        return len(nbrs[u] & nbrs[v]) / len(nbrs[u] | nbrs[v])

    return score


def _neighbor_degree(G):
    def score(u, v):
        return G.degree(v)

    return score


def _reference(G, s, score, rounding):
    """Union over nodes u of u's top-k edges, ties at the cut-off included."""
    kept = set()
    for u in G:
        d = G.degree(u)
        if d == 0:
            continue
        k = max(1, (math.ceil if rounding == "ceil" else math.floor)(d**s))
        ranked = sorted((score(u, v) for v in G[u]), reverse=True)
        cutoff = ranked[k - 1]
        kept |= {frozenset((u, v)) for v in G[u] if score(u, v) >= cutoff}
    return kept


def _kept(H):
    return {frozenset((u, v)) for u, v, keep in H.edges(data="sparsify_keep") if keep}


EXPONENTS = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]


def _test_graphs():
    yield nx.karate_club_graph()
    yield nx.convert_node_labels_to_integers(nx.les_miserables_graph())
    for seed in range(3):
        yield nx.gnm_random_graph(40, 120, seed=seed)


def test_lspar_matches_reference():
    for i, G in enumerate(_test_graphs()):
        G = nx.Graph(G.edges())
        for s in EXPONENTS:
            expected = _reference(G, s, _jaccard(G), "ceil")
            assert _kept(lspar(G, s=s)) == expected, f"graph {i}, s={s}"


def test_local_degree_matches_reference():
    for i, G in enumerate(_test_graphs()):
        G = nx.Graph(G.edges())
        for s in EXPONENTS:
            expected = _reference(G, s, _neighbor_degree(G), "floor")
            assert _kept(local_degree(G, s=s)) == expected, f"graph {i}, s={s}"


def test_sparsify_degree_filter_is_normalization_invariant(karate):
    for s in EXPONENTS:
        ranked = sparsify(karate, escore="jaccard", normalize="rank", s=s)
        raw = sparsify(karate, escore="jaccard", normalize=None, s=s)
        assert _kept(ranked) == _kept(raw) == _kept(lspar(karate, s=s)), s


def test_karate_counts_known_answer():
    G = nx.karate_club_graph()
    assert len(_kept(lspar(G, s=0.5))) == 61
    assert len(_kept(local_degree(G, s=0.3))) == len(
        _reference(G, 0.3, _neighbor_degree(G), "floor")
    )


def test_lspar_score_is_critical_exponent():
    """A single threshold on sparsify_score reproduces any exponent."""
    G = nx.karate_club_graph()
    H = lspar(G, s=0.5)
    for s2 in EXPONENTS:
        via_score = threshold_filter(H, "sparsify_score", s2, mode="below")
        direct = boolean_filter(lspar(G, s=s2), "sparsify_keep")
        assert {frozenset(e) for e in via_score.edges()} == {
            frozenset(e) for e in direct.edges()
        }


def test_local_degree_score_is_critical_exponent():
    G = nx.karate_club_graph()
    H = local_degree(G, s=0.3)
    for s2 in EXPONENTS:
        via_score = {
            frozenset((u, v)) for u, v, sc in H.edges(data="sparsify_score") if sc <= s2
        }
        assert via_score == _kept(local_degree(G, s=s2))


def test_local_degree_ranks_by_neighbor_degree():
    """Regression: the old symmetric max(deg u, deg v) score kept (x, y)."""
    G = nx.Graph()
    G.add_edges_from((0, i) for i in range(1, 6))  # hub 0, degree 5
    G.add_edges_from((7, i) for i in range(8, 13))  # hub 7, degree 5
    G.add_edges_from([("x", 0), ("x", "y"), ("y", 7), ("y", "z")])
    G.add_edges_from([("z", 20), ("z", 21), ("z", 22)])
    # x ranks hub 0 first; y ranks hub 7 first (then z with degree 4).
    H = local_degree(G, s=0.0)
    assert H["x"][0]["sparsify_keep"]
    assert H["y"][7]["sparsify_keep"]
    assert not H["x"]["y"]["sparsify_keep"]
    # Leaves always keep their only edge.
    assert H[0][1]["sparsify_score"] == -math.inf


def test_local_degree_ties_kept_together():
    G = nx.star_graph(3)  # hub 0 with three degree-1 leaves -> all tied
    H = local_degree(G, s=0.0)
    assert all(keep for *_, keep in H.edges(data="sparsify_keep"))


class TestSparsifyOptions:
    def test_unknown_options_raise(self, karate):
        with pytest.raises(ValueError):
            sparsify(karate, filter="invalid")
        with pytest.raises(ValueError, match="normalize"):
            sparsify(karate, normalize="zscore")
        with pytest.raises(ValueError, match="scoring"):
            sparsify(karate, escore="bogus")

    def test_scoring_methods_sparsify(self, karate):
        for method in ["jaccard", "degree", "triangles"]:
            H = sparsify(karate, escore=method, s=0.5)
            backbone = boolean_filter(H, "sparsify_keep")
            assert backbone.number_of_edges() < karate.number_of_edges(), method

    def test_does_not_mutate_input(self, karate):
        before = {e: dict(d) for e, d in karate.edges.items()}
        sparsify(karate, escore="random", umst=True, s=0.0, seed=1)
        assert {e: dict(d) for e, d in karate.edges.items()} == before

    def test_random_seed_reproducible(self, karate):
        def scores(seed):
            H = sparsify(
                karate, escore="random", filter="threshold", normalize=None, seed=seed
            )
            return dict(nx.get_edge_attributes(H, "sparsify_score"))

        assert scores(7) == scores(7)
        assert scores(7) != scores(8)
        assert scores(random.Random(3)) == scores(random.Random(3))
        assert all(0.0 <= v < 1.0 for v in scores(None).values())

    @pytest.mark.skipif(
        bool(os.environ.get("NETWORKX_TEST_BACKEND")),
        reason="the dispatch test harness runs sparsify twice, and both runs "
        "draw from the one NumPy generator that py_random_state wraps",
    )
    def test_numpy_seed_reproducible(self, karate):
        np = pytest.importorskip("numpy")

        def scores(seed):
            H = sparsify(
                karate, escore="random", filter="threshold", normalize=None, seed=seed
            )
            return dict(nx.get_edge_attributes(H, "sparsify_score"))

        assert scores(np.random.default_rng(5)) == scores(np.random.default_rng(5))

    def test_quadrangles_known_answer(self):
        H = sparsify(
            nx.cycle_graph(4),
            escore="quadrangles",
            normalize=None,
            filter="threshold",
            s=1,
        )
        assert set(nx.get_edge_attributes(H, "sparsify_score").values()) == {1.0}
        H = sparsify(
            nx.complete_graph(4),
            escore="quadrangles",
            normalize=None,
            filter="threshold",
            s=3,
        )
        assert set(nx.get_edge_attributes(H, "sparsify_score").values()) == {2.0}
        assert not any(nx.get_edge_attributes(H, "sparsify_keep").values())

    def test_triangles_and_degree_threshold_raw(self):
        G = nx.complete_graph(4)
        G.add_edge(3, 4)
        H = sparsify(G, escore="triangles", normalize=None, filter="threshold", s=2)
        assert H[0][1]["sparsify_keep"] and not H[3][4]["sparsify_keep"]
        H = sparsify(G, escore="degree", normalize=None, filter="threshold", s=4)
        # Raw neighbor-degree scores combine by max over the two endpoints.
        assert H[3][4]["sparsify_score"] == 4.0
        assert H[0][1]["sparsify_score"] == 3.0

    def test_rank_threshold_normalized_values(self):
        G = nx.star_graph(3)
        G.add_edge(1, 2)
        H = sparsify(G, escore="jaccard", normalize="rank", filter="threshold", s=1.0)
        for _, _, sc in H.edges(data="sparsify_score"):
            assert 0.0 < sc <= 1.0
        # Each node's best edge has normalized rank 1 and is kept.
        assert H[1][2]["sparsify_keep"]
        assert H[0][3]["sparsify_keep"]  # only edge of leaf 3

    def test_umst_restores_connectivity(self):
        G = nx.barbell_graph(5, 0)
        plain = sparsify(G, escore="jaccard", s=0.0)
        assert not nx.is_connected(boolean_filter(plain, "sparsify_keep"))
        H = sparsify(G, escore="jaccard", s=0.0, umst=True)
        backbone = boolean_filter(H, "sparsify_keep")
        assert nx.is_connected(backbone)
        assert _kept(plain) <= _kept(H)

    def test_umst_threshold_filter(self):
        G = nx.barbell_graph(4, 1)
        H = sparsify(
            G, escore="triangles", normalize=None, filter="threshold", s=10, umst=True
        )
        backbone = boolean_filter(H, "sparsify_keep")
        assert nx.is_connected(backbone)
        # The s=10 threshold keeps nothing.  Every clique edge has 2 triangles
        # and both bridges 0, so every edge is in some maximum spanning tree:
        # the UMST is the whole graph, not one 8-edge tree.
        assert _kept(H) == {frozenset(e) for e in G.edges()}

    def test_umst_equal_score_cycle_keeps_every_edge(self):
        # All Jaccard scores on C6 are 0, so the s=1 threshold keeps nothing
        # and every spanning path of the cycle is a minimum spanning tree.
        G = nx.cycle_graph(6)
        kw = {"escore": "jaccard", "normalize": None, "filter": "threshold", "s": 1}
        assert not _kept(sparsify(G, **kw))
        assert _kept(sparsify(G, umst=True, **kw)) == {frozenset(e) for e in G.edges()}

    def test_umst_unique_tree_hand_built(self):
        # Raw neighbor-degree scores (max over endpoints):
        #   (0,1): 2, (1,2): 3, (0,2): 3, (2,3): 3, (3,4): 2.
        # The three 3-edges form a tree; (3,4) attaches 4 and (0,1) closes a
        # cycle of more important edges, so the maximum spanning tree is
        # unique and equals the UMST.
        G = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)])
        H = sparsify(
            G, escore="degree", normalize=None, filter="threshold", s=10, umst=True
        )
        expected = {frozenset(e) for e in [(1, 2), (0, 2), (2, 3), (3, 4)]}
        assert _kept(H) == expected
        for u, v in G.edges():
            G[u][v]["w"] = H[u][v]["sparsify_score"]
        single = nx.maximum_spanning_edges(G, weight="w", data=False)
        assert {frozenset(e) for e in single} == expected

    def test_umst_independent_of_insertion_order(self, karate):
        for kw in [
            {"escore": "degree", "normalize": None, "filter": "threshold", "s": 99},
            {"escore": "jaccard", "s": 0.0},
            {"escore": "triangles", "s": 0.2},
        ]:
            reference = _kept(sparsify(karate, umst=True, **kw))
            rng = random.Random(42)
            for _ in range(5):
                nodes = list(karate)
                edges = list(karate.edges())
                rng.shuffle(nodes)
                rng.shuffle(edges)
                edges = [(v, u) if rng.random() < 0.5 else (u, v) for u, v in edges]
                G = nx.Graph()
                G.add_nodes_from(nodes)
                G.add_edges_from(edges)
                assert _kept(sparsify(G, umst=True, **kw)) == reference, kw

    @staticmethod
    def _connected_gnp(n, p, rng):
        while True:
            G = nx.gnp_random_graph(n, p, seed=rng.randrange(10**6))
            if nx.is_connected(G):
                return G

    def test_umst_matches_brute_force(self):
        # Enumerate every spanning tree of a small connected graph and take
        # the union of those of maximum total triangle score.  The s=inf
        # threshold keeps nothing, so the backbone is exactly the UMST.
        failed = []
        for seed in range(12):
            G = self._connected_gnp(7, 0.5, random.Random(seed))
            H = sparsify(
                G,
                escore="triangles",
                normalize=None,
                filter="threshold",
                s=math.inf,
                umst=True,
            )
            score = {frozenset(e): H.edges[e]["sparsify_score"] for e in G.edges()}
            n = G.number_of_nodes()
            best, union = -math.inf, set()
            for tree in itertools.combinations(score, n - 1):
                T = nx.Graph(tuple(e) for e in tree)
                if T.number_of_nodes() != n or not nx.is_tree(T):
                    continue
                total = sum(score[e] for e in tree)
                if total > best:
                    best, union = total, set(tree)
                elif total == best:
                    union |= set(tree)
            if _kept(H) != union:
                failed.append(seed)
        assert not failed, f"UMST differs from brute force for seeds {failed}"

    def test_umst_helper_matches_spanning_tree_iterator(self):
        from networkx_backbone.unweighted import _umst_edges

        failed = []
        for seed in range(8):
            rng = random.Random(seed)
            G = self._connected_gnp(6, 0.6, rng)
            for u, v in G.edges():
                G[u][v]["w"] = rng.randint(0, 2)
            badness = {frozenset(e): w for *e, w in G.edges(data="w")}
            # iter() is required on networkx < 3.5, where __iter__ sets up state.
            trees = iter(nx.SpanningTreeIterator(G, weight="w", minimum=True))
            first = next(trees)
            best = first.size(weight="w")
            union = {frozenset(e) for e in first.edges()}
            for T in trees:
                if T.size(weight="w") != best:
                    break
                union |= {frozenset(e) for e in T.edges()}
            if _umst_edges(G, badness) != union:
                failed.append(seed)
        assert not failed, f"_umst_edges differs for seeds {failed}"

    def test_umst_noop_when_connected(self, karate):
        H1 = sparsify(karate, s=1.0)
        H2 = sparsify(karate, s=1.0, umst=True)
        assert _kept(H1) == _kept(H2)

    def test_umst_disconnected_input(self):
        G = nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(4))
        H = sparsify(G, s=0.0, umst=True)
        assert nx.number_connected_components(boolean_filter(H, "sparsify_keep")) == 2

    def test_umst_mixed_type_nodes(self):
        G = nx.Graph([("a", 1), (1, 2.5), ("a", 2.5), (2.5, (0, 1))])
        H = sparsify(G, s=0.0, umst=True)
        assert nx.is_connected(boolean_filter(H, "sparsify_keep"))

    def test_self_loops_never_kept(self):
        G = nx.cycle_graph(4)
        G.add_edge(0, 0)
        H = sparsify(G, s=1.0)
        assert H[0][0]["sparsify_keep"] is False
        assert H[0][0]["sparsify_score"] == math.inf
        assert H[0][1]["sparsify_keep"]
        H = sparsify(G, filter="threshold", s=0.0)
        assert H[0][0]["sparsify_keep"] is False
        assert H[0][0]["sparsify_score"] == -math.inf

    def test_isolated_nodes(self):
        G = nx.empty_graph(3)
        G.add_edge(0, 1)
        H = local_degree(G)
        assert H[0][1]["sparsify_keep"] and set(H) == {0, 1, 2}


# ---------------------------------------------------------------------
# convention="r" (R backbone's lspar; exactness is tested against R's
# output in test_reference_implementations.py)
# ---------------------------------------------------------------------


def test_lspar_convention_argument():
    G = nx.karate_club_graph()
    with pytest.raises(ValueError, match="convention"):
        lspar(G, convention="netbone")
    # "satuluri" is the default.
    for s in EXPONENTS:
        assert _kept(lspar(G, s=s, convention="satuluri")) == _kept(lspar(G, s=s))


def test_lspar_r_convention_edge_cases():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])  # triangle with a tail
    G.add_edge(5, 6)  # isolated edge: Jaccard 0
    G.add_edge(0, 0)  # self-loop: ignored
    # R's Jaccard drops the endpoints from the union: (0, 1) scores 1/1,
    # (1, 2) and (0, 2) score 1/2, and the tail and the isolated edge 0.
    triangle = {frozenset(e) for e in [(0, 1), (1, 2), (2, 0)]}
    H = lspar(G, s=0.0, convention="r")
    # Every triangle edge is top-ranked at one of its endpoints.
    assert _kept(H) == triangle
    assert {H.edges[tuple(e)]["sparsify_score"] for e in triangle} == {0.0}
    assert H.edges[2, 3]["sparsify_score"] == math.inf
    assert H.edges[5, 6]["sparsify_score"] == math.inf
    assert H.edges[0, 0]["sparsify_keep"] is False
    # floor(d**s) has no minimum of one: below s = 0 nothing is kept, while
    # the default convention keeps every node's top-ranked edges.
    assert _kept(lspar(G, s=-0.5, convention="r")) == set()
    assert _kept(lspar(G, s=-0.5))
    for s in EXPONENTS:
        H = lspar(G, s=s, convention="r")
        assert _kept(H) == triangle
        assert _kept(H) == {
            frozenset((u, v))
            for u, v, x in H.edges(data="sparsify_score")
            if x <= s and u != v
        }
