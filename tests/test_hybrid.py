"""Known-answer tests for networkx_backbone.hybrid (GLAB).

Shared behavior (attributes, weight validation, directed support,
multigraph rejection, empty graphs, no input mutation) is covered by
``test_api_contract.py``.
"""

import networkx as nx
import pytest

from networkx_backbone import glab_filter


def _out_star(n):
    """Directed star with edges hub -> leaf (nx.star_graph rejects DiGraph on networkx < 3.5)."""
    return nx.DiGraph((0, i) for i in range(1, n + 1))


class TestGLABPaperEquations:
    """Known answers from Zhang et al. (2014), Eqs. 1 and 4."""

    def test_involvement_sums_to_one_per_node(self, les_miserables_weighted):
        from networkx_backbone.hybrid import _involvement

        G = les_miserables_weighted

        def length(u, v, d):
            return 1.0 / d["weight"]

        for node in list(G)[:10]:
            inv = _involvement(G, node, length)
            assert sum(inv.values()) == pytest.approx(1.0)

    def test_involvement_splits_tied_paths(self):
        from networkx_backbone.hybrid import _involvement

        # From node 0 in C4: node 1 via (0,1), node 3 via (0,3), node 2 split.
        assert _involvement(nx.cycle_graph(4), 0, None) == {1: 0.5, 3: 0.5}

    def test_cycle_hand_computed(self):
        # Every node has k=2 and I=1/2, so SI = (1 - 1/2) ** (1 ** c) = 0.5.
        H = glab_filter(nx.cycle_graph(4), weight=None, c=0.7)
        assert all(
            d["glab_pvalue"] == pytest.approx(0.5) for *_, d in H.edges(data=True)
        )

    def test_star_leaf_uses_hub_side(self):
        # Hub of degree 3: I = 1/3 for each spoke, SI = (2/3) ** (2 ** c).
        H = glab_filter(nx.star_graph(3), weight=None, c=1.0)
        for *_, d in H.edges(data=True):
            assert d["glab_pvalue"] == pytest.approx((2 / 3) ** 2)

    def test_c_zero_ignores_degree(self):
        H = glab_filter(nx.star_graph(5), weight=None, c=0.0)
        for *_, d in H.edges(data=True):
            assert d["glab_pvalue"] == pytest.approx(0.8)

    def test_isolated_dyad_is_not_significant(self):
        G = nx.Graph()
        G.add_edge("a", "b", weight=3.0)
        H = glab_filter(G)
        assert H["a"]["b"]["glab_pvalue"] == 1.0

    def test_heavy_link_more_important(self):
        # Path 0-1 is heavy, so 2 reaches 0 through 1 and the heavy link
        # dominates node 1's involvement.
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 10.0), (1, 2, 1.0), (0, 2, 0.1), (2, 3, 1.0)])
        H = glab_filter(G)
        assert H[0][1]["glab_pvalue"] < H[0][2]["glab_pvalue"]

    def test_les_miserables_reference(self, les_miserables_weighted):
        H = glab_filter(les_miserables_weighted)
        assert H["Valjean"]["Cosette"]["glab_pvalue"] == pytest.approx(0.0425, abs=1e-4)
        # The pinned count of edges with p < 0.05 is in test_les_miserables.py.

    def test_negative_c_raises(self, weighted_triangle):
        with pytest.raises(ValueError):
            glab_filter(weighted_triangle, c=-1)

    def test_disconnected_graph(self):
        G = nx.disjoint_union(nx.cycle_graph(4), nx.path_graph(3))
        H = glab_filter(G, weight=None)
        assert all(0.0 <= d["glab_pvalue"] <= 1.0 for *_, d in H.edges(data=True))


class TestGLABDirected:
    """Directed variant, Zhang et al. (2014) Section 2.3, Eqs. 5 and 6."""

    def test_directed_cycle_has_no_null_model(self):
        # Every node has out-degree 1 and in-degree 1, so neither SI^(out)
        # nor SI^(in) is defined and every link gets p = 1.
        H = glab_filter(nx.cycle_graph(4, create_using=nx.DiGraph), weight=None)
        assert all(p == 1.0 for *_, p in H.edges(data="glab_pvalue"))

    def test_directed_cycle_with_chord_hand_computed(self):
        # 0->1->2->3->0 plus the chord 0->2.
        # Out of 0 (reaches 1, 2, 3): 1 via (0,1); 2 and 3 via (0,2).
        #   I_out(0,1) = 1/3, I_out(0,2) = 2/3, k_out(0) = 2.
        # Into 2 (reached from 0, 1, 3): 1 via (1,2); 0 and 3 (3->0->2) via (0,2).
        #   I_in(1,2) = 1/3, I_in(0,2) = 2/3, k_in(2) = 2.
        # All other out-/in-degrees are 1, so (2,3) and (3,0) have p = 1.
        G = nx.cycle_graph(4, create_using=nx.DiGraph)
        G.add_edge(0, 2)
        H = glab_filter(G, weight=None, c=0.7)
        expected = {
            (0, 1): 2 / 3,
            (0, 2): 1 / 3,
            (1, 2): 2 / 3,
            (2, 3): 1.0,
            (3, 0): 1.0,
        }
        got = {(u, v): p for u, v, p in H.edges(data="glab_pvalue")}
        assert got == pytest.approx(expected)

    def test_stars_use_out_and_in_involvement(self):
        for c in (0.0, 0.5, 1.0):
            # The hub has out-degree 3 and I_out = 1/3 for each spoke; leaves
            # have in-degree 1, so only SI^(out) = (2/3) ** (2 ** c) applies.
            H = glab_filter(_out_star(3), weight=None, c=c)
            assert H.number_of_edges() == 3
            for *_, p in H.edges(data="glab_pvalue"):
                assert p == pytest.approx((2 / 3) ** (2**c)), c
            # Leaves -> hub: the hub has in-degree 3 and I_in = 1/3 per spoke.
            H = glab_filter(_out_star(3).reverse(), weight=None, c=c)
            assert all(v == 0 for _, v in H.edges())
            for *_, p in H.edges(data="glab_pvalue"):
                assert p == pytest.approx((2 / 3) ** (2**c)), c

    def test_in_involvement_follows_reverse_paths(self):
        # In-star into 0 plus 1 -> 2.  Node 1 reaches 0 more cheaply via 2
        # (length 0.1 + 0.1) than directly (length 1), so (2, 0) carries the
        # in-paths of both 1 and 2: I_in(2,0) = 2/3, I_in(1,0) = 0,
        # I_in(3,0) = 1/3, with k_in(0) = 3.
        G = nx.DiGraph()
        G.add_weighted_edges_from(
            [(1, 0, 1.0), (2, 0, 10.0), (3, 0, 1.0), (1, 2, 10.0)]
        )
        H = glab_filter(G, c=1.0)
        assert H[2][0]["glab_pvalue"] == pytest.approx((1 / 3) ** 2)
        assert H[1][0]["glab_pvalue"] == pytest.approx(1.0)
        assert H[3][0]["glab_pvalue"] == pytest.approx((2 / 3) ** 2)
        # Out of 1 (k_out = 2): both 0 and 2 are reached via (1, 2).
        assert H[1][2]["glab_pvalue"] == pytest.approx(0.0)

    def test_symmetric_digraph_matches_undirected(
        self, les_miserables_weighted, karate
    ):
        # On a symmetric DiGraph, I_out(i,j) = I_ij, I_in(i,j) = I_ji and the
        # out-/in-degrees equal the undirected degree, so Eq. 6 reduces to Eq. 4.
        split = nx.disjoint_union(nx.star_graph(3), nx.path_graph(4))
        split.add_edge(7, 8)
        for name, G, weight in [
            ("les_miserables", les_miserables_weighted, "weight"),
            ("karate", karate, None),
            ("split", split, None),
        ]:
            undirected = glab_filter(G, weight=weight)
            directed = glab_filter(nx.DiGraph(G), weight=weight)
            assert directed.number_of_edges() == 2 * undirected.number_of_edges()
            for u, v, p in directed.edges(data="glab_pvalue"):
                expected = undirected[u][v]["glab_pvalue"]
                assert p == pytest.approx(expected, abs=1e-12), (name, u, v)

    # A directed input gives a directed copy (input untouched): see
    # test_api_contract.py.
