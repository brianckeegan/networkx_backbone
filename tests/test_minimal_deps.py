"""The core package must work with only NetworkX installed.

numpy, scipy and matplotlib are optional extras.  These tests run the
pure-Python methods in a subprocess in which importing those packages raises
:class:`ImportError`, so they check the optional-dependency contract even when
the extras are installed.
"""

import subprocess
import sys
import textwrap

_BLOCKER = textwrap.dedent(
    """
    import sys

    class _Blocker:
        BLOCKED = {"numpy", "scipy", "matplotlib", "pandas"}

        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in self.BLOCKED:
                raise ImportError(f"{name} is blocked for this test")
            return None

    sys.meta_path.insert(0, _Blocker())
    """
)


def _run(code):
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_import_without_optional_dependencies():
    _run(
        """
        import nx_backbone as nb
        assert "disparity_filter" in nb.__all__
        """
    )


PURE_PYTHON_WEIGHTED = [
    "disparity_filter",
    "noise_corrected_filter",
    "lans_filter",
    "multiple_linkage_analysis",
    "strongest_n_ties",
    "global_sparsification",
    "primary_linkage_analysis",
    "edge_betweenness_filter",
    "node_degree_filter",
    "high_salience_skeleton",
    "metric_backbone",
    "ultrametric_backbone",
    "h_backbone",
    "modularity_backbone",
    "planar_maximally_filtered_graph",
    "maximum_spanning_tree_backbone",
    "jaccard_backbone",
    "dice_backbone",
    "cosine_backbone",
    "adamic_adar_index",
    "resource_allocation_index",
    "glab_filter",
    "local_degree",
    "lspar",
]


def test_pure_python_methods():
    # One subprocess for every method; report all failures at once.
    _run(
        f"""
        import traceback
        import networkx as nx
        import nx_backbone as nb
        G = nx.les_miserables_graph()
        failures = []
        for name in {PURE_PYTHON_WEIGHTED!r}:
            try:
                H = getattr(nb, name)(G)
                assert H.number_of_nodes() == G.number_of_nodes(), "lost nodes"
            except Exception:
                failures.append(name + ":\\n" + traceback.format_exc())
        if failures:
            raise SystemExit("methods failed without numpy/scipy:\\n" + "\\n".join(failures))
        """
    )


def test_pure_python_pipeline():
    _run(
        """
        import networkx as nx
        import nx_backbone as nb
        G = nx.les_miserables_graph()
        scored = nb.disparity_filter(G)
        bb = nb.threshold_filter(scored, "disparity_pvalue", 0.05)
        assert 0 < bb.number_of_edges() < G.number_of_edges()
        bh = nb.threshold_filter(scored, "disparity_pvalue", 0.05, mtc="bh")
        assert bh.number_of_edges() <= bb.number_of_edges()
        assert 0 < nb.edge_fraction(G, bb) < 1
        # Graph input never needs the numpy/scipy/pandas converters.
        assert nb.to_graph(G) is G
        assert 0 < nb.backbone_from_weighted(G).number_of_edges() < G.number_of_edges()
        assert nb.backbone(G, method="lspar").number_of_nodes() == G.number_of_nodes()
        H = [(1, 2, 3, 4), (1, 2, 3), (2, 3, 4), (8, 9)]
        assert nb.maximal_hyperedges(H) == [frozenset({1, 2, 3, 4}), frozenset({8, 9})]
        """
    )


def test_default_disparity_and_callable_transform_need_no_scipy():
    # disparity_filter(a=1) keeps the closed form; transform_weights with a
    # plain callable runs per edge in pure Python.  a != 1 needs SciPy.
    _run(
        """
        import math
        import networkx as nx
        import nx_backbone as nb
        G = nx.les_miserables_graph()
        H = nb.transform_weights(G, math.log1p)
        assert H.graph["weight_transforms"][0]["transform"] == "math.log1p"
        for a in (1, 1.0):
            bb = nb.threshold_filter(nb.disparity_filter(G, a=a), "disparity_pvalue", 0.05)
            assert bb.number_of_edges() == 9
        nb.disparity_filter(H)
        try:
            nb.disparity_filter(G, a=2.0)
        except ImportError:
            pass
        else:
            raise AssertionError("expected ImportError for a != 1")
        """
    )


def test_missing_optional_dependency_raises_importerror():
    _run(
        """
        import networkx as nx
        import nx_backbone as nb
        try:
            nb.ecm_filter(nx.les_miserables_graph())
        except ImportError as exc:
            assert "numpy" in str(exc) or "scipy" in str(exc)
        else:
            raise AssertionError("expected ImportError")
        """
    )
