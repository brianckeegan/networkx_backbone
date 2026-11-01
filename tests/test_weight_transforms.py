"""Tests for transform_weights and the invariance table in the user guide.

:data:`ROWS` holds one entry per row of the table in
``docs/user_guide/weight_transforms.rst``: whether the method's scores stay the
same when the weights are rescaled, and when they go through a strictly
increasing nonlinear transform.  :func:`test_invariance_claim` checks every
claim on Les Misérables, and :func:`test_docs_table_matches_tests` checks that
the table says the same.
"""

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

import networkx_backbone as nb

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "user_guide" / "weight_transforms.rst"

# Multiplying by a power of two is exact in floating point, so a
# scale-invariant method must give exactly the same arithmetic.
RESCALE = ("x4", lambda w: 4 * w)
INCREASING = [
    ("sqrt", math.sqrt),
    ("square", lambda w: w * w),
    ("log1p", math.log1p),
]


@dataclass(frozen=True)
class Row:
    label: str  # first cell of the docs table row
    func: Any
    kwargs: dict = field(default_factory=dict)
    rescaling: bool = True  # invariant to w -> c w
    increasing: bool = False  # invariant to w -> f(w), f strictly increasing


def _row(name, qualifier="", **kw):
    label = f":func:`~networkx_backbone.{name}`" + (
        f" {qualifier}" if qualifier else ""
    )
    return label, getattr(nb, name)


ROWS = [
    Row(*_row("lans_filter"), increasing=True),
    Row(*_row("strongest_n_ties"), {"n": 2}, increasing=True),
    Row(*_row("global_sparsification"), {"s": 0.5}, increasing=True),
    Row(*_row("primary_linkage_analysis"), increasing=True),
    Row(*_row("maximum_spanning_tree_backbone"), increasing=True),
    Row(*_row("planar_maximally_filtered_graph"), increasing=True),
    Row(*_row("ultrametric_backbone"), increasing=True),
    Row(*_row("disparity_filter", "(any ``a``)")),
    Row(*_row("disparity_filter", "(any ``a``)"), {"a": 2.0}),
    Row(*_row("disparity_filter", "(any ``a``)"), {"a": "ml"}),
    Row(*_row("multiple_linkage_analysis")),
    Row(
        *_row("polya_filter", "(``approximate=True``)"),
        {"a": 1.0, "approximate": True},
    ),
    Row(*_row("gloss_filter")),
    Row(*_row("doubly_stochastic_filter")),
    Row(*_row("metric_backbone")),
    Row(*_row("high_salience_skeleton")),
    Row(*_row("edge_betweenness_filter"), {"s": 0.5}),
    Row(*_row("glab_filter")),
    Row(*_row("modularity_backbone"), {"seed": 1}),
    Row(*_row("marginal_likelihood_filter"), rescaling=False),
    Row(*_row("ecm_filter"), rescaling=False),
    Row(*_row("noise_corrected_filter"), rescaling=False),
    Row(*_row("polya_filter", "(exact p-values)"), rescaling=False),
    Row(*_row("h_backbone"), rescaling=False),
    Row(
        *_row("local_path_index", "(``weight=...``)"),
        {"weight": "weight"},
        rescaling=False,
    ),
]


def _scores(row, G):
    """Every edge attribute the method adds, keyed by undirected edge."""
    with warnings.catch_warnings():
        # doubly_stochastic_filter does not converge on Les Misérables.
        warnings.simplefilter("ignore", RuntimeWarning)
        H = row.func(G, **row.kwargs)
    return {
        frozenset((u, v)): {k: x for k, x in d.items() if k != "weight"}
        for u, v, d in H.edges(data=True)
    }


def _same(a, b):
    if a.keys() != b.keys():
        return False
    for edge, attrs in a.items():
        if attrs.keys() != b[edge].keys():
            return False
        for key, x in attrs.items():
            y = b[edge][key]
            if isinstance(x, float) or isinstance(y, float):
                if not math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12):
                    return False
            elif x != y:
                return False
    return True


def _row_id(row):
    extra = ",".join(f"{k}={v}" for k, v in row.kwargs.items())
    return f"{row.func.__name__}({extra})"


@pytest.mark.parametrize("row", ROWS, ids=[_row_id(r) for r in ROWS])
def test_invariance_claim(row, les_miserables_weighted):
    pytest.importorskip("scipy")
    G = les_miserables_weighted
    base = _scores(row, G)
    rescaled = _same(base, _scores(row, nb.transform_weights(G, RESCALE[1])))
    assert rescaled == row.rescaling, f"{row.label}: rescaling"
    if not row.rescaling:
        return  # a rescaling is itself an increasing transform
    same = {
        name: _same(base, _scores(row, nb.transform_weights(G, f)))
        for name, f in INCREASING
    }
    if row.increasing:
        assert all(same.values()), f"{row.label}: changed by {same}"
    else:
        assert not any(same.values()), f"{row.label}: unchanged by {same}"


def _docs_table():
    """``{first cell: (rescaling, increasing)}`` from the guide's list-table."""
    rows, cells = [], None
    for line in GUIDE.read_text(encoding="utf-8").splitlines():
        if line.startswith("   * - "):
            cells = [line[len("   * - ") :].strip()]
            rows.append(cells)
        elif cells is not None and line.startswith("     - "):
            cells.append(line[len("     - ") :].strip())
    table = {}
    for cells in rows[1:]:  # skip the header row
        assert cells[1] in ("yes", "no") and cells[2] in ("yes", "no"), cells
        table[cells[0]] = (cells[1] == "yes", cells[2] == "yes")
    return table


def test_docs_table_matches_tests():
    if not GUIDE.exists():
        pytest.skip("docs not available")
    expected = {r.label: (r.rescaling, r.increasing) for r in ROWS}
    assert _docs_table() == expected


def test_weight_free_methods_are_unaffected(les_miserables_weighted):
    G = les_miserables_weighted
    T = nb.transform_weights(G, math.sqrt)
    for func in (nb.node_degree_filter, nb.jaccard_backbone, nb.local_degree):
        row = Row("", func)
        assert _same(_scores(row, G), _scores(row, T)), func.__name__


@pytest.mark.parametrize(("name", "f"), [RESCALE, *INCREASING])
def test_global_threshold_is_equivariant(name, f, les_miserables_weighted):
    G = les_miserables_weighted
    base = nb.global_threshold_filter(G, threshold=2)
    got = nb.global_threshold_filter(nb.transform_weights(G, f), threshold=f(2))
    assert nx.get_edge_attributes(got, "global_threshold_keep") == (
        nx.get_edge_attributes(base, "global_threshold_keep")
    )


# ---------------------------------------------------------------------------
# transform_weights
# ---------------------------------------------------------------------------


class Doubler:
    """Minimal duck-typed transformer: records its input, doubles it."""

    def __init__(self):
        self.calls = []

    def fit_transform(self, X):
        self.calls.append(X)
        return 2 * X

    def __repr__(self):
        return "Doubler()"


class FittedShift:
    """An already-fitted transformer: only ``transform``."""

    def transform(self, X):
        return (X + 1).ravel()

    def __repr__(self):
        return "FittedShift()"


@pytest.fixture
def small():
    G = nx.Graph()
    G.add_weighted_edges_from([("a", "b", 4.0), ("b", "c", 1.0), ("a", "c", 9.0)])
    G.graph["name"] = "small"
    return G


class TestTransformWeights:
    def test_callable(self, small):
        before = nx.get_edge_attributes(small, "weight")
        H = nb.transform_weights(small, math.sqrt)
        assert nx.get_edge_attributes(H, "weight") == {
            e: math.sqrt(w) for e, w in before.items()
        }
        assert nx.get_edge_attributes(small, "weight") == before  # input intact
        assert H.graph["name"] == "small"
        assert H.graph["weight_transforms"] == [
            {"transform": "math.sqrt", "weight": "weight", "out": "weight"}
        ]
        assert "weight_transforms" not in small.graph

    def test_out_keeps_original(self, small):
        H = nb.transform_weights(small, math.log, out="logw")
        for u, v, d in H.edges(data=True):
            assert d["logw"] == math.log(d["weight"])
            assert d["weight"] == small[u][v]["weight"]
        assert H.graph["weight_transforms"][0]["out"] == "logw"

    def test_custom_weight_attribute(self):
        G = nx.Graph()
        G.add_edge(0, 1, flow=3.0, weight=1.0)
        G.add_edge(1, 2, flow=5.0, weight=1.0)
        H = nb.transform_weights(G, lambda w: w * w, weight="flow")
        assert H[0][1] == {"flow": 9.0, "weight": 1.0}
        assert H.graph["weight_transforms"][0]["transform"] == "<lambda>"

    def test_chained_transforms_are_all_recorded(self, small):
        H = nb.transform_weights(nb.transform_weights(small, math.sqrt), math.log1p)
        assert [r["transform"] for r in H.graph["weight_transforms"]] == [
            "math.sqrt",
            "math.log1p",
        ]

    @pytest.mark.parametrize("directed", [False, True])
    def test_neighbor_order_is_kept(self, directed, les_miserables_weighted):
        G = les_miserables_weighted
        if directed:
            G = nx.DiGraph(G)
            G.add_edge("Valjean", "Myriel", weight=3)  # asymmetric weight
        H = nb.transform_weights(G, float)
        # G.copy() reorders neighbors on this graph; the transform must not.
        assert list(H.nodes) == list(G.nodes)
        assert list(H.edges) == list(G.edges)
        for u in G:
            assert list(H.adj[u]) == list(G.adj[u])
            if directed:
                assert list(H.pred[u]) == list(G.pred[u])
        # Each edge has one attribute dict, shared by both directions.
        if not directed:
            H["Valjean"]["Myriel"]["weight"] = -1
            assert H["Myriel"]["Valjean"]["weight"] == -1
        assert G["Valjean"]["Myriel"]["weight"] != -1

    def test_identity_keeps_tie_breaking(self, les_miserables_weighted):
        # strongest_n_ties breaks weight ties by neighbor order.
        G = les_miserables_weighted
        H = nb.transform_weights(G, lambda w: w)
        assert nx.utils.edges_equal(
            nb.strongest_n_ties(H, n=2).edges(data=True),
            nb.strongest_n_ties(G, n=2).edges(data=True),
        )

    def test_self_loop(self):
        G = nx.Graph()
        G.add_edge(0, 0, weight=4.0)
        G.add_edge(0, 1, weight=9.0)
        H = nb.transform_weights(G, math.sqrt)
        assert H[0][0]["weight"] == 2.0
        assert list(H.adj[0]) == list(G.adj[0])

    def test_fit_transform_is_called_once_on_a_column(self, small):
        np = pytest.importorskip("numpy")
        t = Doubler()
        H = nb.transform_weights(small, t)
        # NetworkX's dispatch test harness (NETWORKX_TEST_BACKEND) runs the
        # function a second time to compare backends.
        testing = getattr(nx.utils.backends._dispatchable, "_is_testing", False)
        assert len(t.calls) == (2 if testing else 1)
        X = t.calls[0]
        assert X.shape == (3, 1)
        assert X.ravel().tolist() == [d["weight"] for *_, d in small.edges(data=True)]
        assert isinstance(X, np.ndarray)
        for u, v, d in H.edges(data=True):
            assert d["weight"] == 2 * small[u][v]["weight"]
        assert H.graph["weight_transforms"][0]["transform"] == "Doubler()"

    def test_transform_only_object(self, small):
        pytest.importorskip("numpy")
        H = nb.transform_weights(small, FittedShift())
        assert H["a"]["b"]["weight"] == 5.0

    def test_empty_graph(self):
        t = Doubler()
        H = nb.transform_weights(nx.empty_graph(3), t)
        assert t.calls == []
        assert H.number_of_nodes() == 3
        assert H.graph["weight_transforms"][0]["transform"] == "Doubler()"

    def test_wrong_shape(self, small):
        pytest.importorskip("numpy")

        class TwoColumns:
            def fit_transform(self, X):
                return X.repeat(2, axis=1)

        with pytest.raises(ValueError, match="returned shape"):
            nb.transform_weights(small, TwoColumns())

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_non_finite_output(self, bad, small):
        with pytest.raises(ValueError, match="NaN or infinite"):
            nb.transform_weights(small, lambda w: bad)

    def test_constant_output(self, small):
        with pytest.raises(ValueError, match=r"maps every weight to 1\.0"):
            nb.transform_weights(small, lambda w: 1.0)
        # Constant weights may stay constant.
        G = nx.path_graph(4)
        nx.set_edge_attributes(G, 2.0, "weight")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            H = nb.transform_weights(G, math.log)
        assert set(nx.get_edge_attributes(H, "weight").values()) == {math.log(2.0)}

    def test_increasing_transform_does_not_warn(self, les_miserables_weighted):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            nb.transform_weights(les_miserables_weighted, math.log1p)

    @pytest.mark.parametrize(
        ("f", "match"),
        [
            (lambda w: -w, "does not preserve the order"),
            (lambda w: 1 / w, "does not preserve the order"),
            (lambda w: (w - 4) ** 2, "does not preserve the order"),
            (lambda w: w // 5, "creating ties"),
        ],
    )
    def test_order_warnings(self, f, match, les_miserables_weighted):
        with pytest.warns(UserWarning, match=match) as rec:
            nb.transform_weights(les_miserables_weighted, f)
        assert rec[0].filename == __file__

    def test_equal_weights_given_different_values_warns(self, small):
        np = pytest.importorskip("numpy")

        class RowNumber:
            def fit_transform(self, X):
                return np.arange(len(X), dtype=float) + 1

        small["b"]["c"]["weight"] = 4.0  # ties with a-b
        with pytest.warns(UserWarning, match="equal weights get different values"):
            nb.transform_weights(small, RowNumber())

    @pytest.mark.parametrize("bad", [None, "heavy", math.nan, math.inf, True])
    def test_invalid_weights(self, bad, small):
        small["a"]["b"]["weight"] = bad
        with pytest.raises(nx.NetworkXError, match="'weight' attribute"):
            nb.transform_weights(small, math.sqrt)
        del small["a"]["b"]["weight"]
        with pytest.raises(nx.NetworkXError, match="'weight' attribute"):
            nb.transform_weights(small, math.sqrt)

    def test_invalid_transformer(self, small):
        with pytest.raises(TypeError, match="transformer must be callable"):
            nb.transform_weights(small, 5)

    def test_multigraph_is_rejected(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            nb.transform_weights(nx.MultiGraph([(0, 1, {"weight": 1.0})]), math.sqrt)

    def test_describe_backbone_reports_transform(self, les_miserables_weighted):
        G = les_miserables_weighted
        H = nb.transform_weights(G, math.log1p)
        bb = nb.threshold_filter(nb.lans_filter(H), "lans_pvalue", 0.05)
        text = nb.describe_backbone("lans_filter", alpha=0.05, original=G, backbone=bb)
        assert (
            "Before scoring, the edge weights ('weight') were transformed with "
            "math.log1p, so the null model applies to the transformed weights."
        ) in text
        H = nb.transform_weights(G, math.sqrt, out="w")
        md = nb.describe_backbone("lans_filter", original=H, style="markdown")
        assert "transformed with `math.sqrt` into the attribute 'w'" in md


class TestScikitLearn:
    """Real scikit-learn transformers (optional; CI does not install it)."""

    @pytest.fixture(autouse=True)
    def _sklearn(self):
        pytest.importorskip("sklearn")

    def test_function_transformer_matches_callable(self, les_miserables_weighted):
        import numpy as np
        from sklearn.preprocessing import FunctionTransformer

        G = les_miserables_weighted
        got = nb.transform_weights(G, FunctionTransformer(np.log1p))
        want = nb.transform_weights(G, math.log1p)
        assert nx.get_edge_attributes(got, "weight") == pytest.approx(
            nx.get_edge_attributes(want, "weight"), rel=1e-12
        )
        assert got.graph["weight_transforms"][0]["transform"].startswith(
            "FunctionTransformer("
        )

    def test_box_cox_is_increasing(self, les_miserables_weighted):
        from sklearn.preprocessing import PowerTransformer

        G = les_miserables_weighted
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            H = nb.transform_weights(
                G, PowerTransformer(method="box-cox", standardize=False)
            )
        # Monotone-invariant methods are unchanged.  (Box-Cox maps w = 1 to
        # 0, which the statistical scorers would reject.)
        mst = nb.maximum_spanning_tree_backbone
        assert nx.get_edge_attributes(mst(H), "mst_keep") == (
            nx.get_edge_attributes(mst(G), "mst_keep")
        )

    def test_normalizer_is_rejected(self, les_miserables_weighted):
        from sklearn.preprocessing import Normalizer

        with pytest.raises(ValueError, match=r"maps every weight to 1\.0"):
            nb.transform_weights(les_miserables_weighted, Normalizer())

    def test_binning_warns_about_ties(self, les_miserables_weighted):
        from sklearn.preprocessing import KBinsDiscretizer

        binner = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
        with pytest.warns(UserWarning, match="creating ties"):
            nb.transform_weights(les_miserables_weighted, binner)
