"""API contract shared by every public graph-scoring function.

Each scorer is registered once in :data:`REGISTRY` with the input it expects,
the attributes it adds and the ways it legitimately differs from the others
(directed support, weight validation, ...).  :func:`test_contract` then checks
every invariant that applies, so adding a scorer means adding one registry row.
It runs once per kind of input and loops over the scorers, so a failure lists
every scorer that broke the contract instead of stopping at the first.

Method-specific behavior (known answers, paper equations, reference
implementations) lives in the per-module test files.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pytest

import networkx_backbone as nb

P = (0.0, 1.0)  # bounds of a p-value or other unit-interval score
NONNEG = (0.0, math.inf)
ANY = (-math.inf, math.inf)


class _Missing:
    def __repr__(self):
        return "<missing>"


MISSING = _Missing()

# Bad edge weights each validating scorer must reject, with the error message
# pattern where the function documents one.
STATISTICAL_BAD = dict.fromkeys([math.nan, math.inf, 0, -1.0, MISSING])
_NO_WEIGHT = "no 'weight' attribute"
_NON_NUMERIC = {
    MISSING: _NO_WEIGHT,
    math.nan: "NaN",
    "heavy": "non-numeric",
    None: "non-numeric",
}
POSITIVE_BAD = {**_NON_NUMERIC, 0.0: "strictly positive", -1.0: "strictly positive"}
NONNEGATIVE_BAD = {**_NON_NUMERIC, -1.0: "negative"}


@dataclass(frozen=True)
class Spec:
    """How to call a scorer and what its output must look like.

    kind
        ``"weighted"`` (Les Misérables), ``"unweighted"`` (Les Misérables
        without weights), ``"bipartite"`` (Davis Southern Women; called as
        ``func(B, agent_nodes, **kwargs)``) or ``"temporal"`` (called on
        snapshots of Les Misérables, see :func:`_check_temporal_scorer`).
    scores
        Numeric edge attributes added to every edge, mapped to ``(lo, hi)``.
    keep
        Boolean edge attributes added to every edge.
    node_attrs
        Attributes added to every node.
    directed
        Accepts directed graphs; otherwise must raise NetworkXNotImplemented.
    order_invariant
        Scores do not depend on node/edge insertion order (false when ties
        are broken by input order).  Checked by comparing *G* with a copy
        built in reverse node and edge order, with every edge flipped.
    label_ties
        Ties are broken by node label, so relabeling may change which tied
        edge is kept.
    allow_inf
        Scores may be +/-inf by design (never NaN).
    bad_weights
        Weight values the function must reject with NetworkXError, mapped to
        the expected message pattern (or None).
    all_pairs
        Bipartite scorer returns every agent pair, not just co-occurring ones.
    """

    func: Any
    kind: str
    scores: dict = field(default_factory=dict)
    keep: tuple = ()
    node_attrs: tuple = ()
    kwargs: dict = field(default_factory=dict)
    directed: bool = False
    order_invariant: bool = True
    label_ties: bool = False
    allow_inf: bool = False
    bad_weights: dict = field(default_factory=dict)
    all_pairs: bool = False

    @property
    def name(self):
        return getattr(self.func, "__name__", "scorer")

    def __call__(self, G, *args):
        return self.func(G, *args, **self.kwargs)


W, U, B, TM = "weighted", "unweighted", "bipartite", "temporal"

REGISTRY = [
    # --- statistical -------------------------------------------------------
    Spec(nb.disparity_filter, W, {"disparity_pvalue": P}, directed=True,
         bad_weights=STATISTICAL_BAD),
    Spec(nb.disparity_filter, W, {"disparity_pvalue": P}, kwargs={"a": 2.0},
         directed=True, bad_weights=STATISTICAL_BAD),
    Spec(nb.disparity_filter, W, {"disparity_pvalue": P}, kwargs={"a": "ml"},
         directed=True, bad_weights=STATISTICAL_BAD),
    Spec(nb.noise_corrected_filter, W,
         {"nc_score": (-1.0, 1.0), "nc_sdev": NONNEG, "nc_pvalue": P},
         directed=True, bad_weights=STATISTICAL_BAD),
    Spec(nb.noise_corrected_filter, W, {"nc_score": ANY, "nc_pvalue": P},
         kwargs={"method": "zscore"}, directed=True, bad_weights=STATISTICAL_BAD),
    Spec(nb.marginal_likelihood_filter, W, {"ml_pvalue": P}, directed=True,
         bad_weights=STATISTICAL_BAD),
    Spec(nb.ecm_filter, W, {"ecm_pvalue": P}, directed=True,
         bad_weights=STATISTICAL_BAD),
    Spec(nb.lans_filter, W, {"lans_pvalue": P}, directed=True,
         bad_weights=STATISTICAL_BAD),
    Spec(nb.multiple_linkage_analysis, W, keep=("mla_keep",),
         node_attrs=("mla_rank",), directed=True, bad_weights=STATISTICAL_BAD),
    Spec(nb.multiple_linkage_analysis, W, {"mla_pvalue": P, "lans_pvalue": P},
         keep=("mla_keep",), kwargs={"method": "lans"}, directed=True,
         bad_weights=STATISTICAL_BAD),
    Spec(nb.polya_filter, W, {"polya_pvalue": P}, directed=True,
         bad_weights=STATISTICAL_BAD),
    Spec(nb.gloss_filter, W, {"gloss_pvalue": P}, directed=True,
         bad_weights=STATISTICAL_BAD),
    # --- structural --------------------------------------------------------
    Spec(nb.global_threshold_filter, W, keep=("global_threshold_keep",),
         kwargs={"threshold": 2.0}, directed=True),
    Spec(nb.strongest_n_ties, W, keep=("strongest_n_ties_keep",), directed=True),
    Spec(nb.global_sparsification, W, keep=("global_sparsification_keep",),
         directed=True, label_ties=True),
    Spec(nb.primary_linkage_analysis, W, keep=("primary_linkage_keep",),
         directed=True, label_ties=True),
    Spec(nb.edge_betweenness_filter, W, {"edge_betweenness": P},
         keep=("edge_betweenness_keep",), directed=True, label_ties=True),
    Spec(nb.node_degree_filter, W, keep=("node_degree_keep",),
         node_attrs=("node_degree_keep",), directed=True),
    Spec(nb.high_salience_skeleton, W, {"salience": P}, bad_weights=POSITIVE_BAD),
    Spec(nb.metric_backbone, W, keep=("metric_keep",), bad_weights=POSITIVE_BAD),
    Spec(nb.ultrametric_backbone, W, keep=("ultrametric_keep",),
         bad_weights=POSITIVE_BAD),
    Spec(nb.doubly_stochastic_filter, W, {"ds_weight": P}, keep=("ds_keep",),
         bad_weights=NONNEGATIVE_BAD),
    Spec(nb.h_backbone, W, {"h_bridge": NONNEG}, keep=("h_backbone_keep",),
         bad_weights=_NON_NUMERIC),
    Spec(nb.modularity_backbone, W, keep=("modularity_keep",),
         node_attrs=("vitality",), kwargs={"seed": 1}, order_invariant=False),
    Spec(nb.planar_maximally_filtered_graph, W, keep=("pmfg_keep",),
         order_invariant=False, bad_weights=_NON_NUMERIC),
    Spec(nb.maximum_spanning_tree_backbone, W, keep=("mst_keep",),
         order_invariant=False),
    # --- hybrid ------------------------------------------------------------
    Spec(nb.glab_filter, W, {"glab_pvalue": P}, directed=True,
         bad_weights={**STATISTICAL_BAD, None: None}),
    # --- proximity ---------------------------------------------------------
    Spec(nb.neighborhood_overlap, U, {"overlap": NONNEG}, directed=True),
    Spec(nb.jaccard_backbone, U, {"jaccard": P}, directed=True),
    Spec(nb.dice_backbone, U, {"dice": P}, directed=True),
    Spec(nb.cosine_backbone, U, {"cosine": P}, directed=True),
    Spec(nb.hub_promoted_index, U, {"hpi": P}, directed=True),
    Spec(nb.hub_depressed_index, U, {"hdi": P}, directed=True),
    Spec(nb.lhn_local_index, U, {"lhn_local": P}, directed=True),
    Spec(nb.preferential_attachment_score, U, {"pa": NONNEG}, directed=True),
    Spec(nb.adamic_adar_index, U, {"adamic_adar": NONNEG}, directed=True),
    Spec(nb.resource_allocation_index, U, {"resource_allocation": NONNEG},
         directed=True),
    Spec(nb.graph_distance_proximity, U, {"dist": P}, directed=True),
    Spec(nb.local_path_index, U, {"lp": NONNEG}, directed=True),
    # --- unweighted sparsification -------------------------------------------
    Spec(nb.sparsify, U, {"sparsify_score": ANY}, keep=("sparsify_keep",),
         allow_inf=True),
    Spec(nb.lspar, U, {"sparsify_score": ANY}, keep=("sparsify_keep",),
         allow_inf=True),
    Spec(nb.local_degree, U, {"sparsify_score": ANY}, keep=("sparsify_keep",),
         allow_inf=True),
    # --- bipartite projections (Davis Southern Women) ------------------------
    Spec(nb.simple_projection, B, {"weight": NONNEG}),
    Spec(nb.hyper_projection, B, {"weight": NONNEG}),
    Spec(nb.hyper_projection, B, {"weight": NONNEG}, kwargs={"newman": True}),
    Spec(nb.probs_projection, B, {"weight": NONNEG}),
    Spec(nb.ycn_projection, B, {"weight": NONNEG}),
    Spec(nb.bipartite_projection, B, {"weight": NONNEG}),
    Spec(nb.bicm, B),  # returns the probability matrix
    Spec(nb.sdsm, B, {"sdsm_pvalue": P, "weight": NONNEG}, all_pairs=True),
    Spec(nb.fdsm, B, {"fdsm_pvalue": P, "weight": NONNEG},
         kwargs={"trials": 20, "seed": 0}, all_pairs=True),
    Spec(nb.fixedfill, B, {"fixedfill_pvalue": P, "weight": NONNEG}),
    Spec(nb.fixedrow, B, {"fixedrow_pvalue": P, "weight": NONNEG}),
    Spec(nb.fixedcol, B, {"fixedcol_pvalue": P, "weight": NONNEG}),
    Spec(nb.backbone_from_projection, B, {"fixedrow_pvalue": P, "weight": NONNEG},
         kwargs={"method": "fixedrow"}),
    # --- temporal (snapshots of Les Misérables) ------------------------------
    Spec(nb.observation_fraction, TM,
         {"obs_fraction": P, "n_obs": NONNEG, "n_eligible": NONNEG}),
    Spec(nb.persistence, TM, {"persistence": P, "max_run": NONNEG,
                              "mean_run": NONNEG, "n_runs": NONNEG}),
    Spec(nb.backbone_stability, TM, {"stability": P, "n_kept": NONNEG},
         kwargs={"method": "disparity_filter"}),
    Spec(nb.weight_trend, TM, {"weight_mean": NONNEG, "weight_std": NONNEG,
                               "weight_cv": NONNEG, "weight_slope": ANY}),
    Spec(nb.edge_persistence_hmm, TM, {"hmm_score": P, "hmm_active_fraction": P,
                                       "hmm_n_active": NONNEG}, keep=("hmm_keep",)),
]  # fmt: skip

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _inputs(kind):
    """Fresh input(s) for one scorer of *kind* (as in conftest.py's fixtures)."""
    if kind == B:
        Bg = nx.davis_southern_women_graph()
        return Bg, [n for n, d in Bg.nodes(data=True) if d["bipartite"] == 0]
    G = nx.les_miserables_graph()
    if kind == U:
        U_ = nx.Graph()
        U_.add_nodes_from(G.nodes(data=True))
        U_.add_edges_from(G.edges())
        return (U_,)
    return (G,)


def _label(spec):
    kwargs = ", ".join(f"{k}={v!r}" for k, v in spec.kwargs.items())
    return f"{spec.name}({kwargs})" if kwargs else spec.name


def _failures(specs, check):
    """Run *check* on every spec; one line per failing spec (empty if none)."""
    failures = []
    for spec in specs:
        try:
            check(spec)
        except (Exception, pytest.fail.Exception) as err:
            msg = str(err).split("\n", 1)[0]
            failures.append(f"{_label(spec)}: {type(err).__name__}: {msg}")
    return failures


def _mixed_labels(G):
    """Relabel to a mix of int, str and tuple labels (insertion order kept)."""
    labels = {}
    for i, n in enumerate(G):
        labels[n] = (i, (str(i), i), str(i))[i % 3]
    return nx.relabel_nodes(G, labels), labels


def _reversed(G):
    """Same graph with nodes and edges inserted in reverse order."""
    R = G.__class__()
    R.add_nodes_from(reversed(list(G.nodes(data=True))))
    R.add_edges_from((v, u, d) for u, v, d in reversed(list(G.edges(data=True))))
    return R


def _snapshot(G):
    return (
        dict(G.graph),
        {n: dict(d) for n, d in G.nodes(data=True)},
        {(u, v): dict(d) for u, v, d in G.edges(data=True)},
    )


def _edge_set(G):
    if G.is_directed():
        return set(G.edges())
    return {frozenset(e) for e in G.edges()}


def _scores(spec, H):
    """{edge: {attr: value}} for every score/keep attribute, direction-aware."""
    attrs = (*spec.scores, *spec.keep)
    out = {}
    for u, v, d in H.edges(data=True):
        key = (u, v) if H.is_directed() else frozenset((u, v))
        out[key] = {a: d[a] for a in attrs}
    return out


def _assert_same_scores(spec, expected, got, what):
    assert expected.keys() == got.keys(), f"{spec.name}: edge set changed {what}"
    for edge, values in expected.items():
        for attr, val in values.items():
            assert got[edge][attr] == pytest.approx(val, rel=1e-6, abs=1e-9), (
                f"{spec.name}: {attr!r} on {set(edge)} changed {what}"
            )


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------


def _check_scored_graph(spec, G, H, nodes=None, edges=None):
    name = spec.name
    assert H is not G, f"{name}: must return a new graph, not the input"
    assert type(H) is type(G) or spec.kind == B, (
        f"{name}: returned {type(H).__name__} for a {type(G).__name__} input"
    )
    assert set(H) == set(G if nodes is None else nodes), f"{name}: node set changed"
    if edges is not None:
        assert _edge_set(H) == edges, f"{name}: edge set changed"
    for attr in spec.node_attrs:
        missing = [n for n, d in H.nodes(data=True) if attr not in d]
        assert not missing, f"{name}: node attribute {attr!r} missing on {missing[:3]}"
    for u, v, d in H.edges(data=True):
        for attr in spec.keep:
            assert attr in d, f"{name}: edge ({u!r}, {v!r}) lacks {attr!r}"
            assert isinstance(d[attr], bool), f"{name}: {attr!r} is not a bool"
        for attr, (lo, hi) in spec.scores.items():
            assert attr in d, f"{name}: edge ({u!r}, {v!r}) lacks {attr!r}"
            val = d[attr]
            assert not math.isnan(val), f"{name}: {attr!r} is NaN on ({u!r}, {v!r})"
            assert spec.allow_inf or math.isfinite(val), (
                f"{name}: {attr!r} = {val} is not finite on ({u!r}, {v!r})"
            )
            assert math.isinf(val) or lo <= val <= hi, (
                f"{name}: {attr!r} = {val} outside [{lo}, {hi}] on ({u!r}, {v!r})"
            )


def _check_graph_scorer(spec, G):
    name = spec.name
    before = _snapshot(G)
    H = spec(G)
    assert _snapshot(G) == before, f"{name}: input graph was mutated"
    _check_scored_graph(spec, G, H, edges=_edge_set(G))
    scores = _scores(spec, H)

    if spec.order_invariant:
        _assert_same_scores(
            spec,
            scores,
            _scores(spec, spec(_reversed(G))),
            "when insertion order is reversed",
        )

    M, labels = _mixed_labels(G)
    HM = spec(M)
    _check_scored_graph(spec, M, HM, edges=_edge_set(M))
    if spec.order_invariant and not spec.label_ties:
        relabeled = {
            frozenset(labels[n] for n in edge): vals for edge, vals in scores.items()
        }
        _assert_same_scores(spec, relabeled, _scores(spec, HM), "with mixed labels")

    D = nx.DiGraph(G)
    if spec.directed:
        before = _snapshot(D)
        HD = spec(D)
        assert _snapshot(D) == before, f"{name}: directed input was mutated"
        _check_scored_graph(spec, D, HD, edges=_edge_set(D))
    else:
        with pytest.raises(nx.NetworkXNotImplemented):
            spec(D)

    for E in (nx.Graph(), nx.empty_graph(3)):
        HE = spec(E)
        assert set(HE) == set(E), f"{name}: empty graph lost or gained nodes"
        assert HE.number_of_edges() == 0, f"{name}: empty graph gained edges"

    for cls in (nx.MultiGraph, nx.MultiDiGraph):
        multi = cls()
        multi.add_edge(0, 1, weight=5.0)
        multi.add_edge(0, 1, weight=1.0)
        multi.add_edge(1, 2, weight=2.0)
        with pytest.raises(nx.NetworkXNotImplemented):
            spec(multi)


def _check_bipartite_scorer(spec, Bg, women):
    name = spec.name
    before = _snapshot(Bg)
    H = spec(Bg, women)
    assert _snapshot(Bg) == before, f"{name}: input graph was mutated"
    from_gen = spec(Bg, (w for w in women))

    if isinstance(H, np.ndarray):
        n_artifacts = Bg.number_of_nodes() - len(women)
        assert H.shape == (len(women), n_artifacts), f"{name}: wrong matrix shape"
        assert np.all((H >= 0) & (H <= 1)), f"{name}: probabilities outside [0, 1]"
        np.testing.assert_allclose(from_gen, H, err_msg=f"{name}: generator input")
    else:
        _check_scored_graph(spec, Bg, H, nodes=women)
        n = len(women)
        if spec.all_pairs:
            assert H.number_of_edges() == n * (n - 1) // 2, f"{name}: not all pairs"
        assert _edge_set(from_gen) == _edge_set(H), f"{name}: generator input"
        _check_scored_graph(spec, Bg, from_gen, nodes=women)

    M, labels = _mixed_labels(Bg)
    HM = spec(M, [labels[w] for w in women])
    if not isinstance(HM, np.ndarray):
        _check_scored_graph(spec, M, HM, nodes=[labels[w] for w in women])

    E = spec(nx.Graph(), [])
    assert len(E) == 0, f"{name}: empty bipartite graph should give empty output"

    small = nx.Graph([("a", "x"), ("b", "x"), ("a", "y"), ("c", "y")])
    for agents in (["a", "x"], ["a", "b"]):  # an edge inside / outside the set
        with pytest.raises(nx.NetworkXError, match="one side"):
            spec(small, agents)
    for cls in (nx.DiGraph, nx.MultiGraph):
        with pytest.raises(nx.NetworkXNotImplemented):
            spec(cls(small), ["a", "b", "c"])


def _check_temporal_scorer(spec, G):
    """Snapshot k of G lacks every third edge from the k-th; all nodes present.

    Every edge is then eligible in all three steps and observed in two, so
    no score is NaN.
    """
    name = spec.name
    edges = list(G.edges(data=True))
    S = {}
    for k in range(3):
        S[k] = G.__class__()
        S[k].add_nodes_from(G.nodes(data=True))
        S[k].add_edges_from(e for i, e in enumerate(edges) if i % 3 != k)
    before = {t: _snapshot(g) for t, g in S.items()}
    H = spec(S)
    assert {t: _snapshot(g) for t, g in S.items()} == before, f"{name}: mutated"
    _check_scored_graph(spec, nx.Graph(), H, nodes=G, edges=_edge_set(G))
    scores = _scores(spec, H)
    reverse = {t: _reversed(S[t]) for t in reversed(S)}
    for other, what in ((reverse, "in reverse order"), (list(S.values()), "as a list")):
        _assert_same_scores(spec, scores, _scores(spec, spec(other)), what)

    _, labels = _mixed_labels(G)
    HM = spec({t: nx.relabel_nodes(g, labels) for t, g in S.items()})
    relabeled = {frozenset(labels[n] for n in e): x for e, x in scores.items()}
    _assert_same_scores(spec, relabeled, _scores(spec, HM), "with mixed labels")

    D = {t: nx.DiGraph(g) for t, g in S.items()}
    HD = spec(D)
    _check_scored_graph(spec, nx.DiGraph(), HD, nodes=G, edges=_edge_set(nx.DiGraph(G)))

    with pytest.raises(nx.NetworkXNotImplemented):
        spec({**S, 1: nx.MultiGraph(S[1])})
    HE = spec([nx.empty_graph(3), nx.Graph()])
    assert set(HE) == {0, 1, 2} and HE.number_of_edges() == 0


def _check(spec):
    if spec.kind == B:
        _check_bipartite_scorer(spec, *_inputs(B))
    elif spec.kind == TM:
        _check_temporal_scorer(spec, *_inputs(W))
    else:
        _check_graph_scorer(spec, *_inputs(spec.kind))


# One test per input kind, each looping over its scorers; a failure lists every
# scorer that broke the contract, with the first line of its error.
# Les Misérables has leaves, so it cannot be scaled to a doubly stochastic matrix.
@pytest.mark.filterwarnings("ignore:doubly_stochastic_filter:RuntimeWarning")
@pytest.mark.parametrize("kind", [W, U, B, TM])
def test_contract(kind):
    specs = [spec for spec in REGISTRY if spec.kind == kind]
    assert specs
    failures = _failures(specs, _check)
    assert not failures, "\n".join(failures)


def _check_rejects_invalid_weights(spec):
    for bad, pattern in spec.bad_weights.items():
        G = nx.path_graph(4)
        nx.set_edge_attributes(G, 1.0, "weight")
        if bad is MISSING:
            del G[1][2]["weight"]
        else:
            G[1][2]["weight"] = bad
        try:
            spec(G)
        except nx.NetworkXError as err:
            assert pattern is None or re.search(pattern, str(err)), (
                f"weight {bad!r} raised {err!r}, expected /{pattern}/"
            )
        else:
            pytest.fail(f"accepted invalid weight {bad!r}")


def test_rejects_invalid_weights():
    validating = [spec for spec in REGISTRY if spec.bad_weights]
    failures = _failures(validating, _check_rejects_invalid_weights)
    assert not failures, "\n".join(failures)
