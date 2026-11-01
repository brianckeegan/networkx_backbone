"""Cross-implementation reference tests against R ``backbone`` and netbone.

The fixtures in ``tests/fixtures/`` hold p-values, scores and retained edges
computed by the reference implementations on the same data networkx ships:

- ``r_backbone.json``: R package ``backbone`` (version in ``meta``), written
  by ``scripts/make_reference_fixtures.R``.
- ``netbone.json``: the netbone Python package, written by
  ``scripts/make_netbone_fixtures.py``.
- ``reference_inputs/*.csv``: the input data both generators read, written by
  ``scripts/export_reference_inputs.py``.

Neither R nor netbone is needed to run these tests; ``docs/contributing.rst``
explains how to regenerate the fixtures.  Where networkx-backbone
intentionally differs from a reference, the test asserts the documented
relationship instead of equality and the method's docstring Notes explain
the difference.
"""

import csv
import functools
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import networkx_backbone as nb
from networkx_backbone.statistical import _nc_score_sdev

FIXTURES = Path(__file__).parent / "fixtures"
INPUTS = FIXTURES / "reference_inputs"
ALPHA = 0.05
# Any alpha above 1 keeps every pair (including p = 1) in the fixed-model
# scorers, which threshold at alpha themselves.
KEEP_ALL = 1.01


@functools.cache
def r_fixture():
    return json.loads((FIXTURES / "r_backbone.json").read_text(encoding="utf-8"))


@functools.cache
def netbone_fixture():
    return json.loads((FIXTURES / "netbone.json").read_text(encoding="utf-8"))


def values(xs):
    """Fixture list to a float array (``None`` -> NaN)."""
    return np.array([np.nan if x is None else x for x in xs], dtype=float)


WEIGHTED = {
    "les_miserables": (nx.les_miserables_graph, str),
    "karate": (nx.karate_club_graph, int),
}


def weighted_case(dataset, fixture):
    """``(G, nodes, edges)`` for a weighted dataset in fixture edge order."""
    make, conv = WEIGHTED[dataset]
    section = fixture[dataset]
    nodes = [conv(n) for n in section["nodes"]]
    edges = []
    for e in section["edges"]:
        if isinstance(e[0], str):  # netbone stores labels
            edges.append((conv(e[0]), conv(e[1])))
        else:  # R stores 0-based node indices
            edges.append((nodes[e[0]], nodes[e[1]]))
    return make(), nodes, edges


def edge_values(H, edges, attr):
    return np.array([float(H.edges[e][attr]) for e in edges])


def r_backbone_set(entries, nodes):
    """R's retained ``[i, j, sign]`` triples as ``{(frozenset, sign)}``."""
    return {(frozenset((nodes[int(i)], nodes[int(j)])), int(s)) for i, j, s in entries}


def our_backbone_set(H):
    return {(frozenset(e), H.edges[e].get("sign", 1)) for e in H.edges}


def r_two_sided(upper, lower):
    """R's signed test as networkx-backbone reports it.

    R retains an edge when ``min(upper, lower) < alpha / 2`` and signs it
    ``+1`` when ``upper < lower``; networkx-backbone stores the equivalent
    two-sided p-value ``min(1, 2 min(upper, lower))``.
    """
    return np.minimum(1.0, 2.0 * np.minimum(upper, lower)), np.where(
        upper < lower, 1, -1
    )


def assert_signs(sign, upper, lower, atol=1e-12):
    """Signs agree with R's wherever the two tails differ.

    R signs a test ``+1`` only when ``upper < lower``; networkx-backbone signs
    it ``+1`` when ``upper <= lower``.  At an exact tie R reports ``-1`` and
    networkx-backbone ``+1``, and tails equal up to rounding can fall either
    way, so ties are not compared; the two-sided p-value, ``2 * upper``, is
    the same either way.
    """
    tie = np.abs(upper - lower) <= atol
    np.testing.assert_array_equal(sign[~tie], np.where(upper < lower, 1, -1)[~tie])


# =====================================================================
# Inputs
# =====================================================================


def _read_csv(name):
    with open(INPUTS / name, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


@pytest.mark.parametrize("dataset", sorted(WEIGHTED))
def test_weighted_inputs_match_networkx(dataset):
    make, conv = WEIGHTED[dataset]
    G = make()
    rows = _read_csv(f"{dataset}.csv")
    assert rows[0] == ["source", "target", "weight"]
    expected = [(u, v, float(w)) for u, v, w in G.edges(data="weight")]
    assert [(conv(u), conv(v), float(w)) for u, v, w in rows[1:]] == expected


def test_davis_input_matches_networkx():
    B = nx.davis_southern_women_graph()
    rows = _read_csv("davis_southern_women.csv")
    women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    events = [n for n, d in B.nodes(data=True) if d["bipartite"] == 1]
    assert rows[0] == ["woman", *events]
    assert [r[0] for r in rows[1:]] == women
    for w, row in zip(women, rows[1:]):
        assert [int(x) for x in row[1:]] == [int(B.has_edge(w, e)) for e in events]


def test_fixture_metadata():
    meta = r_fixture()["meta"]
    assert meta["backbone"].startswith("3.")
    assert meta["alpha"] == ALPHA
    assert "R" in meta
    assert "netbone" in netbone_fixture()["meta"]


# =====================================================================
# R backbone: weighted models (disparity, mlf, lans, global)
# =====================================================================

WEIGHTED_MODELS = {
    "disparity": (nb.disparity_filter, "disparity_pvalue"),
    "mlf": (nb.marginal_likelihood_filter, "ml_pvalue"),
    "lans": (nb.lans_filter, "lans_pvalue"),
}


@pytest.mark.parametrize("dataset", sorted(WEIGHTED))
@pytest.mark.parametrize("model", sorted(WEIGHTED_MODELS))
def test_r_weighted_upper_tail(dataset, model):
    """One-tailed p-values equal R's for every edge."""
    G, _, edges = weighted_case(dataset, r_fixture())
    func, attr = WEIGHTED_MODELS[model]
    ours = edge_values(func(G), edges, attr)
    theirs = values(r_fixture()[dataset][model]["upper"])
    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("dataset", sorted(WEIGHTED))
@pytest.mark.parametrize("model", sorted(WEIGHTED_MODELS))
@pytest.mark.parametrize("mtc", ["none", "bonferroni", "holm", "BH"])
def test_r_weighted_one_tailed_backbone(dataset, model, mtc):
    G, nodes, _ = weighted_case(dataset, r_fixture())
    H = nb.backbone_from_weighted(G, method=model, alpha=ALPHA, mtc=mtc)
    expected = r_fixture()[dataset][model]["backbones"][mtc]
    assert our_backbone_set(H) == r_backbone_set(expected, nodes)


@pytest.mark.parametrize("dataset", sorted(WEIGHTED))
def test_r_mlf_signed(dataset):
    """Signed MLF: two-sided p-values, signs and backbones equal R's."""
    G, nodes, edges = weighted_case(dataset, r_fixture())
    section = r_fixture()[dataset]["mlf"]
    H = nb.marginal_likelihood_filter(G, signed=True)
    p, _ = r_two_sided(values(section["upper"]), values(section["lower"]))
    np.testing.assert_allclose(edge_values(H, edges, "ml_pvalue"), p, rtol=1e-12)
    assert_signs(
        edge_values(H, edges, "sign"),
        values(section["upper"]),
        values(section["lower"]),
    )
    for mtc in ("none", "bonferroni", "holm", "BH"):
        B = nb.backbone_from_weighted(G, method="mlf", signed=True, mtc=mtc)
        expected = section["backbones"][f"signed_{mtc}"]
        assert our_backbone_set(B) == r_backbone_set(expected, nodes)


@pytest.mark.parametrize("dataset", sorted(WEIGHTED))
def test_r_disparity_signed(dataset):
    """Signed disparity: documented lower-tail difference.

    R's lower tail is ``1 - upper`` with ``upper = min(alpha_u, alpha_v)``;
    networkx-backbone applies the OR rule to each tail, so its lower tail
    ``min(1 - alpha_u, 1 - alpha_v)`` is never larger than R's.  Hence a
    positive edge has the same p-value in both, a negative one a p-value no
    larger than R's.
    """
    G, nodes, edges = weighted_case(dataset, r_fixture())
    section = r_fixture()[dataset]["disparity"]
    upper, lower = values(section["upper"]), values(section["lower"])
    np.testing.assert_allclose(lower, 1.0 - upper, atol=1e-12)
    r_p, r_sign = r_two_sided(upper, lower)
    H = nb.disparity_filter(G, signed=True)
    p = edge_values(H, edges, "disparity_pvalue")
    sign = edge_values(H, edges, "sign")
    pos = sign == 1
    np.testing.assert_allclose(p[pos], np.minimum(1.0, 2.0 * upper[pos]), rtol=1e-12)
    assert np.all(r_sign[pos] == 1)
    assert np.all(p[~pos] <= r_p[~pos] + 1e-12)
    # On these datasets the retained edges are the same as R's.
    for mtc in ("none", "bonferroni", "holm", "BH"):
        B = nb.backbone_from_weighted(G, method="disparity", signed=True, mtc=mtc)
        expected = section["backbones"][f"signed_{mtc}"]
        assert our_backbone_set(B) == r_backbone_set(expected, nodes)


@pytest.mark.parametrize("dataset", sorted(WEIGHTED))
def test_r_lans_signed(dataset):
    """Signed LANS: documented lower-tail difference.

    Both use the upper tail ``#{weights > w} / k``.  R's lower tail is the
    mirror image ``#{weights < w} / k``; networkx-backbone's is the empirical
    CDF ``#{weights <= w} / k``, which counts the edge itself, so it is
    larger by at least ``1 / k``.  Every edge R signs ``+1`` has the same sign
    and p-value here, and networkx-backbone retains a subset of R's
    negative edges: R marks the weakest edge of any node as significantly
    weak, networkx-backbone only when ``1 / k < alpha / 2``.
    """
    G, nodes, edges = weighted_case(dataset, r_fixture())
    section = r_fixture()[dataset]["lans"]
    upper, lower = values(section["upper"]), values(section["lower"])
    r_p, r_sign = r_two_sided(upper, lower)
    H = nb.lans_filter(G, signed=True)
    p = edge_values(H, edges, "lans_pvalue")
    sign = edge_values(H, edges, "sign")
    pos = r_sign == 1
    np.testing.assert_allclose(p[pos], r_p[pos], atol=1e-12)
    assert np.all(sign[pos] == 1)
    assert np.all(p >= r_p - 1e-12)
    # R signs an exact tie upper == lower as -1 (a node whose links all have
    # the same weight makes each of them both its strongest and its weakest);
    # networkx-backbone signs it +1.
    ties = {frozenset(e) for e, u, lo in zip(edges, upper, lower) if u == lo}
    for mtc in ("none", "bonferroni", "holm", "BH"):
        B = our_backbone_set(
            nb.backbone_from_weighted(G, method="lans", signed=True, mtc=mtc)
        )
        R = r_backbone_set(section["backbones"][f"signed_{mtc}"], nodes)
        ours_pos = {e for e, s in B if s == 1}
        r_pos = {e for e, s in R if s == 1}
        assert r_pos <= ours_pos
        assert ours_pos - r_pos <= ties & {e for e, s in R if s == -1}
        assert {e for e in B if e[1] == -1} <= {e for e in R if e[1] == -1}


def test_r_p_adjust():
    """``adjust_pvalues`` reproduces R's ``stats::p.adjust`` (with ties)."""
    section = r_fixture()["les_miserables"]
    p = values(section["mlf"]["upper"])
    for method, expected in section["p_adjust"].items():
        ours = nb.adjust_pvalues(p.tolist(), method=method.lower())
        np.testing.assert_allclose(ours, values(expected), rtol=1e-12, atol=1e-15)


def test_r_global_threshold():
    """R keeps ``w > t``; :func:`global_threshold_filter` keeps ``w >= t``."""
    section = r_fixture()["les_miserables"]
    G, nodes, _ = weighted_case("les_miserables", r_fixture())
    bb = section["global"]["backbones"]
    for t, key in ((2, "threshold_2"), (4, "signed_1_4")):
        H = nb.global_threshold_filter(G, threshold=math.nextafter(t, math.inf))
        kept = nb.boolean_filter(H, "global_threshold_keep")
        # No Les Miserables weight is below 1, so R's signed (1, 4) backbone
        # has only positive edges.
        assert our_backbone_set(kept) == r_backbone_set(bb[key], nodes)


# =====================================================================
# R backbone: unweighted sparsification (karate club)
# =====================================================================


def _karate_unweighted():
    G = nx.Graph()
    G.add_edges_from(nx.karate_club_graph().edges())
    return G


@pytest.mark.parametrize("s", [0.3, 0.5, 0.7])
def test_r_local_degree(s):
    G = _karate_unweighted()
    nodes = [int(n) for n in r_fixture()["karate"]["nodes"]]
    H = nb.boolean_filter(nb.local_degree(G, s=s), "sparsify_keep")
    expected = r_fixture()["karate"]["unweighted"][f"degree_{s}"]
    assert our_backbone_set(H) == r_backbone_set(expected, nodes)


def _r_lspar_rule(G, s):
    """R's ``lspar`` as documented in :func:`networkx_backbone.lspar` Notes."""
    nbrs = {u: set(G[u]) for u in G}
    kept = set()
    for u in G:
        jac = {}
        for v in G[u]:
            union = len(nbrs[u] | nbrs[v]) - 2  # the endpoints are excluded
            jac[v] = len(nbrs[u] & nbrs[v]) / union if union else 0.0
        positive = sorted({x for x in jac.values() if x > 0}, reverse=True)
        cut = math.floor(len([x for x in jac.values() if x > 0]) ** s)
        kept |= {
            frozenset((u, v))
            for v, x in jac.items()
            if x > 0 and positive.index(x) + 1 <= cut  # dense rank
        }
    return kept


@pytest.mark.parametrize("s", [0.3, 0.5, 0.7])
def test_r_lspar_documented_difference(s):
    """R's ``lspar`` uses different conventions (see ``lspar`` Notes)."""
    G = _karate_unweighted()
    nodes = [int(n) for n in r_fixture()["karate"]["nodes"]]
    expected = {
        e
        for e, _ in r_backbone_set(
            r_fixture()["karate"]["unweighted"][f"lspar_{s}"], nodes
        )
    }
    assert _r_lspar_rule(G, s) == expected
    ours = {
        frozenset(e) for e in nb.boolean_filter(nb.lspar(G, s=s), "sparsify_keep").edges
    }
    assert ours != expected  # the conventions matter on this graph


def _unweighted(dataset):
    G, nodes, _ = weighted_case(dataset, r_fixture())
    U = nx.Graph()
    U.add_nodes_from(G)
    U.add_edges_from(G.edges())
    return U, nodes


@pytest.mark.parametrize("dataset", sorted(WEIGHTED))
@pytest.mark.parametrize("s", [0.3, 0.5, 0.7])
def test_r_lspar_convention_r(dataset, s):
    """``lspar(convention="r")`` reproduces R's ``lspar`` exactly."""
    G, nodes = _unweighted(dataset)
    expected = {
        e
        for e, _ in r_backbone_set(
            r_fixture()[dataset]["unweighted"][f"lspar_{s}"], nodes
        )
    }
    H = nb.lspar(G, s=s, convention="r")
    assert {frozenset(e) for e in nb.boolean_filter(H, "sparsify_keep").edges} == (
        expected
    )
    assert _r_lspar_rule(G, s) == expected
    # The critical exponent reproduces the same backbone.
    assert {
        frozenset(e)
        for e, x in nx.get_edge_attributes(H, "sparsify_score").items()
        if x <= s
    } == expected
    # Through the dispatcher too.
    B = nb.backbone_from_unweighted(G, method="lspar", s=s, convention="r")
    assert {frozenset(e) for e in B.edges} == expected


# =====================================================================
# R backbone: bipartite projections (Davis Southern Women)
# =====================================================================


@functools.cache
def davis():
    section = r_fixture()["davis"]
    B = nx.davis_southern_women_graph()
    women = section["agents"]
    pairs = [(women[i], women[j]) for i, j in section["pairs"]]
    return B, women, pairs, np.array(section["cooccurrence"])


def pair_values(H, pairs, attr):
    return np.array([float(H.edges[e][attr]) for e in pairs])


def test_davis_agents_and_counts():
    B, women, pairs, k = davis()
    assert women == [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    assert len(pairs) == len(women) * (len(women) - 1) // 2
    assert [len(set(B[u]) & set(B[v])) for u, v in pairs] == k.tolist()


def test_r_bicm():
    B, women, _, _ = davis()
    P, agents, artifacts = nb.bicm(B, agent_nodes=women, return_labels=True)
    order = [agents.index(w) for w in women]
    cols = [artifacts.index(e) for e in r_fixture()["davis"]["artifacts"]]
    np.testing.assert_allclose(
        P[order][:, cols], r_fixture()["davis"]["bicm"], atol=1e-7
    )


def test_r_sdsm_rna():
    """With ``pvalue_method="rna"`` SDSM reproduces R up to the BiCM fit.

    R evaluates the refined normal approximation even for pairs that never
    co-occur (``missing_as_zero=TRUE``); networkx-backbone returns the exact
    value ``P(X >= 0) = 1`` for them.
    """
    B, women, pairs, k = davis()
    section = r_fixture()["davis"]["sdsm"]
    upper, lower = values(section["upper"]), values(section["lower"])
    ours = pair_values(sdsm_scores(B, women, "rna", False), pairs, "sdsm_pvalue")
    np.testing.assert_allclose(ours[k > 0], upper[k > 0], atol=1e-7)
    assert np.all(ours[k == 0] == 1.0)
    signed = sdsm_scores(B, women, "rna", True)
    r_p, _ = r_two_sided(upper, lower)
    np.testing.assert_allclose(
        pair_values(signed, pairs, "sdsm_pvalue"), r_p, atol=1e-7
    )
    assert_signs(pair_values(signed, pairs, "sign"), upper, lower, atol=1e-7)


def sdsm_scores(B, women, method, signed):
    return nb.sdsm(B, agent_nodes=women, pvalue_method=method, signed=signed)


def test_r_sdsm_exact_vs_rna():
    """Davis (14 events) uses the exact Poisson-binomial tail by default.

    The exact tail differs from R's refined normal approximation by at most
    0.017 here; both find no significant pair at alpha = 0.05 (smallest
    p-value 0.088 exact, 0.097 in R).
    """
    B, women, pairs, k = davis()
    upper = values(r_fixture()["davis"]["sdsm"]["upper"])
    ours = pair_values(nb.sdsm(B, agent_nodes=women), pairs, "sdsm_pvalue")
    assert np.max(np.abs(ours - upper)[k > 0]) < 0.02
    assert ours.min() == pytest.approx(0.0884, abs=5e-4)
    assert upper[k > 0].min() == pytest.approx(0.0974, abs=5e-4)
    assert (ours < ALPHA).sum() == (upper < ALPHA).sum() == 0


def test_r_sdsm_ec():
    """SDSM-EC: different null-model fits, same (empty) backbones.

    R fits the edge probabilities of the unconstrained cells with a logistic
    regression on the row and column sums; networkx-backbone refits the BiCM
    on those cells, so expected degrees match the observed ones exactly.  The
    p-values track each other closely but not exactly.
    """
    B, women, pairs, k = davis()
    section = r_fixture()["davis"]["sdsm_ec"]
    kwargs = {
        "prohibited": [tuple(c) for c in section["prohibited"]],
        "required": [tuple(c) for c in section["required"]],
    }
    upper = values(section["upper"])
    H = nb.sdsm(B, agent_nodes=women, pvalue_method="rna", **kwargs)
    ours = pair_values(H, pairs, "sdsm_pvalue")
    assert np.corrcoef(ours[k > 0], upper[k > 0])[0, 1] > 0.99
    assert np.max(np.abs(ours - upper)[k > 0]) < 0.1
    for signed in (False, True):
        for mtc in ("none", "bonferroni", "holm", "BH"):
            bb = nb.backbone_from_projection(
                B, agent_nodes=women, signed=signed, mtc=mtc, **kwargs
            )
            key = f"signed_{mtc}" if signed else mtc
            expected = section["backbones"][key]
            assert our_backbone_set(bb) == r_backbone_set(expected, women)


@pytest.mark.parametrize("model", ["fixedfill", "fixedrow"])
def test_r_fixed_models_exact(model):
    """Fixed fill and fixed row tails are exact in both implementations."""
    B, women, pairs, _ = davis()
    section = r_fixture()["davis"][model]
    func = getattr(nb, model)
    upper, lower = values(section["upper"]), values(section["lower"])
    H = func(B, women, alpha=KEEP_ALL)
    np.testing.assert_allclose(
        pair_values(H, pairs, f"{model}_pvalue"), upper, rtol=1e-10, atol=1e-15
    )
    S = func(B, women, alpha=KEEP_ALL, signed=True)
    r_p, _ = r_two_sided(upper, lower)
    np.testing.assert_allclose(
        pair_values(S, pairs, f"{model}_pvalue"), r_p, rtol=1e-10, atol=1e-15
    )
    assert_signs(pair_values(S, pairs, "sign"), upper, lower, atol=1e-12)


def test_r_fixedcol_exact_vs_rna():
    """``fixedcol`` computes the exact Poisson-binomial tail; R uses the RNA."""
    B, women, pairs, _ = davis()
    section = r_fixture()["davis"]["fixedcol"]
    upper, lower = values(section["upper"]), values(section["lower"])
    H = nb.fixedcol(B, women, alpha=KEEP_ALL)
    np.testing.assert_allclose(
        pair_values(H, pairs, "fixedcol_pvalue"), upper, atol=0.01
    )
    S = nb.fixedcol(B, women, alpha=KEEP_ALL, signed=True)
    r_p, r_sign = r_two_sided(upper, lower)
    np.testing.assert_allclose(pair_values(S, pairs, "fixedcol_pvalue"), r_p, atol=0.02)
    clear = np.abs(upper - lower) > 0.01
    np.testing.assert_array_equal(pair_values(S, pairs, "sign")[clear], r_sign[clear])


@pytest.mark.parametrize("model", ["sdsm", "fixedfill", "fixedrow", "fixedcol"])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("mtc", ["none", "bonferroni", "holm", "BH"])
def test_r_projection_backbones(model, signed, mtc):
    B, women, _, _ = davis()
    H = nb.backbone_from_projection(
        B, agent_nodes=women, method=model, alpha=ALPHA, signed=signed, mtc=mtc
    )
    key = f"signed_{mtc}" if signed else mtc
    expected = r_fixture()["davis"][model]["backbones"][key]
    assert our_backbone_set(H) == r_backbone_set(expected, women)


@pytest.mark.parametrize("model", ["fixedfill", "fixedrow"])
@pytest.mark.parametrize("missing_as_zero", [False, True])
def test_r_projection_missing_as_zero(model, missing_as_zero):
    """``missing_as_zero`` reproduces R's choice of tested pairs exactly.

    With ``False`` (R's default) the 14 Davis pairs that never co-occur are
    neither counted by ``mtc`` nor retained; with ``True`` (the default here)
    every pair is a test.  The fixture uses alphas at which this matters.
    """
    B, women, _, _ = davis()
    section = r_fixture()["davis"]["missing_as_zero"]
    for alpha in section["alphas"]:
        expected = section[model][str(missing_as_zero).lower()][str(alpha)]
        for key, edges in expected.items():
            signed = key.startswith("signed_")
            H = nb.backbone_from_projection(
                B,
                agent_nodes=women,
                method=model,
                alpha=alpha,
                signed=signed,
                mtc=key.removeprefix("signed_"),
                missing_as_zero=missing_as_zero,
            )
            assert our_backbone_set(H) == r_backbone_set(edges, women), (alpha, key)


def test_r_projection_missing_as_zero_matters():
    """The ``missing_as_zero`` fixture discriminates between the two scopes."""
    section = r_fixture()["davis"]["missing_as_zero"]
    differ = [
        (model, alpha, key)
        for model in ("fixedfill", "fixedrow")
        for alpha in section["alphas"]
        for key, edges in section[model]["false"][str(alpha)].items()
        if edges != section[model]["true"][str(alpha)][key]
    ]
    assert len(differ) >= 10
    # Without mtc an unsigned backbone does not depend on it.
    assert not [d for d in differ if d[2] == "none"]


def _mc_tolerance(q, *trials):
    """Four Monte Carlo standard errors of a difference of p-value estimates."""
    var = np.clip(q * (1 - q), 1e-4, None) * sum(1 / t for t in trials)
    return 4 * np.sqrt(var) + 1 / (min(trials) + 1)


def test_r_fdsm():
    """FDSM p-values agree with a well-mixed R reference within Monte Carlo error.

    R's ``fdsm()`` p-value is ``count / trials`` over independent samples,
    each drawn by ``fastball()`` from the observed matrix with 5 trades per
    row (70 trades on Davis, rotated to 14 event rows).  That is too few to
    forget the observed matrix, and its p-values are biased upward.  The
    fixture's ``fdsm_mixed`` repeats R's test with 100 trades per row using
    backbone's own ``fastball()``; ours, ``(count + 1) / (trials + 1)`` from
    one curveball chain, matches it.  Both R variants and ours find the same
    5 significant pairs at alpha = 0.05.
    """
    B, women, pairs, _ = davis()
    default = r_fixture()["davis"]["fdsm"]
    mixed = r_fixture()["davis"]["fdsm_mixed"]
    trials = 2000
    upper, lower = values(mixed["upper"]), values(mixed["lower"])

    one_tailed = pair_values(
        nb.fdsm(B, agent_nodes=women, trials=trials, seed=14), pairs, "fdsm_pvalue"
    )
    tol = _mc_tolerance(upper, trials, mixed["trials"])
    assert np.all(np.abs(one_tailed - upper) <= tol)
    S = nb.fdsm(B, agent_nodes=women, trials=trials, seed=14, signed=True)
    r_p, _ = r_two_sided(upper, lower)
    tail = np.minimum(upper, lower)
    assert np.all(
        np.abs(pair_values(S, pairs, "fdsm_pvalue") - r_p)
        <= 2 * _mc_tolerance(tail, trials, mixed["trials"])
    )

    # R's default sampler is measurably biased on Davis ...
    default_upper = values(default["upper"])
    se = np.sqrt(np.clip(upper * (1 - upper), 1e-4, None) * 2 / mixed["trials"])
    assert np.max((default_upper - upper) / se) > 4
    # ... but not enough to change the backbone at alpha = 0.05.
    assert len(default["backbones"]["none"]) == (default_upper < ALPHA).sum() == 5
    assert (upper < ALPHA).sum() == 5
    clear = np.abs(upper - ALPHA) > tol
    np.testing.assert_array_equal((one_tailed < ALPHA)[clear], (upper < ALPHA)[clear])
    assert (one_tailed < ALPHA).sum() == 5


# =====================================================================
# netbone
# =====================================================================


@pytest.mark.parametrize("dataset", sorted(WEIGHTED))
@pytest.mark.parametrize(
    ("model", "func", "attr"),
    [
        ("disparity", nb.disparity_filter, "disparity_pvalue"),
        ("lans", nb.lans_filter, "lans_pvalue"),
        ("marginal_likelihood", nb.marginal_likelihood_filter, "ml_pvalue"),
    ],
)
def test_netbone_exact(dataset, model, func, attr):
    G, _, edges = weighted_case(dataset, netbone_fixture())
    theirs = values(netbone_fixture()[dataset][model]["p_value"])
    np.testing.assert_allclose(
        edge_values(func(G), edges, attr), theirs, rtol=1e-12, atol=1e-15
    )


@pytest.mark.parametrize("dataset", sorted(WEIGHTED))
def test_netbone_ecm(dataset):
    """ECM p-values agree up to the solvers' tolerances.

    netbone's JAX solve is not bit-for-bit reproducible: regenerating the
    fixture changes its karate p-values by up to about 1e-5.
    """
    G, _, edges = weighted_case(dataset, netbone_fixture())
    theirs = values(netbone_fixture()[dataset]["ecm"]["p_value"])
    ours = edge_values(nb.ecm_filter(G), edges, "ecm_pvalue")
    np.testing.assert_allclose(ours, theirs, rtol=1e-4, atol=5e-5)


@pytest.mark.parametrize("dataset", sorted(WEIGHTED))
def test_netbone_noise_corrected(dataset):
    """netbone uses total weight ``W``; networkx-backbone ``2W`` (see Notes).

    netbone sums each undirected edge once for the total but at both
    endpoints for the strengths.  With that total, the score and standard
    deviation formulas are the same.
    """
    G, _, edges = weighted_case(dataset, netbone_fixture())
    section = netbone_fixture()[dataset]["noise_corrected"]
    W = G.size(weight="weight")
    s = dict(G.degree(weight="weight"))
    with_w = np.array(
        [_nc_score_sdev(G.edges[u, v]["weight"], s[u], s[v], W) for u, v in edges]
    )
    np.testing.assert_allclose(with_w[:, 0], values(section["score"]), rtol=1e-12)
    np.testing.assert_allclose(with_w[:, 1], values(section["sdev"]), rtol=1e-10)
    ours = edge_values(nb.noise_corrected_filter(G), edges, "nc_score")
    assert not np.allclose(ours, values(section["score"]))


def _inverse_weight(u, v, data):
    return 1.0 / data["weight"]


def _shortest_path_ties(G):
    """Per-edge salience bounds and the roots that reach a node via a tie."""
    n = len(G)
    lo = {frozenset(e): 0 for e in G.edges}
    hi = dict(lo)
    tied_roots = set()
    for root in G:
        pred, _ = nx.dijkstra_predecessor_and_distance(G, root, weight=_inverse_weight)
        for x, preds in pred.items():
            if len(preds) > 1:
                tied_roots.add(root)
            for p in preds:
                hi[frozenset((x, p))] += 1
                lo[frozenset((x, p))] += len(preds) == 1
    return (
        {e: v / n for e, v in lo.items()},
        {e: v / n for e, v in hi.items()},
        tied_roots,
    )


@pytest.mark.parametrize("dataset", sorted(WEIGHTED))
def test_netbone_high_salience_skeleton(dataset):
    """Same salience except where shortest paths tie (see Notes).

    netbone credits the one path Dijkstra happens to return; networkx-backbone
    splits the credit over the tied predecessors.  Both lie between the
    fraction of trees in which the edge is the only shortest-path predecessor
    and the fraction in which it is one of them, and agree when those are
    equal.
    """
    G, _, edges = weighted_case(dataset, netbone_fixture())
    theirs = values(netbone_fixture()[dataset]["high_salience_skeleton"]["salience"])
    ours = edge_values(nb.high_salience_skeleton(G), edges, "salience")
    lo_d, hi_d, _ = _shortest_path_ties(G)
    lo = np.array([lo_d[frozenset(e)] for e in edges])
    hi = np.array([hi_d[frozenset(e)] for e in edges])
    for x in (ours, theirs):
        assert np.all((lo - 1e-12 <= x) & (x <= hi + 1e-12))
    np.testing.assert_allclose(ours[lo == hi], theirs[lo == hi], atol=1e-12)
    assert ours.sum() == pytest.approx(theirs.sum()) == pytest.approx(len(G) - 1)


@pytest.mark.parametrize("c", [0.5, 1.0])
def test_netbone_glab(c):
    """GLAB p-values agree for links whose endpoints see no tied paths.

    netbone's involvement counts shortest paths (a target reached by ``g``
    tied paths weighs ``g`` times); networkx-backbone uses the paper's
    average over targets of the fraction of paths through the link.
    """
    G, _, edges = weighted_case("les_miserables", netbone_fixture())
    theirs = values(netbone_fixture()["les_miserables"][f"glanb_c{c}"]["SI"])
    ours = edge_values(nb.glab_filter(G, c=c), edges, "glab_pvalue")
    _, _, tied = _shortest_path_ties(G)
    free = np.array([u not in tied and v not in tied for u, v in edges])
    assert free.sum() > 100
    np.testing.assert_allclose(ours[free], theirs[free], atol=1e-12)


@pytest.mark.parametrize("dataset", sorted(WEIGHTED))
@pytest.mark.parametrize(
    ("model", "func", "attr"),
    [
        ("metric_distance_backbone", nb.metric_backbone, "metric_keep"),
        ("ultrametric_distance_backbone", nb.ultrametric_backbone, "ultrametric_keep"),
    ],
)
def test_netbone_distance_backbones(dataset, model, func, attr):
    G, _, edges = weighted_case(dataset, netbone_fixture())
    theirs = netbone_fixture()[dataset][model]["in_backbone"]
    H = func(G)
    ours = [bool(H.edges[e][attr]) for e in edges]
    assert ours == theirs


@pytest.mark.parametrize("dataset", sorted(WEIGHTED))
def test_netbone_h_backbone(dataset):
    """Same h-strength edges; different bridge values (see Notes).

    netbone computes edge betweenness with the weights as distances and
    rounds the bridge to three decimals; networkx-backbone follows the paper
    and uses the unweighted topology, so only the h-strength part agrees.
    """
    G, _, edges = weighted_case(dataset, netbone_fixture())
    theirs = np.array(netbone_fixture()[dataset]["h_backbone"]["in_backbone"])
    ours = np.array(
        [bool(x) for x in edge_values(nb.h_backbone(G), edges, "h_backbone_keep")]
    )
    w = np.array([G.edges[e]["weight"] for e in edges])
    h = max(i for i in range(len(w) + 1) if (w >= i).sum() >= i)
    strong = w >= h
    assert strong.any()
    assert np.all(ours[strong]) and np.all(theirs[strong])


@pytest.mark.parametrize("dataset", sorted(WEIGHTED))
def test_netbone_mla_differs(dataset):
    """netbone's ``mlam`` is a different procedure (see MLA Notes).

    It fits ideal distributions by the rounded squared correlation and stops
    at the first fit that does not improve.  On these datasets every link it
    keeps is also kept by the default ``method="dominance"``, which keeps
    more.
    """
    G, _, edges = weighted_case(dataset, netbone_fixture())
    theirs = np.array(netbone_fixture()[dataset]["mlam"]["in_backbone"])
    H = nb.multiple_linkage_analysis(G)
    ours = np.array([bool(H.edges[e]["mla_keep"]) for e in edges])
    assert np.all(ours[theirs])
    assert ours.sum() > theirs.sum()
