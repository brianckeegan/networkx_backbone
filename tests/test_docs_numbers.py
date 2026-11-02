"""The numbers quoted in the prose of the how-to and user guide pages.

The narrative tutorials (``docs/examples/``) compute every number when the
documentation is built, but the reST pages under ``docs/tutorials/`` and
``docs/user_guide/`` state results as plain text, which nothing else would
catch going stale after a change in results (#137).  Each entry below names a
page, a passage of its text with ``{}`` where a computed value goes (``{*}``
for a value that contains spaces), and a function that computes the values
with the library.  The test fails when the page no longer says what the
library computes, and names the page, the text it found and the text it
expects.  ``FACTS`` does the same for comparisons stated in words.

Pages whose code runs as doctests (``validating_backbones`` and
``temporal_backbones``) are run once here, and their values come from the
doctest namespace, so the prose is checked against the output it describes.

Not checked: dates, timings labeled as measured (the table on
``user_guide/backends``), the inputs of examples (sizes, probabilities and
parameters of synthetic data), values marked "e.g.", and results quoted from
the literature.  Printed output in doctests is checked by the doctests.
"""

import doctest
import functools
import re
from fractions import Fraction
from pathlib import Path

import networkx as nx
import pytest

import nx_backbone as nb

pytest.importorskip("numpy")
pytest.importorskip("scipy")

DOCS = Path(__file__).resolve().parents[1] / "docs"
ALPHA = 0.05

# structural_methods quotes doubly_stochastic_filter's count without smoothing,
# where Sinkhorn-Knopp does not converge and the filter warns (the page says so).
pytestmark = pytest.mark.filterwarnings("ignore:doubly_stochastic_filter")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@functools.cache
def _text(page):
    path = DOCS / f"{page}.rst"
    if not path.exists():
        pytest.skip("docs not available")
    return " ".join(path.read_text(encoding="utf-8").split())


def _pattern(template):
    parts = re.split(r"(\{\*?\})", template)
    return "".join(
        "(.+?)" if p == "{*}" else r"(\S+)" if p == "{}" else re.escape(p)
        for p in parts
    )


def _fill(template, values):
    return re.sub(r"\{\*?\}", "{}", template).format(*values)


_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def _words(n, capitalize=False):
    """``34`` as ``"thirty-four"``, for numbers from 0 to 99."""
    tens, ones = divmod(n, 10)
    word = _ONES[n] if n < 20 else _TENS[tens - 2] + (f"-{_ONES[ones]}" * bool(ones))
    return word.capitalize() if capitalize else word


@functools.cache
def _namespace(page):
    """The globals of a page's doctests after running all of them."""
    text = (DOCS / f"{page}.rst").read_text(encoding="utf-8")
    test = doctest.DocTestParser().get_doctest(text, {}, page, page, 0)
    runner = doctest.DocTestRunner(
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    result = runner.run(test, clear_globs=False)
    assert result.failed == 0, f"docs/{page}.rst: {result.failed} doctests fail"
    return test.globs


def _mean(H, edges, score):
    # The pages' own helper: the mean over edges, NaN skipped, to 2 decimals.
    values = [x for x in (H.edges[e][score] for e in edges) if x == x]
    return round(sum(values) / len(values), 2)


# ---------------------------------------------------------------------------
# Les Misérables and Davis
# ---------------------------------------------------------------------------


@functools.cache
def _lm():
    return nx.les_miserables_graph()


@functools.cache
def _scored(method, **kw):
    return getattr(nb, method)(_lm(), **kw)


@functools.cache
def _backbone(method, attr, **kw):
    return nb.threshold_filter(_scored(method, **kw), attr, ALPHA)


def _kept(method, attr, **kw):
    return _backbone(method, attr, **kw).number_of_edges()


def _flagged(method, flag="mla_keep", **kw):
    return nb.boolean_filter(_scored(method, **kw), flag).number_of_edges()


def _edge_set(G):
    return {frozenset(e) for e in G.edges()}


def _statistical_counts():
    # The methods the page compares, as it filters them.
    return {
        "disparity": _kept("disparity_filter", "disparity_pvalue"),
        "noise_corrected": _kept("noise_corrected_filter", "nc_pvalue"),
        "marginal_likelihood": _kept("marginal_likelihood_filter", "ml_pvalue"),
        "ecm": _kept("ecm_filter", "ecm_pvalue"),
        "lans": _kept("lans_filter", "lans_pvalue"),
        "polya": _kept("polya_filter", "polya_pvalue"),
        "gloss": _kept("gloss_filter", "gloss_pvalue"),
        "mla": _flagged("multiple_linkage_analysis"),
    }


def _delta_rule_kept():
    H = _scored("noise_corrected_filter")
    return sum(d["nc_score"] > 1.64 * d["nc_sdev"] for *_, d in H.edges(data=True))


def _unit_counts():
    values = []
    for method, attr in [
        ("marginal_likelihood_filter", "ml_pvalue"),
        ("ecm_filter", "ecm_pvalue"),
        ("noise_corrected_filter", "nc_pvalue"),
    ]:
        values += [_kept(method, attr, unit=0.1), _kept(method, attr)]
    return values


@functools.cache
def _davis_projections():
    B = nx.davis_southern_women_graph()
    women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    return nb.hyper_projection(B, women), nb.hyper_projection(B, women, newman=True)


def _fraction(x):
    return str(Fraction(x).limit_denominator(1000))


def _or_rule_rate():
    # The simulation of tests/test_calibration.py::test_or_rule_is_anti_conservative.
    import numpy as np

    rng = np.random.default_rng(7)
    rates = []
    for seed in range(5):
        G = nx.random_regular_graph(8, 400, seed=seed)
        for u, v in G.edges():
            G[u][v]["weight"] = float(rng.exponential())
        H = nb.disparity_filter(G)
        rates.append(np.mean([p < ALPHA for *_, p in H.edges(data="disparity_pvalue")]))
    return round(100 * float(np.mean(rates)))


# ---------------------------------------------------------------------------
# Hypergraphs
# ---------------------------------------------------------------------------

_H = [(1, 2, 3, 4), (1, 2, 3), (2, 3, 4), (8, 9)]
_H2 = [
    (0, 1, 2, 3, 4, 5),
    (0, 1, 2, 3, 4),
    (0, 1, 2, 3, 5),
    (0, 1, 2, 3, 6, 7, 8),
    (2, 3, 4, 5, 9, 10, 11),
]


@functools.cache
def _local(aggregation="majority"):
    return nb.local_mdl_hypergraph_backbone(_H2, aggregation=aggregation)


@functools.cache
def _global():
    return nb.mdl_hypergraph_backbone(_H2)


def _aggregation_lines():
    values = []
    for rule in ["union", "majority", "all"]:
        result = _local(rule)
        values += [len(result.backbone), round(result.compression_ratio, 2)]
    return values


# ---------------------------------------------------------------------------
# API counts
# ---------------------------------------------------------------------------

_DIRECTED = nx.DiGraph()
_DIRECTED.add_weighted_edges_from(
    [(0, 1, 5.0), (0, 2, 3.0), (1, 0, 2.0), (1, 2, 4.0), (2, 3, 6.0), (3, 0, 2.0)]
)
_REQUIRED = {"global_threshold_filter": {"threshold": 1.0}}


def _methods(module):
    """The public functions of a module, without the short aliases."""
    module = getattr(nb, module)
    return [
        name
        for name in module.__all__
        if not (getattr(module, name).__doc__ or "").startswith("Alias for")
    ]


def _accepts_directed(module, name):
    func = getattr(getattr(nb, module), name)
    D = _DIRECTED
    args = {"compare_backbones": (D, {"D": D}), "reachability": (D,)}
    try:
        if module == "measures":
            func(*args.get(name, (D, D)))
        else:
            func(D, **_REQUIRED.get(name, {}))
    except nx.NetworkXNotImplemented:
        return False
    return True


def _directed_count(module):
    names = _methods(module)
    directed = sum(_accepts_directed(module, name) for name in names)
    return directed, len(names)


def _all_directed(module):
    directed, total = _directed_count(module)
    return (total,) if directed == total else (f"{directed} of {total}",)


# ---------------------------------------------------------------------------
# Doctested pages
# ---------------------------------------------------------------------------

_VB = "tutorials/validating_backbones"
_TB = "tutorials/temporal_backbones"


def _vb():
    return _namespace(_VB)


def _vb_votes():
    from collections import Counter

    counts = Counter(_vb()["votes"].values())
    return _words(counts[4], capitalize=True), _words(counts[1])


def _vb_strengths():
    s = dict(_lm().degree(weight="weight"))
    return s["Valjean"], s["Bossuet"]


def _vb_expected_weight():
    # The marginal likelihood null: s_u s_v / (2 T), T the total weight.
    s = dict(_lm().degree(weight="weight"))
    total = _lm().size(weight="weight")
    return (_words(round(s["Valjean"] * s["Bossuet"] / (2 * total))),)


def _tb():
    return _namespace(_TB)


def _tb_kalman_kept(t):
    """(rising, other) edges kept by the smoothed backbone in step *t*."""
    ns = _tb()
    return ns["found"](
        nb.threshold_filter(ns["steps"][t], "kalman_pvalue", ALPHA).edges
    )


def _tb_rising():
    n_rising = len(_tb()["rising"])
    late = all(_tb_kalman_kept(t)[0] == n_rising for t in range(15, 20))
    return _words(_tb_kalman_kept(11)[0]), _words(n_rising) if late else "not all"


def _tb_others():
    ns = _tb()
    others = len(ns["pairs"]) - len(ns["rising"])
    return others, _words(round(ALPHA * others))


def _tb_beta_one_percent():
    ns = _tb()
    MB = nb.edge_persistence_hmm(
        ns["snaps"], weight="weight", min_weight=1, params={"beta": 0.01}
    )
    return _mean(MB, ns["flicker"], "hmm_n_active") > _mean(
        MB, ns["block"], "hmm_n_active"
    )


# ---------------------------------------------------------------------------
# The claims
# ---------------------------------------------------------------------------

_SB = "tutorials/statistical_backbone"
_HB = "tutorials/hypergraph_backbone"
_WT = "user_guide/weight_transforms"
_DG = "user_guide/directed_graphs"

NUMBERS = [
    # Statistical backbones on Les Misérables.
    (
        _SB,
        "the default keeps {} of {} edges at 0.05",
        lambda: (_kept("noise_corrected_filter", "nc_pvalue"), _lm().size()),
    ),
    (_SB, "the ``delta`` rule keeps {}.", lambda: (_delta_rule_kept(),)),
    (
        _SB,
        r"the fit gives :math:`a \approx {}` and the backbone keeps {} edges",
        lambda: (
            f"{_scored('polya_filter').graph['polya_a']:.2f}",
            _kept("polya_filter", "polya_pvalue"),
        ),
    ),
    (
        _SB,
        "on Les Misérables it keeps {} edges.",
        lambda: (_kept("gloss_filter", "gloss_pvalue"),),
    ),
    (
        "tutorials/structural_methods",
        "On Les Misérables this keeps {} edges, against {} without smoothing.",
        lambda: (
            _flagged("doubly_stochastic_filter", "ds_keep", smoothing=1e-4),
            _flagged("doubly_stochastic_filter", "ds_keep"),
        ),
    ),
    (
        "tutorials/les_miserables_benchmark",
        "ORIGINAL_EDGES = G.number_of_edges() # {}",
        lambda: (_lm().size(),),
    ),
    (
        "tutorials/bipartite_backbone",
        "(Olivia and Sylvia: {} versus {})",
        lambda: tuple(
            _fraction(P["Olivia Carleton"]["Sylvia Avondale"]["weight"])
            for P in _davis_projections()
        ),
    ),
    (
        _WT,
        "keeps {} instead of {} edges for MLF at ``alpha = 0.05``, {} instead of "
        "{} for the ECM, and {} instead of {} for the noise-corrected filter.",
        _unit_counts,
    ),
    (
        _WT,
        'H.graph["disparity_a"] # {} nb.threshold_filter(H, "disparity_pvalue", '
        "0.05).number_of_edges() # {}",
        lambda: (
            f"{_scored('disparity_filter', a='ml').graph['disparity_a']:.3f}",
            _kept("disparity_filter", "disparity_pvalue", a="ml"),
        ),
    ),
    (
        _WT,
        "the fitted filter keeps {} edges where the original keeps {}.",
        lambda: (
            _kept("disparity_filter", "disparity_pvalue", a="ml"),
            _kept("disparity_filter", "disparity_pvalue"),
        ),
    ),
    (
        _WT,
        "about {}% of the edges fall below 0.05",
        lambda: (_or_rule_rate(),),
    ),
    # Hypergraphs (code comments that give the output).
    (
        _HB,
        "print(result.backbone) # {*} print(",
        lambda: (nb.mdl_hypergraph_backbone(_H).backbone,),
    ),
    (
        _HB,
        "nb.mdl_hypergraph_backbone(H2).backbone # {*} local =",
        lambda: (_global().backbone,),
    ),
    (_HB, "local.backbone # [{*}, {*}, # {*}] ..", lambda: tuple(_local().backbone)),
    (
        _HB,
        "local.local_backbones[6] # {*} local.",
        lambda: (_local().local_backbones[6],),
    ),
    (_HB, "# union {} {} # majority {} {} # all {} {} The", _aggregation_lines),
    (
        _HB,
        "compress less ({} here against {} for the global backbone)",
        lambda: (
            round(_local().compression_ratio, 2),
            round(_global().compression_ratio, 2),
        ),
    ),
    (_HB, "nb.maximal_hyperedges(H) # {*} -", lambda: (nb.maximal_hyperedges(_H),)),
    (
        _HB,
        "print(result.validated) # {*} print(",
        lambda: (
            nb.statistically_validated_hypergraph(
                [(1, 2)] * 5 + [(3, 4)] * 100, alpha=ALPHA
            ).validated,
        ),
    ),
    # Validating backbones (doctested).
    (
        _VB,
        "({} characters, {} weighted edges)",
        lambda: (_lm().number_of_nodes(), _lm().size()),
    ),
    (
        _VB,
        r"roughly :math:`0.05 \times {} \approx {}` false positives",
        lambda: (_lm().size(), round(ALPHA * _lm().size())),
    ),
    (
        _VB,
        "The uncorrected backbone keeps {} edges, fewer than the {} that chance",
        lambda: (
            _kept("disparity_filter", "disparity_pvalue"),
            round(ALPHA * _lm().size()),
        ),
    ),
    (_VB, "does not survive {} tests", lambda: (_lm().size(),)),
    (
        _VB,
        "so the {}-edge backbone should not",
        lambda: (_kept("disparity_filter", "disparity_pvalue"),),
    ),
    (_VB, "{} edges are kept by all four methods; {} are kept by only one", _vb_votes),
    (
        _VB,
        "Valjean has the largest strength in the network ({}) and Bossuet one of "
        "the larger ones ({})",
        _vb_strengths,
    ),
    (_VB, "expects them to co-appear about {} times", _vb_expected_weight),
    (
        _VB,
        r"the same :math:`\alpha` ({} rather than {} here)",
        lambda: (
            sum(d["sign"] > 0 for *_, d in _vb()["signed_bb"].edges(data=True)),
            _kept("marginal_likelihood_filter", "ml_pvalue"),
        ),
    ),
    # Temporal backbones (doctested).
    (
        _TB,
        "keeps between {} and {} edges per step",
        lambda: (min(_tb()["sizes"]), max(_tb()["sizes"])),
    ),
    (
        _TB,
        "seen in {}% of its steps",
        lambda: (round(100 * _tb()["model"]["alpha"]),),
    ),
    (
        _TB,
        "lasting about {} steps (``1 / a10``)",
        lambda: (_words(round(1 / _tb()["model"]["a10"])),),
    ),
    (
        _TB,
        "than ``persistence`` ({} against {})",
        lambda: tuple(
            _mean(_tb()["P"], _tb()[group], "persistence")
            for group in ("block", "flicker")
        ),
    ),
    (_TB, "It finds {} rising edges by step 11 and all {} from step 15", _tb_rising),
    (
        _TB,
        "no more of the other {} edges than a 5% test should (about {} per step)",
        _tb_others,
    ),
    (_TB, "a correction across the {} edges", lambda: (len(_tb()["pairs"]),)),
    # API counts.
    (
        "user_guide/choosing_method",
        "Proximity methods ({})",
        lambda: (len(_methods("proximity")),),
    ),
    (_DG, "**Statistical methods** (all {}):", lambda: _all_directed("statistical")),
    (_DG, "**Structural ({} of {})**", lambda: _directed_count("structural")),
    (_DG, "**Proximity methods** (all {}):", lambda: _all_directed("proximity")),
    (_DG, "**Measures** (all {}):", lambda: _all_directed("measures")),
]

FACTS = [
    (
        _SB,
        "the same edges as the marginal likelihood filter below",
        lambda: (
            _edge_set(_backbone("noise_corrected_filter", "nc_pvalue"))
            == _edge_set(_backbone("marginal_likelihood_filter", "ml_pvalue"))
        ),
    ),
    (
        _SB,
        "It is the most conservative method here",
        lambda: min(_statistical_counts(), key=_statistical_counts().get) == "gloss",
    ),
    (
        _SB,
        "on this graph Weaver's keeps the most links",
        lambda: (
            _flagged("multiple_linkage_analysis", deviation="weaver")
            > max(
                _flagged("multiple_linkage_analysis", deviation=d)
                for d in ("cod", "profile")
            )
        ),
    ),
    (
        "tutorials/bipartite_backbone",
        "The two differ on every Davis pair",
        lambda: all(
            d["weight"] != _davis_projections()[1].edges[e]["weight"]
            for e, d in _davis_projections()[0].edges.items()
        ),
    ),
    (
        _HB,
        "Local backbones are usually denser than the global one and compress less",
        lambda: (
            len(_local().backbone) > len(_global().backbone)
            and _local().compression_ratio > _global().compression_ratio
        ),
    ),
    (
        _VB,
        "keeps almost two-thirds of the edges",
        lambda: 0.6 < _vb()["results"]["metric"]["edge_fraction"] < 2 / 3,
    ),
    (
        _VB,
        "Valjean has the largest strength in the network",
        lambda: max(_lm().degree(weight="weight"), key=lambda x: x[1])[0] == "Valjean",
    ),
    (
        _TB,
        "Both heavy groups are observed in half of the steps",
        lambda: (
            {
                _mean(_tb()["H"], _tb()[group], "obs_fraction")
                for group in ("block", "flicker")
            }
            == {0.5}
        ),
    ),
    (
        _TB,
        "missed in about a third of their steps",
        lambda: round(3 * _tb()["MH"].graph["temporal"]["hmm"]["beta"]) == 1,
    ),
    (
        _TB,
        "even a 1% miss rate leaves the flicker with more active steps than the block",
        _tb_beta_one_percent,
    ),
    (
        _TB,
        "edges than a 5% test should",
        lambda: (
            max(_tb_kalman_kept(t)[1] for t in (11, 15, 19))
            <= round(ALPHA * _tb_others()[0])
        ),
    ),
]


def _id(claim):
    page, text, _ = claim
    words = re.sub(r"[^\w ]", "", text).split()
    return f"{page.split('/')[-1]}:{'_'.join(words[:5])}"


@pytest.mark.parametrize("claim", NUMBERS, ids=_id)
def test_page_states_the_computed_number(claim):
    page, template, compute = claim
    text = _text(page)
    found = [m.group(0) for m in re.finditer(_pattern(template), text)]
    assert found, (
        f"docs/{page}.rst: passage not found: {template!r}.  If the text was "
        "reworded, update its entry in tests/test_docs_numbers.py."
    )
    expected = _fill(template, compute())
    assert expected in found, (
        f"docs/{page}.rst no longer matches the library:\n"
        f"  the page says:          {found[0]!r}\n"
        f"  the library computes:   {expected!r}\n"
        "Update the page with the new value."
    )


@pytest.mark.parametrize("claim", FACTS, ids=_id)
def test_page_states_a_true_comparison(claim):
    page, passage, holds = claim
    assert passage in _text(page), (
        f"docs/{page}.rst: passage not found: {passage!r}.  If the text was "
        "reworded, update its entry in tests/test_docs_numbers.py."
    )
    assert holds(), (
        f"docs/{page}.rst says {passage!r}, which the library no longer bears "
        "out.  Update the page."
    )


def test_words():
    assert [_words(n) for n in (0, 6, 19, 20, 34)] == [
        "zero",
        "six",
        "nineteen",
        "twenty",
        "thirty-four",
    ]
    assert _words(34, capitalize=True) == "Thirty-four"
