"""Tests for describe_backbone and its method/citation registry."""

import doctest
import inspect
import re
import unicodedata
from pathlib import Path

import networkx as nx
import pytest

import networkx_backbone as nb
from networkx_backbone import describe
from networkx_backbone.describe import (
    ALIASES,
    EXCLUDED,
    HELPER_MODULES,
    METHODS,
    REFERENCES,
)
from tests.test_minimal_deps import _run

ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    return getattr(nb, name).__module__.rsplit(".", 1)[-1]


def _ascii(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


# ---------------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------------


def test_every_public_scorer_has_a_registry_entry():
    # A new public method must be described (or explicitly excluded).
    missing = [
        name
        for name in nb.__all__
        if _module(name) not in HELPER_MODULES
        and name not in METHODS
        and name not in ALIASES
        and name not in EXCLUDED
    ]
    assert not missing, (
        f"add these to networkx_backbone.describe.METHODS (or EXCLUDED): {missing}"
    )


def test_registry_names_are_public():
    public = set(nb.__all__)
    assert set(METHODS) <= public
    assert set(EXCLUDED) <= public
    assert not set(METHODS) & set(EXCLUDED)
    for alias, target in ALIASES.items():
        assert target in METHODS, alias


def test_helper_modules_hold_no_described_methods():
    for name in METHODS:
        assert _module(name) not in HELPER_MODULES, name


def test_references_exist_and_are_used():
    used = {k for entry in METHODS.values() for k in entry.refs}
    used |= {"neal2023", "holm1979", "hochberg1988", "benjamini1995"}
    used |= {"benjamini2001"}
    assert used <= set(REFERENCES)
    assert set(REFERENCES) <= used, set(REFERENCES) - used
    for key, ref in REFERENCES.items():
        assert ref.key == key
        assert ref.kind in {"article", "inproceedings", "book", "misc"}
        assert ref.authors and ref.year and ref.title


def test_citations_match_docstring_references():
    # The registry's references come from each function's References section.
    for name, entry in METHODS.items():
        doc = _ascii(inspect.getdoc(getattr(nb, name)) or "")
        section = doc.split("References\n----------", 1)
        if len(section) == 1:
            assert not entry.refs, f"{name}: cites papers its docstring does not"
            continue
        refs_text = section[1]
        assert entry.refs, f"{name}: its docstring has References; add them"
        for key in entry.refs:
            ref = REFERENCES[key]
            surname = _ascii(ref.authors[0].split(",")[0])
            assert surname in refs_text, f"{name}: {key} not in its References"
            assert str(ref.year) in refs_text, f"{name}: {key} year mismatch"


def test_dois_are_documented_in_the_repository():
    # DOIs are copied from the repository, never invented.
    readme = ROOT / "README.md"
    if not readme.exists():
        pytest.skip("README.md not available")
    text = readme.read_text(encoding="utf-8")
    for key, ref in REFERENCES.items():
        if ref.doi:
            assert ref.doi in text, f"{key}: DOI {ref.doi} not found in README"


def test_module_docstring_lists_registry():
    for name in METHODS:
        assert f"networkx_backbone.{name}`" in describe.__doc__


@pytest.mark.parametrize("name", sorted(METHODS))
@pytest.mark.parametrize("style", ["text", "markdown", "bibtex"])
def test_every_method_renders(name, style):
    entry = METHODS[name]
    kwargs = {"alpha": 0.05} if entry.test else {}
    out = nb.describe_backbone(name, style=style, **kwargs)
    if style == "bibtex":
        assert out.count("@") == len(entry.refs) + 1 + bool(entry.builtin_mtc)
        assert "keegan_networkx_backbone" in out
    else:
        assert entry.label in out
        assert "networkx-backbone version" in out
        for key in entry.refs:
            assert REFERENCES[key].cite() in out


# ---------------------------------------------------------------------------
# describe_backbone
# ---------------------------------------------------------------------------


@pytest.fixture
def les_mis():
    G = nx.les_miserables_graph()
    bb = nb.threshold_filter(nb.disparity_filter(G), "disparity_pvalue", 0.05)
    return G, bb


def test_text_reports_share_kept(les_mis):
    G, bb = les_mis
    text = nb.describe_backbone("disparity_filter", alpha=0.05, original=G, backbone=bb)
    assert "Serrano et al., 2009" in text
    assert "alpha = 0.05, without correction" in text
    assert "9 of 254 edges (3.5%)" in text
    assert "10 of 77 non-isolated nodes (13.0%)" in text
    assert "of the total edge weight" in text
    assert "https://doi.org/10.1073/pnas.0808904106" in text
    version = describe._package_version()
    assert f"version {version}" in text


def test_aliases_and_callables_resolve(les_mis):
    expected = nb.describe_backbone("disparity_filter", alpha=0.05)
    for method in ["disparity", "Disparity_Filter", nb.disparity_filter, nb.disparity]:
        assert nb.describe_backbone(method, alpha=0.05) == expected
    assert "the simple bipartite projection" in nb.describe_backbone("simple")
    assert "the SVH" not in nb.describe_backbone("svh", alpha=0.01)
    assert "(SVH)" in nb.describe_backbone("svh", alpha=0.01)


@pytest.mark.parametrize(
    ("mtc", "phrase", "cite"),
    [
        ("bonferroni", "a Bonferroni correction", None),
        ("holm", "a Holm step-down correction", "Holm, 1979"),
        ("hochberg", "a Hochberg step-up correction", "Hochberg, 1988"),
        ("bh", "Benjamini-Hochberg", "Benjamini & Hochberg, 1995"),
        ("fdr", "Benjamini-Hochberg", "Benjamini & Hochberg, 1995"),
        ("BY", "Benjamini-Yekutieli", "Benjamini & Yekutieli, 2001"),
    ],
)
def test_mtc_wording(mtc, phrase, cite):
    text = nb.describe_backbone("lans_filter", alpha=0.05, mtc=mtc)
    assert phrase in text
    assert "for multiple comparisons" in text
    bib = nb.describe_backbone("lans_filter", alpha=0.05, mtc=mtc, style="bibtex")
    if cite:
        assert cite in text
        assert bib.count("@") == 3
    else:
        assert bib.count("@") == 2


def test_alpha_none_mentions_pvalues():
    text = nb.describe_backbone("disparity_filter", mtc="holm")
    assert "P-values were computed" in text
    assert "Holm" in text


def test_markdown_style(les_mis):
    G, bb = les_mis
    md = nb.describe_backbone(
        "disparity_filter", alpha=0.05, original=G, backbone=bb, style="markdown"
    )
    assert "(`disparity_filter`; Serrano et al., 2009)" in md
    assert "`alpha = 0.05`" in md
    assert "**References**" in md
    assert "- Serrano, M. A." in md
    assert "<https://doi.org/10.1073/pnas.0808904106>" in md


def test_parameters_are_listed():
    text = nb.describe_backbone("lspar", s=0.5)
    assert "Other parameters: s=0.5." in text
    md = nb.describe_backbone("strongest_n_ties", n=2, style="markdown")
    assert "`n=2`" in md


def test_disparity_concentration_is_described():
    G = nx.les_miserables_graph()
    scored = nb.disparity_filter(G, a="ml")
    bb = nb.threshold_filter(scored, "disparity_pvalue", 0.05)
    text = nb.describe_backbone(
        "disparity", alpha=0.05, a="ml", original=G, backbone=bb
    )
    assert (
        "symmetric Dirichlet distribution of each node's weight shares with "
        "concentration 1/a, estimated by maximum likelihood (a = 0.525)"
    ) in text
    assert "Marcaccioli & Livan, 2019" in text
    assert "Marcaccioli, R., & Livan, G. (2019)" in text  # reference list
    assert "Other parameters" not in text
    bib = nb.describe_backbone("disparity_filter", a=2, style="bibtex")
    assert "@article{marcaccioli2019" in bib
    md = nb.describe_backbone("disparity_filter", a=2, style="markdown")
    assert "concentration 1/a, `a = 2`" in md
    # Without a backbone the estimate is unknown; a = 1 is the original filter.
    assert "estimated by maximum likelihood;" in nb.describe_backbone(
        "disparity_filter", a="ml"
    )
    assert nb.describe_backbone("disparity_filter", a=1) == nb.describe_backbone(
        "disparity_filter"
    )


@pytest.mark.parametrize(
    "name", ["marginal_likelihood_filter", "ecm_filter", "noise_corrected_filter"]
)
def test_count_unit_is_described(name):
    text = nb.describe_backbone(name, alpha=0.05, unit=0.25)
    assert (
        "Edge weights were converted to counts in units of unit = 0.25 "
        "(count = weight / unit) before testing."
    ) in text
    assert "Other parameters" not in text
    assert nb.describe_backbone(name, unit=1) == nb.describe_backbone(name)
    # Other methods list unit like any other parameter.
    assert "Other parameters: unit=2." in nb.describe_backbone("lans_filter", unit=2)


def test_bibtex_entries():
    bib = nb.describe_backbone("fdsm", alpha=0.05, mtc="bh", style="bibtex")
    keys = re.findall(r"@\w+\{(\w+),", bib)
    assert keys == [
        "neal2021",
        "godard2022",
        "benjamini1995",
        "keegan_networkx_backbone",
    ]
    assert "doi = {10.1038/s41598-021-03238-3}" in bib
    misc = nb.describe_backbone("ecm_filter", style="bibtex")
    assert "eprint = {1706.00230}" in misc
    assert "archiveprefix = {arXiv}" in misc


def test_sdsm_edge_constraints_cite_sdsm_ec():
    plain = nb.describe_backbone("sdsm", alpha=0.05)
    assert "Neal & Neal, 2023" not in plain
    ec = nb.describe_backbone("sdsm", alpha=0.05, prohibited=[("a", "b")])
    assert "Neal & Neal, 2023" in ec


def test_signed_backbone_counts_signs():
    pytest.importorskip("scipy")
    G = nx.les_miserables_graph()
    scored = nb.marginal_likelihood_filter(G, signed=True)
    bb = nb.threshold_filter(scored, "ml_pvalue", 0.05)
    n_neg = sum(1 for *_, d in bb.edges(data=True) if d["sign"] < 0)
    n_pos = bb.number_of_edges() - n_neg
    text = nb.describe_backbone("mlf", alpha=0.05, signed=True, original=G, backbone=bb)
    assert "two-tailed" in text
    assert f"{n_pos} are significantly strong (sign +1)" in text
    assert f"and {n_neg} significantly weak (sign -1)" in text


def test_bipartite_backbone():
    pytest.importorskip("scipy")
    B = nx.davis_southern_women_graph()
    women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    bb = nb.fixedrow(B, women, alpha=0.05)
    text = nb.describe_backbone("fixedrow", alpha=0.05, original=B, backbone=bb)
    assert "Agent pairs were kept" in text
    assert f"{bb.number_of_edges()} edges among {bb.number_of_nodes()} agents" in text
    proj = nb.describe_backbone("hyper_projection")
    assert proj.startswith("We projected the bipartite network with")


def test_hypergraph_backbone():
    H = [(1, 2, 3, 4), (1, 2, 3), (2, 3, 4), (8, 9), (8, 9)]
    kept = nb.maximal_hyperedges(H)
    text = nb.describe_backbone("maximal_hyperedges", original=H, backbone=kept)
    assert "retains 2 of 4 distinct hyperedges (50.0%)" in text
    result = nb.mdl_hypergraph_backbone(H)
    text = nb.describe_backbone("mdl_hypergraph_backbone", original=H, backbone=result)
    assert f"retains {len(result.backbone)} of 4 distinct hyperedges" in text
    assert "Kirkley et al., 2026" in text
    only = nb.describe_backbone("order_filter", backbone=kept, min_order=3)
    assert "The backbone contains 2 hyperedges." in only


def test_svh_has_builtin_correction():
    text = nb.describe_backbone("svh", alpha=0.01)
    assert "Hyperedges were kept" in text
    assert "applied within the method (Benjamini & Hochberg, 1995)" in text
    with pytest.raises(ValueError, match="applies its own"):
        nb.describe_backbone("svh", alpha=0.01, mtc="holm")


def test_backbone_without_original(les_mis):
    _, bb = les_mis
    text = nb.describe_backbone("disparity_filter", alpha=0.05, backbone=bb)
    assert "The backbone contains 9 edges among 77 nodes." in text


def test_unweighted_original_skips_weight_share():
    G = nx.karate_club_graph()
    for _, _, d in G.edges(data=True):
        d.clear()
    bb = nb.boolean_filter(nb.lspar(G), "sparsify_keep")
    text = nb.describe_backbone("lspar", original=G, backbone=bb)
    assert "of 78 edges" in text
    assert "edge weight" not in text


def test_errors(les_mis):
    G, bb = les_mis
    with pytest.raises(ValueError, match="Unknown method"):
        nb.describe_backbone("no_such_method")
    with pytest.raises(ValueError, match="dispatch wrapper"):
        nb.describe_backbone("backbone_from_weighted")
    with pytest.raises(ValueError, match="result class"):
        nb.describe_backbone("HypergraphBackbone")
    with pytest.raises(TypeError, match="string or a function"):
        nb.describe_backbone(3)
    with pytest.raises(ValueError, match="style"):
        nb.describe_backbone("disparity", style="latex")
    with pytest.raises(ValueError, match="mtc"):
        nb.describe_backbone("disparity", alpha=0.05, mtc="sidak")
    with pytest.raises(ValueError, match="does not produce p-values"):
        nb.describe_backbone("jaccard_backbone", alpha=0.05)
    with pytest.raises(ValueError, match="does not produce p-values"):
        nb.describe_backbone("metric_backbone", mtc="bh")
    with pytest.raises(ValueError, match="no signed"):
        nb.describe_backbone("glab_filter", alpha=0.05, signed=True)
    with pytest.raises(TypeError, match="backbone must be"):
        nb.describe_backbone("disparity", alpha=0.05, original=G, backbone=[1])
    with pytest.raises(TypeError, match="original must be"):
        nb.describe_backbone("disparity", alpha=0.05, original=[1], backbone=bb)


def test_package_version_fallback(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    def missing(_):
        raise PackageNotFoundError

    monkeypatch.setattr(describe, "_pkg_version", missing)
    assert "version unknown" in nb.describe_backbone("lspar")


def test_describe_without_optional_dependencies():
    _run(
        """
        import networkx as nx
        import networkx_backbone as nb
        G = nx.les_miserables_graph()
        bb = nb.threshold_filter(nb.disparity_filter(G), "disparity_pvalue", 0.05)
        text = nb.describe_backbone("disparity", alpha=0.05, mtc="holm",
                                    original=G, backbone=bb)
        assert "Serrano et al., 2009" in text
        assert "@article{serrano2009," in nb.describe_backbone(
            "disparity", style="bibtex")
        """
    )


# ---------------------------------------------------------------------------
# Tutorial
# ---------------------------------------------------------------------------


def test_validating_backbones_tutorial_runs():
    # Every code block of the tutorial, with its printed output, must hold.
    path = ROOT / "docs" / "tutorials" / "validating_backbones.rst"
    if not path.exists():
        pytest.skip("docs not available")
    pytest.importorskip("scipy")
    result = doctest.testfile(
        str(path),
        module_relative=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        encoding="utf-8",
    )
    assert result.attempted > 0
    assert result.failed == 0
