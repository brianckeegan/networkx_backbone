"""Figure coverage of the docs: every how-to and user-guide page has a figure.

Plain text parsing, no Sphinx: a page counts as illustrated when it has a
``page-figure``, ``figure`` or ``image`` directive, or includes a method-gallery
file (``.. include:: ../_gallery/<module>.inc``).  Pages without one yet are
listed in ``PENDING``; give a page a figure, then delete its entry (the test
fails on an entry whose page has a figure or no longer exists).  "Writing docs
figures" in docs/contributing.rst explains how.

The executed tutorials in docs/examples/ that use the tutorial figure helper
(``docs/_figures/tutorial.py``) must also set one gallery thumbnail with it.
"""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
SECTIONS = ("tutorials", "user_guide")

# Pages that have no figure yet, as "<section>/<file stem>".  #121 and #122
# empty this set; never add to it.
PENDING = {
    # How-to tutorials (#121).
    "tutorials/comparing_methods",
    "tutorials/les_miserables_benchmark",
    "tutorials/proximity_scoring",
    "tutorials/statistical_backbone",
    "tutorials/structural_methods",
    "tutorials/unweighted_sparsification",
    # User guide (#122).
    "user_guide/backends",
    "user_guide/choosing_method",
    "user_guide/complexity",
    "user_guide/directed_graphs",
    "user_guide/evaluation",
    "user_guide/temporal",
    "user_guide/weight_transforms",
}

_FIGURE = re.compile(
    r"^\s*\.\.\s+(?:page-figure|figure|image)::\s*\S"
    r"|^\s*\.\.\s+include::\s*\S*_gallery/\S+\.inc\s*$",
    re.MULTILINE,
)
_PAGE_FIGURE = re.compile(r"^\s*\.\.\s+page-figure::\s*(\S+)\s*$", re.MULTILINE)


def _pages():
    return {
        f"{section}/{path.stem}": path
        for section in SECTIONS
        for path in sorted((DOCS / section).glob("*.rst"))
        if path.name != "index.rst"
    }


def _has_figure(path):
    return _FIGURE.search(path.read_text(encoding="utf-8")) is not None


PAGES = _pages()


def test_pages_are_found():
    assert len(PAGES) >= 19
    assert all(page.startswith(SECTIONS) for page in PAGES)


@pytest.mark.parametrize("page", sorted(PAGES))
def test_page_has_a_figure(page):
    if page in PENDING:
        assert not _has_figure(PAGES[page]), (
            f"{page} has a figure now: remove it from PENDING in {__file__}"
        )
    else:
        assert _has_figure(PAGES[page]), (
            f"docs/{page}.rst has no figure: add one with the page-figure "
            "directive (see 'Writing docs figures' in docs/contributing.rst)"
        )


def test_pending_pages_exist():
    missing = sorted(PENDING - set(PAGES))
    assert not missing, f"remove these PENDING entries, the pages are gone: {missing}"


def test_page_figures_name_a_registry_module():
    # The registry has one module per page, docs/_figures/pages/<page>.py;
    # ``gallery:`` references read the method gallery's manifest instead.
    for page, path in PAGES.items():
        for ref in _PAGE_FIGURE.findall(path.read_text(encoding="utf-8")):
            if ref.startswith("gallery:"):
                assert re.fullmatch(r"gallery:[a-z_]+/[\w-]+", ref), (page, ref)
                continue
            module, sep, key = ref.partition("/")
            assert sep and key, f"{page}: write page-figure:: <module>/<key>, not {ref}"
            assert (DOCS / "_figures" / "pages" / f"{module}.py").is_file(), (
                f"{page}: no docs/_figures/pages/{module}.py for {ref}"
            )


_THUMBNAIL = re.compile(
    r'^# sphinx_gallery_thumbnail_path = "_static/example_figures/'
    r'(?P<name>[\w-]+)/thumbnail\.png"$',
    re.MULTILINE,
)
_HELPER = re.compile(r"TutorialFigures\(\s*\"(?P<name>[\w-]+)\"\s*\)")
TUTORIALS = sorted((DOCS / "examples").glob("*/plot_*.py"))


@pytest.mark.parametrize("path", TUTORIALS, ids=lambda p: p.stem)
def test_tutorial_figures_set_a_thumbnail(path):
    source = path.read_text(encoding="utf-8")
    helpers = _HELPER.findall(source)
    if not helpers:
        return  # a classic Sphinx-Gallery example (plt.show): its own thumbnail
    assert len(helpers) == 1, f"{path.name}: one TutorialFigures per file"
    thumbnail = _THUMBNAIL.search(source)
    assert thumbnail and thumbnail["name"] == helpers[0], (
        f"{path.name}: add '# sphinx_gallery_thumbnail_path = "
        f'"_static/example_figures/{helpers[0]}/thumbnail.png"\''
    )
    assert len(re.findall(r"\bthumbnail=True\b", source)) == 1, (
        f"{path.name}: pass thumbnail=True to exactly one figures.show(...)"
    )


def test_tutorial_figure_folders_are_unique():
    names = [
        m["name"] for path in TUTORIALS for m in _HELPER.finditer(path.read_text())
    ]
    assert len(names) == len(set(names)), names
