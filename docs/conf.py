# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

# As in NetworkX's own docs build: render dispatchable functions as the plain
# functions they wrap, without the ``backend=None, **backend_kwargs`` keywords
# the dispatcher adds (docs/user_guide/backends.rst documents those).  Must be
# set before NetworkX is imported.
os.environ["_NETWORKX_BUILDING_DOCS_"] = "True"

# Ensure local package imports resolve during docs builds.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ...and the shared figure style (docs/_figures/style.py), so build hooks,
# Sphinx-Gallery examples and executed reST can ``from _figures import style``.
DOCS = Path(__file__).resolve().parent
if str(DOCS) not in sys.path:
    sys.path.insert(0, str(DOCS))

# -- Project information -----------------------------------------------------

project = "networkx-backbone"
copyright = "2025-2026, Brian C. Keegan"
author = "Brian C. Keegan"
# Single source of truth: the version setuptools-scm derives from git tags.
try:
    release = _pkg_version("networkx-backbone")
except PackageNotFoundError:  # building docs from an uninstalled checkout
    release = "0.0.0+unknown"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_gallery", "Thumbs.db", ".DS_Store"]

# -- numpydoc settings -------------------------------------------------------
# numpydoc alone parses the NumPy-style docstrings.  (sphinx.ext.napoleon used
# to run first and rewrite every section into field lists, which hid the
# sections from numpydoc's validation below.)

numpydoc_show_class_members = False
numpydoc_xref_param_type = False

# Validate every docstring autodoc renders; a failure is a Sphinx warning, so
# the ``-W`` docs build fails on it.  The checks and exclusions live in the
# ``[tool.numpydoc_validation]`` table of pyproject.toml (with the reasons),
# shared with the faster ``python -m numpydoc lint networkx_backbone/*.py``.
with open(ROOT / "pyproject.toml", "rb") as _f:
    _numpydoc_cfg = tomllib.load(_f)["tool"]["numpydoc_validation"]
numpydoc_validation_checks = set(_numpydoc_cfg["checks"])
numpydoc_validation_exclude = set(_numpydoc_cfg["exclude"])

# -- Nitpick settings --------------------------------------------------------
# Suppress warnings for type references that numpydoc extracts from docstrings
# (e.g. "graph", "string", "optional", "default=...") which are not real
# Python classes.

nitpick_ignore_regex = [
    # Common type names from NumPy-style docstrings that are not real classes
    (r"py:class", r"graph"),
    (r"py:class", r"string"),
    (r"py:class", r"optional"),
    (r"py:class", r"default.*"),
    (r"py:class", r"int"),
    (r"py:class", r"float"),
    (r"py:class", r"bool"),
    (r"py:class", r"dict"),
    (r"py:class", r"list"),
    (r"py:class", r"callable"),
    (r"py:class", r"iterable"),
    (r"py:class", r"integer"),
    (r"py:class", r"array-like"),
    (r"py:class", r"np\.ndarray"),
    (r"py:class", r"random_state"),
    (r"py:class", r"matplotlib Axes"),
    (r"py:class", r"color"),
    (r"py:class", r"fig"),
    (r"py:class", r"ax"),
    (r"py:class", r"diff"),
    (r"py:class", r"networkx\.Graph"),
    (r"py:class", r"networkx\.DiGraph"),
    (r"py:class", r"or"),
    # Quoted string values and set-like type annotations from docstrings
    (r"py:class", r'".*"'),
    (r"py:class", r'\{.*'),
    (r"py:class", r'.*\}'),
    # NetworkX exceptions named without the module in numpydoc Raises sections
    (r"py:exc", r"NetworkXError"),
    (r"py:exc", r"NetworkXNotImplemented"),
    (r"py:exc", r"AmbiguousSolution"),
    # numpydoc renders result-class "Attributes" as a parameter list, which
    # creates no per-attribute link target.
    (r"py:attr", r"(HypergraphBackbone|ValidatedHypergraph)\..*"),
]

suppress_warnings = ["ref.citation", "ref.class"]

# -- Autodoc settings --------------------------------------------------------

autodoc_member_order = "bysource"
# The public API is annotated (and ships ``py.typed``), but the rendered pages
# keep plain signatures: the numpydoc "Parameters" and "Returns" sections
# already give every type in prose, the annotations name private aliases from
# ``networkx_backbone._typing``, and overloads would repeat each signature.
autodoc_typehints = "none"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Sphinx Gallery ----------------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": ["examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"plot_",
    "backreferences_dir": "auto_examples/backreferences",
    "doc_module": ("networkx_backbone",),
    "reference_url": {"networkx_backbone": None},
    "within_subsection_order": "FileNameSortKey",
    "remove_config_comments": True,
    # Reset matplotlib before each example, then apply the shared docs style,
    # so every example's figures match the rest of the docs.
    "reset_modules": ("matplotlib", "_figures.style.reset_gallery"),
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/brianckeegan/networkx_backbone",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
html_context = {
    "github_user": "brianckeegan",
    "github_repo": "networkx_backbone",
    "github_version": "main",
    "doc_path": "docs",
}
html_static_path = ["_static"]


# -- Les Misérables benchmark table ------------------------------------------
# ``.. les-miserables-table::`` runs every case defined in
# benchmarks/benchmarks/les_miserables.py (the definition whose counts
# tests/test_les_miserables.py pins) and renders the edge counts at build
# time, so the tutorial never hardcodes them.


def _les_miserables_table_rst():
    import warnings

    from benchmarks.benchmarks import les_miserables as lm

    original = lm.les_miserables_weighted().number_of_edges()
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 50 30 20",
        "",
        "   * - Method",
        "     - Filter applied",
        f"     - Edges kept (of {original})",
    ]
    for case in lm.CASES:
        with warnings.catch_warnings():
            # Convergence warnings (e.g. Sinkhorn-Knopp) are the tests' concern.
            warnings.simplefilter("ignore")
            kept = case.run(lm.graph_for(case)).number_of_edges()
        method = f":func:`~networkx_backbone.{case.method}`"
        if case.kwargs:
            args = ", ".join(f"{k}={v!r}" for k, v in case.kwargs.items())
            method += f" (``{args}``)"
        if not case.weighted:
            method += ", unweighted"
        lines += [
            f"   * - {method}",
            f"     - {case.describe_filter()}",
            f"     - {kept}",
        ]
    return lines


# -- Roadmap figures -----------------------------------------------------------
# scripts/roadmap_figures.py parses docs/roadmap.rst and renders its status
# figures (light and dark variants) and table into _static/roadmap/ on every
# build, so they always match the page.


def _build_roadmap_figures(app):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "roadmap_figures", ROOT / "scripts" / "roadmap_figures.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up here
    spec.loader.exec_module(module)
    module.build(Path(app.srcdir) / "_static" / "roadmap", Path(app.srcdir) / "roadmap.rst")


# -- Method gallery --------------------------------------------------------------
# scripts/generate_visualizations.py runs every public method and draws its
# backbone over the original graph (light and dark variants), plus one reST
# include per API module, into the gitignored _gallery/ on every build, so the
# gallery on the API pages always shows what the current code returns.


def _build_gallery(app):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_visualizations", ROOT / "scripts" / "generate_visualizations.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up here
    spec.loader.exec_module(module)
    module.build(Path(app.srcdir) / "_gallery")


def setup(app):
    app.connect("builder-inited", _build_roadmap_figures)
    app.connect("builder-inited", _build_gallery)
    from docutils import nodes
    from docutils.statemachine import StringList
    from sphinx.util.docutils import SphinxDirective

    class LesMiserablesTable(SphinxDirective):
        """Render the Les Misérables benchmark table, computed at build time."""

        def run(self):
            # The counts depend on library code, not on this page's source:
            # recompute them on every build, incremental ones included.
            self.env.note_reread()
            container = nodes.container()
            source = self.get_source_info()[0]
            content = StringList(_les_miserables_table_rst(), source=source)
            self.state.nested_parse(content, self.content_offset, container)
            return container.children

    app.add_directive("les-miserables-table", LesMiserablesTable)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
