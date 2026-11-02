"""Sphinx extension: build-time light/dark figures on plain reST pages.

On ``builder-inited`` it draws every figure registered in
``docs/_figures/pages/`` (one module per page, see
:mod:`_figures.pages`) into the gitignored ``docs/_page_figures/``.  The
``page-figure`` directive then shows one::

    .. page-figure:: workflow/score-then-filter
       :width: 80%

or reuses a figure of the method gallery on the API pages, which the build
draws anyway (``scripts/generate_visualizations.py``), at no extra cost::

    .. page-figure:: gallery:statistical/disparity_filter

Either way the page gets one ``<figure>`` with the light and the dark image
(the theme shows the one matching the reader's theme), lazily loaded with
their sizes, the registered alt text, and the caption below a wrapper
``<div>`` (#116: the theme hides a caption right after an ``only-dark``
image).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)

PAGE_FIGURES = "_page_figures"  # under the docs source directory
GALLERY = "_gallery"  # written by scripts/generate_visualizations.py
GALLERY_PREFIX = "gallery:"


def _build_page_figures(app: Sphinx) -> None:
    from _figures import pages

    manifest = pages.build(Path(app.srcdir) / PAGE_FIGURES)
    logger.info("page figures: %d registered", len(manifest))


def _manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return {}


class PageFigureDirective(SphinxDirective):
    """``.. page-figure:: <page>/<key>`` or ``gallery:<module>/<name>``."""

    required_arguments = 1
    optional_arguments = 0
    has_content = False
    option_spec: ClassVar[dict[str, Any]] = {
        "width": directives.length_or_percentage_or_unitless,
        "name": directives.unchanged,
    }

    def run(self) -> list[nodes.Node]:
        from _figures.tutorial import image_size

        srcdir = Path(self.env.srcdir)
        ref = self.arguments[0]
        if ref.startswith(GALLERY_PREFIX):
            key = ref.removeprefix(GALLERY_PREFIX)
            manifest = _manifest(srcdir / GALLERY / "manifest.json")
            source = srcdir.parent / "scripts" / "generate_visualizations.py"
        else:
            key = ref
            manifest = _manifest(srcdir / PAGE_FIGURES / "manifest.json")
            page = key.partition("/")[0]
            source = srcdir / "_figures" / "pages" / f"{page}.py"
        entry = manifest.get(key)
        if entry is None:
            known = ", ".join(sorted(manifest)) or "none"
            raise self.error(f"page-figure: unknown figure {ref!r} (known: {known})")
        # Re-read this page when the code that draws its figure changes.
        if source.is_file():
            self.env.note_dependency(str(source))

        figure = nodes.figure(classes=["nxbb-figure"])
        if "width" in self.options:
            figure["width"] = self.options["width"]
        self.add_name(figure)
        variants = nodes.container(classes=["nxbb-figure-variants"])
        for mode, uri in entry["files"].items():
            width, height = image_size(srcdir / uri.lstrip("/"))
            variants += nodes.image(
                uri=uri,
                alt=entry["alt"],
                classes=[f"only-{mode}"],
                loading="lazy",
                width=str(width),
                height=str(height),
            )
        figure += variants
        text_nodes, messages = self.parse_inline(entry["caption"], lineno=self.lineno)
        figure += nodes.caption(entry["caption"], "", *text_nodes)
        return [figure, *messages]


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("builder-inited", _build_page_figures)
    app.add_directive("page-figure", PageFigureDirective)
    app.add_css_file("nxbb-figures.css")
    return {"parallel_read_safe": True, "parallel_write_safe": True}
