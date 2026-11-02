"""The ``page-figure`` directive (docs/_ext/page_figures.py) in a small Sphinx build."""

import io
import json
import sys
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"

SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="180pt" height="90pt" '
    'viewBox="0 0 180 90"><rect width="180" height="90"/></svg>\n'
)


@pytest.fixture
def build(tmp_path, monkeypatch):
    pytest.importorskip("sphinx")
    pytest.importorskip("matplotlib")
    from sphinx.application import Sphinx

    for path in (DOCS, DOCS / "_ext"):
        monkeypatch.syspath_prepend(str(path))
    src = tmp_path / "src"
    (src / "_gallery" / "les_miserables").mkdir(parents=True)
    (src / "conf.py").write_text('extensions = ["page_figures"]\n')
    # A stand-in for the method gallery's manifest and files.
    gallery = {}
    for mode in ("light", "dark"):
        (src / "_gallery" / "les_miserables" / f"demo-{mode}.svg").write_text(SVG)
    gallery["statistical/demo"] = {
        "files": {
            mode: f"/_gallery/les_miserables/demo-{mode}.svg"
            for mode in ("light", "dark")
        },
        "fmt": "svg",
        "alt": "A gallery figure.",
        "caption": "Kept 1 of 2 edges.",
    }
    (src / "_gallery" / "manifest.json").write_text(json.dumps(gallery))

    def run(body):
        (src / "index.rst").write_text("Test\n====\n\n" + body)
        app = Sphinx(
            str(src),
            str(src),
            str(tmp_path / "out"),
            str(tmp_path / "doctrees"),
            "html",
            status=None,
            warning=(warnings := io.StringIO()),
            freshenv=True,
        )
        app.build()
        html = (tmp_path / "out" / "index.html").read_text(encoding="utf-8")
        return html, warnings.getvalue()

    yield run
    for name in [m for m in sys.modules if m.startswith("_figures.pages.")]:
        sys.modules.pop(name)


def test_page_figure_and_gallery_reuse(build):
    html, warnings = build(
        ".. page-figure:: workflow/score-then-filter\n   :width: 80%\n\n"
        ".. page-figure:: gallery:statistical/demo\n"
    )
    assert "ERROR" not in warnings and "WARNING" not in warnings, warnings
    assert html.count('<figure class="nxbb-figure') == 2
    assert 'style="width: 80%"' in html
    for mode in ("light", "dark"):
        assert f'class="only-{mode}"' in html
        assert f"score-then-filter-{mode}.png" in html
        assert f"demo-{mode}.svg" in html
    assert html.count('loading="lazy"') == 4
    assert 'alt="A gallery figure."' in html and "Kept 1 of 2 edges." in html
    # Width and height reserve the space; the SVG's 180 pt are 240 px.
    assert 'height="120"' in html and 'width="240"' in html
    # The caption follows the wrapper of the two images (#116).
    assert "</div>\n<figcaption>" in html


def test_unknown_page_figure_is_an_error(build):
    _, warnings = build(".. page-figure:: workflow/no-such-figure\n")
    assert "unknown figure 'workflow/no-such-figure'" in warnings
