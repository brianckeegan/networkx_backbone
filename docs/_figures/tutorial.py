"""Light and dark figures for the executed tutorials in ``docs/examples/``.

Every figure of a Sphinx-Gallery tutorial goes through one call::

    from _figures.tutorial import TutorialFigures

    figures = TutorialFigures("start_here")  # one name per tutorial file


    def draw_overview(mode):  # -> matplotlib Figure, colors from style.palette(mode)
        ...


    figures.show(
        draw_overview,
        "overview",
        alt=f"... {n_edges} edges ...",  # computed strings, never literals
        caption=f"... {kept} of {n_edges} edges ...",
        fmt="png",  # network drawings; charts stay SVG
        thumbnail=True,  # at most once per tutorial: the gallery thumbnail
    )

:meth:`TutorialFigures.show` draws the figure once per mode with
:func:`_figures.style.save_variants` and returns a :class:`ThemedFigure`.
Make it the last expression of the code block: Sphinx-Gallery then shows its
``_repr_html_``, a ``<figure>`` with the light and the dark image (the docs
theme shows the one matching the reader's theme) and the caption.

Where the files go
------------------
* **In the docs build** (Sphinx-Gallery runs the file from
  ``docs/examples/<folder>/``): into the gitignored
  ``docs/_static/example_figures/<name>/``, which Sphinx copies with the rest
  of ``_static``; the page *links* to them (#116), lazily, so a reader
  downloads only the variant the theme shows.
* **Anywhere else** (the downloaded script or notebook): into a temporary
  folder, and the images are embedded in the HTML as data URIs, so the output
  stays self-contained.

The gallery thumbnail
---------------------
``thumbnail=True`` also writes the light variant as
``docs/_static/example_figures/<name>/thumbnail.png``; point the tutorial's
thumbnail comment at it (``TutorialFigures.thumbnail_path`` gives the
string)::

    # sphinx_gallery_thumbnail_path = "_static/example_figures/start_here/thumbnail.png"

``tests/test_docs_figures.py`` checks that every tutorial using this module
does both, exactly once.

Alt text and captions
---------------------
Both are strings, or functions of the mode where the words differ between
the variants ("darker" in light, "brighter" in dark).  Alt text is plain
text.  Captions may use a small subset of reST inline markup, rendered to
HTML by :func:`render_caption`: inline literals (double backquotes),
emphasis and strong emphasis (``*`` and ``**``), ``:math:`` (typeset by the
page's MathJax) and ``:doc:`` with an explicit title, ``title </path>`` (a
link in the docs build, the title elsewhere).
"""

from __future__ import annotations

import base64
import html
import re
import shutil
import struct
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from _figures import style

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["ThemedFigure", "TutorialFigures", "image_size", "render_caption"]

#: Resolution of PNG figures (dense network drawings); charts stay SVG.
PNG_DPI = 144
#: Resolution of the light PNG used as the gallery thumbnail.
THUMBNAIL_DPI = 100
#: Where the docs build puts the files, relative to ``docs/``.
STATIC_DIR = "_static/example_figures"

_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")


def image_size(path: str | Path) -> tuple[int, int]:
    """Width and height in CSS pixels of a PNG or a matplotlib SVG.

    The ``width`` and ``height`` attributes of an ``<img>`` reserve its space
    before a lazily loaded file arrives.  PNG sizes are in pixels; matplotlib
    writes SVG sizes in points (1 pt = 4/3 px).
    """
    path = Path(path)
    with path.open("rb") as f:
        head = f.read(1000)
    if head.startswith(b"\x89PNG"):
        width, height = struct.unpack(">II", head[16:24])
        return int(width), int(height)
    size = re.search(rb'width="([\d.]+)pt" height="([\d.]+)pt"', head)
    if size is None:
        raise ValueError(f"cannot read the size of {path}")
    width, height = (round(float(v) * 4 / 3) for v in size.groups())
    return width, height


# Inline markup a caption may use: ``literal``, :math:`tex`, :doc:`title <path>`,
# **strong**, *emphasis*.  Anything else is text.
_INLINE = re.compile(
    r"``(?P<code>.+?)``"
    r"|:math:`(?P<math>[^`]+)`"
    r"|:doc:`(?P<doc>[^`]+)`"
    r"|\*\*(?P<strong>[^*]+)\*\*"
    r"|(?<![\w*])\*(?P<em>[^*\s](?:[^*]*[^*\s])?)\*(?![\w*])"
)
_DOC = re.compile(r"(?P<title>.+?)\s*<(?P<target>/[^>]+)>")


def render_caption(text: str, doc_prefix: str | None = None) -> str:
    """HTML for a caption written in the small reST subset above.

    *doc_prefix* is the relative URL from the page to the docs root (e.g.
    ``"../../"``); without one, ``:doc:`` references render as their title.
    """
    out = []
    pos = 0
    for m in _INLINE.finditer(text):
        out.append(html.escape(text[pos : m.start()]))
        pos = m.end()
        if m["code"] is not None:
            out.append(
                f'<code class="docutils literal">{html.escape(m["code"])}</code>'
            )
        elif m["math"] is not None:
            # The markup Sphinx writes for inline math, which MathJax typesets.
            out.append(
                '<span class="math notranslate nohighlight">'
                f"\\({html.escape(m['math'])}\\)</span>"
            )
        elif m["doc"] is not None:
            ref = _DOC.fullmatch(m["doc"])
            if ref is None:
                raise ValueError(
                    f"write :doc: references as :doc:`title </path>`: {m[0]!r}"
                )
            title = html.escape(ref["title"])
            if doc_prefix is None:
                out.append(title)
            else:
                href = doc_prefix + ref["target"].lstrip("/") + ".html"
                out.append(f'<a class="reference internal" href="{href}">{title}</a>')
        elif m["strong"] is not None:
            out.append(f"<strong>{html.escape(m['strong'])}</strong>")
        else:
            out.append(f"<em>{html.escape(m['em'])}</em>")
    out.append(html.escape(text[pos:]))
    return "".join(out)


def _sphinx_gallery_docs_root() -> tuple[Path, int] | None:
    """``(docs root, page depth)`` when Sphinx-Gallery runs this tutorial.

    Sphinx-Gallery executes each example from its own folder,
    ``docs/examples/<folder>/``, and writes the page to the matching
    ``docs/auto_examples/<folder>/``, so the page sits as many folders below
    the docs root as the working directory does.
    """
    if "sphinx_gallery.gen_rst" not in sys.modules:
        return None
    here = Path.cwd().resolve()
    for depth, folder in enumerate((here, *here.parents)):
        if (folder / "conf.py").is_file() and (folder / "_figures").is_dir():
            parts = here.relative_to(folder).parts
            if parts and parts[0] == "examples":
                return folder, depth
            return None
    return None


@dataclass(frozen=True, repr=False)
class ThemedFigure:
    """The light and dark files of one figure, its alt text and caption.

    The last expression of a code block, it renders as HTML (Sphinx-Gallery
    and Jupyter call ``_repr_html_``).  Its plain ``repr`` names the figure
    only, so a cell never prints the build machine's paths (#119).
    """

    name: str
    paths: dict[str, Path]
    alt: dict[str, str]  # by mode
    caption: dict[str, str]  # by mode
    url: str | None = None  # the files' folder as seen from the page; None: inline
    doc_prefix: str | None = None  # the docs root as seen from the page
    width: str | None = "100%"

    def __repr__(self) -> str:
        return f"<ThemedFigure {self.name!r}>"

    def _img(self, mode: str, path: Path) -> str:
        width, height = image_size(path)
        if self.url is None:
            mime = "image/png" if path.suffix == ".png" else "image/svg+xml"
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            src = f"data:{mime};base64,{data}"
        else:
            src = self.url + path.name
        if self.width is None:  # its own size, at most the column's width
            size = "width: auto; max-width: 100%; height: auto;"
        else:
            size = f"width: {self.width}; height: auto;"
        # Lazy loading: the browser fetches only the variant the theme shows,
        # and the width and height reserve its space before it arrives.
        return (
            f'<img class="only-{mode}" alt="{html.escape(self.alt[mode])}" '
            f'src="{src}" width="{width}" height="{height}" loading="lazy" '
            f'style="{size}">'
        )

    def _repr_html_(self) -> str:
        images = "".join(self._img(mode, path) for mode, path in self.paths.items())
        captions = {
            mode: render_caption(text, self.doc_prefix)
            for mode, text in self.caption.items()
        }
        if len(set(captions.values())) == 1:
            caption = f"<p>{captions[style.MODES[0]]}</p>"
        else:  # worded for each mode, like the images
            caption = "".join(
                f'<p class="only-{mode}">{text}</p>' for mode, text in captions.items()
            )
        # The caption follows a wrapper <div>, not the images themselves: the
        # theme hides a <figcaption> that directly follows an ``only-dark``
        # image in light mode (#116).
        return (
            f'<figure class="nxbb-figure"><div>{images}</div>'
            f"<figcaption>{caption}</figcaption></figure>"
        )


class TutorialFigures:
    """Where one tutorial's figures go, and how its page shows them.

    Parameters
    ----------
    name : str
        A name unique to the tutorial file (lowercase, ``-`` and ``_``
        allowed), usually its folder in ``docs/examples/``.  The docs build
        writes the files to ``docs/_static/example_figures/<name>/``, emptied
        first so no file of an earlier build lingers.
    """

    def __init__(self, name: str) -> None:
        if not _NAME.fullmatch(name):
            raise ValueError(f"figure folder names are lowercase slugs: {name!r}")
        self.name = name
        self.thumbnail_path = f"{STATIC_DIR}/{name}/thumbnail.png"
        self._thumbnail: Path | None = None
        found = _sphinx_gallery_docs_root()
        if found is None:
            self.directory = Path(tempfile.mkdtemp(prefix=f"nxbb-{name}-"))
            self.url = self.doc_prefix = None
        else:
            docs, depth = found
            self.directory = docs / STATIC_DIR / name
            shutil.rmtree(self.directory, ignore_errors=True)
            self.directory.mkdir(parents=True)
            self.doc_prefix = "../" * depth
            self.url = f"{self.doc_prefix}{STATIC_DIR}/{name}/"
        self._names: set[str] = set()

    def __repr__(self) -> str:
        return f"<TutorialFigures {self.name!r}>"

    def show(
        self,
        draw: Callable[[str], Figure],
        name: str,
        *,
        alt: str | Callable[[str], str],
        caption: str | Callable[[str], str],
        fmt: str = "svg",
        width: str | None = "100%",
        dpi: float | None = None,
        thumbnail: bool = False,
        **savefig_kw: Any,
    ) -> ThemedFigure:
        """Draw *draw* in light and dark, save both, and return the pair.

        Parameters
        ----------
        draw : callable
            ``draw(mode) -> Figure``, as for :func:`_figures.style.save_variants`.
        name : str
            File stem, unique within the tutorial.
        alt : str or callable
            Alt text: what the figure shows, for a reader who cannot see it.
            Plain text; format its numbers from the variables the tutorial
            computes.  Where the words depend on the mode ("darker" in light,
            "brighter" in dark), pass ``alt(mode) -> str``.
        caption : str or callable
            The caption under the figure (the small reST subset of
            :func:`render_caption`); ``caption(mode) -> str`` where its words
            depend on the mode.
        fmt : {"svg", "png"}
            SVG for charts, PNG for dense network drawings.
        width : str or None
            CSS width of the image, e.g. ``"85%"`` for a narrow chart; None
            keeps the file's own size, at most the column's width.
        dpi : float, optional
            PNG resolution; :data:`PNG_DPI` by default.
        thumbnail : bool
            Also save the light variant as the gallery thumbnail
            (:attr:`thumbnail_path`).  At most one figure per tutorial.
        **savefig_kw
            Passed to ``Figure.savefig``.

        Returns
        -------
        ThemedFigure
        """
        if not _NAME.fullmatch(name):
            raise ValueError(f"figure names are lowercase slugs: {name!r}")
        if name in self._names:
            raise ValueError(f"figure {name!r} is already shown on this page")
        alts, captions = (
            {
                mode: " ".join((text(mode) if callable(text) else text).split())
                for mode in style.MODES
            }
            for text in (alt, caption)
        )
        if not all(alts.values()) or not all(captions.values()):
            raise ValueError(f"figure {name!r} needs alt text and a caption")
        self._names.add(name)
        if fmt == "png":
            savefig_kw.setdefault("dpi", dpi or PNG_DPI)
        if thumbnail:
            if self._thumbnail is not None:
                raise ValueError(
                    f"{self.name} already has a thumbnail ({self._thumbnail.name})"
                )
            self._thumbnail = self.directory / "thumbnail.png"
            draw = self._saving_thumbnail(draw, self._thumbnail)
        paths = style.save_variants(draw, self.directory / name, fmt=fmt, **savefig_kw)
        return ThemedFigure(
            name=name,
            paths=paths,
            alt=alts,
            caption=captions,
            url=self.url,
            doc_prefix=self.doc_prefix,
            width=width,
        )

    @staticmethod
    def _saving_thumbnail(
        draw: Callable[[str], Figure], path: Path
    ) -> Callable[[str], Figure]:
        """Wrap *draw* so that its light figure is also saved as *path*.

        The light figure is drawn once for both files, inside the light
        style that :func:`~_figures.style.save_variants` sets.
        """

        def drawing(mode: str) -> Figure:
            fig = draw(mode)
            if mode == "light":
                fig.savefig(path, format="png", dpi=THUMBNAIL_DPI)
            return fig

        return drawing
