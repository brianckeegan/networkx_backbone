"""Build-time figures for the plain reST pages (how-tos, user guide).

One module per page, named after the page's file (``workflow.py`` for
``docs/user_guide/workflow.rst``), so work on different pages never touches
the same file.  A module registers its figures with :func:`figure`::

    from _figures import style
    from _figures.pages import figure


    @figure(
        "pipeline",
        alt="...",  # a string, or a callable computed at build time
        caption=lambda: f"... {count()} edges ...",
    )
    def draw_pipeline(mode):  # -> matplotlib Figure, colors from style.palette(mode)
        ...

and the page shows it with the ``page-figure`` directive
(``docs/_ext/page_figures.py``)::

    .. page-figure:: workflow/pipeline

:func:`build`, run by the docs build on ``builder-inited``, discovers every
module, draws each figure in light and dark with
:func:`_figures.style.save_variants` into the gitignored
``docs/_page_figures/<page>/``, and writes ``manifest.json``: per reference,
the files, the alt text and the caption, which the directive reads.

Do the work inside the draw function (or a cached helper it calls), not at
import time: discovery imports every module on every build.  A page whose
module, the shared figure code and ``nx_backbone`` are unchanged since the
last build is not drawn again (see :func:`build`).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import pkgutil
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["PageFigure", "build", "discover", "figure"]

FIGURES = Path(__file__).resolve().parents[1]  # docs/_figures
ROOT = FIGURES.parents[1]
MANIFEST = "manifest.json"
#: Resolution of PNG page figures (dense network drawings).
PNG_DPI = 144

_KEY = re.compile(r"[a-z0-9][a-z0-9_-]*")
_REGISTRY: dict[str, PageFigure] = {}

Text = str | Callable[[], str]


@dataclass(frozen=True)
class PageFigure:
    """One registered figure of a reST page."""

    page: str  # the module name, i.e. the page's file stem
    key: str
    draw: Callable[[str], Figure]
    alt: Text
    caption: Text
    fmt: str = "svg"
    dpi: float | None = None

    @property
    def ref(self) -> str:
        """The directive argument: ``<page>/<key>``."""
        return f"{self.page}/{self.key}"


def figure(
    key: str,
    *,
    alt: Text,
    caption: Text,
    fmt: str = "svg",
    dpi: float | None = None,
) -> Callable[[Callable[[str], Figure]], Callable[[str], Figure]]:
    """Register the decorated ``draw(mode)`` as figure *key* of this page.

    Parameters
    ----------
    key : str
        Lowercase slug, unique within the page.
    alt, caption : str or callable
        Plain-text alt text, and the caption (reST inline markup).  A
        callable is called with no arguments after the figure is drawn.
    fmt : {"svg", "png"}
        SVG for charts, PNG for dense network drawings.
    dpi : float, optional
        PNG resolution; :data:`PNG_DPI` by default.

    Returns
    -------
    callable
        The decorator; it returns the draw function unchanged.
    """
    if not _KEY.fullmatch(key):
        raise ValueError(f"figure keys are lowercase slugs: {key!r}")
    if fmt not in ("svg", "png"):
        raise ValueError(f"fmt must be 'svg' or 'png', got {fmt!r}")

    def register(draw: Callable[[str], Figure]) -> Callable[[str], Figure]:
        page = draw.__module__.rpartition(".")[2]
        entry = PageFigure(page, key, draw, alt, caption, fmt, dpi)
        old = _REGISTRY.get(entry.ref)
        if old is not None and old.draw.__qualname__ != draw.__qualname__:
            raise ValueError(f"figure {entry.ref!r} is registered twice")
        _REGISTRY[entry.ref] = entry
        return draw

    return register


def _page_modules() -> list[str]:
    return sorted(
        info.name
        for info in pkgutil.iter_modules(__path__)
        if not info.name.startswith("_")
    )


def discover() -> dict[str, PageFigure]:
    """Import every page module and return the figures by reference."""
    for name in _page_modules():
        importlib.import_module(f"{__name__}.{name}")
    return dict(sorted(_REGISTRY.items()))


def _text(value: Text) -> str:
    text = value() if callable(value) else value
    text = " ".join(str(text).split())
    if not text:
        raise ValueError("alt text and captions must not be empty")
    return text


def _fingerprint(page: str) -> str:
    """What a page's figures depend on: its module, the shared code, the library.

    The page module, every non-page module of ``docs/_figures`` (the style,
    this registry), the ``nx_backbone`` sources and the plotting versions.
    """
    import matplotlib
    import networkx

    h = hashlib.sha256()
    h.update(f"{matplotlib.__version__} {networkx.__version__}".encode())
    shared = [*FIGURES.glob("*.py"), Path(__file__)]
    library = sorted((ROOT / "nx_backbone").rglob("*.py"))
    for path in [Path(__path__[0]) / f"{page}.py", *sorted(shared), *library]:
        h.update(path.as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def build(out_dir: str | Path, *, static_prefix: str = "/_page_figures") -> dict:
    """Draw every registered figure into *out_dir* and write the manifest.

    A page is drawn again only when its fingerprint (its module, the shared
    figure code, ``nx_backbone``, the matplotlib and NetworkX versions)
    differs from the one recorded in the previous manifest, or a file is
    missing; otherwise its manifest entries are kept.  ``rm -rf
    docs/_page_figures`` forces a full redraw.

    Parameters
    ----------
    out_dir : path-like
        Directory for ``<page>/<key>-{light,dark}.<fmt>`` and the manifest.
    static_prefix : str, optional
        How the pages refer to *out_dir* (an absolute docs path).

    Returns
    -------
    dict
        The manifest: ``{ref: {"files": {mode: path}, "alt", "caption",
        "fingerprint"}}``.
    """
    import matplotlib

    from _figures import style

    matplotlib.use("Agg")
    out = Path(out_dir)
    try:
        old = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    figures = discover()
    manifest: dict[str, Any] = {}
    fingerprints = {
        page: _fingerprint(page) for page in {f.page for f in figures.values()}
    }
    for ref, fig in figures.items():
        fingerprint = fingerprints[fig.page]
        cached = old.get(ref)
        if (
            cached
            and cached.get("fingerprint") == fingerprint
            and cached.get("fmt") == fig.fmt
            and all(
                (out / fig.page / Path(f).name).is_file()
                for f in cached["files"].values()
            )
        ):
            manifest[ref] = cached
            continue
        kw = {"dpi": fig.dpi or PNG_DPI} if fig.fmt == "png" else {}
        paths = style.save_variants(
            fig.draw, out / fig.page / fig.key, fmt=fig.fmt, **kw
        )
        manifest[ref] = {
            "files": {
                mode: f"{static_prefix}/{fig.page}/{path.name}"
                for mode, path in paths.items()
            },
            "fmt": fig.fmt,
            "alt": _text(fig.alt),
            "caption": _text(fig.caption),
            "fingerprint": fingerprint,
        }
    out.mkdir(parents=True, exist_ok=True)
    (out / MANIFEST).write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
