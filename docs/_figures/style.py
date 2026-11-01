"""One figure style for the documentation: palette, defaults, layout, overlay.

Every figure in the docs imports this module (``from _figures import style``)
so the method gallery, the Sphinx-Gallery examples and the executed tutorials
read as one system:

* :func:`palette` -- validated color tokens, light and dark;
* :func:`apply` / :func:`rc_context` -- matplotlib defaults (sizes, thin marks,
  ink colors);
* :func:`layout` -- one deterministic, cached layout per dataset, so every
  method's figure on a dataset puts each node in the same place;
* :func:`draw_backbone` -- a backbone drawn over its original graph (kept
  edges emphasized, removed edges faint), with a legend and the counts;
* :func:`save_variants` and :func:`figure_rst` -- write a light and a dark
  file and the reST that shows the one matching the reader's theme (the
  ``only-light`` / ``only-dark`` pattern of ``docs/roadmap.rst``).

The palette
-----------
The hues and steps are the dataviz reference palette (the same blue ramp the
roadmap figures use), checked against the pydata-sphinx-theme surfaces the
figures sit on: ``#ffffff`` (light) and ``#14181e`` (dark).  The dark tokens
are *selected* steps of the same ramps, validated on the dark surface, not an
inversion of the light ones.  Validator output (``validate_palette.js``,
OKLab Delta E x 100, Machado 2009 CVD simulation):

* categorical, 8 slots, adjacent pairs -- light: all checks pass, worst CVD
  9.1 (yellow/aqua), worst normal-vision 19.6; slots aqua (2.82:1), yellow
  (2.17:1) and magenta (2.69:1) are below 3:1 on white, so a figure using them
  needs direct labels or a table.  Dark: all checks pass, worst CVD 8.4,
  normal-vision 19.3, every slot >= 3:1.
* categorical, first 3 slots, all pairs (small multiples, scatter) -- light:
  pass, worst CVD 9.2, normal-vision 24.0; dark: pass, worst CVD 9.4,
  normal-vision 20.9.  Hence :data:`ALL_PAIRS_CAP` = 3.
* signed (positive, negative) = slot 1 blue and slot 8 red, the diverging
  poles -- light: pass, CVD 21.6, normal-vision 32.3, both >= 3:1; dark:
  pass, CVD 19.2, normal-vision 29.0, both >= 3:1.  Negative edges are also
  dashed, so sign never rests on color alone.
* sequential ramp (``--ordinal``), blue steps 250..650 light and 600..200
  dark -- pass: monotone lightness, adjacent Delta L >= 0.06, single hue,
  low end 2.11:1 (light) / 2.20:1 (dark) against the surface.
* kept edge = categorical slot 1 (4.42:1 light, 4.89:1 dark); backbone nodes
  are the same hue, a step closer to the ink (blue 550 light, 250 dark;
  6.63:1 and 8.44:1) with a surface ring, so they read above the edges.
  They are one identity, not a pair of series, so the categorical pair
  checks do not apply to them.
* removed edge and dropped node grays are deliberately recessive (1.68:1 and
  1.95:1 light, 1.78:1 and 2.36:1 dark): context, not data, and never the
  only cue -- the legend and the counts carry the meaning.
* text: ink 14.81:1 / muted 7.45:1 (light), 12.12:1 / 7.08:1 (dark) -- the
  theme's own text colors.

Re-run the checks after changing any value:
``node .../dataviz/scripts/validate_palette.js "<hex,...>" --mode light
--surface "#ffffff"`` and ``--mode dark --surface "#14181e"``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

__all__ = [
    "ALL_PAIRS_CAP",
    "MODES",
    "BackboneCounts",
    "Palette",
    "apply",
    "backbone_counts",
    "categorical",
    "draw_backbone",
    "figure_rst",
    "layout",
    "legend_handles",
    "palette",
    "rc_context",
    "rc_params",
    "reset_gallery",
    "save_variants",
    "sequential_cmap",
]

MODES = ("light", "dark")

#: Most categorical slots a chart may use when any two marks can sit side by
#: side (small multiples, scatter): the first three validate on all pairs.
ALL_PAIRS_CAP = 3


@dataclass(frozen=True)
class Palette:
    """Color tokens for one mode.  Text wears ``ink``/``muted``, never data colors."""

    mode: str
    surface: str  # the docs theme background the figure sits on
    ink: str  # titles, labels, legend text
    muted: str  # axis ticks, secondary text
    hairline: str  # gridlines and axes
    kept: str  # backbone edges (categorical slot 1)
    removed: str  # edges of the original graph not in the backbone
    node: str  # nodes in the backbone
    node_removed: str  # nodes with no backbone edge
    positive: str  # positive signed edges (solid)
    negative: str  # negative signed edges (dashed)
    categorical: tuple[str, ...]  # fixed order; never cycled past 8
    sequential: tuple[str, ...]  # low -> high magnitude


_CATEGORICAL = {
    # blue, orange, aqua, yellow, magenta, green, violet, red
    "light": (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ),
    "dark": (
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ),
}

# Blue ramp, low -> high.  Light: steps 250..650, so even the lowest step
# clears 2:1 on white; high values are dark.  Dark: the anchor flips -- steps
# 600..200, low values sit nearest the dark surface (still >= 2:1) and high
# values are light.  Both validate as ordinal ramps.
_SEQUENTIAL = {
    "light": ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"),
    "dark": ("#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"),
}

_PALETTES = {
    "light": Palette(
        mode="light",
        surface="#ffffff",
        ink="#222832",
        muted="#48566b",
        hairline="#e5e7ea",
        kept=_CATEGORICAL["light"][0],
        removed="#c3c8cf",
        node="#1c5cab",  # blue step 550: reads on top of the kept edges
        node_removed="#b4bac3",
        positive=_CATEGORICAL["light"][0],
        negative=_CATEGORICAL["light"][7],
        categorical=_CATEGORICAL["light"],
        sequential=_SEQUENTIAL["light"],
    ),
    "dark": Palette(
        mode="dark",
        surface="#14181e",
        ink="#ced6dd",
        muted="#9ca4af",
        hairline="#29313d",
        kept=_CATEGORICAL["dark"][0],
        removed="#3a4350",
        node="#86b6ef",  # blue step 250
        node_removed="#4b5563",
        positive=_CATEGORICAL["dark"][0],
        negative=_CATEGORICAL["dark"][7],
        categorical=_CATEGORICAL["dark"],
        sequential=_SEQUENTIAL["dark"],
    ),
}

# Mark specs, in points (1 pt = 4/3 px at 96 dpi): ~2 px kept lines, hairline
# context, >= 8 px markers with a ~2 px surface ring.
KEPT_WIDTH = 1.5
REMOVED_WIDTH = 0.6
NODE_SIZE = 42.0  # marker area in pt^2 (diameter ~6.5 pt ~ 8.6 px)
RING_WIDTH = 1.5
_DASH = (0, (3.0, 2.0))


def _check_mode(mode: str) -> str:
    if mode not in _PALETTES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return mode


def palette(mode: str = "light") -> Palette:
    """Return the validated color tokens for *mode* (``"light"`` or ``"dark"``)."""
    return _PALETTES[_check_mode(mode)]


def categorical(n: int, mode: str = "light") -> tuple[str, ...]:
    """The first *n* categorical slots, in their fixed order.

    More than eight series do not get generated hues: fold the tail into
    "Other" or facet.  Charts where any two marks can be neighbors (small
    multiples, scatter) should use at most :data:`ALL_PAIRS_CAP` slots.
    """
    slots = palette(mode).categorical
    if not 0 <= n <= len(slots):
        raise ValueError(f"at most {len(slots)} categorical slots, got {n}")
    return slots[:n]


def sequential_cmap(mode: str = "light") -> Colormap:
    """A one-hue (blue) colormap for scores, low -> high.

    For p-values, map ``-log10(p)`` (or ``1 - p``) so that significant edges
    get the emphasized end of the ramp.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        f"nxbb_sequential_{mode}", list(palette(mode).sequential)
    )


def rc_params(mode: str = "light") -> dict[str, Any]:
    """Matplotlib defaults for *mode*: sizes, thin marks, ink, palette cycle."""
    from cycler import cycler

    p = palette(mode)
    return {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.dpi": 200,
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 6.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "legend.labelcolor": p.ink,
        "figure.titlesize": 11.0,
        "figure.titleweight": "bold",
        "text.color": p.ink,
        "axes.titlecolor": p.ink,
        "axes.labelcolor": p.ink,
        "axes.edgecolor": p.hairline,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=list(p.categorical)),
        "grid.color": p.hairline,
        "grid.linewidth": 0.75,
        "grid.linestyle": "-",
        "xtick.color": p.hairline,
        "ytick.color": p.hairline,
        "xtick.labelcolor": p.muted,
        "ytick.labelcolor": p.muted,
        "lines.linewidth": KEPT_WIDTH,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 6.5,
        "patch.linewidth": 0.0,
        "image.cmap": "Blues",
        # Reproducible SVG: fixed ids, no timestamp in the metadata.
        "svg.hashsalt": "networkx-backbone",
    }


def apply(mode: str = "light") -> None:
    """Set the docs style on the global ``matplotlib.rcParams``."""
    import matplotlib as mpl

    mpl.rcParams.update(rc_params(mode))


def rc_context(mode: str = "light") -> contextlib.AbstractContextManager[Any]:
    """A ``with`` block that uses the docs style for *mode* and then restores."""
    import matplotlib as mpl

    return mpl.rc_context(rc_params(mode))


def reset_gallery(gallery_conf: Any, fname: str) -> None:
    """Sphinx-Gallery ``reset_modules`` hook: every example starts styled (light)."""
    apply("light")


def save_variants(
    draw: Callable[[str], Figure],
    stem: str | Path,
    *,
    fmt: str = "svg",
    modes: Iterable[str] = MODES,
    **savefig_kw: Any,
) -> dict[str, Path]:
    """Draw and save one file per mode: ``<stem>-light.<fmt>``, ``<stem>-dark.<fmt>``.

    Parameters
    ----------
    draw : callable
        ``draw(mode) -> Figure``.  It is called inside :func:`rc_context` for
        each mode and should take its colors from ``palette(mode)``.
    stem : str or path
        Output path without the ``-<mode>.<fmt>`` suffix; parents are created.
    fmt : {"svg", "png"}
        SVG by default; PNG (at 200 dpi) where an SVG would be too heavy.
    modes : iterable of str
        The variants to write.
    **savefig_kw
        Passed to ``Figure.savefig``.

    Returns
    -------
    dict
        ``{mode: path}`` of the written files.
    """
    import matplotlib.pyplot as plt

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "svg":
        savefig_kw.setdefault("metadata", {"Date": None})
    paths = {}
    for mode in modes:
        with rc_context(mode):
            fig = draw(mode)
            path = stem.with_name(f"{stem.name}-{mode}.{fmt}")
            fig.savefig(path, format=fmt, **savefig_kw)
        plt.close(fig)
        paths[mode] = path
    return paths


def figure_rst(
    stem: str,
    alt: str,
    caption: str | None = None,
    *,
    fmt: str = "svg",
    width: str = "100%",
    indent: str = "",
) -> list[str]:
    """reST for a light/dark figure pair; the theme shows the matching one.

    *stem* is the document-relative or absolute (``/_static/...``) path given
    to :func:`save_variants`, without the ``-<mode>.<fmt>`` suffix.
    """
    lines: list[str] = []
    for mode in MODES:
        lines += [
            f"{indent}.. figure:: {stem}-{mode}.{fmt}",
            f"{indent}   :figclass: only-{mode}",
            f"{indent}   :class: only-{mode}",
            f"{indent}   :width: {width}",
            f"{indent}   :alt: {' '.join(alt.split())}",
            "",
        ]
        if caption:
            lines += [f"{indent}   {' '.join(caption.split())}", ""]
    return lines


# -- layout ---------------------------------------------------------------------

_LAYOUTS: dict[tuple[Hashable, int, str | None], dict[Any, Any]] = {}


def layout(
    G: nx.Graph, key: Hashable, *, seed: int = 42, weight: str | None = None
) -> dict[Any, Any]:
    """One deterministic layout per dataset, cached under *key*.

    Every figure of a dataset (``key="les_miserables"``, say) gets the same
    positions, so backbones of different methods can be compared at a glance.
    Positions are scaled to fit ``[-1, 1]``.  Asking for a cached *key* with a
    graph on different nodes raises ``ValueError``.
    """
    cache_key = (key, seed, weight)
    pos = _LAYOUTS.get(cache_key)
    if pos is None:
        raw = nx.spring_layout(G, seed=seed, weight=weight, iterations=300)
        pos = nx.rescale_layout_dict(raw, scale=1.0)
        _LAYOUTS[cache_key] = pos
    elif set(pos) != set(G):
        raise ValueError(f"layout {key!r} is cached for a graph on different nodes")
    return dict(pos)


# -- backbone overlay ------------------------------------------------------------


@dataclass(frozen=True)
class BackboneCounts:
    """How much of the original graph a backbone keeps."""

    edges_kept: int
    edges_total: int
    nodes_kept: int
    nodes_total: int

    @property
    def edge_share(self) -> float:
        return self.edges_kept / self.edges_total if self.edges_total else 0.0

    @property
    def caption(self) -> str:
        """E.g. ``"Kept 72 of 254 edges (28%) and 50 of 77 nodes."``"""
        return (
            f"Kept {self.edges_kept} of {self.edges_total} edges "
            f"({self.edge_share:.0%}) and {self.nodes_kept} of "
            f"{self.nodes_total} nodes."
        )

    @property
    def short(self) -> str:
        """E.g. ``"72/254 edges, 50/77 nodes"`` for panel subtitles."""
        return (
            f"{self.edges_kept}/{self.edges_total} edges, "
            f"{self.nodes_kept}/{self.nodes_total} nodes"
        )


def _key(u: Any, v: Any, directed: bool) -> Any:
    return (u, v) if directed else frozenset((u, v))


def _split_edges(
    original: nx.Graph, backbone: nx.Graph
) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    """Original edges split into (kept, removed); a backbone edge counts once."""
    directed = original.is_directed() and backbone.is_directed()
    in_backbone = {_key(u, v, directed) for u, v in backbone.edges()}
    kept, removed = [], []
    seen = set()
    for u, v in original.edges():
        k = _key(u, v, original.is_directed())
        if k in seen:  # multigraph parallel edges
            continue
        seen.add(k)
        (kept if _key(u, v, directed) in in_backbone else removed).append((u, v))
    return kept, removed


def backbone_counts(original: nx.Graph, backbone: nx.Graph) -> BackboneCounts:
    """Edges of *original* found in *backbone*, and nodes touching one of them.

    A node counts as kept when at least one of its edges is kept: isolated
    nodes a filter leaves behind (``include_all_nodes=True``) carry nothing.
    """
    kept, removed = _split_edges(original, backbone)
    nodes = {n for e in kept for n in e}
    return BackboneCounts(
        edges_kept=len(kept),
        edges_total=len(kept) + len(removed),
        nodes_kept=len(nodes),
        nodes_total=original.number_of_nodes(),
    )


def legend_handles(mode: str = "light", *, signed: bool = False) -> list[Any]:
    """Legend entries matching :func:`draw_backbone`'s marks."""
    from matplotlib.lines import Line2D

    p = palette(mode)
    if signed:
        edges = [
            Line2D([], [], color=p.positive, lw=KEPT_WIDTH, label="Kept, positive"),
            Line2D(
                [],
                [],
                color=p.negative,
                lw=KEPT_WIDTH,
                linestyle=_DASH,
                label="Kept, negative",
            ),
        ]
    else:
        edges = [Line2D([], [], color=p.kept, lw=KEPT_WIDTH, label="Kept edge")]
    marker = {"marker": "o", "linestyle": "", "markersize": 6.5}
    return [
        *edges,
        Line2D([], [], color=p.removed, lw=KEPT_WIDTH, label="Removed edge"),
        Line2D(
            [],
            [],
            markerfacecolor=p.node,
            markeredgecolor=p.surface,
            markeredgewidth=RING_WIDTH,
            label="Node in backbone",
            **marker,
        ),
        Line2D(
            [],
            [],
            markerfacecolor=p.node_removed,
            markeredgewidth=0,
            label="Node dropped",
            **marker,
        ),
    ]


def draw_backbone(
    ax: Axes,
    original: nx.Graph,
    backbone: nx.Graph,
    pos: Mapping[Any, Any],
    *,
    mode: str = "light",
    sign: str | None = None,
    legend: bool = True,
    title: str | None = None,
    node_size: float = NODE_SIZE,
    kept_width: float = KEPT_WIDTH,
    removed_width: float = REMOVED_WIDTH,
) -> BackboneCounts:
    """Draw *backbone* over *original*: kept edges emphasized, removed faint.

    Parameters
    ----------
    ax : matplotlib Axes
    original, backbone : graph
        The backbone's edges are matched to the original's by endpoints
        (ignoring direction when either graph is undirected).
    pos : mapping
        Node positions for every node of *original*, e.g. from :func:`layout`.
    mode : {"light", "dark"}
    sign : str or None
        Edge attribute whose sign splits kept edges into positive (solid) and
        negative (dashed) for signed backbones.  Read from the backbone edge,
        falling back to the original edge.
    legend : bool
        Add the legend below the axes (small multiples usually add one shared
        figure legend from :func:`legend_handles` instead).
    title : str or None
    node_size, kept_width, removed_width : float
        Marker area (pt^2) and line widths (pt).

    Returns
    -------
    BackboneCounts
        Use ``.caption`` for captions and alt text.
    """
    from matplotlib.collections import LineCollection

    p = palette(mode)
    kept, removed = _split_edges(original, backbone)

    def segments(edges):
        return [(pos[u], pos[v]) for u, v in edges if u != v]

    ax.add_collection(
        LineCollection(
            segments(removed),
            colors=p.removed,
            linewidths=removed_width,
            capstyle="round",
            zorder=1,
        )
    )
    if sign is None:
        groups = [(kept, p.kept, "solid")]
    else:

        def value(u, v):
            data = backbone.get_edge_data(u, v)
            if data is None and not backbone.is_directed():
                data = backbone.get_edge_data(v, u)
            if not data or sign not in data:
                data = original.get_edge_data(u, v) or {}
            return data.get(sign, 0)

        pos_edges = [e for e in kept if value(*e) >= 0]
        neg_edges = [e for e in kept if value(*e) < 0]
        groups = [(pos_edges, p.positive, "solid"), (neg_edges, p.negative, _DASH)]
    for edges, color, style in groups:
        ax.add_collection(
            LineCollection(
                segments(edges),
                colors=color,
                linewidths=kept_width,
                linestyles=style,
                capstyle="round",
                zorder=2,
            )
        )

    in_backbone = {n for e in kept for n in e}
    dropped = [n for n in original if n not in in_backbone]
    shown = [n for n in original if n in in_backbone]
    if dropped:
        xy = [pos[n] for n in dropped]
        ax.scatter(
            [x for x, _ in xy],
            [y for _, y in xy],
            s=node_size * 0.6,
            c=p.node_removed,
            linewidths=0,
            zorder=3,
        )
    if shown:
        xy = [pos[n] for n in shown]
        ax.scatter(
            [x for x, _ in xy],
            [y for _, y in xy],
            s=node_size,
            c=p.node,
            edgecolors=p.surface,
            linewidths=RING_WIDTH,
            zorder=4,
        )

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.04)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(
            handles=legend_handles(mode, signed=sign is not None),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncols=2,
            handlelength=2.4,
            columnspacing=1.6,
        )
    return BackboneCounts(
        edges_kept=len(kept),
        edges_total=len(kept) + len(removed),
        nodes_kept=len(in_backbone),
        nodes_total=original.number_of_nodes(),
    )
