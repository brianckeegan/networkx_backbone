"""Generate the roadmap figures from ``docs/roadmap.rst``.

The roadmap page is the single source of truth: every bullet that starts with a
status tag (``- **done**``, ``- **next**``, ``- **planned**``, ``- **idea**``) is
one item, filed under the nearest section heading.  This module parses those
items and renders, in a light and a dark variant (the docs theme shows the one
matching the reader's mode):

* ``status_by_section-{light,dark}.svg``: a horizontal stacked bar per section,
  segments ordered idea -> planned -> next -> done;
* ``roadmap_map-{light,dark}.svg``: a network of sections (gray hubs) and
  items (dots, labelled with their issue number when they have one) colored by
  status;
* ``status_table.inc``: the same counts as a reStructuredText table, so the figures are never the
  only way to read them.

``docs/conf.py`` calls :func:`build` on every Sphinx build, so the figures
always match the page.  Run ``python scripts/roadmap_figures.py`` to render
them by hand.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROADMAP = ROOT / "docs" / "roadmap.rst"

# Order of progress; the bars and the legend follow it.
STATUSES = ("idea", "planned", "next", "done")

# One-hue ordinal ramp (blue), validated with the dataviz palette checker
# (monotone lightness, visible steps, light end >= 2:1 against the surface).
# On light the most advanced status is darkest; on dark it is lightest.
THEMES = {
    "light": {
        "status": dict(zip(STATUSES, ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b"))),
        "surface": "#fcfcfb",
        "ink": "#1f1f1d",
        "muted": "#6b6a66",
        "hub": "#b4b3ae",
        "edge": "#d9d8d4",
    },
    "dark": {
        "status": dict(zip(STATUSES, ("#184f95", "#2a78d6", "#6da7ec", "#b7d3f6"))),
        "surface": "#1a1a19",
        "ink": "#ecebe7",
        "muted": "#a3a29d",
        "hub": "#5c5b57",
        "edge": "#3a3936",
    },
}

_ITEM = re.compile(r"^- \*\*(done|next|planned|idea)\*\*(.*)$")
_ISSUE = re.compile(r"`#(\d+) <https://github\.com/[^>]+/issues/\d+>`_")
_UNDERLINE = re.compile(r"^([=\-~^])\1{2,}\s*$")


@dataclass(frozen=True)
class Item:
    """One status-tagged roadmap bullet."""

    status: str
    group: str
    section: str
    issue: int | None
    text: str


def parse_roadmap(path=ROADMAP):
    """Return the roadmap items, in page order.

    Parameters
    ----------
    path : path-like, optional
        The roadmap page (reStructuredText).

    Returns
    -------
    items : list of Item
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    levels = []  # heading underline characters in order of first use
    titles = {}
    items = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        m = _UNDERLINE.match(nxt)
        if (
            line.strip()
            and not line.startswith(" ")
            and m
            and len(nxt.strip()) >= len(line.strip())
        ):
            char = m.group(1)
            if char not in levels:
                levels.append(char)
            depth = levels.index(char)
            titles = {d: t for d, t in titles.items() if d < depth}
            titles[depth] = line.strip()
            i += 2
            continue
        m = _ITEM.match(line)
        if m:
            text = m.group(2)
            j = i + 1
            while j < len(lines) and lines[j].startswith("  "):
                text += " " + lines[j].strip()
                j += 1
            issue = _ISSUE.search(text)
            # depth 0 is the page title; 1 the group; 2 the section
            group = titles.get(1, "")
            section = titles.get(2, group)
            items.append(
                Item(
                    status=m.group(1),
                    group=group,
                    section=section,
                    issue=int(issue.group(1)) if issue else None,
                    text=_plain(text),
                )
            )
            i = j
            continue
        i += 1
    return items


def _plain(text):
    text = _ISSUE.sub(lambda m: f"#{m.group(1)}", text)
    text = re.sub(r"^\s*\((#\d+)\)", r"\1", text)
    text = re.sub(r"`([^`<]+?)\s*<[^>]+>`_", r"\1", text)
    text = re.sub(r":\w+:`~?([^`]+)`", r"\1", text)
    text = text.replace("``", "").replace("**", "")
    return re.sub(r"\s+", " ", text).strip(" :()")


def counts_by_section(items):
    """Return ``[(group, section, {status: count})]`` in page order."""
    order, counts = [], {}
    for it in items:
        key = (it.group, it.section)
        if key not in counts:
            order.append(key)
            counts[key] = dict.fromkeys(STATUSES, 0)
        counts[key][it.status] += 1
    return [(g, s, counts[(g, s)]) for g, s in order]


def status_table_rst(items):
    """The status counts as a reStructuredText list-table."""
    rows = counts_by_section(items)
    lines = [
        ".. list-table:: Roadmap items by section and status",
        "   :header-rows: 1",
        "",
        "   * - Section",
        *[f"     - {s.capitalize()}" for s in STATUSES],
        "     - Total",
    ]
    for group, section, c in rows:
        label = section if section == group else f"{group}: {section}"
        lines.append(f"   * - {label}")
        lines += [f"     - {c[s]}" for s in STATUSES]
        lines.append(f"     - {sum(c.values())}")
    totals = {s: sum(c[s] for *_, c in rows) for s in STATUSES}
    lines.append("   * - **All sections**")
    lines += [f"     - **{totals[s]}**" for s in STATUSES]
    lines.append(f"     - **{sum(totals.values())}**")
    return "\n".join(lines) + "\n"


def _style(ax, theme):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["edge"])
    ax.tick_params(colors=theme["muted"], length=0)


def plot_status_by_section(items, path, mode="light"):
    """Horizontal stacked bars: one per section, segments by status."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    theme = THEMES[mode]
    rows = counts_by_section(items)
    labels = [s for _, s, _ in rows]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(8, 0.42 * n + 1.3), dpi=100)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    height = 0.56
    gap = 0.06  # surface gap between segments, in item units
    for y, (_, _, c) in enumerate(rows):
        left = 0.0
        for status in STATUSES:
            k = c[status]
            if not k:
                continue
            ax.add_patch(
                FancyBboxPatch(
                    (left + gap / 2, y - height / 2),
                    k - gap,
                    height,
                    boxstyle="round,pad=0,rounding_size=0.08",
                    linewidth=0,
                    facecolor=theme["status"][status],
                )
            )
            left += k
        ax.text(
            left + 0.15,
            y,
            str(int(left)),
            va="center",
            ha="left",
            fontsize=9,
            color=theme["muted"],
        )
    widest = max(sum(c.values()) for *_, c in rows)
    ax.set_xlim(0, widest + 1)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n), labels, fontsize=9, color=theme["ink"])
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Roadmap items", color=theme["muted"], fontsize=9)
    ax.grid(axis="x", color=theme["edge"], linewidth=0.6)
    ax.set_axisbelow(True)
    _style(ax, theme)
    handles = [
        matplotlib.patches.Patch(facecolor=theme["status"][s], label=s.capitalize())
        for s in STATUSES
    ]
    leg = ax.legend(
        handles=handles,
        ncols=4,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        frameon=False,
        fontsize=9,
        handlelength=1.2,
    )
    for t in leg.get_texts():
        t.set_color(theme["ink"])
    fig.tight_layout()
    fig.savefig(
        path, format="svg", transparent=True, bbox_inches="tight", pad_inches=0.1
    )
    plt.close(fig)


def plot_roadmap_map(items, path, mode="light", seed=7):
    """Network of sections (hubs) and items (dots colored by status)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    theme = THEMES[mode]
    G = nx.Graph()
    for k, it in enumerate(items):
        hub = ("section", it.section)
        G.add_node(hub, kind="section", label=it.section)
        node = ("item", k)
        G.add_node(
            node,
            kind="item",
            status=it.status,
            label=f"#{it.issue}" if it.issue else "",
        )
        G.add_edge(hub, node)
    hubs = [v for v, d in G.nodes(data=True) if d["kind"] == "section"]
    # Deterministic radial layout: sections on an outer circle, each section's
    # items fanned out on the arc facing the center, labels pointing away from
    # their anchor so they do not collide with dots or edges.
    pos, label_pos = {}, {}
    n_hubs = len(hubs)
    for h, hub in enumerate(hubs):
        theta = math.pi / 2 - 2 * math.pi * h / n_hubs
        hx, hy = 1.6 * math.cos(theta), 1.6 * math.sin(theta)
        pos[hub] = (hx, hy)
        label_pos[hub] = (
            hx + 0.16 * math.cos(theta),
            hy + 0.16 * math.sin(theta),
            math.cos(theta),
        )
        members = list(G[hub])
        spread = min(math.radians(200), math.radians(24) * max(len(members) - 1, 1))
        r = 0.3 if len(members) <= 6 else 0.42
        inward = theta + math.pi
        for k, v in enumerate(members):
            phi = (
                inward
                if len(members) == 1
                else inward - spread / 2 + spread * k / (len(members) - 1)
            )
            px, py = hx + r * math.cos(phi), hy + r * math.sin(phi)
            pos[v] = (px, py)
            label_pos[v] = (
                hx + (r + 0.09) * math.cos(phi),
                hy + (r + 0.09) * math.sin(phi),
                math.cos(phi),
            )

    fig, ax = plt.subplots(figsize=(9, 8), dpi=100)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=theme["edge"], width=1.0)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=hubs,
        ax=ax,
        node_size=90,
        node_color=theme["hub"],
        linewidths=0,
    )
    for status in STATUSES:
        nodes = [
            v
            for v, d in G.nodes(data=True)
            if d["kind"] == "item" and d["status"] == status
        ]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            ax=ax,
            node_size=150,
            node_color=theme["status"][status],
            edgecolors=theme["surface"],
            linewidths=2,
            label=status.capitalize(),
        )

    def align(dx):
        return "left" if dx > 0.3 else "right" if dx < -0.3 else "center"

    for v, d in G.nodes(data=True):
        x, y, dx = label_pos[v]
        if d["kind"] == "section":
            ax.text(
                x,
                y,
                d["label"],
                ha=align(dx),
                va="center",
                fontsize=8.5,
                color=theme["ink"],
                fontweight="bold",
                wrap=True,
            )
        elif d["label"]:
            ax.text(
                x,
                y,
                d["label"],
                ha=align(dx),
                va="center",
                fontsize=7,
                color=theme["muted"],
            )
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=4,
        frameon=False,
        fontsize=9,
        markerscale=0.8,
    )
    for t in leg.get_texts():
        t.set_color(theme["ink"])
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.0, 2.0)
    fig.tight_layout()
    fig.savefig(
        path, format="svg", transparent=True, bbox_inches="tight", pad_inches=0.1
    )
    plt.close(fig)


def build(out_dir, roadmap=ROADMAP):
    """Render every figure and the status table into *out_dir*.

    Parameters
    ----------
    out_dir : path-like
        Directory for the SVG files and ``status_table.inc``; created if needed.
    roadmap : path-like, optional
        The roadmap page to parse.

    Returns
    -------
    items : list of Item
        The parsed roadmap items.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = parse_roadmap(roadmap)
    if not items:
        raise ValueError(f"no status-tagged items found in {roadmap}")
    for mode in THEMES:
        plot_status_by_section(items, out / f"status_by_section-{mode}.svg", mode)
        plot_roadmap_map(items, out / f"roadmap_map-{mode}.svg", mode)
    (out / "status_table.inc").write_text(status_table_rst(items), encoding="utf-8")
    return items


if __name__ == "__main__":
    found = build(ROOT / "docs" / "_static" / "roadmap")
    print(f"Rendered roadmap figures for {len(found)} items.")
