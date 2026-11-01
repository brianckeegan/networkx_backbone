"""Tests for the shared docs figure style (docs/_figures/style.py)."""

import sys
from pathlib import Path

import networkx as nx
import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"


@pytest.fixture(scope="module")
def style():
    sys.path.insert(0, str(DOCS))
    try:
        from _figures import style

        yield style
    finally:
        sys.path.remove(str(DOCS))


def test_palettes_are_the_validated_values(style):
    # These are the values validate_palette.js passed (see the module
    # docstring for its output); change them only after re-running it.
    light, dark = style.palette("light"), style.palette("dark")
    assert (light.surface, dark.surface) == ("#ffffff", "#14181e")
    assert light.categorical == (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    )
    assert dark.categorical == (
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    )
    assert light.sequential == ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
    assert dark.sequential == ("#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4")
    assert (light.kept, light.positive, light.negative) == (
        "#2a78d6",
        "#2a78d6",
        "#e34948",
    )
    assert (dark.kept, dark.positive, dark.negative) == (
        "#3987e5",
        "#3987e5",
        "#e66767",
    )
    with pytest.raises(ValueError, match="mode"):
        style.palette("sepia")


def test_layout_is_cached_per_dataset(style):
    G = nx.path_graph(5)
    pos = style.layout(G, "test-path")
    assert pos == style.layout(G, "test-path")
    with pytest.raises(ValueError, match="different nodes"):
        style.layout(nx.path_graph(6), "test-path")


def test_save_variants_writes_light_and_dark(style, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    G = nx.path_graph(4)
    backbone = nx.Graph([(0, 1), (1, 2)])
    pos = style.layout(G, "test-save")
    counts = []

    def draw(mode):
        fig, ax = plt.subplots()
        counts.append(style.draw_backbone(ax, G, backbone, pos, mode=mode))
        return fig

    paths = style.save_variants(draw, tmp_path / "fig")
    assert sorted(p.name for p in paths.values()) == ["fig-dark.svg", "fig-light.svg"]
    assert all(p.stat().st_size > 0 for p in paths.values())
    assert counts[0].caption == "Kept 2 of 3 edges (67%) and 3 of 4 nodes."
    rst = "\n".join(style.figure_rst("/_static/fig", "Alt text."))
    assert ":class: only-light" in rst and "fig-dark.svg" in rst
