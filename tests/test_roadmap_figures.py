"""Tests for the roadmap figure generator used by the docs build."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rf():
    spec = importlib.util.spec_from_file_location(
        "roadmap_figures", ROOT / "scripts" / "roadmap_figures.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


SAMPLE = """\
Roadmap
=======

Intro text with **done** in the middle, which is not an item.

Group A
-------

Section one
~~~~~~~~~~~

- **done** (`#3 <https://github.com/o/r/issues/3>`_): First thing,
  continued on a second line.
- **next**: Second thing, no issue.

Group B
-------

- **idea** (`#7 <https://github.com/o/r/issues/7>`_): A group-level item.
- **planned**: Another one.
"""


def test_parse_sample(rf, tmp_path):
    path = tmp_path / "roadmap.rst"
    path.write_text(SAMPLE, encoding="utf-8")
    items = rf.parse_roadmap(path)
    assert [(i.status, i.group, i.section, i.issue) for i in items] == [
        ("done", "Group A", "Section one", 3),
        ("next", "Group A", "Section one", None),
        ("idea", "Group B", "Group B", 7),
        ("planned", "Group B", "Group B", None),
    ]
    assert items[0].text == "#3: First thing, continued on a second line."


def test_status_table_totals(rf, tmp_path):
    path = tmp_path / "roadmap.rst"
    path.write_text(SAMPLE, encoding="utf-8")
    table = rf.status_table_rst(rf.parse_roadmap(path))
    assert "* - Section one" not in table  # rows are labeled "group: section"
    assert "* - Group A: Section one" in table
    assert table.rstrip().endswith("- **4**")


def test_real_roadmap_parses(rf):
    items = rf.parse_roadmap()
    assert items, "docs/roadmap.rst has no status-tagged items"
    assert {i.status for i in items} <= set(rf.STATUSES)
    assert all(i.section for i in items)


def test_build_writes_all_outputs(rf, tmp_path):
    pytest.importorskip("matplotlib")
    items = rf.build(tmp_path)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "roadmap_map-dark.svg",
        "roadmap_map-light.svg",
        "status_by_section-dark.svg",
        "status_by_section-light.svg",
        "status_table.inc",
    ]
    assert len(items) == len(rf.parse_roadmap())


def test_build_rejects_empty_roadmap(rf, tmp_path):
    empty = tmp_path / "empty.rst"
    empty.write_text("Roadmap\n=======\n\nNothing yet.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no status-tagged items"):
        rf.build(tmp_path / "out", empty)
