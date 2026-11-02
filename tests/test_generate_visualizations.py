"""Tests for the method gallery generator used by the docs build."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gv():
    spec = importlib.util.spec_from_file_location(
        "generate_visualizations", ROOT / "scripts" / "generate_visualizations.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def test_every_public_method_has_a_gallery_figure(gv):
    # A new public method fails here (and the docs build) until it gets a
    # MethodSpec in scripts/generate_visualizations.py or a NOT_DRAWN entry.
    specs = gv.method_specs()
    gv.validate_coverage(specs)
    assert len({s.file for s in specs}) == len(specs)
    assert {s.module for s in specs} | {"hypergraph"} == set(gv.MODULES)
    with pytest.raises(RuntimeError, match="missing: \\['gloss_filter'\\]"):
        gv.validate_coverage([s for s in specs if s.name != "gloss_filter"])


def test_specs_build_the_documented_backbones(gv):
    # Spot checks against the counts the issue (#88) and the changelog report.
    pytest.importorskip("scipy")
    ctx = gv.context()
    specs = {s.file: s for s in gv.method_specs()}
    assert specs["ecm_filter"].build(ctx).number_of_edges() == 72
    assert specs["disparity_filter"].build(ctx).number_of_edges() == 9
    signed = specs["fixedrow_signed"].build(ctx)
    assert sorted(d["sign"] for *_, d in signed.edges(data=True)) == [-1] + [1] * 5
