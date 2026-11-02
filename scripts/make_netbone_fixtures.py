"""Generate ``tests/fixtures/netbone.json`` from the netbone package.

netbone (Yassin et al., 2023; https://gitlab.liris.cnrs.fr/coregraphie/netbone)
is the Python reference implementation users compare nx-backbone with.
This script runs the netbone methods that nx-backbone shares on the
weighted Les Misérables and karate club networks and stores, for every edge,
netbone's score and (for the structural methods) its ``in_backbone`` flag.

netbone's ``doubly_stochastic`` is not included: it Sinkhorn-scales a
one-directional pivot of the edge list with 1e-4 added to every cell, which
is a different matrix from the symmetric adjacency matrix that
:func:`nx_backbone.doubly_stochastic_filter` scales, so the scores are
not comparable.  netbone's ``noise_corrected(approximation=True)`` binomial
p-value is not compared either: it uses W trials and the strict tail
P(X > w), unlike ``noise_corrected_filter``'s default ``nc_pvalue`` (see its
Notes); its ``approximation=False`` score and standard deviation are compared.
``tests/test_reference_implementations.py`` compares against the fixture, so
the test suite does not need netbone installed.

netbone pins old versions of its dependencies (NumPy 1.26, NetworkX 2.8,
SciPy 1.13, JAX 0.4), so run it in a separate Python 3.12 environment, from
the repository root::

    python scripts/export_reference_inputs.py        # refresh the CSV inputs
    python3.12 -m venv /tmp/netbone-env
    /tmp/netbone-env/bin/pip install netbone==0.2.3
    /tmp/netbone-env/bin/python -I scripts/make_netbone_fixtures.py

The script reads only the CSV files in ``tests/fixtures/reference_inputs/``
and does not import nx-backbone.
"""

import csv
import importlib.metadata as md
import json
import math
import platform
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "tests" / "fixtures" / "reference_inputs"
OUT = ROOT / "tests" / "fixtures" / "netbone.json"

GLAB_C = (0.5, 1.0)


def read_graph(name, node_type):
    """Weighted graph from a reference CSV, plus its edges in file order."""
    G = nx.Graph()
    edges = []
    with open(INPUTS / f"{name}.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            u, v = node_type(row["source"]), node_type(row["target"])
            G.add_edge(u, v, weight=float(row["weight"]))
            edges.append((u, v))
    return G, edges


def finite(x):
    """JSON-safe float (``None`` for NaN/inf, as JSON has no such values)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def edge_attr(graph, edges, attr, default=None):
    """Edge attribute in *edges* order (``default`` when absent)."""
    out = []
    for u, v in edges:
        data = graph.get_edge_data(u, v)
        val = default if data is None else data.get(attr, default)
        out.append(val if isinstance(val, bool) else finite(val))
    return out


def run(G, edges):
    import netbone as nb

    section = {
        "nodes": [str(n) for n in G.nodes()],
        "edges": [[str(u), str(v)] for u, v in edges],
    }

    def store(key, backbone, attrs):
        section[key] = {
            name: edge_attr(backbone.graph, edges, attr, default)
            for name, (attr, default) in attrs.items()
        }

    store("disparity", nb.disparity(G), {"p_value": ("p_value", None)})
    store("lans", nb.lans(G), {"p_value": ("p_value", None)})
    store(
        "noise_corrected",
        nb.noise_corrected(G, approximation=False),
        {"score": ("score", None), "sdev": ("nc_sdev", None)},
    )
    store(
        "marginal_likelihood", nb.marginal_likelihood(G), {"p_value": ("p_value", None)}
    )
    store("ecm", nb.ecm(G), {"p_value": ("p_value", None)})
    for c in GLAB_C:
        store(f"glanb_c{c}", nb.glanb(G, c=c), {"SI": ("SI", None)})
    store(
        "high_salience_skeleton",
        nb.high_salience_skeleton(G),
        {"salience": ("salience", None)},
    )
    store(
        "h_backbone",
        nb.h_backbone(G),
        {"h_bridge": ("h_bridge", None), "in_backbone": ("in_backbone", False)},
    )
    store(
        "metric_distance_backbone",
        nb.metric_distance_backbone(G),
        {"in_backbone": ("in_backbone", False)},
    )
    store(
        "ultrametric_distance_backbone",
        nb.ultrametric_distance_backbone(G),
        {"in_backbone": ("in_backbone", False)},
    )
    store("mlam", nb.mlam(G), {"in_backbone": ("in_backbone", False)})
    return section


def main():
    les, les_edges = read_graph("les_miserables", str)
    karate, karate_edges = read_graph("karate", int)
    fixture = {
        "meta": {
            "generator": "scripts/make_netbone_fixtures.py",
            "python": platform.python_version(),
            "netbone": md.version("netbone"),
            "networkx": md.version("networkx"),
            "numpy": md.version("numpy"),
            "scipy": md.version("scipy"),
            "glab_c": list(GLAB_C),
        },
        "les_miserables": run(les, les_edges),
        "karate": run(karate, karate_edges),
    }
    OUT.write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
