"""Time the shortest-path backbones with and without a NetworkX backend.

The backbone functions are dispatchable under ``backbone_*`` names that no
backend implements yet, so with a backend in ``nx.config.backend_priority``
they still run the NetworkX code; the NetworkX primitives they call
(``all_pairs_dijkstra_path_length``, ``edge_betweenness_centrality``, ...) are
what the backend can take over.  This script times each method both ways on
the seeded Barabási-Albert graph from ``benchmarks/benchmarks/_data.py`` and
checks that the outputs agree.

Usage (nx-parallel is ``pip install nx-parallel``)::

    python scripts/benchmark_backends.py --backend parallel --n 1000

It prints an RST table; the one in ``docs/user_guide/backends.rst`` came
from this script.
"""

import argparse
import math
import os
import platform
import sys
import time

import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, ROOT)

from benchmarks._data import weighted_graph  # noqa: E402

import nx_backbone as nb  # noqa: E402

METHODS = [
    "metric_backbone",
    "edge_betweenness_filter",
    "h_backbone",
    "high_salience_skeleton",
    "glab_filter",
]


def _best_of(func, G, repeat):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        H = func(G)
        best = min(best, time.perf_counter() - start)
    return best, H


def _max_difference(H1, H2):
    """Largest absolute difference between numeric edge attributes."""
    diff = 0.0
    for u, v, d1 in H1.edges(data=True):
        d2 = H2[u][v]
        assert d1.keys() == d2.keys()
        for key, val in d1.items():
            if isinstance(val, bool) or not isinstance(val, int | float):
                assert val == d2[key], (u, v, key)
            else:
                diff = max(diff, abs(val - d2[key]))
    return diff


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="parallel")
    parser.add_argument("--n", type=int, default=1000, help="number of nodes")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    G = weighted_graph(args.n)
    print(
        f"Barabási-Albert graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges; Python {platform.python_version()}, "
        f"NetworkX {nx.__version__}, {os.cpu_count()} CPUs"
    )
    rows = []
    for name in METHODS:
        func = getattr(nb, name)
        nx.config.backend_priority.algos = []
        t_nx, H_nx = _best_of(func, G, args.repeat)
        nx.config.backend_priority.algos = [args.backend]
        try:
            t_be, H_be = _best_of(func, G, args.repeat)
        finally:
            nx.config.backend_priority.algos = []
        diff = _max_difference(H_nx, H_be)
        rows.append((name, t_nx, t_be, t_nx / t_be, diff))

    header = ("Method", "NetworkX (s)", f"{args.backend} (s)", "Ratio", "Max diff")
    table = [header] + [
        (f"``{n}``", f"{a:.2f}", f"{b:.2f}", f"{r:.2f}", f"{d:.1e}")
        for n, a, b, r, d in rows
    ]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    rule = "  ".join("=" * w for w in widths)
    print(rule)
    for i, row in enumerate(table):
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
        if i == 0:
            print(rule)
    print(rule)


if __name__ == "__main__":
    main()
