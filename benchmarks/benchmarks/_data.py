"""Seeded synthetic inputs shared by the benchmarks.

Every generator is deterministic, so timings compare like with like across
commits.  NumPy is used only to draw the inputs.
"""

import random

import networkx as nx
import numpy as np


def weighted_graph(n, m=3, max_weight=20, seed=0):
    """Barabási-Albert graph with integer weights in ``[1, max_weight]``.

    With ``n=2000`` and ``m=3`` it has 2000 nodes and 5991 edges, with the
    heavy-tailed degrees the statistical scorers are designed for.
    """
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, max_weight + 1, size=G.number_of_edges())
    for (u, v), w in zip(G.edges(), weights):
        G[u][v]["weight"] = int(w)
    return G


def bipartite_graph(n_agents, n_artifacts, density=0.05, seed=0):
    """Random bipartite graph with heterogeneous agent and artifact degrees.

    Edge ``(i, k)`` is present with probability proportional to
    ``a[i] * b[k]`` (log-normal propensities), scaled to the given density.

    Returns
    -------
    B : networkx.Graph
        Agents are ``("a", i)`` and artifacts ``("f", k)``.
    agents : list
        The agent nodes.
    """
    rng = np.random.default_rng(seed)
    a = rng.lognormal(0.0, 0.5, n_agents)
    b = rng.lognormal(0.0, 0.5, n_artifacts)
    p = np.outer(a, b)
    p *= density / p.mean()
    rows, cols = np.nonzero(rng.random(p.shape) < np.clip(p, 0.0, 1.0))
    agents = [("a", i) for i in range(n_agents)]
    B = nx.Graph()
    B.add_nodes_from(agents, bipartite=0)
    B.add_nodes_from((("f", k) for k in range(n_artifacts)), bipartite=1)
    B.add_edges_from((("a", int(i)), ("f", int(k))) for i, k in zip(rows, cols))
    return B, agents


def nested_hypergraph(m, seed=0):
    """``m`` hyperedges with the nested structure higher-order data shows.

    A quarter of the hyperedges are "parents" of size 4-10 over ``m // 2``
    nodes; the rest are random sub-groups of a parent (size >= 2), with one
    node swapped for a random node 20% of the time.
    """
    rng = random.Random(seed)
    nodes = range(max(m // 2, 12))
    n_parents = max(m // 4, 1)
    parents = [rng.sample(nodes, rng.randint(4, 10)) for _ in range(n_parents)]
    hyperedges = [list(p) for p in parents]
    while len(hyperedges) < m:
        parent = rng.choice(parents)
        child = rng.sample(parent, rng.randint(2, len(parent) - 1))
        if rng.random() < 0.2:
            child[0] = rng.choice(nodes)
        hyperedges.append(child)
    return hyperedges


def group_hypergraph(m, n_nodes, max_size=8, seed=0):
    """``m`` hyperedges of size 2 to *max_size* with planted groups.

    Half the hyperedges are sub-groups of 20 planted groups of *max_size*
    nodes, so the statistically validated methods have significant groups to
    find (and SVC many sub-groups to enumerate); the rest are uniform noise
    over ``n_nodes`` nodes.
    """
    rng = random.Random(seed)
    nodes = range(n_nodes)
    groups = [rng.sample(nodes, max_size) for _ in range(20)]
    hyperedges = []
    for i in range(m):
        source = rng.choice(groups) if i % 2 == 0 else nodes
        hyperedges.append(rng.sample(source, rng.randint(2, max_size)))
    return hyperedges
