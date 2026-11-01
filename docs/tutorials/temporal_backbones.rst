Temporal Backbones
==================

This tutorial scores the edges of a temporal network with the descriptive
filters of :mod:`networkx_backbone.temporal`:
:func:`~networkx_backbone.observation_fraction`,
:func:`~networkx_backbone.persistence`,
:func:`~networkx_backbone.backbone_stability` and
:func:`~networkx_backbone.weight_trend`. Each returns one static graph, the
union of the observed edges with score attributes, so the usual filters such
as :func:`~networkx_backbone.threshold_filter` pick the backbone. See
:doc:`../user_guide/temporal` for the three input forms (snapshots, events and
spans) and the conversions between them, and :ref:`temporal-eligibility` for
how absent edges are counted.

.. contents:: On this page
   :local:
   :depth: 1

Synthetic snapshots
-------------------

Twenty snapshots of a 30-node network hold three kinds of edges:

- a **persistent block**: five heavy edges, each active in one block of 10
  consecutive steps;
- **flicker**: five heavy edges, each active in 10 steps chosen at random;
- **noise**: light edges (weight 1), each pair active with probability 0.1 in
  every step.

All 30 nodes are present in every step, so every edge is eligible in all 20
steps.

>>> import random
>>> from itertools import combinations
>>> import networkx as nx
>>> import networkx_backbone as nb
>>> rng = random.Random(42)
>>> block = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
>>> flicker = [(10, 11), (12, 13), (14, 15), (16, 17), (18, 19)]
>>> snaps = {t: nx.Graph() for t in range(20)}
>>> for G in snaps.values():
...     G.add_nodes_from(range(30))
>>> for u, v in block:
...     first = rng.randrange(11)
...     for t in range(first, first + 10):
...         snaps[t].add_edge(u, v, weight=10)
>>> for u, v in flicker:
...     for t in rng.sample(range(20), 10):
...         snaps[t].add_edge(u, v, weight=10)
>>> for G in snaps.values():
...     for u, v in combinations(range(30), 2):
...         if (u, v) not in block + flicker and rng.random() < 0.1:
...             G.add_edge(u, v, weight=1)

A small helper averages a score over a group of edges, skipping NaN:

>>> def mean(H, edges, score):
...     values = [H.edges[e][score] for e in edges]
...     values = [x for x in values if x == x]
...     return round(sum(values) / len(values), 2)

How often: observation fraction
-------------------------------

:func:`~networkx_backbone.observation_fraction` divides the number of steps in
which an edge is observed by its number of eligible steps:

>>> H = nb.observation_fraction(snaps)
>>> H.number_of_edges()
382
>>> noise = [e for e in H.edges if e not in block + flicker]
>>> mean(H, block, "obs_fraction"), mean(H, flicker, "obs_fraction")
(0.5, 0.5)
>>> mean(H, noise, "obs_fraction")
0.11

Both heavy groups are observed in half of the steps, so the observation
fraction separates them from the noise but not from each other.

How steadily: persistence
-------------------------

:func:`~networkx_backbone.persistence` looks at runs of consecutive observed
steps. Its ``persistence`` score is the share of an edge's observed steps
whose next step is also observed:

>>> P = nb.persistence(snaps)
>>> mean(P, block, "persistence"), mean(P, flicker, "persistence")
(0.92, 0.49)
>>> mean(P, block, "max_run"), mean(P, flicker, "max_run")
(10.0, 3.8)

The block edges stay on once they appear; the flicker edges switch on and
off about as often as a coin. Filtering on ``persistence`` keeps the block:

>>> B = nb.threshold_filter(P, "persistence", 0.8, mode="above",
...                         include_all_nodes=False)
>>> sorted(tuple(sorted(e)) for e in B.edges)
[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

How often a backbone keeps an edge: backbone stability
------------------------------------------------------

A static backbone of each snapshot is noisy. The disparity filter at
``alpha = 0.05`` keeps between 2 and 8 edges per step:

>>> per_step = [
...     nb.threshold_filter(nb.disparity_filter(G), "disparity_pvalue", 0.05)
...     for G in snaps.values()
... ]
>>> sizes = [bb.number_of_edges() for bb in per_step]
>>> min(sizes), max(sizes)
(2, 8)

:func:`~networkx_backbone.backbone_stability` runs the same method on every
snapshot with the same cut and scores each edge by the share of its eligible
steps in which the backbone keeps it:

>>> S = nb.backbone_stability(snaps, nb.disparity_filter, threshold=0.05)
>>> mean(S, block, "stability"), mean(S, flicker, "stability")
(0.47, 0.48)
>>> max(S.edges[e]["stability"] for e in noise)
0.0

Keeping the edges that are in the backbone in at least 30% of their steps
recovers all ten heavy edges and no noise:

>>> B = nb.threshold_filter(S, "stability", 0.3, mode="above",
...                         include_all_nodes=False)
>>> sorted(tuple(sorted(e)) for e in B.edges) == sorted(block + flicker)
True

Like ``obs_fraction``, stability counts steps, so it does not tell the block
from the flicker; combine it with ``persistence`` for that. Where
:func:`~networkx_backbone.consensus_backbone` intersects the backbones of
several methods, ``backbone_stability`` aggregates one method over time.

Weight trends
-------------

:func:`~networkx_backbone.weight_trend` (which needs numpy) summarizes each
edge's weight series over its eligible steps, with weight 0 where the edge is
absent. The block edges switch on and off once, the flicker edges many times,
but both have the same mean and spread:

>>> W = nb.weight_trend(snaps)
>>> W.edges[0, 1]["weight_mean"], W.edges[0, 1]["weight_std"]
(5.0, 5.0)
>>> W.edges[10, 11]["weight_mean"], W.edges[10, 11]["weight_std"]
(5.0, 5.0)

``weight_slope``, the least-squares slope against the step number, tells an
edge that fades out from one that builds up; ``weight_cv`` is the coefficient
of variation.

Describing the result
---------------------

The scored graph records the steps and parameters in ``H.graph["temporal"]``,
which :func:`~networkx_backbone.describe_backbone` reads:

>>> S.graph["temporal"]["n_steps"], S.graph["temporal"]["input_form"]
(20, 'snapshots')
>>> print(nb.describe_backbone("backbone_stability", original=S, backbone=B))
... # doctest: +ELLIPSIS
We scored the edges of the temporal network with backbone stability, ...
The network was observed in 20 time steps, given as snapshots. ... In every
step the edges were scored with the disparity filter (Serrano et al., 2009),
and the same cut was applied in each step. The backbone retains 10 of 382
edges (2.6%) and 20 of 30 non-isolated nodes (66.7%). ...
