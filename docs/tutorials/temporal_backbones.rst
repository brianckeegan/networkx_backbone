Temporal Backbones
==================

This tutorial scores the edges of a temporal network with the descriptive
filters of :mod:`networkx_backbone.temporal`:
:func:`~networkx_backbone.observation_fraction`,
:func:`~networkx_backbone.persistence`,
:func:`~networkx_backbone.backbone_stability` and
:func:`~networkx_backbone.weight_trend`, and the Kalman smoothing of
per-step significance, :func:`~networkx_backbone.kalman_backbone`. Each
returns one static graph, the
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

Smoothing significance over time: Kalman backbone
-------------------------------------------------

The scores above count steps. :func:`~networkx_backbone.kalman_backbone`
instead tests every edge in every step with a static significance test and
pools each edge's evidence over neighboring steps with a Kalman smoother, so
an edge that is moderately significant in several consecutive steps can be
detected where each step alone is too noisy. It needs numpy and SciPy.

A new example: a fixed stochastic block model with three blocks of 10 nodes
(each pair linked with probability 0.3 within a block and 0.05 across), whose
edge weights are drawn afresh in every one of 20 steps from an exponential
distribution with mean 1. Five planted edges inside the first block are
**rising**: their mean weight is 1 up to step 9 and then grows by 1 per
step, to 11 at step 19.

>>> rng = random.Random(7)
>>> pairs = [
...     (u, v)
...     for u, v in combinations(range(30), 2)
...     if rng.random() < (0.3 if u // 10 == v // 10 else 0.05)
... ]
>>> rising = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
>>> pairs = sorted(set(pairs) | set(rising))
>>> sbm = {}
>>> for t in range(20):
...     sbm[t] = nx.Graph()
...     for u, v in pairs:
...         mean = 1 + max(0, t - 9) if (u, v) in rising else 1
...         sbm[t].add_edge(u, v, weight=rng.expovariate(1 / mean))
>>> len(pairs)
66

A helper counts the rising edges a backbone finds, and the other edges it
keeps:

>>> def found(edges):
...     edges = {tuple(sorted(e)) for e in edges}
...     return len(edges & set(rising)), len(edges - set(rising))

The disparity filter on single snapshots finds some of the rising edges in
the late steps, together with several other edges:

>>> for t in (11, 15, 19):
...     D = nb.disparity_filter(sbm[t])
...     print(t, found(nb.threshold_filter(D, "disparity_pvalue", 0.05).edges))
11 (2, 5)
15 (4, 2)
19 (2, 7)

:func:`~networkx_backbone.kalman_backbone` runs the same filter on every
step, turns each p-value into a probit score, and smooths each edge's series.
``half_life=3`` sets how far the smoother reaches: a step's weight halves
every three steps away. ``kalman_snapshots`` expands the per-step p-values
into one graph per step, to be cut like a single snapshot:

>>> K = nb.kalman_backbone(sbm, "disparity", half_life=3, seed=0)
>>> steps = nb.kalman_snapshots(K)
>>> for t in (11, 15, 19):
...     print(t, found(nb.threshold_filter(steps[t], "kalman_pvalue", 0.05).edges))
11 (4, 0)
15 (5, 1)
19 (5, 2)

It finds four rising edges by step 11 and all five from step 15, and keeps
no more of the other 61 edges than a 5% test should (about three per step).
Each edge also gets one p-value across all steps, ``kalman_pvalue``, which
tests its largest smoothed score. It is one test per edge, so a correction
across the 66 edges is enough, and it keeps exactly the rising edges:

>>> KB = nb.threshold_filter(K, "kalman_pvalue", 0.05, mtc="bh")
>>> found(KB.edges)
(5, 0)

The disparity filter gives an edge the smaller of its two endpoints'
p-values. ``kalman_backbone`` corrects that by default (``sidak=True``), which
makes its tests on top of the disparity filter somewhat conservative;
:func:`~networkx_backbone.noise_corrected_filter` tests each edge once.

Three cautions apply (see the Notes of
:func:`~networkx_backbone.kalman_backbone`). The p-values are exact when no
step carries signal, but a signal in some steps also raises the smoothed
score in nearby steps (temporal leakage; ``mode="filter"`` uses only past
steps). Confounders that persist over time make the tests anti-conservative,
so check that each edge's ``kalman_innovations`` look like noise. And the
per-step p-values are many tests, one per edge and step: correct them within
each step, or use the edge-level ``kalman_pvalue``.

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
