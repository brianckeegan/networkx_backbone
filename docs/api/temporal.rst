Temporal
========

Temporal networks as snapshots, timestamped events, or spans, conversions
between the three forms, and temporal backbone scores and tests. The
:doc:`../user_guide/temporal` guide walks through every conversion with
examples and a table of what each one keeps; the
:doc:`../tutorials/temporal_backbones` tutorial compares the scores on
synthetic snapshots, and :ref:`temporal-eligibility` explains how absent
edges are counted.

.. automodule:: networkx_backbone.temporal
   :no-members:

.. currentmodule:: networkx_backbone

.. rubric:: Temporal backbone scores

Each takes any of the three forms (``freq=`` and ``origin=`` bin events and
spans into steps) and returns one scored static graph, filtered with
:func:`~networkx_backbone.threshold_filter` and the other filters. These
scores are descriptive: they come without p-values. They are not dispatchable to
NetworkX backends, because they take several graphs or a multigraph of events
rather than one graph.

.. autofunction:: observation_fraction

.. autofunction:: persistence

.. autofunction:: backbone_stability

.. autofunction:: weight_trend

.. rubric:: Model-based temporal scores and tests

These fit a model to every edge's series of observations and need numpy.
:func:`~networkx_backbone.edge_persistence_hmm` gives posterior activity
probabilities, which are smoothed membership scores, not p-values.
:func:`~networkx_backbone.kalman_backbone` (which also needs SciPy) tests
every edge in every step and returns p-values.

.. autofunction:: edge_persistence_hmm

.. autofunction:: kalman_backbone

.. autofunction:: kalman_snapshots

.. rubric:: Forms

.. autofunction:: temporal_form

.. autofunction:: convert_temporal

.. rubric:: Collapsing to a static graph

.. autofunction:: flatten_temporal

.. rubric:: Binning events and spans into snapshots

.. autofunction:: snapshots_from_events

.. autofunction:: snapshots_from_spans

.. rubric:: Snapshots to events and spans

.. autofunction:: events_from_snapshots

.. autofunction:: spans_from_snapshots

.. rubric:: Between events and spans

.. autofunction:: events_from_spans

.. autofunction:: spans_from_events

.. rubric:: pandas event tables

.. autofunction:: from_event_table

.. autofunction:: to_event_table
