Temporal
========

Temporal networks as snapshots, timestamped events, or spans, and
conversions between the three forms. The :doc:`../user_guide/temporal` guide
walks through every conversion with examples and a table of what each one
keeps. The temporal backbone methods themselves are planned (see the
:doc:`../roadmap`).

.. automodule:: networkx_backbone.temporal
   :no-members:

.. currentmodule:: networkx_backbone

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
