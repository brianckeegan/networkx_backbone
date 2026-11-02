Structural Methods
==================

Examples in this module use ``nx.les_miserables_graph()``.
These functions return scored full graphs. Extract final backbones with
:func:`~nx_backbone.boolean_filter` on ``*_keep`` flags (or
:func:`~nx_backbone.threshold_filter` for continuous scores such as
``salience`` and ``ds_weight``).
Complexity classes are provided in each function docstring.

.. automodule:: nx_backbone.structural
   :no-members:

.. currentmodule:: nx_backbone

.. autofunction:: global_threshold_filter

.. autofunction:: strongest_n_ties

.. autofunction:: global_sparsification

.. autofunction:: primary_linkage_analysis

.. autofunction:: edge_betweenness_filter

.. autofunction:: node_degree_filter

.. autofunction:: high_salience_skeleton

.. autofunction:: metric_backbone

.. autofunction:: ultrametric_backbone

.. autofunction:: doubly_stochastic_filter

.. autofunction:: h_backbone

.. autofunction:: modularity_backbone

.. autofunction:: planar_maximally_filtered_graph

.. autofunction:: maximum_spanning_tree_backbone

.. minigallery::
   nx_backbone.global_threshold_filter
   nx_backbone.strongest_n_ties
   nx_backbone.global_sparsification
   nx_backbone.primary_linkage_analysis
   nx_backbone.edge_betweenness_filter
   nx_backbone.node_degree_filter
   nx_backbone.high_salience_skeleton
   nx_backbone.metric_backbone
   nx_backbone.ultrametric_backbone
   nx_backbone.doubly_stochastic_filter
   nx_backbone.h_backbone
   nx_backbone.modularity_backbone
   nx_backbone.planar_maximally_filtered_graph
   nx_backbone.maximum_spanning_tree_backbone
   :add-heading: Gallery Examples

.. rubric:: Method gallery

.. include:: ../_gallery/structural.inc
