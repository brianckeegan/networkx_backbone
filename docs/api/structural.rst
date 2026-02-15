Structural Methods
==================

Examples in this module use ``nx.les_miserables_graph()``.
These functions return scored full graphs. Extract final backbones with
:func:`~networkx_backbone.boolean_filter` on ``*_keep`` flags (or
:func:`~networkx_backbone.threshold_filter` for continuous scores such as
``salience`` and ``ds_weight``).
Complexity classes are provided in each function's ``Complexity`` section.

.. automodule:: networkx_backbone.structural
   :no-members:

.. currentmodule:: networkx_backbone

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
   networkx_backbone.global_threshold_filter
   networkx_backbone.strongest_n_ties
   networkx_backbone.global_sparsification
   networkx_backbone.primary_linkage_analysis
   networkx_backbone.edge_betweenness_filter
   networkx_backbone.node_degree_filter
   networkx_backbone.high_salience_skeleton
   networkx_backbone.metric_backbone
   networkx_backbone.ultrametric_backbone
   networkx_backbone.doubly_stochastic_filter
   networkx_backbone.h_backbone
   networkx_backbone.modularity_backbone
   networkx_backbone.planar_maximally_filtered_graph
   networkx_backbone.maximum_spanning_tree_backbone
   :add-heading: Gallery Examples
