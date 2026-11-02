Proximity Methods
=================

Examples in this module use ``nx.les_miserables_graph()``.
Methods return scored full graphs; typical extraction uses
:func:`~nx_backbone.fraction_filter` on similarity attributes.
Complexity classes are provided in each function docstring.

.. automodule:: nx_backbone.proximity
   :no-members:

.. currentmodule:: nx_backbone

.. autofunction:: neighborhood_overlap

.. autofunction:: jaccard_backbone

.. autofunction:: dice_backbone

.. autofunction:: cosine_backbone

.. autofunction:: hub_promoted_index

.. autofunction:: hub_depressed_index

.. autofunction:: lhn_local_index

.. autofunction:: preferential_attachment_score

.. autofunction:: adamic_adar_index

.. autofunction:: resource_allocation_index

.. autofunction:: graph_distance_proximity

.. autofunction:: local_path_index

.. minigallery::
   nx_backbone.neighborhood_overlap
   nx_backbone.jaccard_backbone
   nx_backbone.dice_backbone
   nx_backbone.cosine_backbone
   nx_backbone.hub_promoted_index
   nx_backbone.hub_depressed_index
   nx_backbone.lhn_local_index
   nx_backbone.preferential_attachment_score
   nx_backbone.adamic_adar_index
   nx_backbone.resource_allocation_index
   nx_backbone.graph_distance_proximity
   nx_backbone.local_path_index
   :add-heading: Gallery Examples

.. rubric:: Method gallery

.. include:: ../_gallery/proximity.inc
