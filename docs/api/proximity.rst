Proximity Methods
=================

Examples in this module use ``nx.les_miserables_graph()``.
Methods return scored full graphs; typical extraction uses
:func:`~networkx_backbone.fraction_filter` on similarity attributes.
Complexity classes are provided in each function docstring.

.. automodule:: networkx_backbone.proximity
   :no-members:

.. currentmodule:: networkx_backbone

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
   networkx_backbone.neighborhood_overlap
   networkx_backbone.jaccard_backbone
   networkx_backbone.dice_backbone
   networkx_backbone.cosine_backbone
   networkx_backbone.hub_promoted_index
   networkx_backbone.hub_depressed_index
   networkx_backbone.lhn_local_index
   networkx_backbone.preferential_attachment_score
   networkx_backbone.adamic_adar_index
   networkx_backbone.resource_allocation_index
   networkx_backbone.graph_distance_proximity
   networkx_backbone.local_path_index
   :add-heading: Gallery Examples
