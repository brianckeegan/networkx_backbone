Unweighted Sparsification
=========================

Examples in this module use the unweighted version of
``nx.les_miserables_graph()``.
Methods return scored full graphs with ``sparsify_score`` and
``sparsify_keep``; apply :func:`~networkx_backbone.boolean_filter` to
extract sparse backbones.
Complexity classes are provided in each function docstring.

.. automodule:: networkx_backbone.unweighted
   :no-members:

.. currentmodule:: networkx_backbone

.. autofunction:: sparsify

.. autofunction:: lspar

.. autofunction:: local_degree

.. minigallery::
   networkx_backbone.sparsify
   networkx_backbone.lspar
   networkx_backbone.local_degree
   :add-heading: Gallery Examples
