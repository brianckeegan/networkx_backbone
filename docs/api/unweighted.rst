Unweighted Sparsification
=========================

Examples in this module use the unweighted version of
``nx.les_miserables_graph()``.
Methods return scored full graphs with ``sparsify_score`` and
``sparsify_keep``; apply :func:`~nx_backbone.boolean_filter` to
extract sparse backbones.
Complexity classes are provided in each function docstring.

.. automodule:: nx_backbone.unweighted
   :no-members:

.. currentmodule:: nx_backbone

.. autofunction:: sparsify

.. autofunction:: lspar

.. autofunction:: local_degree

.. minigallery::
   nx_backbone.sparsify
   nx_backbone.lspar
   nx_backbone.local_degree
   :add-heading: Gallery Examples

.. rubric:: Method gallery

.. include:: ../_gallery/unweighted.inc
