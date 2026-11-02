Convert
=======

Turn a NumPy or SciPy sparse adjacency matrix, or a pandas edge-list
``DataFrame``, into a NetworkX graph. The dispatch wrappers
:func:`~nx_backbone.backbone`,
:func:`~nx_backbone.backbone_from_weighted` and
:func:`~nx_backbone.backbone_from_unweighted` accept these inputs
directly. To go back to a ``DataFrame``, use
:func:`~networkx.convert_matrix.to_pandas_edgelist`.
Bipartite incidence matrices are accepted by the
:doc:`bipartite functions <bipartite>` themselves.

.. automodule:: nx_backbone.convert
   :no-members:

.. currentmodule:: nx_backbone

.. autofunction:: to_graph
