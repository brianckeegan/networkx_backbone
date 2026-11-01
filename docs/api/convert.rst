Convert
=======

Turn a NumPy or SciPy sparse adjacency matrix, or a pandas edge-list
``DataFrame``, into a NetworkX graph. The dispatch wrappers
:func:`~networkx_backbone.backbone`,
:func:`~networkx_backbone.backbone_from_weighted` and
:func:`~networkx_backbone.backbone_from_unweighted` accept these inputs
directly. To go back to a ``DataFrame``, use
:func:`~networkx.convert_matrix.to_pandas_edgelist`.
Bipartite incidence matrices are accepted by the
:doc:`bipartite functions <bipartite>` themselves.

.. automodule:: networkx_backbone.convert
   :no-members:

.. currentmodule:: networkx_backbone

.. autofunction:: to_graph
