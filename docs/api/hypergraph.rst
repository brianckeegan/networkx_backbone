Hypergraph Methods
==================

Methods that operate directly on hypergraphs (collections of arbitrary-size
hyperedges) rather than on dyadic graphs. Inputs are plain iterables of
hyperedges (each hyperedge an iterable of node labels); the backbone is returned
as a list of :class:`frozenset` hyperedges inside a
:class:`~networkx_backbone.HypergraphBackbone` result.

The :func:`~networkx_backbone.mdl_hypergraph_backbone` method implements the
parameter-free, information-theoretic (minimum description length) backbone of
Kirkley, Felippe, Malizia & Battiston (2026), which prunes nested and redundant
hyperedges and naturally extends to weighted hypergraphs.

.. automodule:: networkx_backbone.hypergraph
   :no-members:

.. currentmodule:: networkx_backbone

.. autofunction:: mdl_hypergraph_backbone

.. autofunction:: hypergraph_compression_ratio

.. autofunction:: intersection_graph

.. autoclass:: HypergraphBackbone
   :members:

.. rubric:: Structural methods

Purely structural hypergraph backbones and utilities with no dyadic analog.
These return plain hyperedge collections rather than a
:class:`~networkx_backbone.HypergraphBackbone`.

.. autofunction:: maximal_hyperedges

.. autofunction:: order_filter

.. autofunction:: s_components
