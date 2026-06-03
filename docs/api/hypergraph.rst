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

.. rubric:: Statistical methods

Hypothesis-testing hypergraph backbones that validate hyperedges/groups
recurring more than expected under a null model (Musciotto, Battiston &
Mantegna, 2021).  These require ``scipy`` and return a
:class:`~networkx_backbone.ValidatedHypergraph`.

.. autofunction:: statistically_validated_hypergraph

.. autofunction:: statistically_validated_cores

.. autoclass:: ValidatedHypergraph
   :members:

.. rubric:: Interoperability and ingestion

Convert between the hyperedge-list representation and a NetworkX incidence
bipartite graph (enabling the bipartite projection backbones), the HIF
interchange format, and the ``xgi`` / ``HyperNetX`` / ``HypergraphX`` / HAT
hypergraph classes.  The third-party libraries are optional and imported lazily.

.. autofunction:: hypergraph_to_bipartite

.. autofunction:: read_hif

.. autofunction:: write_hif

.. autofunction:: from_xgi

.. autofunction:: to_xgi

.. autofunction:: from_hypernetx

.. autofunction:: to_hypernetx

.. autofunction:: from_hypergraphx

.. autofunction:: to_hypergraphx

.. autofunction:: from_hat

.. autofunction:: to_hat
