networkx-backbone
=================

**Backbone extraction algorithms for complex networks, built on NetworkX.**

``networkx-backbone`` provides about 110 functions across 14 modules for
extracting backbone structures from weighted, unweighted, bipartite,
temporal, and higher-order (hypergraph) networks. The public API is
type-annotated, and the graph-in, graph-out functions dispatch to NetworkX
backends (see :doc:`user_guide/backends`).
Core usage follows a score-then-filter pattern: methods score edges on a full
graph first, then filtering utilities extract the final sparse backbone.

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Module
     - Description
     - Key Functions
   * - :doc:`api/statistical`
     - Hypothesis-testing methods
     - :func:`~networkx_backbone.disparity_filter`, :func:`~networkx_backbone.noise_corrected_filter`
   * - :doc:`api/structural`
     - Topology-based methods
     - :func:`~networkx_backbone.high_salience_skeleton`, :func:`~networkx_backbone.metric_backbone`
   * - :doc:`api/proximity`
     - Neighborhood-similarity scoring
     - :func:`~networkx_backbone.jaccard_backbone`, :func:`~networkx_backbone.cosine_backbone`
   * - :doc:`api/hybrid`
     - Combined approaches
     - :func:`~networkx_backbone.glab_filter`
   * - :doc:`api/bipartite`
     - Bipartite projection backbones
     - :func:`~networkx_backbone.sdsm`, :func:`~networkx_backbone.fdsm`, :func:`~networkx_backbone.fixedrow`
   * - :doc:`api/hypergraph`
     - Hypergraph backbones and interoperability
     - :func:`~networkx_backbone.mdl_hypergraph_backbone`, :func:`~networkx_backbone.statistically_validated_hypergraph`, :func:`~networkx_backbone.read_hif`
   * - :doc:`api/unweighted`
     - Sparsification for unweighted graphs
     - :func:`~networkx_backbone.sparsify`, :func:`~networkx_backbone.lspar`
   * - :doc:`api/filters`
     - Post-hoc filtering utilities
     - :func:`~networkx_backbone.threshold_filter`, :func:`~networkx_backbone.fraction_filter`
   * - :doc:`api/measures`
     - Evaluation and comparison
     - :func:`~networkx_backbone.compare_backbones`, :func:`~networkx_backbone.edge_fraction`
   * - :doc:`api/visualization`
     - Comparison plotting
     - :func:`~networkx_backbone.compare_graphs`, :func:`~networkx_backbone.graph_difference`
   * - :doc:`api/describe`
     - Methods text and citations
     - :func:`~networkx_backbone.describe_backbone`
   * - :doc:`api/convert`
     - Array and data-frame input
     - :func:`~networkx_backbone.to_graph`
   * - :doc:`api/temporal`
     - Temporal networks: snapshots, events, and spans
     - :func:`~networkx_backbone.convert_temporal`, :func:`~networkx_backbone.flatten_temporal`, :func:`~networkx_backbone.from_event_table`

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install
   quickstart
   concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog
   roadmap
   contributing
