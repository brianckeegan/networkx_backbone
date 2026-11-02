nx-backbone
===========

**Backbone extraction algorithms for complex networks, built on NetworkX.**

``nx-backbone`` provides about 110 functions across 14 modules for
extracting backbone structures from weighted, unweighted, bipartite,
temporal, and higher-order (hypergraph) networks. The public API is
type-annotated, and the graph-in, graph-out functions dispatch to NetworkX
backends (see :doc:`user_guide/backends`).
Core usage follows a score-then-filter pattern: methods score edges on a full
graph first, then filtering utilities extract the final sparse backbone.

.. tip::

   **New here?** After the :doc:`quickstart`, start with
   :doc:`auto_examples/start_here/plot_hairball_to_backbone`, a worked case
   study of the whole workflow: why a backbone, scoring versus filtering,
   choosing alpha and corrections, comparing and validating backbones, and
   reporting them.

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Module
     - Description
     - Key Functions
   * - :doc:`api/statistical`
     - Hypothesis-testing methods
     - :func:`~nx_backbone.disparity_filter`, :func:`~nx_backbone.noise_corrected_filter`
   * - :doc:`api/structural`
     - Topology-based methods
     - :func:`~nx_backbone.high_salience_skeleton`, :func:`~nx_backbone.metric_backbone`
   * - :doc:`api/proximity`
     - Neighborhood-similarity scoring
     - :func:`~nx_backbone.jaccard_backbone`, :func:`~nx_backbone.cosine_backbone`
   * - :doc:`api/hybrid`
     - Combined approaches
     - :func:`~nx_backbone.glab_filter`
   * - :doc:`api/bipartite`
     - Bipartite projection backbones
     - :func:`~nx_backbone.sdsm`, :func:`~nx_backbone.fdsm`, :func:`~nx_backbone.fixedrow`
   * - :doc:`api/hypergraph`
     - Hypergraph backbones and interoperability
     - :func:`~nx_backbone.mdl_hypergraph_backbone`, :func:`~nx_backbone.statistically_validated_hypergraph`, :func:`~nx_backbone.read_hif`
   * - :doc:`api/unweighted`
     - Sparsification for unweighted graphs
     - :func:`~nx_backbone.sparsify`, :func:`~nx_backbone.lspar`
   * - :doc:`api/filters`
     - Post-hoc filtering utilities
     - :func:`~nx_backbone.threshold_filter`, :func:`~nx_backbone.fraction_filter`
   * - :doc:`api/measures`
     - Evaluation and comparison
     - :func:`~nx_backbone.compare_backbones`, :func:`~nx_backbone.edge_fraction`
   * - :doc:`api/visualization`
     - Comparison plotting
     - :func:`~nx_backbone.compare_graphs`, :func:`~nx_backbone.graph_difference`
   * - :doc:`api/describe`
     - Methods text and citations
     - :func:`~nx_backbone.describe_backbone`
   * - :doc:`api/convert`
     - Array and data-frame input
     - :func:`~nx_backbone.to_graph`
   * - :doc:`api/temporal`
     - Temporal networks: snapshots, events, and spans
     - :func:`~nx_backbone.convert_temporal`, :func:`~nx_backbone.flatten_temporal`, :func:`~nx_backbone.from_event_table`

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
