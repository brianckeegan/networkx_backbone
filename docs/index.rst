networkx-backbone
=================

**Backbone extraction algorithms for complex networks, built on NetworkX.**

``networkx-backbone`` provides 47 functions across 8 modules for extracting
backbone structures from weighted and unweighted networks.

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
     - :func:`~networkx_backbone.sdsm`, :func:`~networkx_backbone.fdsm`
   * - :doc:`api/unweighted`
     - Sparsification for unweighted graphs
     - :func:`~networkx_backbone.sparsify`, :func:`~networkx_backbone.lspar`
   * - :doc:`api/filters`
     - Post-hoc filtering utilities
     - :func:`~networkx_backbone.threshold_filter`, :func:`~networkx_backbone.fraction_filter`
   * - :doc:`api/measures`
     - Evaluation and comparison
     - :func:`~networkx_backbone.compare_backbones`, :func:`~networkx_backbone.edge_fraction`

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
   contributing
