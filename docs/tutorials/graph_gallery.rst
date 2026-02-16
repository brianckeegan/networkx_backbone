Graph Comparison Gallery
========================

This gallery is generated with Sphinx Gallery and applies a strict
score-then-filter workflow before visualizing each method:

1. Score edges on the full graph.
2. Filter scored edges to extract a backbone.
3. Compare the filtered backbone against the original graph.

If a filtered backbone has the same edge count as the original graph,
a validation warning is raised in the example output to prompt re-testing.

.. toctree::
   :maxdepth: 2

   /auto_examples/graph_comparison/index

Static Function Image Reference
-------------------------------

The static snapshots below are sourced from ``docs/_static/graph_gallery/``
and mapped to the same functions used across the API docs and user guide.

Hybrid (Les Miserables)
^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/gallery/hybrid.rst

Proximity (Les Miserables)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/gallery/proximity.rst

Statistical (Les Miserables)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/gallery/statistical.rst

Structural (Les Miserables)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/gallery/structural.rst

Unweighted (Les Miserables)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/gallery/unweighted.rst

Bipartite (Davis Southern Women)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/gallery/bipartite.rst
