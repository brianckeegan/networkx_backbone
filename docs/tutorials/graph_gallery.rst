Method Gallery
==============

Every method in this package, drawn the same way: the backbone over its
original graph, with kept edges in blue and removed edges in faint gray. All
figures of a dataset share one layout, so a node sits in the same place in
every figure and the methods can be compared at a glance. The figures are
generated when the documentation is built, so they always show what the
current code returns. Each caption gives the edges and nodes the backbone
keeps.

Side by side
------------

The executed example below runs the statistical and structural methods on
Les Misérables and draws them as small multiples, with a bar chart of the
edges each one keeps.

.. toctree::
   :maxdepth: 2

   /auto_examples/graph_comparison/index

One figure per method
---------------------

The full-size figure for each method is on its API page, next to the
function's documentation:

- :doc:`../api/statistical` (Les Misérables)
- :doc:`../api/structural` (Les Misérables)
- :doc:`../api/proximity` (Les Misérables)
- :doc:`../api/hybrid` (Les Misérables)
- :doc:`../api/unweighted` (unweighted Les Misérables)
- :doc:`../api/bipartite` (Davis Southern Women, including a signed backbone)
- :doc:`../api/hypergraph` (Davis Southern Women as a hypergraph)
- :doc:`../api/temporal` (the synthetic snapshots of
  :doc:`temporal_backbones`)

The figures are drawn by ``scripts/generate_visualizations.py`` with the
shared figure style in ``docs/_figures/style.py``.
