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
