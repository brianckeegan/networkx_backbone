Tutorials
=========

The narrative tutorials explain *why* a method behaves as it does, with
figures computed when the documentation is built.  The how-to tutorials show
the calls, method by method.

Start here
----------

New to backbones? Read this worked case study first: it takes two weighted
networks from a tangled drawing to a reported backbone, explaining why each
step is needed (scoring and filtering, choosing alpha and corrections,
comparing methods, validating, reporting), with figures computed when the
docs are built.

.. toctree::
   :maxdepth: 1

   /auto_examples/start_here/plot_hairball_to_backbone

By network type
---------------

Narrative tutorials for each kind of network:

.. toctree::
   :maxdepth: 1

   /auto_examples/bipartite/plot_bipartite_projections
   /auto_examples/hypergraphs/plot_hypergraph_backbones
   /auto_examples/temporal/plot_temporal_backbones
   /auto_examples/directed_signed/plot_directed_signed

Short how-to pages for each kind of network:

.. toctree::
   :maxdepth: 2

   statistical_backbone
   proximity_scoring
   structural_methods
   bipartite_backbone
   hypergraph_backbone
   unweighted_sparsification
   temporal_backbones

Going further
-------------

Narrative deep dives, and comparing, validating, and benchmarking backbones.

.. toctree::
   :maxdepth: 1

   /auto_examples/statistical_nulls/plot_statistical_nulls
   /auto_examples/structural_proximity/plot_structural_proximity_tutorial
   /auto_examples/weight_meaning/plot_what_your_weights_mean
   /auto_examples/r_and_scale/plot_coming_from_r
   /auto_examples/r_and_scale/plot_scaling_up
   /auto_examples/validation/plot_compare_validate_report

.. toctree::
   :maxdepth: 2

   comparing_methods
   validating_backbones
   les_miserables_benchmark
   graph_gallery
