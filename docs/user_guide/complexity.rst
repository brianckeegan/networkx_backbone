Complexity Summary
==================

This page summarizes the asymptotic complexity classes used in API docstrings.

Notation: n = number of nodes, m = number of edges, n_a = agents, n_f = artifacts, e_b = bipartite edges, I = iterations, T = trials; for temporal networks, S = snapshots, B = bins.

Statistical
-----------

.. list-table::
   :header-rows: 1
   :widths: 32 24 24 20

   * - Function
     - Time
     - Space
     - Notes
   * - ``disparity``
     - ``O(n + m)``
     - ``O(n + m)``
     - Alias for disparity_filter.
   * - ``disparity_filter``
     - ``O(n + m)``
     - ``O(n + m)``
     - n=V, m=E.
   * - ``ecm_filter``
     - ``O(I * n^2 + m)``
     - ``O(n + m)``
     - I=max_iter, n=V, m=E.
   * - ``lans``
     - ``O(m log n)``
     - ``O(n + m)``
     - Alias for lans_filter.
   * - ``lans_filter``
     - ``O(m log n)``
     - ``O(n + m)``
     - Worst-case over node-local sorted edge-weight lookups.
   * - ``marginal_likelihood_filter``
     - ``O(n + m)``
     - ``O(n + m)``
     - n=V, m=E.
   * - ``mlf``
     - ``O(n + m)``
     - ``O(n + m)``
     - Alias for marginal_likelihood_filter.
   * - ``multiple_linkage_analysis``
     - ``O(m log n)``
     - ``O(n + m)``
     - Dominated by sorting each node's link weights (or lans_filter).
   * - ``noise_corrected_filter``
     - ``O(n + m)``
     - ``O(n + m)``
     - n=V, m=E.

Structural
----------

.. list-table::
   :header-rows: 1
   :widths: 32 24 24 20

   * - Function
     - Time
     - Space
     - Notes
   * - ``doubly_stochastic_filter``
     - ``O(I * n^2 + m log m)``
     - ``O(n^2 + m)``
     - I=max_iter for Sinkhorn-Knopp normalization; the connectivity stage sorts the edges once.
   * - ``edge_betweenness_filter``
     - ``O(nm + n^2 log n)``
     - ``O(n + m)``
     - Dominated by weighted Brandes edge-betweenness.
   * - ``global_sparsification``
     - ``O(m log m)``
     - ``O(n + m)``
     - 
   * - ``global_threshold_filter``
     - ``O(m)``
     - ``O(n + m)``
     - n=V, m=E.
   * - ``h_backbone``
     - ``O(nm + n^2 log n + m log m)``
     - ``O(n + m)``
     - Dominated by edge-betweenness on residual graph.
   * - ``high_salience_skeleton``
     - ``O(nm + n^2 log n)``
     - ``O(n + m)``
     - Runs a shortest-path tree from each root.
   * - ``maximum_spanning_tree_backbone``
     - ``O(m log n)``
     - ``O(n + m)``
     - 
   * - ``metric_backbone``
     - ``O(nm + n^2 log n)``
     - ``O(n^2 + m)``
     - All-pairs weighted shortest paths.
   * - ``modularity_backbone``
     - ``O(nm)``
     - ``O(n + m)``
     - Practical/heuristic bound due to repeated Louvain runs.
   * - ``node_degree_filter``
     - ``O(n + m)``
     - ``O(n + m)``
     - 
   * - ``planar_maximally_filtered_graph``
     - ``O(mn + m log m)``
     - ``O(n + m)``
     - 
   * - ``primary_linkage_analysis``
     - ``O(m)``
     - ``O(n + m)``
     - 
   * - ``strongest_n_ties``
     - ``O(m log n)``
     - ``O(n + m)``
     - Worst-case across per-node strongest-edge selection.
   * - ``ultrametric_backbone``
     - ``O(mn + m log n)``
     - ``O(n + m)``
     - MST plus per-edge minimax path checks.

Proximity
---------

.. list-table::
   :header-rows: 1
   :widths: 32 24 24 20

   * - Function
     - Time
     - Space
     - Notes
   * - ``adamic_adar_index``
     - ``O(mn)``
     - ``O(n + m)``
     - 
   * - ``cosine_backbone``
     - ``O(mn)``
     - ``O(n + m)``
     - 
   * - ``dice_backbone``
     - ``O(mn)``
     - ``O(n + m)``
     - 
   * - ``graph_distance_proximity``
     - ``O(n(n + m))``
     - ``O(n^2 + m)``
     - Computes all-pairs shortest paths before edge annotation.
   * - ``hub_depressed_index``
     - ``O(mn)``
     - ``O(n + m)``
     - 
   * - ``hub_promoted_index``
     - ``O(mn)``
     - ``O(n + m)``
     - 
   * - ``jaccard_backbone``
     - ``O(mn)``
     - ``O(n + m)``
     - 
   * - ``lhn_local_index``
     - ``O(mn)``
     - ``O(n + m)``
     - 
   * - ``local_path_index``
     - ``O(n^3)``
     - ``O(n^2)``
     - Dense adjacency matrix multiplication.
   * - ``neighborhood_overlap``
     - ``O(mn)``
     - ``O(n + m)``
     - Worst-case over per-edge neighbor-set intersections.
   * - ``preferential_attachment_score``
     - ``O(n + m)``
     - ``O(n + m)``
     - 
   * - ``resource_allocation_index``
     - ``O(mn)``
     - ``O(n + m)``
     - 

Hybrid
------

.. list-table::
   :header-rows: 1
   :widths: 32 24 24 20

   * - Function
     - Time
     - Space
     - Notes
   * - ``glab_filter``
     - ``O(nm + n^2 log n)``
     - ``O(n + m)``
     - Dominated by weighted edge-betweenness centrality.

Bipartite
---------

.. list-table::
   :header-rows: 1
   :widths: 32 24 24 20

   * - Function
     - Time
     - Space
     - Notes
   * - ``backbone``
     - ``O(n + m) dispatch checks + selected method cost``
     - ``O(1) additional beyond selected method``
     - 
   * - ``backbone_from_projection``
     - ``O(T n_a^2 n_f + e_b)``
     - ``O(n_a n_f + n_a^2)``
     - Worst-case over {sdsm, fdsm, fixedfill, fixedrow, fixedcol}.
   * - ``backbone_from_unweighted``
     - ``O(mn)``
     - ``O(n + m)``
     - Worst-case over {sparsify, lspar, local_degree}.
   * - ``backbone_from_weighted``
     - ``O(m log n)``
     - ``O(n + m)``
     - Worst-case over {disparity, mlf, lans, global_threshold}.
   * - ``bicm``
     - ``O(n_a n_f + e_b)``
     - ``O(n_a n_f)``
     - 
   * - ``bipartite_projection``
     - ``O(n_a^2 n_f + I n_a^2 + e_b)``
     - ``O(n_a n_f + n_a^2)``
     - Worst-case over supported projection methods.
   * - ``fastball``
     - ``O(r c + S c)``
     - ``O(r c)``
     - r=rows, c=cols, S=n_swaps.
   * - ``fdsm``
     - ``O(T n_a^2 n_f + e_b)``
     - ``O(n_a n_f + n_a^2)``
     - T=trials for Monte Carlo fixed-degree randomization.
   * - ``fixedcol``
     - ``O(n_a^2 n_f + e_b)``
     - ``O(n_a n_f + n_a^2)``
     - 
   * - ``fixedfill``
     - ``O(n_a^2 n_f + e_b)``
     - ``O(n_a n_f + n_a^2)``
     - 
   * - ``fixedrow``
     - ``O(n_a^2 n_f + e_b)``
     - ``O(n_a n_f + n_a^2)``
     - 
   * - ``hyper_projection``
     - ``O(n_a^2 n_f + e_b)``
     - ``O(n_a n_f + n_a^2)``
     - 
   * - ``probs_projection``
     - ``O(n_a^2 n_f + e_b)``
     - ``O(n_a n_f + n_a^2)``
     - 
   * - ``sdsm``
     - ``O(n_a^2 n_f + e_b)``
     - ``O(n_a n_f + n_a^2)``
     - Includes projection-weight annotation.
   * - ``simple_projection``
     - ``O(n_a^2 n_f + e_b)``
     - ``O(n_a n_f + n_a^2)``
     - n_a=agents, n_f=artifacts, e_b=bipartite edges.
   * - ``ycn_projection``
     - ``O(n_a^2 n_f + I n_a^2 + e_b)``
     - ``O(n_a n_f + n_a^2)``
     - I=max_iter for stationary-flow convergence.

Unweighted
----------

.. list-table::
   :header-rows: 1
   :widths: 32 24 24 20

   * - Function
     - Time
     - Space
     - Notes
   * - ``local_degree``
     - ``O(m + n)``
     - ``O(n + m)``
     - Wrapper over sparsify with degree scoring.
   * - ``lspar``
     - ``O(mn)``
     - ``O(n + m)``
     - Wrapper over sparsify with Jaccard scoring.
   * - ``sparsify``
     - ``O(mn)``
     - ``O(n + m)``
     - Worst-case; depends on escore/filter choices.

Filters
-------

.. list-table::
   :header-rows: 1
   :widths: 32 24 24 20

   * - Function
     - Time
     - Space
     - Notes
   * - ``boolean_filter``
     - ``O(n + m)``
     - ``O(n + m)``
     - 
   * - ``consensus_backbone``
     - ``O(km)``
     - ``O(m)``
     - k=number of input backbones, m=edges per backbone.
   * - ``fraction_filter``
     - ``O(m log m)``
     - ``O(n + m)``
     - Edge mode; node mode is O(n log n).
   * - ``multigraph_to_weighted``
     - ``O(m)``
     - ``O(n + m)``
     - m counts input edge instances for multigraphs.
   * - ``threshold_filter``
     - ``O(n + m)``
     - ``O(n + m)``
     - 

Measures
--------

.. list-table::
   :header-rows: 1
   :widths: 32 24 24 20

   * - Function
     - Time
     - Space
     - Notes
   * - ``compare_backbones``
     - ``O(b * C)``
     - ``O(b * q)``
     - b=backbones, q=measures, C=cost per measure evaluation.
   * - ``edge_fraction``
     - ``O(1)``
     - ``O(1)``
     - 
   * - ``ks_degree``
     - ``O(n)``
     - ``O(n)``
     - 
   * - ``ks_weight``
     - ``O(m)``
     - ``O(m)``
     - 
   * - ``node_fraction``
     - ``O(n)``
     - ``O(n)``
     - 
   * - ``reachability``
     - ``O(n + m)``
     - ``O(n)``
     - 
   * - ``weight_fraction``
     - ``O(m)``
     - ``O(1)``
     - 

Visualization
-------------

.. list-table::
   :header-rows: 1
   :widths: 32 24 24 20

   * - Function
     - Time
     - Space
     - Notes
   * - ``compare_graphs``
     - ``O(n + m) without layout; plus layout cost if pos is None``
     - ``O(n + m)``
     - Spring layout adds iterative graph-drawing overhead.
   * - ``graph_difference``
     - ``O(n + m)``
     - ``O(n + m)``
     - 
   * - ``save_graph_comparison``
     - ``O(n + m) without layout; plus rendering/write cost``
     - ``O(n + m)``
     - 

Convert
-------

.. list-table::
   :header-rows: 1
   :widths: 32 24 24 20

   * - Function
     - Time
     - Space
     - Notes
   * - ``to_graph``
     - ``O(n + m)``
     - ``O(n + m)``
     - Sparse input; a dense n x n array is O(n^2).

Temporal
--------

.. list-table::
   :header-rows: 1
   :widths: 32 24 24 20

   * - Function
     - Time
     - Space
     - Notes
   * - ``convert_temporal``
     - ``that of the converter``
     - ``that of the converter``
     - 
   * - ``events_from_snapshots``
     - ``O(S log S + n_S + m_S)``
     - ``O(n_S + m_S)``
     - n_S, m_S nodes and edges summed over snapshots.
   * - ``events_from_spans``
     - ``O(n + m log B + m_B)``
     - ``O(n + m_B)``
     - m_B span-bin overlaps (m with freq=None).
   * - ``flatten_temporal``
     - ``O(n + m + S log S)``
     - ``O(n + m)``
     - m observations (snapshot edges, events or spans), S snapshots.
   * - ``backbone_stability``
     - ``O(n_S + m_S + m T) plus T runs of the method``
     - ``O(n T + m T)``
     - T steps, m distinct edges; plus binning for events and spans.
   * - ``observation_fraction``
     - ``O(n_S + m_S + m T)``
     - ``O(n T + m T)``
     - T steps, m distinct edges; plus binning for events and spans.
   * - ``persistence``
     - ``O(n_S + m_S + m T)``
     - ``O(n T + m T)``
     - T steps, m distinct edges; plus binning for events and spans.
   * - ``weight_trend``
     - ``O(n_S + m_S + m T)``
     - ``O(n T + m T)``
     - T steps, m distinct edges; plus binning for events and spans.
   * - ``edge_persistence_hmm``
     - ``O(n_S + m_S + n_init I m T) plus T runs of the method``
     - ``O(n T + m T)``
     - T steps, m distinct edges, I EM iterations (at most max_iter); plus binning for events and spans.  The method runs only with observe="keep" or "pvalue".
   * - ``kalman_backbone``
     - ``O(n_S + m_S + I m T + P n_sim T) plus T runs of the method``
     - ``O(n T + m T + n_sim T)``
     - T steps, m distinct edges, P distinct missingness patterns, I likelihood evaluations when q is estimated; with missing_as_zero the method also tests the O(n^2) absent pairs of each snapshot.
   * - ``kalman_snapshots``
     - ``O(m T)``
     - ``O(m T)``
     -
   * - ``from_event_table``
     - ``O(r)``
     - ``O(r)``
     - r rows.
   * - ``snapshots_from_events``
     - ``O(n + m log m + B)``
     - ``O(n + m + B)``
     - B bins.
   * - ``snapshots_from_spans``
     - ``O(n + m log m + m_B + B)``
     - ``O(n + m_B + B)``
     - m_B span-bin overlaps.
   * - ``spans_from_events``
     - ``O(n + m log m)``
     - ``O(n + m)``
     - 
   * - ``spans_from_snapshots``
     - ``O(S log S + n_S + m_S)``
     - ``O(n_S + m_S)``
     - n_S, m_S nodes and edges summed over snapshots.
   * - ``temporal_form``
     - ``O(m + S log S)``
     - ``O(S)``
     - m multigraph edges, S snapshots.
   * - ``to_event_table``
     - ``O(m log m)``
     - ``O(m)``
     - Snapshots are first converted to events.
