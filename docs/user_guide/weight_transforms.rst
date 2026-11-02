Weights, Units, and Transforms
==============================

Every weighted method makes an assumption about what an edge weight is: a
count of events, a share of a node's strength, a distance, or just a value
that can be ranked.  This guide explains how the results change when the
weights are rescaled or transformed, how to transform weights with
:func:`~nx_backbone.transform_weights`, how to set the count unit of the
count-based null models, and how to relax the null model of the disparity
filter.

.. seealso:: :ref:`tutorial-weight-meaning`, a narrative tutorial that works through these choices on data, with figures.

Transforming weights
--------------------

:func:`~nx_backbone.transform_weights` returns a copy of a graph with
transformed edge weights.  It takes a plain function, applied to each weight,
or any object with a scikit-learn style ``fit_transform`` method, which is
called once with the weights as a column.  scikit-learn is not a dependency:
any object with that method works, and a plain function needs neither NumPy
nor scikit-learn.

.. code-block:: python

    import math

    import networkx as nx
    import nx_backbone as nb

    G = nx.les_miserables_graph()
    H = nb.transform_weights(G, math.log1p)          # replaces "weight"
    H = nb.transform_weights(G, math.sqrt, out="w")  # keeps "weight", adds "w"

    # Any object with fit_transform, for example from scikit-learn:
    from sklearn.preprocessing import PowerTransformer

    boxcox = PowerTransformer(method="box-cox", standardize=False)
    H = nb.transform_weights(G, boxcox)

The returned graph keeps every node's neighbor order, so methods that break
weight ties by neighbor order give the same result on an identity transform.
The transform is recorded in ``H.graph["weight_transforms"]``, and
:func:`~nx_backbone.describe_backbone` reports it in the methods text.

The function raises an error when the output is not finite or is the same for
every edge although the weights differ (scikit-learn's ``Normalizer`` scales
each *row*, so it maps a column of weights to all ones).  It warns when the
transform does not preserve the order of the weights, or when it maps
different weights to the same value: both change the methods that only use
ranks.  It does not check that the new weights are positive; the statistical
scorers reject non-positive weights themselves.  Centering or standardizing
transforms (``StandardScaler``, ``PowerTransformer`` with its default
``standardize=True``, ``QuantileTransformer`` with a normal output) produce
negative weights.

**A transform changes the hypothesis.**  The disparity filter applied to
``log1p(w)`` tests whether the shares of the *log*-weights are uniform, not
whether the shares of the weights are.  That can be a deliberate modeling
choice, for example log-flows in gravity-type data, but it is a different
test, and it should be reported as such.

.. page-figure:: weight_transforms/shares

Which methods a transform changes
---------------------------------

The table lists, for each weighted method, whether its scores and backbone
stay the same when every weight is multiplied by a constant :math:`c > 0`, and
when every weight is replaced by :math:`f(w)` for a strictly increasing
function :math:`f` such as a power, log, Box-Cox, or quantile transform.
Every entry is checked on Les Misérables by
``tests/test_weight_transforms.py`` (rescaling by 4; :math:`\sqrt{w}`,
:math:`w^2` and :math:`\log(1 + w)` for increasing transforms), which also
checks that this table matches the tests.

.. list-table::
   :header-rows: 1
   :widths: 34 12 12 42

   * - Method
     - Rescaling
     - Increasing transform
     - Why
   * - :func:`~nx_backbone.lans_filter`
     - yes
     - yes
     - Uses only the order of each node's weights.
   * - :func:`~nx_backbone.strongest_n_ties`
     - yes
     - yes
     - Keeps each node's heaviest links.
   * - :func:`~nx_backbone.global_sparsification`
     - yes
     - yes
     - Keeps the heaviest fraction of all edges.
   * - :func:`~nx_backbone.primary_linkage_analysis`
     - yes
     - yes
     - Keeps each node's heaviest link.
   * - :func:`~nx_backbone.maximum_spanning_tree_backbone`
     - yes
     - yes
     - A maximum spanning tree depends only on the order of the weights.
   * - :func:`~nx_backbone.planar_maximally_filtered_graph`
     - yes
     - yes
     - Adds edges in weight order.
   * - :func:`~nx_backbone.ultrametric_backbone`
     - yes
     - yes
     - Minimax paths depend only on the order of the weights.
   * - :func:`~nx_backbone.disparity_filter` (any ``a``)
     - yes
     - no
     - Tests the shares :math:`w/s`.
   * - :func:`~nx_backbone.multiple_linkage_analysis`
     - yes
     - no
     - Fits each node's weight shares.
   * - :func:`~nx_backbone.polya_filter` (``approximate=True``)
     - yes
     - no
     - The large-strength approximation depends on :math:`w/s` only.
   * - :func:`~nx_backbone.gloss_filter`
     - yes
     - no
     - With the default ``resolution``, the weight lattice and the strengths
       rescale together.
   * - :func:`~nx_backbone.doubly_stochastic_filter`
     - yes
     - no
     - Row and column scaling absorbs a constant factor.
   * - :func:`~nx_backbone.metric_backbone`
     - yes
     - no
     - Shortest paths on distances :math:`1/w`: a factor rescales every path
       alike, a nonlinear transform does not.
   * - :func:`~nx_backbone.high_salience_skeleton`
     - yes
     - no
     - Shortest-path trees on distances :math:`1/w`.
   * - :func:`~nx_backbone.edge_betweenness_filter`
     - yes
     - no
     - Shortest paths on distances :math:`1/w`.
   * - :func:`~nx_backbone.glab_filter`
     - yes
     - no
     - Shortest-path involvement on distances :math:`1/w`.
   * - :func:`~nx_backbone.modularity_backbone`
     - yes
     - no
     - Weighted modularity uses weights relative to the total weight.
   * - :func:`~nx_backbone.marginal_likelihood_filter`
     - no
     - no
     - Weights are counts of unit edges; set ``unit=`` instead.
   * - :func:`~nx_backbone.ecm_filter`
     - no
     - no
     - Weights are integer counts; set ``unit=`` instead.
   * - :func:`~nx_backbone.noise_corrected_filter`
     - no
     - no
     - Weights are counts of interactions; set ``unit=`` instead.
   * - :func:`~nx_backbone.polya_filter` (exact p-values)
     - no
     - no
     - Weights are counts of urn draws.
   * - :func:`~nx_backbone.h_backbone`
     - no
     - no
     - The h-index compares weights with counts of edges.
   * - :func:`~nx_backbone.local_path_index` (``weight=...``)
     - no
     - no
     - Mixes :math:`A^2` and :math:`\varepsilon A^3`, whose ratio changes
       with scale.

Methods that ignore weights, such as
:func:`~nx_backbone.node_degree_filter`, the proximity indices without
``weight=``, :func:`~nx_backbone.sparsify`,
:func:`~nx_backbone.lspar`, and :func:`~nx_backbone.local_degree`,
are unaffected by any transform.
:func:`~nx_backbone.global_threshold_filter` changes with the weights
but in step with them: thresholding :math:`f(w)` at :math:`f(t)` keeps the
same edges as thresholding :math:`w` at :math:`t`.

Some transforms are best avoided:

- **Before the count models** (MLF, ECM, noise-corrected, Pólya with integer
  weights): a continuous transform destroys the meaning of a weight as a
  number of events, and binning (``KBinsDiscretizer``) gives ordinal bin
  labels, not counts.  Use ``unit=`` (below) for a change of unit.
- **Rank transforms before share-based methods**: after a uniform
  ``QuantileTransformer`` the disparity filter tests the shares of ranks.
  :func:`~nx_backbone.lans_filter` is the method for ranks.

Count units
-----------

:func:`~nx_backbone.marginal_likelihood_filter`,
:func:`~nx_backbone.ecm_filter`, and
:func:`~nx_backbone.noise_corrected_filter` treat each weight as a
number of events, so their p-values depend on the unit the weights are
recorded in.  Their ``unit=`` argument sets the weight of one event: the
models use the counts ``weight / unit`` (rounded to integers by MLF and ECM).
The default, ``unit=1``, uses the weights themselves, as before.  On Les
Misérables, ``unit=0.1`` (each co-appearance counted as ten events) keeps 215
instead of 126 edges for MLF at ``alpha = 0.05``, 89 instead of 72 for the ECM,
and 215 instead of 126 for the noise-corrected filter.

The three scorers warn when ``weight / unit`` is not an integer count.  MLF
rounds a count below 0.5 to 0 (the edge then gets p-value 1), and the ECM
raises a count below 1.5 to 1.

.. page-figure:: weight_transforms/edge-counts

.. code-block:: python

    # Durations in hours, where one event is a quarter of an hour:
    H = nb.marginal_likelihood_filter(G, unit=0.25)

The disparity filter's null model
----------------------------------

The disparity filter (Serrano et al., 2009) assumes that a node of degree
:math:`k` splits its strength at :math:`k - 1` uniformly random points, so
that a single weight share :math:`x = w/s` is Beta(1, :math:`k - 1`)
distributed.  This is exact when the node's weights are independent
exponential variables, and wrong otherwise: weights that are more
heterogeneous than exponential make the filter reject too often, and more even
weights too rarely (``tests/test_calibration.py`` checks both).

``disparity_filter(G, a=...)`` relaxes the assumption to a one-parameter
family.  The shares are symmetric Dirichlet with concentration :math:`1/a`, so
a single share is Beta(:math:`1/a`, :math:`(k-1)/a`); this is exact for
independent Gamma(:math:`1/a`) weights of any scale.  ``a=1`` is the original
filter (computed exactly as before, without SciPy), ``a > 1`` tolerates more
heterogeneity, and ``a="ml"`` fits *a* by maximum likelihood and stores it in
``H.graph["disparity_a"]``:

.. code-block:: python

    H = nb.disparity_filter(G, a="ml")
    H.graph["disparity_a"]                                       # 0.525
    nb.threshold_filter(H, "disparity_pvalue", 0.05).number_of_edges()   # 19

Les Misérables' shares are more even than the uniform null expects
(:math:`\hat a < 1`), so the fitted filter keeps 19 edges where the original
keeps 9.  The parameter has the same meaning as the reinforcement parameter of
:func:`~nx_backbone.polya_filter`: with the shares held fixed, the
Pólya filter's beta-binomial null converges to this Beta distribution as the
node strength grows.  Only this symmetric family is offered, because the links
of a node are exchangeable under the null, so every share has mean
:math:`1/k`.

The calibration tests (``tests/test_calibration.py``) simulate weights from
each null model and check that the p-value of a single endpoint is uniform.
The edge p-value of the disparity filter, and of the other two-endpoint
filters, is the *smaller* of its two endpoint p-values, which is not
calibrated even when the null model holds: on 8-regular graphs with
independent exponential weights, about 8% of the edges fall below 0.05.

Not offered
-----------

- **Fitting an arbitrary distribution to the weights and testing each weight
  against it.**  With one distribution fitted to all weights, the p-value
  decreases with the weight, so the backbone is a global weight threshold
  (:func:`~nx_backbone.global_threshold_filter`) that ignores degrees
  and strengths.  With a distribution fitted to each node's few weights, the
  fit reuses the weight being tested.
- **Arbitrary share distributions in the disparity filter**, such as a
  Beta(:math:`c`, :math:`d`) with :math:`d \neq (k-1)c`: shares of
  exchangeable links must have mean :math:`1/k`.

Still open (issue `#55 <https://github.com/brianckeegan/nx-backbone/issues/55>`_):
a continuous-weight ECM, an overdispersed (beta-binomial) marginal likelihood
filter, and user-supplied weight distributions for GloSS.
