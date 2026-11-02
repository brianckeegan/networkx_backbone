Bipartite Methods
=================

Examples in this module use ``nx.davis_southern_women_graph()``.
Projection scoring methods return full graphs with p-values/weights; use
:func:`~nx_backbone.threshold_filter` or
:func:`~nx_backbone.boolean_filter` as needed, or
:func:`~nx_backbone.backbone_from_projection`, which keeps only the
significant pairs.
Complexity classes are provided in each function docstring.

Every function accepts either a bipartite graph with its ``agent_nodes`` or an
agents x artifacts incidence matrix (NumPy array, SciPy sparse array or matrix,
or pandas ``DataFrame``) with optional ``agent_labels`` / ``artifact_labels``.
The incidence is kept sparse and agent pairs are scored in blocks of rows.

The simple, hyperbolic, ProbS and YCN weightings follow Coscia & Rossi (2019,
`doi:10.1145/3341161.3342862 <https://doi.org/10.1145/3341161.3342862>`_,
arXiv:1906.09081). The hyperbolic weight uses ``1/d(z)`` per shared artifact;
``hyper_projection(..., newman=True)`` (``method="newman"``) gives Newman's
(2001) ``1/(d(z) - 1)`` collaboration weight.

For a narrative walkthrough with figures (projection weights, the null models
side by side, the FDSM p-value floor, signed backbones, SDSM-EC, matching R
and sparse input), see
:doc:`/auto_examples/bipartite/plot_bipartite_projections`.

.. automodule:: nx_backbone.bipartite
   :no-members:

.. currentmodule:: nx_backbone

.. autofunction:: simple_projection

.. autofunction:: hyper_projection

.. autofunction:: probs_projection

.. autofunction:: ycn_projection

.. autofunction:: bipartite_projection

.. autofunction:: sdsm

.. autofunction:: fdsm

.. autofunction:: fixedfill

.. autofunction:: fixedrow

.. autofunction:: fixedcol

.. autofunction:: bicm

.. autofunction:: fastball

.. minigallery::
   nx_backbone.simple_projection
   nx_backbone.hyper_projection
   nx_backbone.probs_projection
   nx_backbone.ycn_projection
   nx_backbone.bipartite_projection
   nx_backbone.sdsm
   nx_backbone.fdsm
   nx_backbone.fixedfill
   nx_backbone.fixedrow
   nx_backbone.fixedcol
   :add-heading: Gallery Examples

.. rubric:: Method gallery

.. include:: ../_gallery/bipartite.inc

.. rubric:: High-Level Wrappers

.. autofunction:: backbone_from_projection

.. autofunction:: backbone_from_weighted

.. autofunction:: backbone_from_unweighted

.. autofunction:: backbone
