Bipartite Methods
=================

Examples in this module use ``nx.davis_southern_women_graph()``.
Projection scoring methods return full graphs with p-values/weights; use
:func:`~networkx_backbone.threshold_filter` or
:func:`~networkx_backbone.boolean_filter` as needed.
Complexity classes are provided in each function docstring.

.. automodule:: networkx_backbone.bipartite
   :no-members:

.. currentmodule:: networkx_backbone

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
   networkx_backbone.simple_projection
   networkx_backbone.hyper_projection
   networkx_backbone.probs_projection
   networkx_backbone.ycn_projection
   networkx_backbone.bipartite_projection
   networkx_backbone.sdsm
   networkx_backbone.fdsm
   networkx_backbone.fixedfill
   networkx_backbone.fixedrow
   networkx_backbone.fixedcol
   :add-heading: Gallery Examples

.. rubric:: High-Level Wrappers

.. autofunction:: backbone_from_projection

.. autofunction:: backbone_from_weighted

.. autofunction:: backbone_from_unweighted

.. autofunction:: backbone
