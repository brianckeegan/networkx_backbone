Statistical Methods
===================

Examples in this module use ``nx.les_miserables_graph()``.
All functions return scored full graphs; apply
:func:`~networkx_backbone.threshold_filter` or
:func:`~networkx_backbone.boolean_filter` as the second step.
Complexity classes are provided in each function's ``Complexity`` section.

.. automodule:: networkx_backbone.statistical
   :no-members:

.. currentmodule:: networkx_backbone

.. autofunction:: disparity_filter

.. autofunction:: noise_corrected_filter

.. autofunction:: marginal_likelihood_filter

.. autofunction:: ecm_filter

.. autofunction:: lans_filter

.. autofunction:: multiple_linkage_analysis

.. minigallery::
   networkx_backbone.disparity_filter
   networkx_backbone.noise_corrected_filter
   networkx_backbone.marginal_likelihood_filter
   networkx_backbone.ecm_filter
   networkx_backbone.lans_filter
   networkx_backbone.multiple_linkage_analysis
   :add-heading: Gallery Examples

.. rubric:: Alias Names

.. autofunction:: disparity

.. autofunction:: mlf

.. autofunction:: lans
