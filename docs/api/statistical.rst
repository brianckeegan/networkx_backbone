Statistical Methods
===================

Examples in this module use ``nx.les_miserables_graph()``.
All functions return scored full graphs; apply
:func:`~nx_backbone.threshold_filter` or
:func:`~nx_backbone.boolean_filter` as the second step.
Complexity classes are provided in each function docstring.
For a worked case study of scoring, choosing alpha and a correction,
and reporting, start with :doc:`../auto_examples/start_here/plot_hairball_to_backbone`.
The tutorial :ref:`tutorial-statistical-nulls` compares the null models on one
network, with figures.

.. automodule:: nx_backbone.statistical
   :no-members:

.. currentmodule:: nx_backbone

.. autofunction:: disparity_filter

.. autofunction:: noise_corrected_filter

.. autofunction:: marginal_likelihood_filter

.. autofunction:: ecm_filter

.. autofunction:: lans_filter

.. autofunction:: multiple_linkage_analysis

.. autofunction:: polya_filter

.. autofunction:: gloss_filter

.. minigallery::
   nx_backbone.disparity_filter
   nx_backbone.noise_corrected_filter
   nx_backbone.marginal_likelihood_filter
   nx_backbone.ecm_filter
   nx_backbone.lans_filter
   nx_backbone.multiple_linkage_analysis
   nx_backbone.polya_filter
   nx_backbone.gloss_filter
   :add-heading: Gallery Examples

.. rubric:: Method gallery

.. include:: ../_gallery/statistical.inc

.. rubric:: Alias Names

.. autofunction:: disparity

.. autofunction:: mlf

.. autofunction:: lans
