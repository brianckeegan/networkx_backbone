Describe
========

Generate methods-section text and citations for a backbone -- the
counterpart of the ``narrative`` option of Neal's Backbone 3.0. See the
:doc:`../tutorials/validating_backbones` tutorial for a worked example, and
:doc:`../auto_examples/start_here/plot_hairball_to_backbone` for its place
at the end of the whole workflow.

.. automodule:: nx_backbone.describe
   :no-members:

.. currentmodule:: nx_backbone

.. autofunction:: describe_backbone

Registry
--------

The registry is importable from :mod:`nx_backbone.describe` for
building reference lists; it is not part of the top-level namespace.

.. currentmodule:: nx_backbone.describe

.. autodata:: METHODS
   :no-value:

.. autodata:: ALIASES
   :no-value:

.. autodata:: EXCLUDED
   :no-value:

.. autodata:: SCORE_KEYS
   :no-value:

.. autodata:: REFERENCES
   :no-value:

.. autoclass:: Method
   :no-members:

.. autoclass:: Reference
   :members: cite, text, bibtex
