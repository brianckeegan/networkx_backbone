Describe
========

Generate methods-section text and citations for a backbone -- the
counterpart of the ``narrative`` option of Neal's Backbone 3.0. See the
:doc:`../tutorials/validating_backbones` tutorial for a worked example.

.. automodule:: networkx_backbone.describe
   :no-members:

.. currentmodule:: networkx_backbone

.. autofunction:: describe_backbone

Registry
--------

The registry is importable from :mod:`networkx_backbone.describe` for
building reference lists; it is not part of the top-level namespace.

.. currentmodule:: networkx_backbone.describe

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
