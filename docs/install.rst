Installation
============

Quick install
-------------

Install from PyPI::

    pip install nx-backbone

Upgrading from networkx-backbone
--------------------------------

Versions 0.2 and earlier were published as ``networkx-backbone`` and
imported as ``networkx_backbone``. From 0.3.0 the package is
``nx-backbone``, imported as ``nx_backbone``, with no alias for the old
name. Replace the distribution, then change the imports in your code::

    pip uninstall networkx-backbone
    pip install nx-backbone

Function names and arguments are unchanged. See the :doc:`changelog` for
the other changes in 0.3.0.

Full installation
-----------------

For full functionality, install with optional dependencies. NumPy and SciPy
are needed for statistical methods (:func:`~nx_backbone.marginal_likelihood_filter`,
:func:`~nx_backbone.ecm_filter`), bipartite methods
(:func:`~nx_backbone.sdsm`, :func:`~nx_backbone.fdsm`),
:func:`~nx_backbone.doubly_stochastic_filter`,
:func:`~nx_backbone.local_path_index`, and KS measures
(:func:`~nx_backbone.ks_degree`, :func:`~nx_backbone.ks_weight`),
and statistically validated hypergraphs
(:func:`~nx_backbone.statistically_validated_hypergraph`)::

    pip install "nx-backbone[full]"

The plotting helpers in :mod:`nx_backbone.visualization` need
Matplotlib::

    pip install "nx-backbone[viz]"

Calling a function whose optional dependency is missing raises
:class:`ImportError`; everything else keeps working with NetworkX alone.

pandas is optional and imported only when used: for data-frame input to
:func:`~nx_backbone.to_graph` and the bipartite functions, for the event
tables of :func:`~nx_backbone.from_event_table` and
:func:`~nx_backbone.to_event_table`, and for frequency strings such as
``freq="1D"`` in the temporal module. Install it separately::

    pip install pandas

:func:`~nx_backbone.transform_weights` accepts scikit-learn transformers
by duck typing, so scikit-learn is never required.

Hypergraph interoperability
---------------------------

The ``from_*``/``to_*`` adapters in :mod:`nx_backbone.hypergraph_io`
import their target library lazily, so none of them is a dependency. Install
the ones you use through an extra, or all four with ``[interop]``::

    pip install "nx-backbone[xgi]"          # or [hypernetx], [hypergraphx], [hat]
    pip install "nx-backbone[interop]"      # all four

Reading and writing the HIF interchange format
(:func:`~nx_backbone.read_hif`, :func:`~nx_backbone.write_hif`) needs
only the standard library.

The version floors below are the oldest releases that pass the round-trip
tests in ``tests/test_hypergraph_interop.py``: hyperedges, weights, and node
labels survive ``from_X(to_X(h))`` and ``to_X(from_X(H))``, and each library's
own HIF reader and writer exchange files with ours. The "Interop" CI workflow
runs these tests against the latest releases on every pull request and weekly;
it is not a required check, so an upstream release cannot block unrelated
work.

.. list-table::
   :header-rows: 1
   :widths: 18 30 14 38

   * - Extra
     - PyPI distribution (import name)
     - Version
     - Notes
   * - ``[xgi]``
     - ``xgi``
     - >= 0.9.1
     - Multi-edges kept. xgi's HIF reader ignores the ``weight`` field of
       edge records, so weights read with ``xgi.read_hif`` are lost; use
       ``to_xgi(*read_hif(path, return_weights=True))``.
   * - ``[hypernetx]``
     - ``hypernetx``
     - >= 2.3.10
     - HIF support (``hnx.to_hif``/``hnx.from_hif``) first appears in 2.3.10;
       the adapters themselves also work with 2.3.2. Weights come back as
       floats.
   * - ``[hypergraphx]``
     - ``hypergraphx``
     - >= 1.7.7
     - No multi-edges: :func:`~nx_backbone.to_hypergraphx` merges a
       repeated hyperedge into one edge with the summed weight. Node labels in
       a hyperedge must be mutually comparable (HypergraphX sorts them). Its
       HIF reader relabels nodes ``0..n-1`` and ignores weights, and its HIF
       writer (1.8.0) leaves out the incidences.
   * - ``[hat]``
     - ``HypergraphAnalysisToolbox`` (``HAT``)
     - >= 1.1.18
     - Building from an incidence matrix fails in 1.1.14 and 1.1.16.
       :func:`~nx_backbone.to_hat` stores node labels in
       ``H.nodes["Names"]`` and weights in ``H.edges["weight"]``.

These ranges were tested on Python 3.13 with xgi 0.10.2, HyperNetX 2.4.3,
HypergraphX 1.8.0, and HAT 1.1.18 (the latest releases in November 2026) and
with the floors above.

Conda installation
------------------

Install from anaconda.org::

    conda install -c brianckeegan nx-backbone

Development installation
------------------------

Clone the repository and install in editable mode::

    git clone https://github.com/brianckeegan/nx-backbone.git
    cd nx-backbone
    pip install -e ".[dev]"

Run the test suite (including docstring examples) to verify the installation::

    pytest --doctest-modules nx_backbone tests

Requirements
------------

Supported versions follow NetworkX's: Python 3.12 or newer, and the same
minimum versions of NumPy, SciPy, and Matplotlib. CI tests these minimums
exactly, along with the latest releases.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Package
     - Version
     - Status
   * - Python
     - >= 3.12
     - Required
   * - NetworkX
     - >= 3.4
     - Required
   * - NumPy
     - >= 2.2
     - Optional (for ``[full]``)
   * - SciPy
     - >= 1.15
     - Optional (for ``[full]``)
   * - Matplotlib
     - >= 3.10
     - Optional (for ``[viz]``)
   * - pandas
     - >= 2.3
     - Optional (data-frame input, event tables, frequency strings)
