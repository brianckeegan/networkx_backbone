Installation
============

Quick install
-------------

Install from PyPI::

    pip install networkx-backbone

Full installation
-----------------

For full functionality, install with optional dependencies. NumPy and SciPy
are needed for statistical methods (:func:`~networkx_backbone.marginal_likelihood_filter`,
:func:`~networkx_backbone.ecm_filter`), bipartite methods
(:func:`~networkx_backbone.sdsm`, :func:`~networkx_backbone.fdsm`),
:func:`~networkx_backbone.doubly_stochastic_filter`,
:func:`~networkx_backbone.local_path_index`, and KS measures
(:func:`~networkx_backbone.ks_degree`, :func:`~networkx_backbone.ks_weight`)::

    pip install networkx-backbone[full]

Conda installation
------------------

Install from anaconda.org::

    conda install -c brianckeegan networkx-backbone

Development installation
------------------------

Clone the repository and install in editable mode::

    git clone https://github.com/brianckeegan/networkx_backbone.git
    cd networkx_backbone
    pip install -e ".[full,test]"

Run the test suite to verify the installation::

    pytest

Requirements
------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Package
     - Version
     - Status
   * - Python
     - >= 3.10
     - Required
   * - NetworkX
     - >= 3.0
     - Required
   * - NumPy
     - >= 1.23
     - Optional (for ``[full]``)
   * - SciPy
     - >= 1.9
     - Optional (for ``[full]``)
