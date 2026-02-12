Contributing
============

Contributions to ``networkx-backbone`` are welcome.

Development setup
-----------------

Clone the repository and install in editable mode with all development
dependencies::

    git clone https://github.com/brianckeegan/networkx_backbone.git
    cd networkx_backbone
    pip install -e ".[full,test,docs]"

Running tests
-------------

Run the test suite with pytest::

    pytest

Run with verbose output::

    pytest tests/ -v

Building documentation
-----------------------

Build the documentation locally::

    cd docs
    make html

The built documentation will be in ``docs/_build/html/``.

Code style
----------

- Follow PEP 8 for Python code
- Use NumPy-style docstrings for all public functions
- Include ``Parameters``, ``Returns``, ``Raises`` (if applicable),
  ``References``, and ``Examples`` sections in docstrings
- Use RST ``.. math::`` directives for mathematical formulas
- Use ``:func:`` and ``:mod:`` roles for cross-references within docstrings

Submitting changes
------------------

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite to verify
5. Submit a pull request
