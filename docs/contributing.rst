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


Publishing to PyPI
------------------

PyPI releases are published automatically from the
``.github/workflows/publish-to-PyPI.yml`` workflow when a GitHub Release is
published. The workflow follows the PyPA trusted publishing guide:
https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/

Package versions are derived automatically from Git tags using
``setuptools-scm``. Do not edit a version string in ``pyproject.toml``.

Maintainers should configure PyPI trusted publishing once in the PyPI project
settings:

1. Open the ``networkx-backbone`` project on PyPI and add a trusted publisher
   for this GitHub repository.
2. Use the workflow filename ``publish-to-PyPI.yml`` and environment name
   ``pypi``.
3. Create and push a version tag (for example ``v0.2.0``) and then publish a
   GitHub Release from that tag; the workflow will run tests, build
   distributions, verify them with ``twine check``, and publish to PyPI using
   that tag as the package version.

Submitting changes
------------------

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite to verify
5. Submit a pull request
