Contributing
============

Contributions to ``nx-backbone`` are welcome.

Development setup
-----------------

Clone the repository and install in editable mode with all development
dependencies::

    git clone https://github.com/brianckeegan/nx-backbone.git
    cd nx-backbone
    pip install -e ".[dev,docs]"

Running tests
-------------

Run the test suite and the docstring examples with pytest::

    pytest --doctest-modules nx_backbone tests

Measure coverage::

    pytest --cov=nx_backbone --cov-report=term-missing

``tests/test_minimal_deps.py`` runs the pure-Python methods in a subprocess in
which numpy, scipy, and matplotlib cannot be imported. If you add a method that
needs an optional dependency, import it inside the function (not at module
level) so ``import nx_backbone`` keeps working with NetworkX alone.

Run the suite under NetworkX's backend-dispatch test harness, which sends
every dispatchable call through the ``nx_loopback`` test backend and checks
the result against plain NetworkX (see :doc:`user_guide/backends`)::

    NETWORKX_TEST_BACKEND=nx_loopback pytest --doctest-modules nx_backbone tests

Continuous integration (``.github/workflows/tests.yml``) runs lint, mypy, the test
suite on Python 3.12-3.14 (with doctests and coverage on 3.13), a job pinned to
the oldest supported versions, the dispatch tests (``tests/test_dispatch.py``
and ``tests/test_api_contract.py``, which call every dispatchable function)
under ``nx_loopback``, and the minimal-dependency smoke tests on every push and
pull request.

A test that sweeps a registry (every method, style, scorer or dispatchable
function) loops over it inside one test and collects the failures, so that a
failure message names every failing case; it is not parametrized into one
test per case.

Benchmarks
----------

``benchmarks/`` holds an `airspeed velocity (asv)
<https://asv.readthedocs.io/>`_ suite, as NumPy, SciPy and NetworkX use. It
times the statistical scorers on Les Misérables and on a 2000-node synthetic
weighted graph, ``sdsm``/``fdsm``/``fixedfill``/``fixedrow``/``fixedcol`` on a
400 x 300 bipartite graph, ``mdl_hypergraph_backbone`` (``edge``, ``node``,
``auto``) at m = 500 and 2000, the statistically validated hypergraph and cores,
and the shortest-path methods (salience, metric backbone, GLAB).

Install asv and compare your branch with ``main``; asv builds each commit in
its own virtual environment, so commit your changes first::

    pip install -e ".[benchmark]"
    cd benchmarks
    asv machine --yes            # once per machine
    asv continuous main HEAD     # flags benchmarks that changed by > 10%

``asv continuous -b MDL main HEAD`` runs only the benchmarks matching a regular
expression. After editing a benchmark, ``asv run -E existing --quick`` runs each
one once against the nx-backbone already installed in your environment,
without building anything. Benchmarks are plain Python: a class with ``setup`` and ``time_*`` methods and optional
``params``; seeded inputs come from ``benchmarks/benchmarks/_data.py``.

The *Benchmarks* workflow (``.github/workflows/benchmarks.yml``) runs the same
comparison on GitHub Actions when started by hand from the *Actions* tab
(``workflow_dispatch``); its log lists the benchmarks that got faster or
slower.

``benchmarks/benchmarks/les_miserables.py`` is also the single definition of
the :doc:`Les Misérables benchmark <tutorials/les_miserables_benchmark>`: the
tests pin the edge count each case keeps, and the docs build renders the table
from it. When a method's count changes on purpose, update ``EXPECTED_EDGES`` in
``tests/test_les_miserables.py``; the docs follow automatically.

Reference implementations
-------------------------

``tests/test_reference_implementations.py`` compares p-values and backbones
with R `backbone <https://cran.r-project.org/package=backbone>`_ and
`netbone <https://gitlab.liris.cnrs.fr/coregraphie/netbone>`_ on Les
Misérables, the karate club, and Davis Southern Women.  It reads JSON
fixtures in ``tests/fixtures/``, so neither R nor netbone is needed to run
the tests, and CI does not install them.  Where nx-backbone
intentionally differs from a reference, the test checks the documented
relationship and the method's docstring Notes explain the difference.
The `reference conventions note
<https://github.com/brianckeegan/nx-backbone/blob/main/docs/design/reference-conventions.md>`__
lists every such difference, measures its effect on the three datasets, and
records the options and the decision on each (issue #59).

Regenerate the fixtures when a reference implementation releases a new
version, or when you add a comparison.  Both generators read the same
CSV files, which are exported from the datasets that ship with NetworkX::

    python scripts/export_reference_inputs.py   # tests/fixtures/reference_inputs/*.csv

R ``backbone`` (3.0 or later; the fixture records the R and package
versions used)::

    Rscript -e 'install.packages(c("backbone", "jsonlite"))'
    Rscript scripts/make_reference_fixtures.R   # tests/fixtures/r_backbone.json

The R script seeds the random number generator, so FDSM results are
reproducible for a given R and ``backbone`` version.

netbone pins old dependencies (NumPy 1.26, NetworkX 2.8, SciPy 1.13, JAX
0.4), so install it in its own Python 3.12 environment and run the script
with that interpreter from the repository root::

    python3.12 -m venv /tmp/netbone-env
    /tmp/netbone-env/bin/pip install netbone==0.2.3
    /tmp/netbone-env/bin/python -I scripts/make_netbone_fixtures.py   # tests/fixtures/netbone.json

netbone's ECM solver is not bit-for-bit reproducible, so its p-values can
change by about ``1e-5`` between runs; the test allows for that.  After
regenerating, run the reference tests and review every change in the
fixtures with ``git diff``.

Building documentation
-----------------------

Build the documentation locally, treating warnings as errors as CI does::

    cd docs
    make html SPHINXOPTS="-W --keep-going -n"

The built documentation will be in ``docs/_build/html/``.

The build validates every rendered docstring with numpydoc: each public
function needs a docstring with a summary, every parameter documented in
signature order, and ``Returns`` and ``Examples`` sections. The checks are
listed in ``[tool.numpydoc_validation]`` in ``pyproject.toml``; run them in
seconds, without building the docs, with::

    python -m numpydoc lint nx_backbone/*.py

Code style
----------

The project follows the conventions of NetworkX and the wider scientific-Python
stack:

- Lint and format with `ruff <https://docs.astral.sh/ruff/>`_ (configured in
  ``pyproject.toml``)::

      ruff check nx_backbone tests scripts docs/conf.py docs/_figures docs/examples
      ruff format nx_backbone tests scripts docs/conf.py docs/_figures docs/examples

- Take a graph ``G`` as the first argument, return a new graph, and never
  mutate the input.
- Decorate a new function that takes a graph and returns a new graph with
  ``nx_backbone._dispatch._dispatchable`` (``edge_attrs="weight"`` if it
  reads edge weights), innermost, below ``not_implemented_for`` and
  ``py_random_state`` as in NetworkX, and add it to ``tests/test_dispatch.py``.
- Name edge-weight parameters ``weight`` and random-state parameters ``seed``;
  accept ``None``, an ``int``, or a NumPy ``Generator``/``RandomState``, as
  NetworkX does.
- Raise :class:`networkx.NetworkXError` for invalid graph input and
  :class:`networkx.NetworkXNotImplemented` (via
  :func:`networkx.utils.decorators.not_implemented_for`) for unsupported graph types;
  raise :class:`ValueError` for invalid non-graph arguments.
  Functions that also accept a matrix (the bipartite scorers take NumPy,
  SciPy sparse, or pandas incidence matrices) use the module's own check,
  which raises the same exception for graph input, because
  ``not_implemented_for`` assumes a graph argument.
- Annotate public functions (the package ships ``py.typed``). Start each
  module with ``from __future__ import annotations`` and import NumPy, SciPy,
  pandas, and Matplotlib types only under ``if TYPE_CHECKING:``, so the core
  package still imports with only NetworkX installed. Shared aliases (``Seed``,
  ``GraphInput``, ``Hyperedges``, ...) live in ``nx_backbone._typing``;
  import ``not_implemented_for`` and ``py_random_state`` from there too. They
  are NetworkX's own decorators, typed so that they keep the decorated
  function's signature (``_dispatchable`` is typed the same way, and also
  accepts ``backend=...``). Check with mypy (non-strict, configured in
  ``pyproject.toml``; the ``typing`` extra pins mypy and ``types-networkx``)::

      mypy nx_backbone

- Name score attributes ``<method>_pvalue``, ``<method>_score``, or
  ``<method>_keep`` so they compose with the filters in
  :mod:`nx_backbone.filters`.
- Use NumPy-style docstrings for all public functions
- Include ``Parameters``, ``Returns``, ``Raises`` (if applicable),
  ``References``, and ``Examples`` sections in docstrings
- Use RST ``.. math::`` directives for mathematical formulas
- Use ``:func:`` and ``:mod:`` roles for cross-references within docstrings


Making a release
----------------

Publishing a GitHub Release triggers three workflows: ``publish-to-PyPI.yml``
(PyPI), ``publish-to-conda.yml`` (anaconda.org), and ``deploy-docs.yml``
(GitHub Pages). The PyPI workflow follows the PyPA trusted publishing guide:
https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/

The package version comes from the release tag through ``setuptools-scm``:
tag ``v0.3.0`` builds version ``0.3.0``. Do not set a version in
``pyproject.toml`` or in the conda recipe. (The ``v0.2.0`` tag never reached
PyPI because ``pyproject.toml`` still set ``version = "0.1.0"`` there.)

One-time setup:

1. On PyPI, add a trusted publisher for the ``nx-backbone`` project:
   owner ``brianckeegan``, repository ``nx-backbone``, workflow filename
   ``publish-to-PyPI.yml``, environment name ``pypi``. Before the first
   upload the project does not exist yet, so add it as a *pending*
   publisher (PyPI account settings, "Publishing"); the first release then
   creates the project. (Versions 0.2 and earlier were published as
   ``networkx-backbone``, which is not updated.)
2. In the repository settings, create the ``pypi`` environment. If it has
   deployment branch or tag rules, allow tags matching ``v*``: a release
   run deploys from the tag, not from ``main``.
3. Do the same for the ``github-pages`` environment, which GitHub creates
   for Pages with a rule that allows only the default branch: under
   *Deployment branches and tags*, add a tag rule ``v*``. Without it the
   documentation deploy of a release fails with "Tag "vX.Y.Z" is not
   allowed to deploy to github-pages due to environment protection rules"
   (as for v0.3.0). To publish the docs after such a failure, run the
   *Deploy Documentation* workflow by hand on ``main`` while ``main`` is
   still at the release commit.
4. For anaconda.org uploads, add an ``ANACONDA_TOKEN`` repository secret.

For each release:

1. Merge everything for the release into ``main`` with green CI, and set the
   release date in the changelog heading (``Version X.Y.Z (YYYY-MM-DD)``).
2. Draft a GitHub Release whose tag is ``vX.Y.Z`` (the tag can be new; GitHub
   creates it on ``main`` when the release is published) and paste the
   release notes. Curated notes for each release live in
   ``docs/release-notes/``.
3. Publish the release. The PyPI workflow checks out the tag with its full
   history, runs the tests, builds the wheel and sdist, checks them with
   ``twine check --strict``, and **fails before uploading unless both carry
   version X.Y.Z**. The conda workflow checks the same thing before building.
4. Check https://pypi.org/project/nx-backbone/ and the documentation
   site. If an upload fails, fix the cause (for example the trusted
   publisher or the environment rules) and use "Re-run failed jobs" on the
   *release* run, which rebuilds from the same tag.

Running either publishing workflow by hand (``workflow_dispatch``) is a dry
run: it tests and builds, but never uploads. Never publish from a manual run
on ``main``; it would build a development version such as
``0.2.0.post1.dev9``.

Submitting changes
------------------

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite to verify
5. Submit a pull request
