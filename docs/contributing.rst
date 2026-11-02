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

Writing docs figures
--------------------

Every figure in the documentation is drawn when the docs are built, from the
current code, in a light and a dark variant; the theme shows the one that
matches the reader's theme. Nothing generated is committed. There are two
helpers, one for each kind of page:

- executed tutorials (Sphinx-Gallery files in ``docs/examples/``) use
  :class:`!TutorialFigures` from ``docs/_figures/tutorial.py``;
- plain reST pages (the how-to tutorials in ``docs/tutorials/`` and the user
  guide) use the ``page-figure`` directive, with one registry module per page
  in ``docs/_figures/pages/``.

Both draw with the shared style, ``docs/_figures/style.py``
(``from _figures import style``): take every color from
``style.palette(mode)``, at most ``style.ALL_PAIRS_CAP`` (3) categorical
series in a small multiple or scatter, ``style.layout(G, key)`` for the fixed
layout of each dataset, and ``style.draw_backbone`` for a backbone over its
graph. The conventions:

- **A drawing function of the mode.** ``draw(mode) -> Figure``, called once
  per mode (``"light"``, ``"dark"``) inside the matching Matplotlib style.
- **Alt text and a caption, computed.** Alt text says what the figure shows
  for a reader who cannot see it; the caption says what to take from it.
  Format every number in either from the variables the code computes, never
  as a literal, so the text stays true when the code changes.
- **Not by color alone.** Kept and removed edges, positive and negative
  edges, and series differ in line style, marker or label too. Three of the
  light-mode categorical colors are below 3:1 contrast, so label series
  directly or print a table of the plotted values.
- **Formats.** SVG for charts; PNG (about 144 dpi, the helpers' default) for
  dense network drawings, whose SVG would be large.
- **Linked, not embedded.** The page links the image files and loads them
  lazily, so a reader downloads only the variant the theme shows (#116).
- **Fast.** Use the small bundled datasets (Les Misérables, karate club,
  Davis, Florentine families) or small seeded synthetic graphs, no
  downloads; an executed tutorial should build in under about 30 s
  (``docs/sg_execution_times.rst`` lists the times after a build).

**In an executed tutorial**, create one ``TutorialFigures`` per file, named
after it, and make each ``figures.show(...)`` call the last expression of its
code block; Sphinx-Gallery then shows the figure below the code.

Readers should see the narrative and the analysis, not the figure machinery:
``_figures`` lives in the docs, and readers cannot import it. Wrap everything
that uses it in ``# sphinx_gallery_start_ignore`` /
``# sphinx_gallery_end_ignore``: the ``_figures`` imports (after a
``# isort: split`` line, so that ruff keeps them out of the import block),
the ``TutorialFigures`` setup, every drawing function and plotting helper,
and every ``figures.show(...)`` call. Hidden code still runs and its figures
still show; the rendered page leaves it out, and the downloaded script and
notebook keep it. Visible code must not use a name defined only in hidden
code. Where a cell is all hidden, the build drops the empty code block
Sphinx-Gallery would show (``docs/_ext/hidden_code.py``)::

    # sphinx_gallery_thumbnail_path = "_static/example_figures/my_topic/thumbnail.png"

    import matplotlib.pyplot as plt
    import networkx as nx

    import nx_backbone as nb

    # sphinx_gallery_start_ignore
    # isort: split
    from _figures import style
    from _figures.tutorial import TutorialFigures

    figures = TutorialFigures("my_topic")
    # sphinx_gallery_end_ignore

    G = nx.les_miserables_graph()
    backbone = nb.threshold_filter(nb.disparity_filter(G), "disparity_pvalue", 0.05)
    print(f"Kept {backbone.number_of_edges()} of {G.number_of_edges()} edges.")

    # %%
    # The backbone over the original graph:

    # sphinx_gallery_start_ignore
    counts = style.backbone_counts(G, backbone)


    def draw_backbone(mode):
        fig, ax = plt.subplots(figsize=(6, 4))
        style.draw_backbone(ax, G, backbone, style.layout(G, "les_miserables"), mode=mode)
        return fig


    figures.show(
        draw_backbone,
        "backbone",
        alt=f"Les Misérables with the {counts.edges_kept} backbone edges in solid "
        f"blue over the {counts.edges_total - counts.edges_kept} removed ones.",
        caption=f"The disparity backbone at p < 0.05. {counts.caption}",
        fmt="png",
        thumbnail=True,
    )
    # sphinx_gallery_end_ignore

In the docs build the files go to the gitignored
``docs/_static/example_figures/<name>/``; run anywhere else (as a script or
as the downloaded notebook), the figure is embedded in the output instead, so
it stays self-contained. ``figures.show`` takes:

- ``alt`` and ``caption``: strings, or functions of the mode where the words
  differ between the variants ("darker" in light, "brighter" in dark).
  Captions may use a little reST markup: inline literals (double
  backquotes), emphasis (``*...*``), ``:math:``, and ``:doc:`` with an
  explicit title (``title </path>``). Alt text is plain text.
- ``fmt``, ``dpi`` and ``width`` (a CSS width such as ``"85%"``; ``None``
  keeps the file's own size, at most the column's width).
- ``thumbnail=True`` on exactly one figure: it also saves the light variant as
  ``_static/example_figures/<name>/thumbnail.png``, which the
  ``sphinx_gallery_thumbnail_path`` comment at the top of the file must name.

``figures.show`` returns an object that renders as HTML, and whose plain
``repr`` names the figure only, so a cell never prints the build machine's
paths. Do not print the paths the style functions return.

**On a reST page**, register the figure in ``docs/_figures/pages/<page>.py``,
named after the page's file (``workflow.py`` for
``docs/user_guide/workflow.rst``), so that work on different pages never
touches the same file. The module is discovered automatically::

    """Figures of ``docs/user_guide/workflow.rst``."""

    import functools

    import matplotlib.pyplot as plt
    import networkx as nx

    import nx_backbone as nb
    from _figures import style
    from _figures.pages import figure


    @functools.cache
    def _backbone():
        G = nx.les_miserables_graph()
        H = nb.disparity_filter(G)
        return G, nb.threshold_filter(H, "disparity_pvalue", 0.05)


    def _caption():
        return f"The disparity backbone at p < 0.05. {style.backbone_counts(*_backbone()).caption}"


    @figure("backbone", alt="Les Misérables with its disparity backbone ...", caption=_caption, fmt="png")
    def draw_backbone(mode):
        G, backbone = _backbone()
        fig, ax = plt.subplots(figsize=(6, 4))
        style.draw_backbone(ax, G, backbone, style.layout(G, "les_miserables"), mode=mode)
        return fig

and show it on the page with::

    .. page-figure:: workflow/backbone
       :width: 80%

The ``alt`` and ``caption`` of a page figure are strings, or functions with no
arguments called at build time; the caption is reST inline markup, so it can
link with ``:func:``. Page figures have one alt text for both variants, so
describe them in words that fit both. Do the work inside the drawing function
or a cached helper, not at import time: every build imports every page module.
The build draws the figures on ``builder-inited`` into the gitignored
``docs/_page_figures/`` and redraws a page's figures only when its module, the
shared figure code (``docs/_figures/*.py``) or ``nx_backbone`` changes;
delete ``docs/_page_figures/`` to force a redraw.

A page can also show any figure of the method gallery on the API pages at no
cost, since the build draws those anyway (``scripts/generate_visualizations.py``
lists them; the name is ``<API module>/<method>``)::

    .. page-figure:: gallery:statistical/disparity_filter

**Figure coverage.** ``tests/test_docs_figures.py`` requires a figure (a
``page-figure``, ``figure`` or ``image`` directive, or a gallery include) on
every page of ``docs/tutorials/`` and ``docs/user_guide/`` except their
indexes. The pages not done yet are listed in its ``PENDING`` set. When you
give a page its first figure, delete the page from ``PENDING`` in the same
pull request: the test fails while a page listed there has a figure, and
while an entry names a page that does not exist. Never add a page to
``PENDING``. The same test checks that every tutorial using
``TutorialFigures`` sets its thumbnail as above.

Look at every figure in the built pages in both themes (the theme switcher is
at the top right) before opening a pull request.

Code style
----------

The project follows the conventions of NetworkX and the wider scientific-Python
stack:

- Lint and format with `ruff <https://docs.astral.sh/ruff/>`_ (configured in
  ``pyproject.toml``)::

      ruff check nx_backbone tests scripts docs/conf.py docs/_figures docs/_ext docs/examples
      ruff format nx_backbone tests scripts docs/conf.py docs/_figures docs/_ext docs/examples

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
4. Check https://pypi.org/project/nx-backbone/,
   https://anaconda.org/brianckeegan/nx-backbone and the documentation
   site. If an upload fails, fix the cause (for example the trusted
   publisher or the environment rules) and use "Re-run failed jobs" on the
   *release* run, which rebuilds from the same tag.

A re-run uses each workflow file as it was **at the tag**, not as it is on
``main``. It therefore only helps when the cause lies outside the file (a
publisher, an environment rule, a secret, an outage). When the workflow file
itself was wrong, fix it on ``main`` first, then:

- **conda:** run *Publish to Conda* by hand (Actions, *Publish to Conda*,
  *Run workflow*) on ``main`` with the ``tag`` input set to the release tag,
  for example ``v0.3.0``. The run checks out that tag with its full history,
  fails unless its setuptools-scm version equals the tag without its ``v``,
  builds the package with the recipe at the tag, runs the recipe's tests, and
  uploads to anaconda.org (the ``ANACONDA_TOKEN`` secret must be set). Use
  it, for example, for ``v0.3.0``, whose release run failed in the conda
  workflow.
- **Docs:** run *Deploy Documentation* by hand on ``main`` while ``main`` is
  still at the release commit (see the one-time setup above).
- **PyPI:** there is no manual upload of an existing tag, and PyPI refuses a
  second upload of the same version; a broken PyPI workflow is fixed in the
  next patch release.

Running the PyPI workflow by hand (``workflow_dispatch``), or the conda
workflow without a ``tag``, is a dry run: it tests and builds the selected
branch, but never uploads. Never publish a build of ``main`` itself; it
would carry a development version such as ``0.2.0.post1.dev9``.

The conda workflow installs only ``conda-build`` and ``anaconda-client``,
from conda-forge with strict channel priority. Do not add ``conda-verify``:
it is an optional linter for ``conda-build`` with no build for Python 3.12 or
later, and installing it made the 0.3.0 release run fail before building. To
check the recipe locally (it never uploads)::

    conda create -n conda-build -c conda-forge --override-channels conda-build
    conda activate conda-build
    NX_BACKBONE_VERSION=0.3.0 SETUPTOOLS_SCM_PRETEND_VERSION=0.3.0 \
        conda build .github/workflows/conda-recipe/ --output-folder /tmp/conda-bld \
        --override-channels -c conda-forge --no-anaconda-upload

Submitting changes
------------------

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite to verify
5. Submit a pull request
