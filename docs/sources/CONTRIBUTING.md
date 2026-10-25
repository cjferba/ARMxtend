# How to Contribute

I would be very happy about any kind of contribution that helps to improve and extend the
functionality of ARMxtend.

## Getting started

- If you don't have a [GitHub](https://github.com) account yet, please create one to contribute to
  this project.
- Please open an issue on the [issue tracker](https://github.com/cjferba/ARMxtend/issues) to
  discuss the fix or new feature before spending too much time and effort on the implementation.
- Fork the `ARMxtend` repository from the GitHub web interface, then clone your fork:

```bash
git clone https://github.com/<your_username>/ARMxtend.git
cd ARMxtend
```

- Install the library in editable mode (see [Installation](installation.md) for the `spark` extra):

```bash
pip install -e ".[spark,dev]"   # pyspark + pytest, pytest-cov, ruff, mypy, build, twine, mkdocs...
```

The tests that run on a real local Spark need a JDK 17 or 21 (set `JAVA_HOME`; on macOS
`brew install openjdk@21`); without one they are skipped automatically.

## Making changes

1. Create a new branch for your change: `git checkout -b my-feature`.
2. Implement your change, following the style of the surrounding code.
3. Add or update unit tests under `tests/` covering it.
   Tests for the Spark-based algorithms (`FIM.apriori`, `FIM.Eclat`, `FIM.BD_ARE`, `FIM.BD_FARE`,
   `SARE`, `SFIM`) should run against `tests/spark_stub.py` so they don't require a JVM/cluster.
4. Run the same checks as the CI and make sure everything passes:

```bash
ruff check .                       # unused/undefined names, syntax errors
mypy                               # type check (the public API is typed; the package ships py.typed)
pytest --cov=ARMxtend              # tests + coverage (the CI requires >= 80 %)
```

   `pytest` alone runs everything (`testpaths` is set in `pyproject.toml`). Without `pyspark` the Spark-related
   tests are skipped, which is also how the CI checks that the core works without it.

5. If your change affects the public API or adds a new module, update the relevant page under
   `docs/sources/user_guide/` and add it to `docs/mkdocs.yml`'s `nav`, then check it renders
   correctly:

```bash
cd docs
mkdocs serve   # preview at http://127.0.0.1:8000
```

6. Commit your changes with a descriptive message and open a pull request against
   [cjferba/ARMxtend](https://github.com/cjferba/ARMxtend).

## Building and publishing the documentation

From the `docs` directory:

```bash
mkdocs build --strict   # builds the static site into docs/site/, failing on broken links
mkdocs gh-deploy        # builds and publishes docs/site/ to the gh-pages branch
```

## Releasing a new version

Releases are published to PyPI by `.github/workflows/release.yml` using PyPI *trusted publishing* (OpenID
Connect): no API token is stored in the repository or the CI.

**One-time setup** (needs the maintainer's PyPI account):

1. On <https://pypi.org/manage/account/publishing/> add a *pending publisher* for the project `ARMxtend`:
   owner `cjferba`, repository `ARMxtend`, workflow `release.yml`, environment `pypi`.
2. Optionally do the same on <https://test.pypi.org> with environment `testpypi`, to rehearse a release
   (Actions -> *Release* -> *Run workflow* publishes the current commit to TestPyPI).
3. In the repository settings create the environments `pypi` and `testpypi` (a required reviewer on `pypi`
   adds a manual approval before anything is uploaded).

**For each release:**

1. Update `__version__` in `ARMxtend/__init__.py` and move the entries of `CHANGELOG.md` from *unreleased* to the
   new version.
2. Commit, then create a GitHub release whose tag is `v<version>` (e.g. `v0.1.0`). The workflow refuses to publish
   if the tag does not match `__version__`.
3. The workflow builds the sdist and the wheel, runs `twine check --strict` and uploads them.

Locally, `python -m build && twine check --strict dist/*` does the same build and check without uploading.
