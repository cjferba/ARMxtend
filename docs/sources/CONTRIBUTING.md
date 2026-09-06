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
pip install -e ".[spark]"
pip install pytest mkdocs
```

## Making changes

1. Create a new branch for your change: `git checkout -b my-feature`.
2. Implement your change, following the style of the surrounding code.
3. Add or update unit tests under `tests/` (or `ARMxtend/FIM/BD_ARE/TestFunctions.py`) covering it.
   Tests for the Spark-based algorithms (`FIM.apriori`, `FIM.Eclat`, `FIM.BD_ARE`, `FIM.BD_FARE`,
   `SARE`, `SFIM`) should run against `tests/spark_stub.py` so they don't require a JVM/cluster.
4. Run the test suite and make sure everything passes:

```bash
pytest tests/ ARMxtend/FIM/BD_ARE/TestFunctions.py
```

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
