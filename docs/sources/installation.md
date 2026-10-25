# Installing ARMxtend

---

ARMxtend is being prepared for its first release on PyPI (see the [changelog](https://github.com/cjferba/ARMxtend/blob/master/CHANGELOG.md)).
Until it is published, install it from the [GitHub repository](https://github.com/cjferba/ARMxtend):

```bash
pip install git+https://github.com/cjferba/ARMxtend.git
# or, to work on it, from a local checkout:
git clone https://github.com/cjferba/ARMxtend.git
cd ARMxtend
pip install -e .
```

Once published, `pip install armxtend` will be enough. It requires Python 3.9 or newer (tested on 3.9-3.12).
The installation also provides the [`armxtend` command](user_guide/cli.md).

This installs the core, single-machine part of the library: `ARM` (crisp association rules and
meta-rules), `FIM.FARE` (fuzzy association rules), `FFIM` (crisp and fuzzy FP-Growth), `VizARM`
and `preprocessing`. Its only dependencies are `numpy`, `pandas` and `networkx`.

### Big Data / streaming extra

The Big Data and streaming algorithms (`FIM.apriori`, `FIM.Eclat`, `FIM.BD_ARE`, `FIM.BD_FARE`,
`SARE`, `SFIM`) run on [Apache Spark](https://spark.apache.org/) and additionally require
`pyspark`. Install it with the `spark` extra:

```bash
pip install -e ".[spark]"
```

`pyspark` itself requires a Java runtime (JDK 17 or 21 for PySpark 4) to be installed and on your `PATH`; see the
[PySpark installation guide](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)
if `pyspark` fails to start a `SparkContext`.

### Running the tests

```bash
pip install -e ".[spark,dev]"
pytest
```

The tests that run the algorithms on a real local Spark (including the FIMoTS streaming driver) need a JDK; they are
skipped when none is found.

The tests covering the Spark-based algorithms run against `tests/spark_stub.py`, a local double of
the small part of the Spark RDD/broadcast API these algorithms use, so they exercise the exact same
production code without requiring a real cluster or a JVM.
