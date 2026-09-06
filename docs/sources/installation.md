# Installing ARMxtend

---

ARMxtend is not yet published on PyPI. Install it directly from a local checkout of the
[GitHub repository](https://github.com/cjferba/ARMxtend):

```bash
git clone https://github.com/cjferba/ARMxtend.git
cd ARMxtend
pip install -e .
```

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

`pyspark` itself requires a Java runtime (JDK 8/11/17) to be installed and on your `PATH`; see the
[PySpark installation guide](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)
if `pyspark` fails to start a `SparkContext`.

### Running the tests

```bash
pip install pytest
pytest tests/ ARMxtend/FIM/BD_ARE/TestFunctions.py
```

The tests covering the Spark-based algorithms run against `tests/spark_stub.py`, a local double of
the small part of the Spark RDD/broadcast API these algorithms use, so they exercise the exact same
production code without requiring a real cluster or a JVM.
