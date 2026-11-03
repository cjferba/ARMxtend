# -*- coding: utf-8 -*-
"""
Integration tests with a REAL Spark (SparkContext in local[2] mode), unlike
the rest of the Big Data tests, which run against the in-memory stand-in
``tests/spark_stub.py``.

For each algorithm it is checked that the result on real Spark matches
(a) the result obtained with the local stand-in, that is, that the stub is
faithful, and (b) an independent sequential implementation (FP-Growth /
``generate_rules``) or the hand-computed values. Spark runs the
closures in different Python processes, so these tests detect
problems the stub cannot see: non-serializable closures, broadcast
variables read in the wrong place, reduction between partitions, etc.

They are skipped automatically if there is no ``pyspark`` or JDK (>= 17) available.
Locally: ``export JAVA_HOME=$(/usr/libexec/java_home)`` or equivalent.
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("pyspark")

sys.path.insert(0, os.path.dirname(__file__))
from spark_stub import FakeSparkContext  # noqa: E402

from ARMxtend.FFIM import fpgrowth, fuzzy_fpgrowth  # noqa: E402
from ARMxtend.FIM.apriori import DApriori, DAprioriTID  # noqa: E402
from ARMxtend.FIM.BD_ARE import association_rules_bd, generate_rules  # noqa: E402
from ARMxtend.FIM.BD_FARE import FuzzyDAprioriTID  # noqa: E402
from ARMxtend.FIM.Eclat import DECLAT, FuzzyDECLAT  # noqa: E402

TABLE2_TRANSACTIONS = [
    "A,B,C", "B,D", "A,C,D", "B,C", "A,C",
    "B,D", "A,B,C", "B,C,D", "A,B,D", "B,C,D",
]

# Table 2 of Ruiz et al. (2016): membership degree of i1..i5 in t1..t6
_DEGREES = {
    "i1": [1, 1, 0.4, 0.6, 0.4, 0],
    "i2": [0.2, 1, 0.1, 0, 0.1, 1],
    "i3": [1, 0.8, 0.7, 0.4, 0, 0],
    "i4": [0.8, 0, 0.6, 0.4, 0, 0],
    "i5": [0.9, 0, 0, 0.5, 0, 0],
}
FUZZY_TRANSACTIONS = [[(item, _DEGREES[item][t]) for item in _DEGREES] for t in range(6)]
NUM_ALPHA = 5
PARTITIONS = 3  # several partitions: forces the reduction between partitions


@pytest.fixture
def sc(real_sc):
    return real_sc


@pytest.mark.parametrize("algorithm", [DApriori, DAprioriTID, DECLAT])
def test_crisp_algorithms_on_real_spark_match_stub_and_fpgrowth(sc, algorithm):
    realResult = algorithm.run(sc, sc.parallelize(TABLE2_TRANSACTIONS, PARTITIONS), min_supp=0.3)

    stub = FakeSparkContext()
    stubResult = algorithm.run(stub, stub.parallelize(TABLE2_TRANSACTIONS), min_supp=0.3)
    sequentialResult = fpgrowth([line.split(",") for line in TABLE2_TRANSACTIONS], min_supp=0.3)

    assert realResult == pytest.approx(stubResult)
    assert realResult == pytest.approx(sequentialResult)


def test_association_rules_bd_on_real_spark_matches_sequential(sc):
    freqItemsets = DApriori.run(sc, sc.parallelize(TABLE2_TRANSACTIONS, PARTITIONS), min_supp=0.3)

    distributedRules = association_rules_bd(sc, freqItemsets, min_conf=0.7)

    assert distributedRules == pytest.approx(generate_rules(freqItemsets, min_conf=0.7))
    assert set(distributedRules) == {("A", "C"), ("C", "B"), ("D", "B")}


@pytest.mark.parametrize("algorithm", [FuzzyDECLAT, FuzzyDAprioriTID])
def test_fuzzy_algorithms_on_real_spark_match_stub_and_fuzzy_fpgrowth(sc, algorithm):
    realResult = algorithm.run(sc, sc.parallelize(FUZZY_TRANSACTIONS, PARTITIONS),
                               min_supp=0.1, num_alpha=NUM_ALPHA)

    stub = FakeSparkContext()
    stubResult = algorithm.run(stub, stub.parallelize(FUZZY_TRANSACTIONS),
                               min_supp=0.1, num_alpha=NUM_ALPHA)
    sequentialResult = fuzzy_fpgrowth(FUZZY_TRANSACTIONS, min_supp=0.1, num_alpha=NUM_ALPHA)

    assert set(realResult) == set(stubResult) == set(sequentialResult)
    for key, vector in realResult.items():
        assert np.allclose(vector, stubResult[key])
        assert np.allclose(vector, sequentialResult[key])


def test_declat_vertical_arrays_on_real_spark_match_fpgrowth_on_many_partitions(sc):
    """The numpy arrays are built on the executors, travel to the driver and are broadcast: same result as FP-Growth."""
    rng = np.random.RandomState(7)
    probability = rng.uniform(0.4, 0.85, 9)
    lines = [",".join("i{}".format(k) for k in range(9) if rng.rand() < probability[k]) for _ in range(400)]

    result = DECLAT.run(sc, sc.parallelize(lines, 6), min_supp=0.1)
    expected = fpgrowth([line.split(",") if line else [] for line in lines], min_supp=0.1)

    assert max(len(key.split("-")) for key in result) >= 4
    assert set(result) == set(expected) and result == pytest.approx(expected)


def test_fuzzy_declat_vertical_arrays_on_real_spark_match_fuzzy_fpgrowth_on_many_partitions(sc):
    rng = np.random.RandomState(8)
    grid = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.33, 0.97])
    transactions = [[("i{}".format(k), float(rng.choice(grid))) for k in range(7) if rng.rand() < 0.8] for _ in range(300)]

    result = FuzzyDECLAT.run(sc, sc.parallelize(transactions, 6), min_supp=0.05, num_alpha=5)
    expected = fuzzy_fpgrowth(transactions, min_supp=0.05, num_alpha=5)

    assert set(result) == set(expected) and max(len(key.split("-")) for key in result) >= 3
    for key, vector in result.items():
        assert np.allclose(vector, expected[key], atol=1e-12)


def test_fuzzy_declat_on_real_spark_rejects_degrees_outside_the_unit_interval(sc):
    with pytest.raises(Exception, match="membership degree"):
        FuzzyDECLAT.run(sc, sc.parallelize([[("a", 0.5), ("b", 1.5)]] * 4, 2), min_supp=0.1, num_alpha=3)


# ------------------------------------------------------------ entrada desde DataFrames / Parquet

def test_crisp_algorithms_read_parquet_and_dataframes(sc, tmp_path):
    from pyspark.sql import SparkSession

    from ARMxtend.FIM import read_transactions, transactions_from_dataframe

    session = SparkSession.builder.getOrCreate()
    baskets = [line.split(",") for line in TABLE2_TRANSACTIONS]
    path = str(tmp_path / "baskets.parquet")
    session.createDataFrame([(b,) for b in baskets], "items array<string>").repartition(3).write.parquet(path)
    expected = fpgrowth(baskets, min_supp=0.3)

    assert DECLAT.run(sc, read_transactions(session, path), 0.3) == pytest.approx(expected)
    text = session.createDataFrame([(" , ".join(b) + ",",) for b in baskets] + [(None,)], "line string")
    assert sorted(transactions_from_dataframe(text, "line", sep=",").collect())[:2] == ["", "A,B,C"]
    assert DApriori.run(sc, transactions_from_dataframe(text.na.drop(), "line", sep=","), 0.3) == pytest.approx(expected)
    for call in (lambda: transactions_from_dataframe(text, "line"), lambda: transactions_from_dataframe(text, "nope"),
                 lambda: transactions_from_dataframe(session.createDataFrame([(["a"],)], "x array<string>"), "x", sep=",")):
        with pytest.raises(ValueError):
            call()


def test_fuzzy_algorithms_read_struct_and_parallel_columns(sc, tmp_path):
    from pyspark.sql import SparkSession

    from ARMxtend.FIM import fuzzy_transactions_from_dataframe, read_transactions

    session = SparkSession.builder.getOrCreate()
    expected = fuzzy_fpgrowth(FUZZY_TRANSACTIONS, min_supp=0.1, num_alpha=NUM_ALPHA)
    structs = session.createDataFrame([([{"item": i, "degree": float(d)} for i, d in t],) for t in FUZZY_TRANSACTIONS],
                                      "items array<struct<item:string,degree:double>>")
    path = str(tmp_path / "fuzzy.parquet")
    structs.write.parquet(path)
    parallel = session.createDataFrame([([i for i, _ in t], [float(d) for _, d in t]) for t in FUZZY_TRANSACTIONS],
                                       "items array<string>, degrees array<double>")
    for rdd in (read_transactions(session, path, fuzzy=True), fuzzy_transactions_from_dataframe(parallel, "items", "degrees")):
        result = FuzzyDECLAT.run(sc, rdd, 0.1, NUM_ALPHA)
        assert set(result) == set(expected)
        for key, vector in result.items():
            assert np.allclose(vector, expected[key])
    with pytest.raises(ValueError):
        fuzzy_transactions_from_dataframe(parallel, "nope")
    bad = session.createDataFrame([(["a", "b"], [0.5])], "items array<string>, degrees array<double>")
    with pytest.raises(Exception, match="same size"):
        fuzzy_transactions_from_dataframe(bad, "items", "degrees").collect()


def test_prefixspan_spark_accepts_dataframes_without_collecting(sc, tmp_path):
    from pyspark.sql import SparkSession

    from ARMxtend.FIM.sequential import prefixspan, prefixspan_spark

    session = SparkSession.builder.getOrCreate()
    database = [[["a"], ["b", "c"], ["a"]], [["a"], ["c"]], [["b"], ["a", "a"], []], [["a", "b"], ["c"]], [[], ["c"]]]
    path = str(tmp_path / "seq.parquet")
    session.createDataFrame([(s,) for s in database], "events array<array<string>>").write.parquet(path)

    expected = prefixspan(database, 0.4, max_len=4)
    for result in (prefixspan_spark(session, session.read.parquet(path), 0.4, 4, sequence_column="events"),
                   prefixspan_spark(session, sc.parallelize(database, 2), 0.4, 4)):
        assert dict(zip(result["sequence"], result["support"])) == pytest.approx(dict(zip(expected["sequence"], expected["support"])))
    empty = session.createDataFrame([], "events array<array<string>>")
    assert len(prefixspan_spark(session, empty, 0.4, sequence_column="events")) == 0
