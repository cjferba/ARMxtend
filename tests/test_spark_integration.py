# -*- coding: utf-8 -*-
"""
Tests de integracion con un Spark REAL (SparkContext en modo local[2]), a
diferencia del resto de tests de Big Data, que corren contra el doble en
memoria ``tests/spark_stub.py``.

Para cada algoritmo se comprueba que el resultado sobre Spark real coincide
con (a) el resultado obtenido con el doble local, es decir, que el stub es
fiel, y (b) una implementacion secuencial independiente (FP-Growth /
``generate_rules``) o los valores calculados a mano. Spark ejecuta los
closures en procesos Python distintos, por lo que estos tests detectan
problemas que el stub no puede ver: closures no serializables, variables
broadcast leidas en el sitio equivocado, reduccion entre particiones, etc.

Se omiten automaticamente si no hay ``pyspark`` o un JDK (>= 17) disponible.
En local: ``export JAVA_HOME=$(/usr/libexec/java_home)`` o equivalente.
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

# Tabla 2 de Ruiz et al. (2016): grado de pertenencia de i1..i5 en t1..t6
_DEGREES = {
    "i1": [1, 1, 0.4, 0.6, 0.4, 0],
    "i2": [0.2, 1, 0.1, 0, 0.1, 1],
    "i3": [1, 0.8, 0.7, 0.4, 0, 0],
    "i4": [0.8, 0, 0.6, 0.4, 0, 0],
    "i5": [0.9, 0, 0, 0.5, 0, 0],
}
FUZZY_TRANSACTIONS = [[(item, _DEGREES[item][t]) for item in _DEGREES] for t in range(6)]
NUM_ALPHA = 5
PARTITIONS = 3  # varias particiones: fuerza la reduccion entre particiones


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
