# -*- coding: utf-8 -*-
"""
Tests de regresion para los algoritmos de mineria de itemsets frecuentes en
Big Data (ARMxtend.FIM.apriori, ARMxtend.FIM.Eclat) y de reglas de asociacion
(ARMxtend.FIM.BD_ARE), usando el dataset de ejemplo (Tabla 2) de
Fernandez-Basso, Ruiz & Martin-Bautista (2024), "New Spark solutions for
distributed frequent itemset and association rule mining algorithms",
Cluster Computing, 27, 1217-1234.

Se ejecutan contra `spark_stub.FakeSparkContext`, un doble local de la
API de Spark usada por estos algoritmos (map/flatMap/reduceByKey/broadcast),
por lo que corren sin necesidad de una JVM/cluster real, ejecutando
exactamente el mismo codigo de produccion que se ejecutaria sobre un
SparkContext autentico.
"""
import numpy as np
import pytest

from spark_stub import FakeSparkContext

from ARMxtend.FIM.apriori import DApriori, DAprioriTID
from ARMxtend.FIM.Eclat import DECLAT, FuzzyDECLAT
from ARMxtend.FIM.BD_ARE import association_rules_bd
from ARMxtend.FIM.BD_FARE import FuzzyDAprioriTID

TABLE2_TRANSACTIONS = [
    "A,B,C", "B,D", "A,C,D", "B,C", "A,C",
    "B,D", "A,B,C", "B,C,D", "A,B,D", "B,C,D",
]

EXPECTED_FREQ_ITEMSETS = {
    "A": 0.5, "B": 0.8, "C": 0.7, "D": 0.6,
    "A-B": 0.3, "A-C": 0.4, "B-C": 0.5, "B-D": 0.5, "C-D": 0.3,
}


@pytest.fixture
def sc():
    return FakeSparkContext()


@pytest.mark.parametrize("algorithm", [DApriori, DAprioriTID, DECLAT])
def test_crisp_fim_algorithms_agree_with_hand_computed_itemsets(sc, algorithm):
    result = algorithm.run(sc, sc.parallelize(TABLE2_TRANSACTIONS), min_supp=0.3)
    assert result == pytest.approx(EXPECTED_FREQ_ITEMSETS)


def test_association_rules_bd_matches_sequential_generate_rules(sc):
    from ARMxtend.FIM.BD_ARE import generate_rules

    freqItemsets = DApriori.run(sc, sc.parallelize(TABLE2_TRANSACTIONS), min_supp=0.3)
    sequentialRules = generate_rules(freqItemsets, min_conf=0.7)
    distributedRules = association_rules_bd(sc, freqItemsets, min_conf=0.7)

    assert distributedRules == pytest.approx(sequentialRules)
    assert set(distributedRules.keys()) == {("A", "C"), ("C", "B"), ("D", "B")}


def test_fuzzy_declat_with_full_membership_matches_crisp_declat(sc):
    fuzzyTransactions = [[(item, 1.0) for item in line.split(",")] for line in TABLE2_TRANSACTIONS]

    crispResult = DECLAT.run(sc, sc.parallelize(TABLE2_TRANSACTIONS), min_supp=0.3)
    fuzzyResult = FuzzyDECLAT.run(sc, sc.parallelize(fuzzyTransactions), min_supp=0.3, num_alpha=3)

    assert set(fuzzyResult.keys()) == set(crispResult.keys())
    for key, crispSupport in crispResult.items():
        # Con pertenencia total (grado 1.0), el itemset satisface TODOS los
        # alpha-cortes (incluido el mas exigente, alpha=1, via >=), luego su
        # soporte coincide con el crisp en cada uno de ellos
        assert fuzzyResult[key] == pytest.approx(np.full(3, crispSupport))


def test_fuzzy_declat_partial_membership_is_non_decreasing_by_looseness(sc):
    # Con grado de pertenencia parcial (0.6) y niveles alpha=[1, 0.67, 0.33],
    # solo se supera (>=) el nivel mas laxo (0.33): el soporte debe ser 0 en
    # los niveles estrictos y coincidir con el crisp solo en el mas laxo.
    partialTransactions = [[(item, 0.6) for item in line.split(",")] for line in TABLE2_TRANSACTIONS]

    crispResult = DECLAT.run(sc, sc.parallelize(TABLE2_TRANSACTIONS), min_supp=0.3)
    # min_supp ligeramente por debajo de 0.3/3 para evitar el borde exacto
    # (0.3 no es representable exactamente en punto flotante binario)
    fuzzyResult = FuzzyDECLAT.run(sc, sc.parallelize(partialTransactions), min_supp=0.09, num_alpha=3)

    for key, crispSupport in crispResult.items():
        assert fuzzyResult[key] == pytest.approx([0.0, 0.0, crispSupport])
        # El soporte es no-decreciente al relajar la exigencia del alpha-corte
        assert np.all(np.diff(fuzzyResult[key]) >= -1e-9)


def test_fuzzy_daprioritid_with_full_membership_matches_crisp_dapriori(sc):
    fuzzyTransactions = [[(item, 1.0) for item in line.split(",")] for line in TABLE2_TRANSACTIONS]

    crispResult = DApriori.run(sc, sc.parallelize(TABLE2_TRANSACTIONS), min_supp=0.3)
    fuzzyResult = FuzzyDAprioriTID.run(sc, sc.parallelize(fuzzyTransactions), min_supp=0.3, num_alpha=3)

    assert set(fuzzyResult.keys()) == set(crispResult.keys())
    for key, crispSupport in crispResult.items():
        assert np.all(fuzzyResult[key] == pytest.approx(crispSupport))
