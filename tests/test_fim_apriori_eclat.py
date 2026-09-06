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
        # A alpha=0 (presencia binaria) el soporte difuso coincide con el crisp
        assert fuzzyResult[key][0] == pytest.approx(crispSupport)
        # El soporte es no-creciente al aumentar la exigencia del alpha-corte
        assert np.all(np.diff(fuzzyResult[key]) <= 1e-9)


def test_fuzzy_daprioritid_with_full_membership_matches_crisp_dapriori(sc):
    fuzzyTransactions = [[(item, 1.0) for item in line.split(",")] for line in TABLE2_TRANSACTIONS]

    crispResult = DApriori.run(sc, sc.parallelize(TABLE2_TRANSACTIONS), min_supp=0.3)
    fuzzyResult = FuzzyDAprioriTID.run(sc, sc.parallelize(fuzzyTransactions), min_supp=0.3, num_alpha=3)

    assert set(fuzzyResult.keys()) == set(crispResult.keys())
    for key, crispSupport in crispResult.items():
        assert np.all(fuzzyResult[key] == pytest.approx(crispSupport))
