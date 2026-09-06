# -*- coding: utf-8 -*-
"""
Tests de regresion para FFIM (FP-Growth crisp y difuso), verificados contra
el mismo dataset de ejemplo (Tabla 2) usado en test_fim_apriori_eclat.py y,
para el caso difuso, contra ARMxtend.FIM.Eclat.FuzzyDECLAT.
"""
import numpy as np
import pytest

from spark_stub import FakeSparkContext

from ARMxtend.FFIM import fpgrowth, fuzzy_fpgrowth
from ARMxtend.FIM.Eclat import FuzzyDECLAT

TABLE2_TRANSACTIONS = [
    "A,B,C", "B,D", "A,C,D", "B,C", "A,C",
    "B,D", "A,B,C", "B,C,D", "A,B,D", "B,C,D",
]

EXPECTED_FREQ_ITEMSETS = {
    "A": 0.5, "B": 0.8, "C": 0.7, "D": 0.6,
    "A-B": 0.3, "A-C": 0.4, "B-C": 0.5, "B-D": 0.5, "C-D": 0.3,
}


def test_fpgrowth_matches_hand_computed_itemsets():
    transactions = [line.split(",") for line in TABLE2_TRANSACTIONS]
    result = fpgrowth(transactions, min_supp=0.3)
    assert result == pytest.approx(EXPECTED_FREQ_ITEMSETS)


def test_fpgrowth_empty_transactions_returns_empty():
    assert fpgrowth([], min_supp=0.3) == {}


def test_fuzzy_fpgrowth_with_full_membership_matches_crisp_fpgrowth():
    fuzzyTransactions = [[(item, 1.0) for item in line.split(",")] for line in TABLE2_TRANSACTIONS]
    result = fuzzy_fpgrowth(fuzzyTransactions, min_supp=0.3, num_alpha=3)

    assert set(result.keys()) == set(EXPECTED_FREQ_ITEMSETS.keys())
    for key, crispSupport in EXPECTED_FREQ_ITEMSETS.items():
        assert np.all(result[key] == pytest.approx(crispSupport))


def test_fuzzy_fpgrowth_matches_fuzzy_declat_on_partial_membership():
    partialTransactions = [[(item, 0.6) for item in line.split(",")] for line in TABLE2_TRANSACTIONS]

    sc = FakeSparkContext()
    declatResult = FuzzyDECLAT.run(sc, sc.parallelize(partialTransactions), min_supp=0.1, num_alpha=4)
    fpgrowthResult = fuzzy_fpgrowth(partialTransactions, min_supp=0.1, num_alpha=4)

    for key, declatSupport in declatResult.items():
        assert np.allclose(fpgrowthResult.get(key, np.zeros(4)), declatSupport)
