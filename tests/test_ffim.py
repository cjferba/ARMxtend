# -*- coding: utf-8 -*-
"""
Regression tests for FFIM (crisp and fuzzy FP-Growth), verified against
the same example dataset (Table 2) used in test_fim_apriori_eclat.py and,
for the fuzzy case, against ARMxtend.FIM.Eclat.FuzzyDECLAT.
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


def test_fpgrowth_with_many_repeated_transactions_matches_brute_force():
    """Equal transactions are grouped with weights: the result must be that of counting one by one."""
    from itertools import combinations
    from ARMxtend.FFIM import fpgrowth
    from ARMxtend.FIM._shared import key_to_itemset

    base = [["a", "b", "c"], ["a", "b"], ["b", "c", "d"], ["a", "d"], ["c"], []]
    transactions = [base[i % len(base)] for i in range(60)] + [["a", "b", "c", "d"]] * 7
    found = {key_to_itemset(key): supp for key, supp in fpgrowth(transactions, 0.2).items()}
    expected = {}
    for size in range(1, 5):
        for itemset in combinations("abcd", size):
            supp = sum(set(itemset) <= set(t) for t in transactions) / len(transactions)
            if supp >= 0.2 - 1e-9:
                expected[frozenset(itemset)] = supp
    assert found.keys() == expected.keys()
    assert all(abs(found[k] - expected[k]) < 1e-12 for k in expected)
