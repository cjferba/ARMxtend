# -*- coding: utf-8 -*-
"""
Regression tests for candidate generation (`FIM._shared`) and for
association rule generation (`FIM.BD_ARE.generate_rules`), using the example
dataset (Table 2) of Fernandez-Basso, Ruiz & Martin-Bautista (2024),
"New Spark solutions for distributed frequent itemset and association rule
mining algorithms", Cluster Computing, 27, 1217-1234.

Replaces the file of the same name that existed previously, which was not a
real test but a fragment of manual test code (with no asserts) that
could not be run on its own.

Run: pytest tests/test_bd_are.py
"""
import pytest

from ARMxtend.FIM._shared import generate_candidates
from ARMxtend.FIM.BD_ARE import generate_rules

# Expected frequent itemsets for the dataset of Table 2 of the paper,
# with minimum support 0.3 (computed by hand and also verified against
# DApriori/DAprioriTID/DECLAT en tests/test_fim_apriori_eclat.py)
FREQ_ITEMSETS = {
    "A": 0.5, "B": 0.8, "C": 0.7, "D": 0.6,
    "A-B": 0.3, "A-C": 0.4, "B-C": 0.5, "B-D": 0.5, "C-D": 0.3,
}


def test_generate_candidates_from_singletons():
    candidates = generate_candidates(["A", "B", "C", "D"])
    assert candidates == ["A-B", "A-C", "A-D", "B-C", "B-D", "C-D"]


def test_generate_candidates_prunes_infrequent_subsets():
    # A-D is not frequent in Table 2, so no candidate of length 3
    # that contains it must be generated from the frequent pairs
    frequentPairs = ["A-B", "A-C", "B-C", "B-D", "C-D"]
    candidates = generate_candidates(frequentPairs)
    assert candidates == ["A-B-C", "B-C-D"]


def test_generate_candidates_empty_when_no_pairs_share_prefix():
    assert generate_candidates(["A-D"]) == []
    assert generate_candidates([]) == []


def test_generate_rules_matches_hand_computed_confidences():
    rules = generate_rules(FREQ_ITEMSETS, min_conf=0.0)

    expected = {
        ("A", "B"): 0.3 / 0.5,
        ("B", "A"): 0.3 / 0.8,
        ("A", "C"): 0.4 / 0.5,
        ("C", "A"): 0.4 / 0.7,
        ("B", "C"): 0.5 / 0.8,
        ("C", "B"): 0.5 / 0.7,
        ("B", "D"): 0.5 / 0.8,
        ("D", "B"): 0.5 / 0.6,
        ("C", "D"): 0.3 / 0.7,
        ("D", "C"): 0.3 / 0.6,
    }
    assert rules.keys() == expected.keys()
    for rule, confidence in expected.items():
        assert rules[rule] == pytest.approx(confidence)


def test_generate_rules_filters_by_min_confidence():
    rules = generate_rules(FREQ_ITEMSETS, min_conf=0.7)
    # Only A->C (0.8), C->B (0.714), D->B (0.833) exceed 0.7 confidence
    assert set(rules.keys()) == {("A", "C"), ("C", "B"), ("D", "B")}
    assert all(confidence >= 0.7 for confidence in rules.values())


def test_generate_rules_ignores_singleton_itemsets():
    rules = generate_rules({"A": 0.5}, min_conf=0.0)
    assert rules == {}
