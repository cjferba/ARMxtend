# -*- coding: utf-8 -*-
"""
Regression tests for ARMxtend.FIM.FARE (fuzzy association rules via
alpha-cuts / RL-theory).

The main test reproduces, figure by figure, the numerical example of
Section 4.4 of:
    Ruiz, M.D., Gomez-Romero, J., Molina-Solana, M., Campana, J.R.,
    Martin-Bautista, M.J. (2016). "Meta-association rules for mining
    interesting associations in multiple datasets". Applied Soft Computing,
    49, 212-223.

using the fuzzy database of Table 2 of that paper (5 items, 6
transactions), the alpha-cuts Lambda={1, 0.8, 0.6, 0.4, 0.2} of Table 6,
and the itemset A={i1,i3} -> B={i4}, for which the paper reports
FSupp=0.266, FConf=0.5 and FCF=0.33.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "tests")
from spark_stub import FakeSparkContext  # noqa: E402

from ARMxtend.FIM.Eclat import FuzzyDECLAT  # noqa: E402
from ARMxtend.FIM.FARE import fuzzy_association_rules  # noqa: E402

# Table 2 of the paper: membership degree of each item i1..i5 in each
# transaccion t1..t6
_DEGREES = {
    "i1": [1, 1, 0.4, 0.6, 0.4, 0],
    "i2": [0.2, 1, 0.1, 0, 0.1, 1],
    "i3": [1, 0.8, 0.7, 0.4, 0, 0],
    "i4": [0.8, 0, 0.6, 0.4, 0, 0],
    "i5": [0.9, 0, 0, 0.5, 0, 0],
}
_NUM_ALPHA = 5  # Lambda = {1, 0.8, 0.6, 0.4, 0.2}


def _fuzzy_transactions():
    return [[(item, _DEGREES[item][t]) for item in _DEGREES] for t in range(6)]


@pytest.fixture
def freq_itemsets():
    sc = FakeSparkContext()
    return FuzzyDECLAT.run(sc, sc.parallelize(_fuzzy_transactions()), min_supp=0.0, num_alpha=_NUM_ALPHA)


def test_fsupp_fconf_fcf_match_paper_worked_example(freq_itemsets):
    rules = fuzzy_association_rules(freq_itemsets, num_alpha=_NUM_ALPHA, metric="support", min_threshold=0.0)
    rule = rules[(rules["antecedents"] == frozenset(["i1", "i3"])) & (rules["consequents"] == frozenset(["i4"]))]

    assert len(rule) == 1
    assert rule.iloc[0]["support"] == pytest.approx(0.2667, abs=1e-3)
    assert rule.iloc[0]["confidence"] == pytest.approx(0.5, abs=1e-9)
    assert rule.iloc[0]["certainty_factor"] == pytest.approx(0.33, abs=1e-9)


def test_rule_support_equals_union_itemset_support(freq_itemsets):
    """FSupp(A -> B) must match FSupp(A U B) exactly (Eq. 3)."""
    rules = fuzzy_association_rules(freq_itemsets, num_alpha=_NUM_ALPHA, metric="support", min_threshold=0.0)
    rule = rules[(rules["antecedents"] == frozenset(["i1", "i3"])) & (rules["consequents"] == frozenset(["i4"]))]
    assert rule.iloc[0]["support"] == pytest.approx(freq_itemsets["i1-i3-i4"] @
                                                     np.array([0.2, 0.2, 0.2, 0.2, 0.2]))


def test_min_threshold_filters_rules(freq_itemsets):
    allRules = fuzzy_association_rules(freq_itemsets, num_alpha=_NUM_ALPHA, metric="confidence", min_threshold=0.0)
    strictRules = fuzzy_association_rules(freq_itemsets, num_alpha=_NUM_ALPHA, metric="confidence",
                                          min_threshold=0.9)
    assert len(strictRules) <= len(allRules)
    assert all(strictRules["confidence"] >= 0.9)


def test_invalid_metric_raises(freq_itemsets):
    # 'conviction' is not supported in the fuzzy case: it is unbounded level by level
    with pytest.raises(ValueError):
        fuzzy_association_rules(freq_itemsets, num_alpha=_NUM_ALPHA, metric="conviction", min_threshold=0.5)
