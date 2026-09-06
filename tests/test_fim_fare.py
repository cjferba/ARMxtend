# -*- coding: utf-8 -*-
"""
Tests de regresion para ARMxtend.FIM.FARE (reglas de asociacion difusas via
alpha-cortes / RL-theory).

El test principal reproduce, cifra a cifra, el ejemplo numerico de la
Seccion 4.4 de:
    Ruiz, M.D., Gomez-Romero, J., Molina-Solana, M., Campana, J.R.,
    Martin-Bautista, M.J. (2016). "Meta-association rules for mining
    interesting associations in multiple datasets". Applied Soft Computing,
    49, 212-223.

usando la base de datos difusa de la Tabla 2 de ese articulo (5 items, 6
transacciones), los alpha-cortes Lambda={1, 0.8, 0.6, 0.4, 0.2} de la Tabla 6,
y el itemset A={i1,i3} -> B={i4}, para el que el articulo reporta
FSupp=0.266, FConf=0.5 y FCF=0.33.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "tests")
from spark_stub import FakeSparkContext  # noqa: E402

from ARMxtend.FIM.Eclat import FuzzyDECLAT  # noqa: E402
from ARMxtend.FIM.FARE import fuzzy_association_rules  # noqa: E402

# Tabla 2 del articulo: grado de pertenencia de cada item i1..i5 en cada
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
    """FSupp(A -> B) debe coincidir exactamente con FSupp(A U B) (Ec. 3)."""
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
    with pytest.raises(ValueError):
        fuzzy_association_rules(freq_itemsets, num_alpha=_NUM_ALPHA, metric="lift", min_threshold=0.5)
