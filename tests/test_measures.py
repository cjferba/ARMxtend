# -*- coding: utf-8 -*-
"""
Tests de las medidas de interes adicionales (ARM._measures), de su uso en
ARM.association_rules y de su generalizacion difusa por niveles en
FIM.FARE (RL-theory: medida crisp en cada alpha-corte + agregacion ponderada).
"""
import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import association_rules
from ARMxtend.ARM._measures import MEASURES, EXTRA_MEASURES
from ARMxtend.FIM._shared import alpha_levels, alpha_weights, weighted_alpha_aggregate
from ARMxtend.FIM.FARE import fuzzy_association_rules
from ARMxtend.FFIM import fuzzy_fpgrowth

# supp(AB)=.4, supp(A)=.5, supp(B)=.7 -> a=.4, b=.1, c=.3, d=.2
S_AB, S_A, S_B = 0.4, 0.5, 0.7

EXPECTED = {
    "confidence": 0.8,
    "lift": 0.8 / 0.7,
    "leverage": 0.4 - 0.35,
    "jaccard": 0.4 / 0.8,
    "cosine": 0.4 / np.sqrt(0.35),
    "kulczynski": 0.5 * (0.8 + 0.4 / 0.7),
    "all_confidence": 0.4 / 0.7,
    "max_confidence": 0.8,
    "added_value": 0.8 - 0.7,
    "zhangs_metric": 0.05 / 0.2,
    "yules_q": (0.4 * 0.2 - 0.1 * 0.3) / (0.4 * 0.2 + 0.1 * 0.3),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_measure_matches_hand_computed_value(name):
    assert MEASURES[name](S_AB, S_A, S_B) == pytest.approx(EXPECTED[name])


@pytest.mark.parametrize("name", sorted(MEASURES))
def test_measures_accept_arrays_and_return_floats_for_scalars(name):
    assert isinstance(MEASURES[name](S_AB, S_A, S_B), float)
    vector = MEASURES[name]([S_AB, S_AB], [S_A, S_A], [S_B, S_B])
    assert vector.shape == (2,)
    assert vector[0] == pytest.approx(MEASURES[name](S_AB, S_A, S_B))


@pytest.mark.parametrize("name", sorted(MEASURES))
def test_measures_are_neutral_without_evidence(name):
    """Sin soporte (0/0) las medidas no producen NaN/inf: devuelven el valor neutro 0."""
    value = MEASURES[name](0.0, 0.0, 0.0)
    assert np.isfinite(value)


def test_symmetric_measures_do_not_depend_on_rule_direction():
    for name in ("jaccard", "cosine", "kulczynski", "all_confidence", "max_confidence", "yules_q"):
        assert MEASURES[name](S_AB, S_A, S_B) == pytest.approx(MEASURES[name](S_AB, S_B, S_A))


def test_independence_gives_neutral_values():
    # supp(AB) = supp(A) supp(B)
    sA, sB = 0.4, 0.5
    sAB = sA * sB
    assert MEASURES["lift"](sAB, sA, sB) == pytest.approx(1.0)
    for name in ("leverage", "added_value", "zhangs_metric", "yules_q", "certainty_factor"):
        assert MEASURES[name](sAB, sA, sB) == pytest.approx(0.0, abs=1e-12)


def test_perfect_positive_association():
    # A <-> B siempre juntos: supp(AB) = supp(A) = supp(B)
    assert MEASURES["zhangs_metric"](0.3, 0.3, 0.3) == pytest.approx(1.0)
    assert MEASURES["yules_q"](0.3, 0.3, 0.3) == pytest.approx(1.0)
    assert MEASURES["jaccard"](0.3, 0.3, 0.3) == pytest.approx(1.0)


def test_association_rules_exposes_all_extra_measures():
    itemsets = pd.DataFrame({
        "support": [0.5, 0.7, 0.4],
        "itemsets": [frozenset("A"), frozenset("B"), frozenset("AB")],
    })
    rules = association_rules(itemsets, metric="confidence", min_threshold=0.0)
    ruleAB = rules[rules["antecedents"] == frozenset("A")].iloc[0]
    for name in EXTRA_MEASURES:
        assert ruleAB[name] == pytest.approx(EXPECTED[name])


def test_association_rules_can_filter_by_extra_metric():
    itemsets = pd.DataFrame({
        "support": [0.5, 0.7, 0.4],
        "itemsets": [frozenset("A"), frozenset("B"), frozenset("AB")],
    })
    rules = association_rules(itemsets, metric="kulczynski", min_threshold=0.7)
    # kulczynski es simetrica (0.6857 < 0.7): ninguna de las dos direcciones pasa
    assert len(rules) == 0
    rules = association_rules(itemsets, metric="kulczynski", min_threshold=0.6)
    assert len(rules) == 2


def test_association_rules_invalid_metric_lists_valid_ones():
    itemsets = pd.DataFrame({"support": [0.5], "itemsets": [frozenset("A")]})
    with pytest.raises(ValueError, match="kulczynski"):
        association_rules(itemsets, metric="not-a-metric")


# --- caso difuso ---------------------------------------------------------

def _fuzzy_itemsets(num_alpha=4):
    transactions = [
        [("A", 1.0), ("B", 0.9)],
        [("A", 0.7), ("B", 0.6)],
        [("A", 0.4), ("B", 0.3)],
        [("A", 0.2)],
        [("B", 0.8)],
    ]
    return fuzzy_fpgrowth(transactions, min_supp=0.01, num_alpha=num_alpha)


def test_fuzzy_measure_equals_weighted_aggregate_of_crisp_measure_per_level():
    num_alpha = 4
    itemsets = _fuzzy_itemsets(num_alpha)
    weights = alpha_weights(alpha_levels(num_alpha))

    rules = fuzzy_association_rules(itemsets, num_alpha, metric="support", min_threshold=0.0,
                                    measures=["jaccard", "yules_q", "lift"])
    rule = rules[(rules["antecedents"] == frozenset("A")) & (rules["consequents"] == frozenset("B"))].iloc[0]

    vAB, vA, vB = itemsets["A-B"], itemsets["A"], itemsets["B"]
    for name in ("jaccard", "yules_q", "lift"):
        expected = weighted_alpha_aggregate(MEASURES[name](vAB, vA, vB), weights)
        assert rule[name] == pytest.approx(expected)


def test_fuzzy_default_columns_are_unchanged():
    rules = fuzzy_association_rules(_fuzzy_itemsets(), 4, metric="support", min_threshold=0.0)
    assert list(rules.columns) == ["antecedents", "consequents", "support", "confidence", "certainty_factor"]


def test_fuzzy_can_filter_by_extra_metric_without_returning_it():
    itemsets = _fuzzy_itemsets()
    everything = fuzzy_association_rules(itemsets, 4, metric="jaccard", min_threshold=0.0, measures=["jaccard"])
    filtered = fuzzy_association_rules(itemsets, 4, metric="jaccard", min_threshold=0.99)
    assert "jaccard" not in filtered.columns
    assert len(filtered) == int((everything["jaccard"] >= 0.99).sum())


def test_fuzzy_unknown_measure_raises():
    with pytest.raises(ValueError):
        fuzzy_association_rules(_fuzzy_itemsets(), 4, measures=["nope"])


# --- factor de certeza frente a la Ec. (7) de Ruiz et al. (2016) -------------

def _cf_from_4ft(a, b, c, d):
    """CF(a, b, c, d), Ec. (7), con a + b + c + d = 1 (tabla de frecuencias relativas)."""
    if a * d > b * c:
        return (a * d - b * c) / ((a + b) * (b + d))
    if a * d < b * c:
        return (a * d - b * c) / ((a + b) * (a + c))
    return 0.0


def test_certainty_factor_matches_equation_7_on_random_tables():
    rng = np.random.RandomState(0)
    for _ in range(500):
        a, b, c, d = rng.dirichlet([1, 1, 1, 1])
        assert MEASURES["certainty_factor"](a, a + b, a + c) == pytest.approx(_cf_from_4ft(a, b, c, d))


def test_certainty_factor_is_zero_when_the_antecedent_has_no_support():
    # a = b = 0 -> ad = bc -> CF = 0 (no -1: sin antecedente no hay evidencia ni a favor ni en contra)
    assert MEASURES["certainty_factor"](0.0, 0.0, 0.4) == 0.0
    assert np.allclose(MEASURES["certainty_factor"]([0.0, 0.2], [0.0, 0.5], [0.4, 0.4]),
                       [0.0, _cf_from_4ft(0.2, 0.3, 0.2, 0.3)])


def test_fuzzy_certainty_factor_ignores_levels_where_the_antecedent_never_appears():
    # A solo llega a grado 0.5: en los niveles 1 y 0.75 (de 4) no tiene soporte y su CF es 0
    transactions = [[("A", 0.5), ("B", 0.5)], [("A", 0.5)], [("B", 1.0)], [("B", 0.5)]]
    numAlpha = 4
    itemsets = fuzzy_fpgrowth(transactions, min_supp=0.01, num_alpha=numAlpha)
    rule = fuzzy_association_rules(itemsets, numAlpha, metric="support", min_threshold=0.0).pipe(
        lambda df: df[(df["antecedents"] == frozenset("A")) & (df["consequents"] == frozenset("B"))].iloc[0])

    weights = alpha_weights(alpha_levels(numAlpha))
    perLevel = [_cf_from_4ft(*_table(itemsets, level)) for level in range(numAlpha)]
    assert perLevel[0] == 0.0 and perLevel[1] == 0.0
    assert rule["certainty_factor"] == pytest.approx(weighted_alpha_aggregate(perLevel, weights))


def _table(itemsets, level):
    sAB, sA, sB = itemsets["A-B"][level], itemsets["A"][level], itemsets["B"][level]
    return sAB, sA - sAB, sB - sAB, 1 - sA - sB + sAB
