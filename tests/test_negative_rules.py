# -*- coding: utf-8 -*-
"""
Tests de las reglas con items ausentes (ARM.negative_rules), crisp y
difusas, segun la seccion 4 de Delgado, Ruiz, Sanchez & Serrano (2011).
"""
import numpy as np
import pytest

from ARMxtend.FFIM import fuzzy_fpgrowth
from ARMxtend.FIM._shared import alpha_levels, alpha_weights, weighted_alpha_aggregate

from ARMxtend.ARM.negative_rules import (
    add_absent_items, negative_association_rules, fuzzy_negative_association_rules, is_negated, negate,
)

# A aparece en 4/6 transacciones y nunca junto a C; C aparece en 2/6.
TRANSACTIONS = [
    {"A", "B"}, {"A", "B"}, {"A", "B"}, {"C"}, {"C", "B"}, {"A"},
]


def _rule(rules, antecedents, consequents):
    match = rules[(rules["antecedents"] == frozenset(antecedents)) & (rules["consequents"] == frozenset(consequents))]
    return match.iloc[0] if len(match) else None


def test_negate_helpers():
    assert negate("A") == "~A"
    assert is_negated("~A") and not is_negated("A")


def test_add_absent_items_adds_complement_of_each_transaction():
    extended = add_absent_items([{"A"}, {"B"}])
    assert extended == [{"A", "~B"}, {"B", "~A"}]
    assert add_absent_items([{"A"}, {"B"}], negate_items=["A"]) == [{"A"}, {"B", "~A"}]


def test_crisp_finds_rule_with_absent_consequent():
    rules = negative_association_rules(TRANSACTIONS, min_supp=0.3, min_threshold=0.9)
    rule = _rule(rules, {"A"}, {"~C"})
    assert rule is not None
    assert rule["support"] == pytest.approx(4 / 6)       # A y no C: T1, T2, T3, T6
    assert rule["confidence"] == pytest.approx(1.0)
    # el consecuente de A es la ausencia de C, y C -> ~A es la regla simetrica
    assert _rule(rules, {"C"}, {"~A"}) is not None


def test_crisp_never_mixes_an_item_with_its_own_absence():
    rules = negative_association_rules(TRANSACTIONS, min_supp=0.1, min_threshold=0.0, only_negative=False)
    for _, rule in rules.iterrows():
        items = set(rule["antecedents"]) | set(rule["consequents"])
        assert not any(negate(item) in items for item in items if not is_negated(item))


def test_crisp_only_negative_filters_positive_rules():
    positive = negative_association_rules(TRANSACTIONS, min_supp=0.3, min_threshold=0.0, only_negative=False)
    onlyNeg = negative_association_rules(TRANSACTIONS, min_supp=0.3, min_threshold=0.0)
    assert len(onlyNeg) < len(positive)
    assert all(any(is_negated(i) for i in set(r["antecedents"]) | set(r["consequents"]))
               for _, r in onlyNeg.iterrows())
    assert _rule(positive, {"A"}, {"B"}) is not None and _rule(onlyNeg, {"A"}, {"B"}) is None


def test_han_beheshti_restriction_only_negates_frequent_items():
    transactions = TRANSACTIONS + [{"D"}]       # D: soporte 1/7 < min_supp
    restricted = negative_association_rules(transactions, min_supp=0.2, min_threshold=0.0)
    unrestricted = negative_association_rules(transactions, min_supp=0.2, min_threshold=0.0,
                                              only_frequent_positive=False)
    items = lambda df: set().union(*(set(a) | set(c) for a, c in zip(df["antecedents"], df["consequents"])))
    assert "~D" not in items(restricted)
    assert "~D" in items(unrestricted)


def test_negate_items_limits_which_items_are_negated():
    rules = negative_association_rules(TRANSACTIONS, min_supp=0.3, min_threshold=0.0, negate_items=["C"])
    items = set().union(*(set(a) | set(c) for a, c in zip(rules["antecedents"], rules["consequents"])))
    assert {i for i in items if is_negated(i)} == {"~C"}


def test_crisp_empty_input():
    assert len(negative_association_rules([], min_supp=0.5)) == 0


# --- difuso ---------------------------------------------------------------

def _crisp_as_fuzzy(transactions):
    return [[(item, 1.0) for item in t] for t in transactions]


def test_fuzzy_with_crisp_degrees_matches_crisp_negative_rules():
    crisp = negative_association_rules(TRANSACTIONS, min_supp=0.3, min_threshold=0.0)
    fuzzy = fuzzy_negative_association_rules(_crisp_as_fuzzy(TRANSACTIONS), min_supp=0.3, num_alpha=4,
                                             min_threshold=0.0)

    key = lambda df: {(a, c): (s, cf) for a, c, s, cf in
                      zip(df["antecedents"], df["consequents"], df["support"], df["confidence"])}
    crispRules, fuzzyRules = key(crisp), key(fuzzy)
    assert set(crispRules) == set(fuzzyRules)
    for ruleKey, (support, confidence) in crispRules.items():
        assert fuzzyRules[ruleKey][0] == pytest.approx(support)
        assert fuzzyRules[ruleKey][1] == pytest.approx(confidence)


def test_fuzzy_absence_is_level_complement_not_one_minus_degree():
    """
    Ec. (27) de Delgado et al.: en cada nivel, supp(X ^ ~A) = supp(X) - supp(X ^ A),
    luego FSupp(X ^ A) + FSupp(X ^ ~A) = FSupp(X). Con la negacion de
    conjuntos difusos (1 - grado) esta identidad no se cumpliria.
    """
    rng = np.random.RandomState(0)
    transactions = [[("X", float(rng.choice([0.3, 0.6, 1.0]))), ("A", float(rng.rand()))] for _ in range(30)]
    numAlpha = 5

    rules = fuzzy_negative_association_rules(transactions, min_supp=0.01, num_alpha=numAlpha, metric="support",
                                             min_threshold=0.0, only_negative=False)
    withA = _rule(rules, {"X"}, {"A"})["support"]
    withoutA = _rule(rules, {"X"}, {"~A"})["support"]

    xSupportVector = fuzzy_fpgrowth(transactions, min_supp=0.01, num_alpha=numAlpha)["X"]
    fsuppX = weighted_alpha_aggregate(xSupportVector, alpha_weights(alpha_levels(numAlpha)))

    assert withA + withoutA == pytest.approx(fsuppX)
    assert withoutA > 0


def test_fuzzy_complement_values_for_single_transaction():
    # un solo grado 0.6 con Lambda={1,.8,.6,.4,.2}: A vale en los niveles .6,.4,.2 (FSupp=.6);
    # ~A vale en 1 y .8 (FSupp=.4 = 1 - .6). Con 1 - grado seria [0,0,0,1,1].
    transactions = [[("A", 0.6), ("B", 1.0)]]
    rules = fuzzy_negative_association_rules(transactions, min_supp=0.01, num_alpha=5, metric="support",
                                             min_threshold=0.0, only_negative=False)
    assert _rule(rules, {"B"}, {"A"})["support"] == pytest.approx(0.6)
    assert _rule(rules, {"B"}, {"~A"})["support"] == pytest.approx(0.4)


def test_fuzzy_empty_input_and_extra_measures():
    assert len(fuzzy_negative_association_rules([], 0.5, 4)) == 0
    rules = fuzzy_negative_association_rules(_crisp_as_fuzzy(TRANSACTIONS), 0.3, 4, min_threshold=0.0,
                                             measures=["zhangs_metric"])
    assert "zhangs_metric" in rules.columns
