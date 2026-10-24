# -*- coding: utf-8 -*-
"""Tests de ARM.gradual_rules (reglas graduales, soporte por pares al estilo Hullermeier)."""
import itertools

import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM.gradual_rules import (
    gradual_itemsets, gradual_rules, format_gradual_item, format_gradual_itemset,
)


def _brute_force_support(table, itemset):
    """Cuenta pares NO ordenados que satisfacen el itemset en alguna orientacion."""
    n = len(table)
    count = 0
    for i, j in itertools.combinations(range(n), 2):
        for s, t in ((i, j), (j, i)):
            if all((table[a][s] < table[a][t]) if d == "+" else (table[a][s] > table[a][t]) for a, d in itemset):
                count += 1
                break
    return count / (n * (n - 1) / 2)


@pytest.fixture
def random_table():
    rng = np.random.RandomState(1)
    return pd.DataFrame(rng.randint(0, 6, size=(25, 3)), columns=["A", "B", "C"])  # con empates


def test_pair_support_matches_brute_force(random_table):
    itemsets = gradual_itemsets(random_table, min_supp=0.0)
    assert itemsets  # no vacio
    for itemset, support in itemsets.items():
        assert support == pytest.approx(_brute_force_support(random_table, itemset)), itemset


def test_perfectly_monotone_attributes():
    table = pd.DataFrame({"A": range(10), "B": range(10), "C": range(10, 0, -1)})
    itemsets = gradual_itemsets(table, min_supp=0.99)
    assert itemsets[(("A", "+"), ("B", "+"))] == pytest.approx(1.0)       # cuanto mas A, mas B
    assert itemsets[(("A", "+"), ("C", "-"))] == pytest.approx(1.0)       # cuanto mas A, menos C
    assert (("A", "+"), ("C", "+")) not in itemsets                       # opuesto: soporte 0
    assert (("A", "+"), ("B", "+"), ("C", "-")) in itemsets


def test_only_canonical_itemsets_are_generated(random_table):
    for itemset in gradual_itemsets(random_table, min_supp=0.0):
        assert itemset[0][1] == "+"
        assert len({attribute for attribute, _ in itemset}) == len(itemset)


def test_mirror_itemset_has_the_same_support(random_table):
    support = gradual_itemsets(random_table, 0.0)[(("A", "+"), ("B", "-"))]
    assert _brute_force_support(random_table, (("A", "-"), ("B", "+"))) == pytest.approx(support)


def test_support_is_antimonotone_so_min_supp_prunes(random_table):
    loose = gradual_itemsets(random_table, min_supp=0.05)
    strict = gradual_itemsets(random_table, min_supp=0.3)
    assert set(strict) <= set(loose)
    assert all(support >= 0.3 for support in strict.values())
    # lo que el umbral estricto descarta no esta en el resultado aunque sea valido en el laxo
    assert any(s < 0.3 for s in loose.values())


def test_max_len_limits_itemset_size(random_table):
    assert max(len(i) for i in gradual_itemsets(random_table, 0.0, max_len=1)) == 1
    assert max(len(i) for i in gradual_itemsets(random_table, 0.0, max_len=2)) == 2


def test_gradual_rules_confidence_and_columns():
    table = pd.DataFrame({"age": range(20), "chol": range(20), "sport": range(20, 0, -1)})
    rules = gradual_rules(table, min_supp=0.9, min_conf=0.9)
    assert list(rules.columns) == ["antecedents", "consequents", "antecedent support",
                                   "consequent support", "support", "confidence"]
    rule = rules[(rules["antecedents"] == frozenset([("age", "+"), ("sport", "-")]))
                 & (rules["consequents"] == frozenset([("chol", "+")]))].iloc[0]
    assert rule["confidence"] == pytest.approx(1.0)
    assert rule["support"] == pytest.approx(1.0)


def test_gradual_rule_confidence_is_support_ratio(random_table):
    rules = gradual_rules(random_table, min_supp=0.0, min_conf=0.0)
    for _, rule in rules.iterrows():
        assert rule["confidence"] == pytest.approx(rule["support"] / rule["antecedent support"])
        assert 0.0 <= rule["confidence"] <= 1.0 + 1e-12


def test_min_conf_filters(random_table):
    everything = gradual_rules(random_table, 0.0, 0.0)
    threshold = everything["confidence"].median()       # datos independientes: confianzas ~0.5
    strict = gradual_rules(random_table, 0.0, threshold)
    assert 0 < len(strict) < len(everything)
    assert (strict["confidence"] >= threshold).all()


def test_sampling_is_reproducible_and_limits_rows():
    rng = np.random.RandomState(0)
    table = pd.DataFrame(rng.rand(300, 2), columns=["x", "y"])
    first = gradual_itemsets(table, 0.0, max_rows=40, random_state=3)
    second = gradual_itemsets(table, 0.0, max_rows=40, random_state=3)
    assert first == second
    assert first != gradual_itemsets(table, 0.0, max_rows=40, random_state=4)


def test_array_input_and_error_cases():
    itemsets = gradual_itemsets(np.array([[1, 1], [2, 2], [3, 3]]), min_supp=0.5)
    assert (("0", "+"),) not in itemsets and ((0, "+"), (1, "+")) in itemsets
    with pytest.raises(ValueError):
        gradual_itemsets(pd.DataFrame({"a": [1.0]}), 0.1)
    with pytest.raises(ValueError):
        gradual_itemsets(pd.DataFrame({"a": ["x", "y"]}), 0.1)


def test_formatting():
    assert format_gradual_item(("A", "+")) == "A↑"
    assert format_gradual_item(("A", "-")) == "A↓"
    assert format_gradual_itemset([("B", "-"), ("A", "+")]) == "A↑, B↓"
