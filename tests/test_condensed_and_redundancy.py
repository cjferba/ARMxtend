# -*- coding: utf-8 -*-
"""Tests de itemsets cerrados/maximales (FIM.condensed) y poda de reglas redundantes (ARM.redundancy)."""
import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import association_rules
from ARMxtend.ARM.redundancy import prune_redundant_rules
from ARMxtend.FFIM import fpgrowth, fuzzy_fpgrowth
from ARMxtend.FIM._shared import key_to_itemset
from ARMxtend.FIM.condensed import closed_itemsets, maximal_itemsets, support_from_closed

# a,b siempre juntos; c solo con ellos; d aislado
TRANSACTIONS = [{"a", "b", "c"}] * 3 + [{"a", "b"}] + [{"d"}]


@pytest.fixture
def frequent():
    return fpgrowth(TRANSACTIONS, min_supp=0.2)


def test_closed_itemsets_hand_computed(frequent):
    closed = closed_itemsets(frequent)
    assert set(closed) == {"a-b", "a-b-c", "d"}
    assert closed["a-b"] == pytest.approx(0.8) and closed["a-b-c"] == pytest.approx(0.6)


def test_maximal_itemsets_hand_computed(frequent):
    assert set(maximal_itemsets(frequent)) == {"a-b-c", "d"}


def test_maximal_are_always_closed(frequent):
    assert set(maximal_itemsets(frequent)) <= set(closed_itemsets(frequent))


def test_support_of_every_frequent_itemset_is_recoverable_from_closed(frequent):
    closed = closed_itemsets(frequent)
    for key, support in frequent.items():
        assert support_from_closed(key, closed) == pytest.approx(support)
    assert support_from_closed(["a", "c"], closed) == pytest.approx(0.6)
    assert support_from_closed("a-d", closed) is None            # no es frecuente


def test_no_proper_superset_of_a_closed_itemset_has_equal_support(frequent):
    closed = closed_itemsets(frequent)
    for key, support in closed.items():
        for other, otherSupport in frequent.items():
            if key_to_itemset(key) < key_to_itemset(other):
                assert otherSupport < support - 1e-12


def test_closed_itemsets_with_a_tolerance():
    freq = {"a": 0.5, "b": 0.5000001, "a-b": 0.5}
    # a y a-b valen exactamente lo mismo: a no es cerrado; b difiere en 1e-7
    assert set(closed_itemsets(freq, tol=1e-12)) == {"b", "a-b"}
    assert set(closed_itemsets(freq, tol=1e-3)) == {"a-b"}


def test_fuzzy_closure_is_level_wise():
    # b >= a siempre: el bit-list de {a,b} coincide con el de a -> a no es cerrado
    same = fuzzy_fpgrowth([[("a", 1.0), ("b", 1.0)], [("a", 0.5), ("b", 0.5)]], 0.1, 2)
    assert "a" not in closed_itemsets(same) and "a-b" in closed_itemsets(same)

    # mismos grados maximos pero niveles distintos: {a,b} tiene un bit-list diferente
    different = fuzzy_fpgrowth([[("a", 1.0), ("b", 0.5)], [("a", 0.5), ("b", 1.0)]], 0.1, 2)
    assert "a" in closed_itemsets(different) and "b" in closed_itemsets(different)
    assert set(maximal_itemsets(different)) == {"a-b"}
    # y se recupera el bit-list (no solo un escalar)
    assert np.allclose(support_from_closed("a", closed_itemsets(different)), different["a"])


def test_empty_input():
    assert closed_itemsets({}) == {} and maximal_itemsets({}) == {}


# --- poda de reglas redundantes -------------------------------------------

def _rules(*specs):
    return pd.DataFrame([
        {"antecedents": frozenset(a), "consequents": frozenset(c), "support": s, "confidence": cf}
        for a, c, s, cf in specs
    ])


def test_prune_removes_rules_with_larger_antecedent_or_smaller_consequent():
    rules = _rules(
        ({"a"}, {"b", "c"}, 0.6, 0.75),         # general: se conserva
        ({"a"}, {"b"}, 0.6, 0.75),              # menor consecuente -> redundante
        ({"a", "d"}, {"b", "c"}, 0.6, 0.75),    # mayor antecedente -> redundante
        ({"a", "d"}, {"b"}, 0.6, 0.75),         # ambas -> redundante
        ({"a"}, {"e"}, 0.6, 0.75),              # no comparable -> se conserva
        ({"a"}, {"b"}, 0.6, 0.80),              # distinta confianza -> se conserva
    )
    pruned = prune_redundant_rules(rules)
    kept = {(a, c) for a, c in zip(pruned["antecedents"], pruned["consequents"])}
    assert kept == {
        (frozenset({"a"}), frozenset({"b", "c"})),
        (frozenset({"a"}), frozenset({"e"})),
        (frozenset({"a"}), frozenset({"b"})),
    }
    assert len(pruned) == 3


def test_prune_keeps_one_copy_of_duplicate_rules():
    rules = _rules(({"a"}, {"b"}, 0.5, 0.9), ({"a"}, {"b"}, 0.5, 0.9), ({"a"}, {"b"}, 0.5, 0.9))
    assert len(prune_redundant_rules(rules)) == 1


def test_prune_respects_tolerance():
    rules = _rules(({"a"}, {"b", "c"}, 0.6, 0.75), ({"a"}, {"b"}, 0.6, 0.75 + 1e-6))
    assert len(prune_redundant_rules(rules, tol=1e-9)) == 2
    assert len(prune_redundant_rules(rules, tol=1e-4)) == 1


def test_prune_on_real_rules_leaves_a_general_rule_for_each_removed_one():
    frequent = fpgrowth(TRANSACTIONS, min_supp=0.2)
    itemsets = pd.DataFrame({"support": list(frequent.values()),
                             "itemsets": [key_to_itemset(k) for k in frequent]})
    allRules = association_rules(itemsets, min_threshold=0.0)
    pruned = prune_redundant_rules(allRules)
    assert 0 < len(pruned) < len(allRules)

    keptKeys = set(zip(pruned["antecedents"], pruned["consequents"]))
    for _, rule in allRules.iterrows():
        if (rule["antecedents"], rule["consequents"]) in keptKeys:
            continue
        assert any(
            a <= rule["antecedents"] and c >= rule["consequents"]
            and k["support"] == pytest.approx(rule["support"]) and k["confidence"] == pytest.approx(rule["confidence"])
            for a, c, k in ((r["antecedents"], r["consequents"], r) for _, r in pruned.iterrows())
        )


def test_prune_empty():
    assert len(prune_redundant_rules(pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence"]))) == 0
