# -*- coding: utf-8 -*-
"""Tests of closed/maximal itemsets (FIM.condensed) and redundant rule pruning (ARM.redundancy)."""
import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import association_rules
from ARMxtend.ARM import redundancy
from ARMxtend.ARM.redundancy import prune_redundant_rules, prune_unproductive_rules
from ARMxtend.FFIM import fpgrowth, fuzzy_fpgrowth
from ARMxtend.FIM._shared import key_to_itemset
from ARMxtend.FIM.condensed import closed_itemsets, maximal_itemsets, support_from_closed

# a,b always together; c only with them; d isolated
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
    assert support_from_closed("a-d", closed) is None            # not frequent


def test_no_proper_superset_of_a_closed_itemset_has_equal_support(frequent):
    closed = closed_itemsets(frequent)
    for key, support in closed.items():
        for other, otherSupport in frequent.items():
            if key_to_itemset(key) < key_to_itemset(other):
                assert otherSupport < support - 1e-12


def test_closed_itemsets_with_a_tolerance():
    freq = {"a": 0.5, "b": 0.5000001, "a-b": 0.5}
    # a and a-b are worth exactly the same: a is not closed; b differs by 1e-7
    assert set(closed_itemsets(freq, tol=1e-12)) == {"b", "a-b"}
    assert set(closed_itemsets(freq, tol=1e-3)) == {"a-b"}


def test_fuzzy_closure_is_level_wise():
    # b >= a always: the bit-list of {a,b} equals that of a -> a is not closed
    same = fuzzy_fpgrowth([[("a", 1.0), ("b", 1.0)], [("a", 0.5), ("b", 0.5)]], 0.1, 2)
    assert "a" not in closed_itemsets(same) and "a-b" in closed_itemsets(same)

    # same maximum degrees but different levels: {a,b} has a different bit-list
    different = fuzzy_fpgrowth([[("a", 1.0), ("b", 0.5)], [("a", 0.5), ("b", 1.0)]], 0.1, 2)
    assert "a" in closed_itemsets(different) and "b" in closed_itemsets(different)
    assert set(maximal_itemsets(different)) == {"a-b"}
    # and the bit-list is recovered (not just a scalar)
    assert np.allclose(support_from_closed("a", closed_itemsets(different)), different["a"])


def test_empty_input():
    assert closed_itemsets({}) == {} and maximal_itemsets({}) == {}


# --- pruning of redundant rules -------------------------------------------

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
        ({"a"}, {"e"}, 0.6, 0.75),              # not comparable -> kept
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


def confidence_rules(*rules):
    return pd.DataFrame({"antecedents": [frozenset(a) for a, _, _ in rules],
                         "consequents": [frozenset(c) for _, c, _ in rules],
                         "confidence": [p for _, _, p in rules], "support": [0.1] * len(rules)})


def test_unproductive_rules_hand_computed():
    rules = confidence_rules(({"a"}, {"c"}, 0.80), ({"a", "b"}, {"c"}, 0.82), ({"a", "b", "d"}, {"c"}, 0.95),
                             ({"b"}, {"c"}, 0.50), ({"a", "b"}, {"e"}, 0.50))
    # a,b => c gains 0.02 over a => c: unproductive with 0.05, productive with 0. a,b,d => c gains 0.13 over a,b => c
    kept = prune_unproductive_rules(rules, min_improvement=0.05)
    assert [(set(a), set(c)) for a, c in zip(kept["antecedents"], kept["consequents"])] == [
        ({"a"}, {"c"}), ({"a", "b", "d"}, {"c"}), ({"b"}, {"c"}), ({"a", "b"}, {"e"})]      # a,b => e: no more general rule
    assert len(prune_unproductive_rules(rules, min_improvement=0.0)) == 5
    equal = confidence_rules(({"a"}, {"c"}, 0.8), ({"a", "b"}, {"c"}, 0.8))
    assert len(prune_unproductive_rules(equal)) == 1                                        # no mejora: sobra


def test_unproductive_rules_compare_with_any_more_general_rule_even_if_it_is_dropped():
    # a,b => c gains 0.03 over a => c and is discarded; a,b,d => c gains 0.07 over a => c but only 0.04 over
    # a,b => c (which is in `rules`, even if discarded): the improvement is the maximum over all the sub-rules
    rules = confidence_rules(({"a"}, {"c"}, 0.75), ({"a", "b"}, {"c"}, 0.78), ({"a", "b", "d"}, {"c"}, 0.82))
    assert len(prune_unproductive_rules(rules, 0.05)) == 1


def test_unproductive_rules_large_antecedents_use_the_pairwise_path(monkeypatch):
    rules = confidence_rules(({"a"}, {"c"}, 0.80), ({"a", "b"}, {"c"}, 0.82), ({"a", "b", "d"}, {"c"}, 0.95),
                             ({"b"}, {"c"}, 0.50))
    expected = prune_unproductive_rules(rules, 0.05)
    monkeypatch.setattr(redundancy, "_EXHAUSTIVE_LIMIT", 1)
    assert prune_unproductive_rules(rules, 0.05).equals(expected)
    assert prune_unproductive_rules(rules.iloc[:0]).empty


def test_unproductive_rules_keep_the_order_and_the_most_general_rules_of_real_mining():
    rng = np.random.RandomState(0)
    table = rng.rand(400, 5) < 0.5
    table[:, 4] = table[:, 0] | (rng.rand(400) < 0.05)
    transactions = [{"i{}".format(j) for j in range(5) if row[j]} for row in table]
    frequent = fpgrowth(transactions, 0.05)
    itemsets = pd.DataFrame({"support": list(frequent.values()), "itemsets": [key_to_itemset(k) for k in frequent]})
    rules = association_rules(itemsets, metric="confidence", min_threshold=0.0)
    kept = prune_unproductive_rules(rules, 0.02)
    assert 0 < len(kept) < len(rules)
    confidence = {(a, c): p for a, c, p in zip(kept["antecedents"], kept["consequents"], kept["confidence"])}
    for (antecedent, consequent), value in confidence.items():
        for (other, other_consequent), other_value in confidence.items():
            if other_consequent == consequent and other < antecedent:
                assert value - other_value > 0.02 - 1e-9
