# -*- coding: utf-8 -*-
"""Tests of the significance of fuzzy rules (ARM.fuzzy_significance)."""
import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import fuzzy_rule_significance, fuzzy_significant_rules
from ARMxtend.FFIM import fuzzy_fpgrowth
from ARMxtend.FIM.FARE import fuzzy_association_rules


def make_data(rows=300, seed=0):
    """a and b related (b follows a), c independent."""
    rng = np.random.default_rng(seed)
    a = rng.random(rows)
    b = np.clip(a + rng.normal(0, 0.1, rows), 0, 1)
    c = rng.random(rows)
    return pd.DataFrame({"a": a, "b": b, "c": c})


def pairs(table):
    return [[(name, float(v)) for name, v in row.items() if v > 0] for _, row in table.iterrows()]


def mined(table, min_supp=0.2):
    frequent = fuzzy_fpgrowth(pairs(table), min_supp, 5)
    return fuzzy_association_rules(frequent, 5, metric="confidence", min_threshold=0.0)


def row_of(rules, antecedent, consequent):
    mask = [a == frozenset(antecedent) and c == frozenset(consequent) for a, c in zip(rules["antecedents"], rules["consequents"])]
    return rules[mask].iloc[0]


def test_related_items_get_small_p_values_and_independent_ones_do_not():
    table = make_data()
    scored = fuzzy_rule_significance(mined(table), table, n_permutations=499, random_state=1)
    assert row_of(scored, "a", "b")["p_value"] <= 0.01 and row_of(scored, "b", "a")["p_value"] <= 0.01
    assert row_of(scored, "a", "c")["p_value"] > 0.05 and row_of(scored, "c", "a")["p_value"] > 0.05
    assert scored["p_value"].between(1 / 500, 1).all() and scored.attrs["n_permutations"] == 499 and scored.attrs["n_transactions"] == 300


def test_statistic_is_the_scalar_cardinality_confidence():
    table = make_data(50, 3)
    scored = fuzzy_rule_significance(mined(table), table, n_permutations=9, random_state=0)
    one = row_of(scored, "a", "b")
    expected = np.minimum(table["a"], table["b"]).sum() / table["a"].sum()
    assert one["fuzzy_confidence"] == pytest.approx(expected)


def test_table_and_pair_inputs_give_the_same_result_and_are_reproducible():
    table = make_data(120, 4)
    rules = mined(table)
    from_pairs = fuzzy_rule_significance(rules, pairs(table), 99, random_state=7)
    from_table = fuzzy_rule_significance(rules, table, 99, random_state=7)
    assert np.allclose(from_pairs["p_value"], from_table["p_value"]) and np.allclose(from_pairs["fuzzy_confidence"], from_table["fuzzy_confidence"])
    assert np.array_equal(from_table["p_value"], fuzzy_rule_significance(rules, table, 99, random_state=7)["p_value"])


def test_significant_rules_keep_the_related_pair_and_use_the_candidate_family():
    table = make_data()
    rules = mined(table)
    assert rules.attrs["n_candidates"] >= len(rules) > 0
    kept = fuzzy_significant_rules(rules, table, alpha=0.05, n_permutations=499, random_state=2)
    names = {(tuple(a), tuple(c)) for a, c in zip(kept["antecedents"], kept["consequents"])}
    assert (("a",), ("b",)) in names and (("a",), ("c",)) not in names
    assert (kept["p_adjusted"] <= 0.05).all() and kept.attrs["n_tests"] == rules.attrs["n_candidates"]
    assert kept["p_adjusted"].is_monotonic_increasing
    assert len(fuzzy_significant_rules(rules, table, alpha=0.05, n_permutations=499, n_tests=10 ** 6, random_state=2)) == 0


def test_uncorrelated_data_gives_few_false_discoveries():
    table = pd.DataFrame(np.random.default_rng(5).random((300, 4)), columns=list("abcd"))
    rules = mined(table, 0.15)
    kept = fuzzy_significant_rules(rules, table, alpha=0.05, n_permutations=499, random_state=3)
    assert len(rules) > 10 and len(kept) <= 1


def test_edge_cases_and_validation():
    table = make_data(40, 6)
    rules = mined(table)
    empty = fuzzy_rule_significance(rules.iloc[0:0], table, 9)
    assert len(empty) == 0 and {"p_value", "fuzzy_confidence"} <= set(empty.columns)
    never = pd.DataFrame({"antecedents": [frozenset({"a"})], "consequents": [frozenset({"b"})]})
    zero = pd.DataFrame({"a": [0.0] * 5, "b": [0.5] * 5})
    result = fuzzy_rule_significance(never, zero, 9)
    assert result["p_value"].iloc[0] == 1.0 and result["fuzzy_confidence"].iloc[0] == 0.0
    with pytest.raises(ValueError, match="are not in"):
        fuzzy_rule_significance(never, pd.DataFrame({"a": [0.5]}), 9)
    with pytest.raises(ValueError, match="n_permutations"):
        fuzzy_rule_significance(rules, table, 0)
    with pytest.raises(ValueError, match="there are no transactions"):
        fuzzy_rule_significance(never, [], 9)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fuzzy_rule_significance(never, pd.DataFrame({"a": [2.0], "b": [0.5]}), 9)
    with pytest.raises(ValueError, match="alpha"):
        fuzzy_significant_rules(rules, table, alpha=1.5)
    repeated = fuzzy_rule_significance(never, [[("a", 0.2), ("a", 0.9), ("b", 0.5)]] * 4, 9, random_state=0)
    assert repeated["fuzzy_confidence"].iloc[0] == pytest.approx(0.5 / 0.9)


def test_fconf_statistic_matches_the_aggregated_confidence_of_fare():
    table = make_data()
    rules = mined(table)
    scored = fuzzy_rule_significance(rules, table, n_permutations=199, random_state=1, statistic="fconf", num_alpha=5)
    assert np.allclose(scored["fuzzy_confidence"].to_numpy(), rules["confidence"].to_numpy(), atol=1e-9)
    assert row_of(scored, "a", "b")["p_value"] <= 0.02 and row_of(scored, "a", "c")["p_value"] > 0.05
    kept = fuzzy_significant_rules(rules, table, n_permutations=499, random_state=1, statistic="fconf", num_alpha=5)
    assert {(tuple(a), tuple(c)) for a, c in zip(kept["antecedents"], kept["consequents"])} >= {(("a",), ("b",))}
    with pytest.raises(ValueError, match="statistic"):
        fuzzy_rule_significance(rules, table, statistic="nope")
