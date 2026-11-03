# -*- coding: utf-8 -*-
"""Tests of ARM.explain: evidence, contributions, alternatives and row explanations."""
import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import explain_row, explain_rule, rule_evidence


def table():
    rows = [
        {"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 1, "c": 0}, {"a": 1, "b": 0, "c": 1},
        {"a": 1, "b": 0, "c": 0}, {"a": 0, "b": 1, "c": 1}, {"a": 0, "b": 0, "c": 1}, {"a": 0, "b": 0, "c": 0},
    ]
    for index, row in enumerate(rows):
        row["d"] = index % 2 == 0
    return pd.DataFrame(rows).astype(bool)


def rules_frame():
    spec = [({"a", "b"}, {"c"}), ({"a"}, {"c"}), ({"b"}, {"c"}), ({"a", "b"}, {"d"}), ({"a"}, {"b"})]
    frame = pd.DataFrame({"antecedents": [frozenset(a) for a, _ in spec], "consequents": [frozenset(c) for _, c in spec]})
    frame["confidence"] = [2 / 3, 3 / 5, 3 / 4, 2 / 3, 3 / 5]
    return frame


def test_evidence_is_the_contingency_table():
    items = table()
    evidence = rule_evidence(items, {"a", "b"}, {"c"})
    assert (evidence["both"], evidence["only_antecedent"], evidence["only_consequent"], evidence["neither"]) == (2, 1, 3, 2)
    assert evidence["n_rows"] == 8 and evidence["support"] == pytest.approx(2 / 8)
    assert evidence["confidence"] == pytest.approx(2 / 3) and evidence["lift"] == pytest.approx((2 / 3) / (5 / 8))
    assert evidence["supporting"] == [0, 1] and evidence["violating"] == [2]


def test_evidence_with_lags_uses_only_rows_with_history():
    items = pd.DataFrame({"x": [True, False, True, False, True], "y": [False, True, False, True, True]})
    evidence = rule_evidence(items, {"x@lag1"}, {"y"})
    assert evidence["n_rows"] == 4 and evidence["both"] == 2 and evidence["only_antecedent"] == 0
    assert evidence["confidence"] == pytest.approx(1.0)


def test_explain_rule_contributions_and_alternatives():
    items, rules = table(), rules_frame()
    explained = explain_rule(items, rules, 0)
    assert explained["rule"] == "a (t), b (t) => c (t)"
    contributions = explained["contributions"].set_index("item")
    assert contributions.loc["a", "confidence_without"] == pytest.approx(3 / 4)       # b => c
    assert contributions.loc["b", "confidence_without"] == pytest.approx(3 / 5)       # a => c
    assert contributions.loc["a", "gain"] == pytest.approx(2 / 3 - 3 / 4)
    alternatives = explained["alternatives"]
    assert set(alternatives["relation"]) == {"more general", "same antecedent"}
    general = alternatives[alternatives["relation"] == "more general"].sort_values("confidence", ascending=False)
    assert general["rule"].tolist() == ["b (t) => c (t)", "a (t) => c (t)"]
    assert general["delta_confidence"].iloc[0] == pytest.approx(3 / 4 - 2 / 3)
    single = explain_rule(items, rules, 1)
    assert single["contributions"]["confidence_without"].iloc[0] == pytest.approx(5 / 8)       # base rate of c
    assert "more specific" in set(single["alternatives"]["relation"])


def test_explain_row_statuses_and_missing_items():
    items, rules = table(), rules_frame()
    first = explain_row(items, rules, 1)
    assert first.loc[0, "status"] == "confirmed" and first.loc[3, "status"] == "violated"
    third = explain_row(items, rules, 2)
    assert third.loc[0, "status"] == "violated" and third.loc[0, "missing"] == []
    fourth = explain_row(items, rules, 3)
    assert fourth.loc[0, "status"] == "not fired" and fourth.loc[0, "missing"] == ["b"]
    assert fourth.loc[1, "status"] == "confirmed" and fourth.loc[1, "confidence"] == pytest.approx(3 / 5)
    lagged = pd.DataFrame({"x": [True, True], "y": [False, True]})
    result = explain_row(lagged, pd.DataFrame({"antecedents": [frozenset({"x@lag1"})], "consequents": [frozenset({"y"})]}), 0)
    assert result["status"].iloc[0] == "no history" and np.isnan(result["confidence"].iloc[0])


def test_validation():
    items, rules = table(), rules_frame()
    with pytest.raises(ValueError, match="not in `rules`"):
        explain_rule(items, rules, 99)
    with pytest.raises(ValueError, match="not in `items`"):
        explain_row(items, rules, 99)
    with pytest.raises(ValueError, match="missing the columns"):
        explain_row(items, rules[["antecedents"]], 0)
