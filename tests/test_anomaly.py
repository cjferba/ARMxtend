# -*- coding: utf-8 -*-
"""
Tests of ``ARM.anomaly``: rule violations (with and without lag) against hand-computed tables,
rule weights, explanations, and detection of planted faults with mined rules.
"""
import math

import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import anomaly_scores, detect_anomalies, rule_violations, temporal_association_rules

ITEMS = pd.DataFrame({"a": [1, 1, 0, 1, 0], "b": [1, 0, 0, 1, 1], "c": [0, 1, 1, 1, 0]}).astype(bool)


def make_rules(*rules):
    """Rules (antecedent, consequent, confidence) as a DataFrame."""
    return pd.DataFrame({"antecedents": [frozenset(a) for a, _, _ in rules],
                         "consequents": [frozenset(c) for _, c, _ in rules],
                         "confidence": [p for _, _, p in rules]})


def test_violations_match_a_hand_computed_table():
    rules = make_rules(({"a"}, {"b"}, 0.75), ({"a", "b"}, {"c"}, 0.9))
    table = rule_violations(ITEMS, rules)
    # a => b: row 1 has a and not b. a,b => c: row 0 has a and b and not c (row 3 does have c)
    assert table[0].tolist() == [False, True, False, False, False]
    assert table[1].tolist() == [True, False, False, False, False]
    assert table.index.equals(ITEMS.index) and list(table.columns) == [0, 1]


def test_lagged_rule_reads_the_previous_rows_and_skips_the_first_ones():
    rules = make_rules(({"a@lag1"}, {"b"}, 0.9))
    # row t: a at t-1 and not b at t; row 0 has no past and cannot violate
    assert rule_violations(ITEMS, rules)[0].tolist() == [False, True, True, False, False]


def test_scores_weights_and_applicable_counts():
    rules = make_rules(({"a"}, {"b"}, 0.75), ({"a", "b"}, {"c"}, 0.9))
    surprise = anomaly_scores(ITEMS, rules)
    assert surprise["score"].tolist() == pytest.approx([-math.log2(0.1), -math.log2(0.25), 0, 0, 0])
    assert surprise["violations"].tolist() == [1, 1, 0, 0, 0]
    # a holds in rows 0, 1 and 3; a and b together, in rows 0 and 3
    assert surprise["applicable"].tolist() == [2, 1, 0, 2, 0]
    assert anomaly_scores(ITEMS, rules, weight="count")["score"].tolist() == [1, 1, 0, 0, 0]
    assert anomaly_scores(ITEMS, rules, weight="confidence")["score"].tolist() == pytest.approx([0.9, 0.75, 0, 0, 0])


def test_a_perfect_rule_weighs_at_most_log2_of_one_over_floor():
    rules = make_rules(({"a"}, {"b"}, 1.0))
    assert anomaly_scores(ITEMS, rules)["score"].max() == pytest.approx(-math.log2(1e-3))
    assert anomaly_scores(ITEMS, rules, floor=0.25)["score"].max() == pytest.approx(2.0)


def test_scores_are_the_weighted_sum_of_the_violation_table():
    rng = np.random.RandomState(0)
    items = pd.DataFrame(rng.rand(200, 4) < 0.4, columns=list("abcd"))
    rules = make_rules(({"a"}, {"b"}, 0.6), ({"c@lag1"}, {"d"}, 0.9), ({"a", "d@lag2"}, {"b", "c"}, 0.8))
    weights = -np.log2(1 - rules["confidence"].to_numpy())
    scores = anomaly_scores(items, rules)
    table = rule_violations(items, rules)
    assert scores["score"].to_numpy() == pytest.approx(table.to_numpy() @ weights)
    assert scores["violations"].tolist() == table.sum(axis=1).tolist()


def test_detect_anomalies_orders_explains_and_limits():
    rules = make_rules(({"a"}, {"b"}, 0.75), ({"a", "b"}, {"c"}, 0.9))
    flagged = detect_anomalies(ITEMS, rules)
    assert flagged.index.tolist() == [0, 1]                      # mas sorpresa primero (3.3 bits frente a 2)
    assert flagged["rules"].tolist() == [["a (t), b (t) => c (t)"], ["a (t) => b (t)"]]
    assert flagged["violations"].tolist() == [1, 1]
    assert detect_anomalies(ITEMS, rules, top=1).index.tolist() == [0]
    assert detect_anomalies(ITEMS, rules, min_score=3.0).index.tolist() == [0]
    assert detect_anomalies(ITEMS, rules, weight="count").index.tolist() == [0, 1]          # empate: orden cronologico
    assert detect_anomalies(ITEMS, make_rules(({"c"}, {"a"}, 0.9)).iloc[:0]).empty


def test_explanations_write_the_lags_of_temporal_rules():
    flagged = detect_anomalies(ITEMS, make_rules(({"a@lag1", "c@lag1"}, {"b"}, 0.9)))
    assert flagged["rules"].tolist() == [["a (t-1), c (t-1) => b (t)"]] and flagged.index.tolist() == [2]


def test_input_errors():
    rules = make_rules(({"a"}, {"b"}, 0.9))
    with pytest.raises(ValueError, match="weight"):
        anomaly_scores(ITEMS, rules, weight="bits")
    with pytest.raises(ValueError, match="floor"):
        anomaly_scores(ITEMS, rules, floor=0.0)
    with pytest.raises(ValueError, match="is not a column"):
        rule_violations(ITEMS, make_rules(({"zzz"}, {"b"}, 0.9)))
    with pytest.raises(ValueError, match="boolean"):
        rule_violations(ITEMS.astype(float) * 0.5, rules)
    with pytest.raises(ValueError, match="confidence"):
        anomaly_scores(ITEMS, rules[["antecedents", "consequents"]])
    with pytest.raises(ValueError, match="missing the columns"):
        rule_violations(ITEMS, rules[["antecedents"]])
    # los 0/1 enteros valen
    assert not rule_violations(ITEMS.astype(int), rules).empty


def faulty_log(rows, faults, seed):
    """a at random; b is exactly a one step earlier, except at `faults` instants where it fails."""
    rng = np.random.RandomState(seed)
    a = rng.rand(rows) < 0.3
    b = np.r_[False, a[:-1]]
    broken = np.zeros(rows, dtype=bool)
    if faults:
        broken[rng.choice(np.flatnonzero(b), faults, replace=False)] = True
        b = b & ~broken
    return pd.DataFrame({"a": a, "b": b, "noise": rng.rand(rows) < 0.5}), broken


def test_planted_faults_are_found_with_rules_mined_on_clean_data():
    clean, _ = faulty_log(2000, 0, seed=1)
    rules = temporal_association_rules(clean, 1, 0.05, min_threshold=0.99, same_time_antecedent=False,
                                       exclude_persistence=True)
    assert len(rules) >= 1
    test, broken = faulty_log(1500, 25, seed=2)
    flagged = detect_anomalies(test, rules)
    assert set(np.flatnonzero(broken)) == set(flagged.index)
    assert all("a (t-1) => b (t)" in explanation for explanation in flagged["rules"])


def test_surprise_separates_faults_better_than_counting_when_a_weak_rule_is_often_violated():
    rng = np.random.RandomState(7)
    rows = 3000
    motion, door = rng.rand(rows) < 0.3, rng.rand(rows) < 0.3
    light = np.r_[False, motion[:-1]]                               # regla fiable: movimiento => luz
    chime = np.r_[False, door[:-1]] & (rng.rand(rows) < 0.8)         # regla debil: puerta => timbre (80 %)
    fault = np.zeros(rows, dtype=bool)
    fault[rng.choice(np.flatnonzero(light), 30, replace=False)] = True
    light = light & ~fault
    items = pd.DataFrame({"motion": motion, "door": door, "light": light, "chime": chime})
    rules = make_rules(({"motion@lag1"}, {"light"}, 0.99), ({"door@lag1"}, {"chime"}, 0.8))

    def precision_at_faults(score):
        top = np.argsort(-score, kind="stable")[:fault.sum()]          # empates: orden cronologico
        return fault[top].mean()

    # with 'surprise' the first 30 rows are the 30 faults; when counting rules, the faults
    # are mixed up with the ~180 expected violations of the weak rule (all score 1)
    assert precision_at_faults(anomaly_scores(items, rules)["score"].to_numpy()) == 1.0
    assert precision_at_faults(anomaly_scores(items, rules, weight="count")["score"].to_numpy()) < 0.5
