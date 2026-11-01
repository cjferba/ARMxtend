# -*- coding: utf-8 -*-
"""
Tests de ``ARM.anomaly``: violaciones de reglas (con y sin retardo) frente a tablas calculadas a
mano, pesos de las reglas, explicaciones, y deteccion de fallos plantados con reglas minadas.
"""
import math

import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import anomaly_scores, detect_anomalies, rule_violations, temporal_association_rules

ITEMS = pd.DataFrame({"a": [1, 1, 0, 1, 0], "b": [1, 0, 0, 1, 1], "c": [0, 1, 1, 1, 0]}).astype(bool)


def make_rules(*rules):
    """Reglas (antecedente, consecuente, confianza) como DataFrame."""
    return pd.DataFrame({"antecedents": [frozenset(a) for a, _, _ in rules],
                         "consequents": [frozenset(c) for _, c, _ in rules],
                         "confidence": [p for _, _, p in rules]})


def test_violations_match_a_hand_computed_table():
    rules = make_rules(({"a"}, {"b"}, 0.75), ({"a", "b"}, {"c"}, 0.9))
    table = rule_violations(ITEMS, rules)
    # a => b: la fila 1 tiene a y no b. a,b => c: la fila 0 tiene a y b y no c (la 3 si tiene c)
    assert table[0].tolist() == [False, True, False, False, False]
    assert table[1].tolist() == [True, False, False, False, False]
    assert table.index.equals(ITEMS.index) and list(table.columns) == [0, 1]


def test_lagged_rule_reads_the_previous_rows_and_skips_the_first_ones():
    rules = make_rules(({"a@lag1"}, {"b"}, 0.9))
    # fila t: a en t-1 y no b en t; la fila 0 no tiene pasado y no puede violar
    assert rule_violations(ITEMS, rules)[0].tolist() == [False, True, True, False, False]


def test_scores_weights_and_applicable_counts():
    rules = make_rules(({"a"}, {"b"}, 0.75), ({"a", "b"}, {"c"}, 0.9))
    surprise = anomaly_scores(ITEMS, rules)
    assert surprise["score"].tolist() == pytest.approx([-math.log2(0.1), -math.log2(0.25), 0, 0, 0])
    assert surprise["violations"].tolist() == [1, 1, 0, 0, 0]
    # a se cumple en las filas 0, 1 y 3; a y b juntas, en las 0 y 3
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
    with pytest.raises(ValueError, match="no es una columna"):
        rule_violations(ITEMS, make_rules(({"zzz"}, {"b"}, 0.9)))
    with pytest.raises(ValueError, match="booleanos"):
        rule_violations(ITEMS.astype(float) * 0.5, rules)
    with pytest.raises(ValueError, match="confidence"):
        anomaly_scores(ITEMS, rules[["antecedents", "consequents"]])
    with pytest.raises(ValueError, match="columnas"):
        rule_violations(ITEMS, rules[["antecedents"]])
    # los 0/1 enteros valen
    assert not rule_violations(ITEMS.astype(int), rules).empty


def faulty_log(rows, faults, seed):
    """a al azar; b es exactamente a un paso antes, salvo en `faults` instantes en que falla."""
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

    # con 'surprise' las 30 primeras filas son los 30 fallos; contando reglas, los fallos se
    # confunden con las ~180 violaciones esperadas de la regla debil (todas puntuan 1)
    assert precision_at_faults(anomaly_scores(items, rules)["score"].to_numpy()) == 1.0
    assert precision_at_faults(anomaly_scores(items, rules, weight="count")["score"].to_numpy()) < 0.5
