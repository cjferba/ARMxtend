# -*- coding: utf-8 -*-
"""
Tests de ``ARM.temporal_rules``: las medidas de cada regla se comparan con un
recuento directo sobre los datos originales (oraculo), y se comprueba que un
patron con retardo plantado se recupera.
"""
import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import (
    format_lagged_item, fuzzy_temporal_association_rules, prune_redundant_rules, rule_profile,
    temporal_association_rules,
)
from ARMxtend.preprocessing.timeseries import split_lag


def planted_series(rows=1500, seed=0):
    """A y C aleatorios; B ocurre dos pasos despues de A el 90 % de las veces."""
    rng = np.random.RandomState(seed)
    a = rng.rand(rows) < 0.3
    b = np.zeros(rows, dtype=bool)
    b[2:] = a[:-2] & (rng.rand(rows - 2) < 0.9)
    b |= rng.rand(rows) < 0.05
    c = rng.rand(rows) < 0.5
    return pd.DataFrame({"A": a, "B": b, "C": c})


def count_rule(table, antecedent, consequent, max_lag):
    """Soporte y confianza de la regla contando directamente, sobre las filas con todos los retardos."""
    def holds(names, t):
        return all(table[base].iloc[t - lag] for base, lag in (split_lag(name) for name in names))

    times = range(max_lag, len(table))
    both = sum(holds(antecedent, t) and holds(consequent, t) for t in times)
    onlyAntecedent = sum(holds(antecedent, t) for t in times)
    return both / len(times), both / onlyAntecedent


def test_every_rule_matches_a_direct_count_on_the_data():
    table = planted_series(rows=400, seed=3)
    rules = temporal_association_rules(table, max_lag=3, min_supp=0.08, min_threshold=0.5)
    assert len(rules) > 5
    for _, rule in rules.iterrows():
        support, confidence = count_rule(table, rule["antecedents"], rule["consequents"], max_lag=3)
        assert rule["support"] == pytest.approx(support)
        assert rule["confidence"] == pytest.approx(confidence)


def test_the_planted_lag_two_rule_is_found_and_the_reverse_is_not():
    table = planted_series()
    rules = temporal_association_rules(table, max_lag=3, min_supp=0.05, min_threshold=0.8)
    found = rules[(rules["antecedents"] == frozenset({"A@lag2"})) & (rules["consequents"] == frozenset({"B"}))]
    assert len(found) == 1
    assert found.iloc[0]["confidence"] == pytest.approx(0.9, abs=0.03)
    assert found.iloc[0]["lags"] == (2,) and found.iloc[0]["max_lag"] == 2
    # B no predice A: no hay ninguna regla con A de consecuente
    assert not any("A" in consequent for consequent in rules["consequents"])
    # y la aleatoria C no aparece en reglas con confianza alta
    assert not any("C" in consequent for consequent in rules["consequents"])


def test_structure_of_the_rules():
    table = planted_series(rows=600, seed=1)
    rules = temporal_association_rules(table, max_lag=3, min_supp=0.05, min_threshold=0.5)
    for _, rule in rules.iterrows():
        assert all(split_lag(item)[1] == 0 for item in rule["consequents"])
        antecedentLags = sorted({split_lag(item)[1] for item in rule["antecedents"] if split_lag(item)[1] > 0})
        assert antecedentLags, "el antecedente debe mirar al pasado"
        assert rule["lags"] == tuple(antecedentLags) and rule["max_lag"] == max(antecedentLags)
        assert max(antecedentLags) <= 3


def test_lags_option_restricts_the_offsets():
    table = planted_series(rows=800, seed=2)
    rules = temporal_association_rules(table, max_lag=3, min_supp=0.05, min_threshold=0.5, lags=[2])
    assert len(rules) > 0
    assert all(set(lags) == {2} for lags in rules["lags"])
    with pytest.raises(ValueError, match="1..3"):
        temporal_association_rules(table, max_lag=3, min_supp=0.05, lags=[5])
    with pytest.raises(ValueError, match="max_lag"):
        temporal_association_rules(table, max_lag=0, min_supp=0.05)


def test_targets_filter_by_item_or_by_variable():
    table = pd.DataFrame({"temp=high": planted_series()["A"], "heater": planted_series()["B"]})
    byVariable = temporal_association_rules(table, 2, 0.05, min_threshold=0.8, targets=["heater"])
    byItem = temporal_association_rules(table, 2, 0.05, min_threshold=0.8, targets=["temp=high"])
    assert len(byVariable) > 0 and all(consequent == frozenset({"heater"}) for consequent in byVariable["consequents"])
    assert all(consequent == frozenset({"temp=high"}) for consequent in byItem["consequents"])
    onlyVariable = pd.DataFrame({"temp=high": [True, False] * 100, "temp=low": [False, True] * 100})
    mined = temporal_association_rules(onlyVariable, 1, 0.2, min_threshold=0.9, targets=["temp"])
    assert len(mined) > 0 and all(all(item.startswith("temp") for item in consequent) for consequent in mined["consequents"])


def test_exclude_persistence_removes_same_variable_rules():
    alternating = pd.DataFrame({"temp=high": [True, True, False, False] * 60, "door": [True, False] * 120})
    withPersistence = temporal_association_rules(alternating, 1, 0.1, min_threshold=0.9)
    without = temporal_association_rules(alternating, 1, 0.1, min_threshold=0.9, exclude_persistence=True)
    assert any(split_lag(next(iter(rule["consequents"])))[0] == "temp=high"
               and any(split_lag(item)[0] == "temp=high" for item in rule["antecedents"])
               for _, rule in withPersistence.iterrows())
    for _, rule in without.iterrows():
        consequentVariables = {item.split("=")[0] for item in rule["consequents"]}
        assert not any(split_lag(item)[1] > 0 and split_lag(item)[0].split("=")[0] in consequentVariables
                       for item in rule["antecedents"])


def test_same_time_antecedent_and_lagged_antecedent_switches():
    table = planted_series(rows=600, seed=4)
    strict = temporal_association_rules(table, 2, 0.05, min_threshold=0.5, same_time_antecedent=False)
    assert all(split_lag(item)[1] > 0 for antecedent in strict["antecedents"] for item in antecedent)
    loose = temporal_association_rules(table, 2, 0.05, min_threshold=0.5, require_lagged_antecedent=False)
    assert any(all(split_lag(item)[1] == 0 for item in rule) for rule in loose["antecedents"])
    assert len(loose) >= len(strict) > 0


def test_rules_can_be_pruned_and_formatted():
    table = planted_series(rows=800, seed=5)
    rules = temporal_association_rules(table, 3, 0.05, min_threshold=0.8)
    pruned = prune_redundant_rules(rules)
    assert 0 < len(pruned) <= len(rules)
    assert format_lagged_item("temp=high@lag2") == "temp=high (t-2)"
    assert format_lagged_item("heater") == "heater (t)"


def test_no_rules_gives_an_empty_frame_with_the_temporal_columns():
    rules = temporal_association_rules(planted_series(rows=200), 2, 0.9, min_threshold=0.99)
    assert len(rules) == 0
    assert {"antecedents", "consequents", "support", "confidence", "lags", "max_lag"} <= set(rules.columns)


def test_input_validation():
    table = planted_series(rows=50)
    with pytest.raises(ValueError, match="'-'"):
        temporal_association_rules(table.rename(columns={"A": "a-b"}), 2, 0.1)
    with pytest.raises(ValueError, match="mas de"):
        temporal_association_rules(table.iloc[:2], 3, 0.1)


# ---------------------------------------------------------------- difusas

def test_fuzzy_version_with_crisp_memberships_matches_the_crisp_rules():
    table = planted_series(rows=500, seed=6)
    crisp = temporal_association_rules(table, 2, 0.06, min_threshold=0.6)
    fuzzy = fuzzy_temporal_association_rules(table.astype(float), 2, 0.06, num_alpha=5, min_threshold=0.6)

    def keyed(rules):
        return {(rule["antecedents"], rule["consequents"]): rule for _, rule in rules.iterrows()}

    crispRules, fuzzyRules = keyed(crisp), keyed(fuzzy)
    assert set(crispRules) == set(fuzzyRules) and len(crispRules) > 3
    for key, rule in crispRules.items():
        assert fuzzyRules[key]["support"] == pytest.approx(rule["support"])
        assert fuzzyRules[key]["confidence"] == pytest.approx(rule["confidence"])
        assert fuzzyRules[key]["lags"] == rule["lags"]


def test_fuzzy_temporal_rule_with_graded_memberships_is_found():
    rng = np.random.RandomState(7)
    rows = 600
    hot = np.clip(rng.rand(rows), 0, 1)
    heater = np.zeros(rows)
    heater[1:] = np.clip(hot[:-1] + rng.normal(0, 0.05, rows - 1), 0, 1)           # el calentador sigue a hot
    memberships = pd.DataFrame({"hot": hot, "heater": heater, "noise": rng.rand(rows)})
    rules = fuzzy_temporal_association_rules(memberships, 2, 0.15, num_alpha=10, metric="confidence",
                                             min_threshold=0.6, targets=["heater"])
    follow = rules[(rules["antecedents"] == frozenset({"hot@lag1"})) & (rules["consequents"] == frozenset({"heater"}))]
    assert len(follow) == 1
    assert follow.iloc[0]["confidence"] > 0.6 and follow.iloc[0]["lags"] == (1,)


# ----------------------------------------------------------- rule_profile

def test_rule_profile_matches_a_manual_count_per_group():
    table = planted_series(rows=600, seed=8)
    groups = np.arange(len(table)) % 3
    profile = rule_profile(table, ["A@lag2"], ["B"], groups)
    start = 2
    a = table["A"].to_numpy()[:-start]
    b = table["B"].to_numpy()[start:]
    grp = groups[start:]
    assert set(profile.index) == {0, 1, 2}
    for group in (0, 1, 2):
        mask = grp == group
        both = int((a[mask] & b[mask]).sum())
        antecedent = int(a[mask].sum())
        assert profile.loc[group, "n"] == mask.sum()
        assert profile.loc[group, "antecedent_count"] == antecedent
        assert profile.loc[group, "joint_count"] == both
        assert profile.loc[group, "support"] == pytest.approx(both / mask.sum())
        assert profile.loc[group, "confidence"] == pytest.approx(both / antecedent)
        assert profile.loc[group, "lift"] == pytest.approx((both / antecedent) / (b[mask].mean()))


def test_rule_profile_finds_a_rule_that_only_holds_at_some_hours():
    index = pd.date_range("2024-01-01", periods=24 * 60, freq="h")
    rng = np.random.RandomState(9)
    door = rng.rand(len(index)) < 0.5
    light = np.zeros(len(index), dtype=bool)
    evening = index.hour >= 18
    light[1:] = np.where(evening[1:], door[:-1], rng.rand(len(index) - 1) < 0.5)      # solo de tarde sigue a la puerta
    table = pd.DataFrame({"door": door, "light": light}, index=index)

    profile = rule_profile(table, ["door@lag1"], ["light"], lambda stamps: stamps.hour)
    eveningConfidence = profile.loc[18:, "confidence"]
    daytimeConfidence = profile.loc[2:17, "confidence"]
    assert eveningConfidence.min() > 0.99
    assert daytimeConfidence.max() < 0.8
    overall = rule_profile(table, ["door@lag1"], ["light"], np.zeros(len(table)))
    assert len(overall) == 1 and overall["confidence"].iloc[0] < 0.9


def test_rule_profile_validation():
    table = planted_series(rows=20)
    with pytest.raises(ValueError, match="columnas"):
        rule_profile(table, ["Z"], ["B"], np.zeros(20))
    with pytest.raises(ValueError, match="una entrada por fila"):
        rule_profile(table, ["A"], ["B"], np.zeros(5))
