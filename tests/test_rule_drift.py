# -*- coding: utf-8 -*-
"""Tests de ``ARM.rule_drift``: ventanas, estados de las reglas y comparacion de conjuntos de reglas."""
import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import compare_rule_sets, fuzzy_rule_miner, rule_drift, window_rules
from ARMxtend.ARM.meta_rules import rule_key

WINDOW_ROWS = 20


def build_rows(window):
    """
    Filas de una ventana de 20 instantes, determinista:

    * D y E siempre juntos (regla estable D <-> E) y en todas las filas;
    * A y B juntos en las ventanas 0-3 (regla que desaparece), C y B juntos
      en las 4-7 (regla que aparece);
    * F y G juntos solo en las ventanas 2, 3 y 5 (regla intermitente, sin
      estar en la primera ni en la ultima ventana);
    * H y J juntos solo en las ventanas 0 y 7 (extremos con huecos: intermitente).
    """
    rows = []
    for position in range(WINDOW_ROWS):
        row = {"D", "E"}
        group = position % 5
        if group in (0, 1):
            row |= {"A", "B"} if window < 4 else {"A"}
        elif group in (2, 3):
            row |= {"C"} if window < 4 else {"C", "B"}
        if window in (2, 3, 5) and position % 5 in (0, 2):
            row |= {"F", "G"}
        if window in (0, 7) and position % 5 in (1, 3):
            row |= {"H", "J"}
        rows.append(row)
    return rows


@pytest.fixture(scope="module")
def drifting_items():
    rows = [row for window in range(8) for row in build_rows(window)]
    table = pd.DataFrame([{item: True for item in row} for row in rows]).fillna(False).astype(bool)
    table.index = pd.date_range("2024-01-01", periods=len(table), freq="h")
    return table


@pytest.fixture(scope="module")
def windowed(drifting_items):
    return window_rules(drifting_items, WINDOW_ROWS, min_supp=0.3, min_threshold=0.9)


def status_of(drift, antecedent, consequent):
    match = drift[(drift["antecedents"] == frozenset(antecedent)) & (drift["consequents"] == frozenset(consequent))]
    assert len(match) == 1, "regla {} -> {} no encontrada".format(antecedent, consequent)
    return match.iloc[0]["status"]


def test_window_rules_metadata(windowed, drifting_items):
    assert windowed.attrs["n_windows"] == 8
    assert sorted(windowed["window"].unique()) == list(range(8))
    assert (windowed["n_rows"] == WINDOW_ROWS).all()
    first = windowed[windowed["window"] == 3].iloc[0]
    assert first["start"] == drifting_items.index[3 * WINDOW_ROWS]
    assert first["end"] == drifting_items.index[4 * WINDOW_ROWS - 1]
    assert {"antecedents", "consequents", "support", "confidence"} <= set(windowed.columns)


def test_rule_drift_statuses(windowed):
    drift = rule_drift(windowed)
    assert status_of(drift, {"D"}, {"E"}) == "stable"
    assert status_of(drift, {"E"}, {"D"}) == "stable"
    assert status_of(drift, {"A"}, {"B"}) == "vanishing"
    assert status_of(drift, {"B"}, {"A"}) == "vanishing"
    assert status_of(drift, {"C"}, {"B"}) == "emerging"
    assert status_of(drift, {"F"}, {"G"}) == "intermittent"          # ventanas 2, 3 y 5: ni primera ni ultima
    assert status_of(drift, {"H"}, {"J"}) == "intermittent"          # extremos pero con persistencia 2/8
    assert set(drift["status"]) <= {"stable", "emerging", "vanishing", "intermittent"}


def test_rule_drift_counts_windows_and_summarises_the_measure(windowed):
    drift = rule_drift(windowed, measure="confidence")
    row = drift[(drift["antecedents"] == frozenset({"A"})) & (drift["consequents"] == frozenset({"B"}))].iloc[0]
    assert row["windows"] == 4 and row["persistence"] == pytest.approx(0.5)
    assert row["first_window"] == 0 and row["last_window"] == 3
    assert row["mean_confidence"] == pytest.approx(1.0) and row["min_confidence"] == row["max_confidence"] == 1.0
    assert np.isfinite(row["slope"]) and row["slope"] == pytest.approx(0.0, abs=1e-12)
    stable = drift[(drift["antecedents"] == frozenset({"D"})) & (drift["consequents"] == frozenset({"E"}))].iloc[0]
    assert stable["windows"] == 8 and stable["persistence"] == 1.0
    two = drift[(drift["antecedents"] == frozenset({"H"})) & (drift["consequents"] == frozenset({"J"}))].iloc[0]
    assert np.isnan(two["slope"])                                   # menos de 3 ventanas: sin pendiente


def test_rule_drift_slope_follows_a_trend():
    rows = []
    for window in range(6):
        confidence = 0.5 + 0.1 * window                              # A -> B sube del 50 % al 100 %
        antecedent = 20
        withB = int(round(confidence * antecedent))
        for position in range(antecedent):
            rows.append({"A", "B"} if position < withB else {"A"})
    table = pd.DataFrame([{item: True for item in row} for row in rows]).fillna(False).astype(bool)
    windows = window_rules(table, 20, min_supp=0.4, min_threshold=0.5)
    drift = rule_drift(windows)
    row = drift[(drift["antecedents"] == frozenset({"A"})) & (drift["consequents"] == frozenset({"B"}))].iloc[0]
    assert row["status"] == "stable" and row["slope"] == pytest.approx(0.1, abs=1e-9)


def test_rule_drift_sort_order_and_errors(windowed):
    drift = rule_drift(windowed)
    order = ["stable", "emerging", "vanishing", "intermittent"]
    positions = [order.index(status) for status in drift["status"]]
    assert positions == sorted(positions)
    with pytest.raises(ValueError, match="measure"):
        rule_drift(windowed, measure="nope")


def test_overlapping_windows_and_row_windows(drifting_items):
    sliding = window_rules(drifting_items, 40, step=20, min_supp=0.3, min_threshold=0.9)
    assert sliding.attrs["n_windows"] == 7                           # 160 filas, ventanas de 40 cada 20
    assert sliding.groupby("window")["start"].first().iloc[1] == drifting_items.index[20]


def test_duration_windows_match_row_windows_on_regular_data(drifting_items, windowed):
    byTime = window_rules(drifting_items, "20h", min_supp=0.3, min_threshold=0.9)
    assert byTime.attrs["n_windows"] == 8
    assert len(byTime) == len(windowed)
    assert byTime["start"].tolist() == windowed["start"].tolist()


def test_incomplete_last_window_is_skipped(drifting_items):
    trimmed = drifting_items.iloc[:-5]                               # la ultima ventana de 20h solo tiene 15 filas
    full = window_rules(trimmed, "20h", min_supp=0.3, min_threshold=0.9, min_rows=20)
    assert full.attrs["n_windows"] == 7
    partial = window_rules(trimmed, "20h", min_supp=0.3, min_threshold=0.9, min_rows=2)
    assert partial.attrs["n_windows"] == 8


def test_windows_without_rules_still_count_for_persistence():
    rows = [{"A", "B"}] * 10 + [{"A"}, {"B"}, {"C"}, {"D"}, set(), {"E"}, {"F"}, {"G"}, {"H"}, {"J"}] * 1
    table = pd.DataFrame([{item: True for item in row} for row in rows]).fillna(False).astype(bool)
    windows = window_rules(table, 10, min_supp=0.5, min_threshold=0.9)
    assert windows.attrs["n_windows"] == 2 and windows["window"].nunique() == 1
    drift = rule_drift(windows)
    assert (drift["persistence"] == 0.5).all() and set(drift["status"]) == {"vanishing"}


def test_custom_and_fuzzy_miners(drifting_items):
    calls = []

    def miner(window):
        calls.append(len(window))
        return pd.DataFrame({"antecedents": [frozenset({"D"})], "consequents": [frozenset({"E"})],
                             "support": [1.0], "confidence": [1.0]})

    windows = window_rules(drifting_items, WINDOW_ROWS, miner=miner)
    assert calls == [WINDOW_ROWS] * 8 and len(windows) == 8

    degrees = pd.DataFrame({"a": [0.9, 0.8, 0.1, 0.9] * 10, "b": [0.8, 0.9, 0.2, 0.8] * 10})
    fuzzy = window_rules(degrees, 20, miner=fuzzy_rule_miner(0.3, num_alpha=4, min_threshold=0.5))
    assert fuzzy.attrs["n_windows"] == 2 and {"antecedents", "consequents", "confidence"} <= set(fuzzy.columns)
    assert len(fuzzy) > 0


def test_window_validation(drifting_items):
    with pytest.raises(ValueError, match="DatetimeIndex"):
        window_rules(drifting_items.reset_index(drop=True), "1h")
    with pytest.raises(ValueError, match=">= 1"):
        window_rules(drifting_items, 0)
    with pytest.raises(ValueError, match="ninguna ventana"):
        window_rules(drifting_items, 1000)
    with pytest.raises(ValueError, match="positivas"):
        window_rules(drifting_items, "0h")


def test_compare_rule_sets():
    before = pd.DataFrame({"antecedents": [frozenset("A"), frozenset("B"), frozenset("C"), frozenset("D")],
                           "consequents": [frozenset("X")] * 4, "confidence": [0.9, 0.8, 0.7, 0.95]})
    after = pd.DataFrame({"antecedents": [frozenset("A"), frozenset("B"), frozenset("D"), frozenset("E")],
                          "consequents": [frozenset("X")] * 4, "confidence": [0.5, 0.82, 0.96, 0.99]})
    result = compare_rule_sets(before, after, tolerance=0.05)
    byRule = {rule_key(row["antecedents"], row["consequents"]): row for _, row in result.iterrows()}
    assert byRule["A=>X"]["status"] == "changed" and byRule["A=>X"]["delta"] == pytest.approx(-0.4)
    assert byRule["B=>X"]["status"] == "stable" and byRule["B=>X"]["delta"] == pytest.approx(0.02)
    assert byRule["C=>X"]["status"] == "lost" and np.isnan(byRule["C=>X"]["confidence_after"])
    assert byRule["E=>X"]["status"] == "new" and np.isnan(byRule["E=>X"]["confidence_before"])
    assert byRule["D=>X"]["status"] == "stable"
    assert result["status"].tolist()[0] == "changed"                  # primero lo que mas cambia
    with pytest.raises(ValueError, match="columnas"):
        compare_rule_sets(before, after.drop(columns="confidence"))
