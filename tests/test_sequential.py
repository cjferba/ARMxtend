# -*- coding: utf-8 -*-
"""
Tests de ``FIM.sequential`` (PrefixSpan) y ``ARM.sequential_rules``: los
resultados se comparan con una enumeracion exhaustiva (oraculo), con ejemplos
calculados a mano y, para Spark, con el PrefixSpan de MLlib.
"""
import itertools
import random

import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM.sequential_rules import contains_rule, sequential_rules
from ARMxtend.FIM.sequential import (
    contains_sequence, format_sequence, prefixspan, prefixspan_spark, sequences_from_events, sequences_from_items,
)

ITEMS = "abcd"


def random_database(count=50, seed=0):
    rng = random.Random(seed)
    return [[rng.sample(ITEMS, rng.randint(1, 2)) for _ in range(rng.randint(2, 6))] for _ in range(count)]


def all_patterns(max_items):
    """Todos los patrones secuenciales sobre ITEMS con a lo sumo `max_items` items en total."""
    elements = [frozenset(c) for size in range(1, max_items + 1) for c in itertools.combinations(ITEMS, size)]
    patterns = []

    def grow(prefix, used):
        if prefix:
            patterns.append(tuple(prefix))
        for element in elements:
            if used + len(element) <= max_items:
                grow(prefix + [element], used + len(element))

    grow([], 0)
    return patterns


def as_dict(result):
    return {pattern: support for support, pattern in zip(result["support"], result["sequence"])}


# ------------------------------------------------------------- PrefixSpan

@pytest.mark.parametrize("seed,min_supp,max_len", [(0, 0.15, 4), (1, 0.25, 4), (2, 0.1, 3)])
def test_prefixspan_matches_exhaustive_enumeration(seed, min_supp, max_len):
    database = random_database(count=45, seed=seed)
    mined = as_dict(prefixspan(database, min_supp, max_len=max_len))
    expected = {}
    for pattern in all_patterns(max_len):
        support = sum(contains_sequence(sequence, pattern) for sequence in database) / len(database)
        if support >= min_supp - 1e-9:
            expected[pattern] = support
    assert len(expected) > 10
    assert mined.keys() == expected.keys()
    for pattern, support in expected.items():
        assert mined[pattern] == pytest.approx(support)


def test_prefixspan_hand_computed_example():
    database = [["a", "b", "c"], ["a", "c"], ["b", "c"]]
    mined = as_dict(prefixspan(database, 0.6))
    f = frozenset
    assert mined == {
        (f("a"),): pytest.approx(2 / 3), (f("b"),): pytest.approx(2 / 3), (f("c"),): pytest.approx(1.0),
        (f("a"), f("c")): pytest.approx(2 / 3), (f("b"), f("c")): pytest.approx(2 / 3),
    }


def test_prefixspan_distinguishes_simultaneous_items_from_ordered_items():
    database = [[{"a", "b"}, "c"], [{"a", "b"}, "c"], ["a", "b", "c"]]
    mined = as_dict(prefixspan(database, 0.3))
    f = frozenset
    assert mined[(f("ab"),)] == pytest.approx(2 / 3)                  # {a, b} a la vez: solo en dos secuencias
    assert mined[(f("a"), f("b"))] == pytest.approx(1 / 3)           # a y luego b: solo en la tercera
    assert mined[(f("ab"), f("c"))] == pytest.approx(2 / 3)
    assert mined[(f("a"), f("c"))] == pytest.approx(1.0)
    assert (f("b"), f("a")) not in mined


def test_prefixspan_counts_each_sequence_once():
    mined = as_dict(prefixspan([["a", "a", "a", "a"], ["b"]], 0.5))
    assert mined[(frozenset("a"),)] == pytest.approx(0.5)
    assert (frozenset("a"), frozenset("a")) in mined                  # a ... a esta en la primera secuencia
    assert mined[(frozenset("a"), frozenset("a"))] == pytest.approx(0.5)


def test_prefixspan_length_limits_and_ordering():
    database = random_database(seed=3)
    limited = prefixspan(database, 0.15, max_len=2)
    assert limited["length"].max() == 2
    onlyLong = prefixspan(database, 0.15, max_len=3, min_len=3)
    assert set(onlyLong["length"]) == {3}
    full = prefixspan(database, 0.15, max_len=3)
    assert as_dict(onlyLong).items() <= as_dict(full).items()
    assert full["support"].is_monotonic_decreasing
    assert prefixspan([], 0.5).empty


def test_format_and_contains_sequence():
    assert format_sequence((frozenset("ba"), frozenset("c"))) == "<{a, b} {c}>"
    assert contains_sequence([{"a"}, {"b", "c"}, {"d"}], [{"a"}, {"c"}, {"d"}])
    assert not contains_sequence([{"a"}, {"b", "c"}, {"d"}], [{"d"}, {"a"}])
    assert not contains_sequence([{"a", "b"}], [{"a"}, {"b"}])        # no estan en elementos distintos


# ---------------------------------------------------- construccion de secuencias

def test_sequences_from_events():
    events = pd.DataFrame({
        "patient": ["p2", "p1", "p1", "p1", "p2", "p1"],
        "time": pd.to_datetime(["2024-01-02 08:00", "2024-01-01 08:00", "2024-01-01 08:01", "2024-01-01 12:00",
                                "2024-01-02 09:00", "2024-01-01 12:00"]),
        "event": ["fever", "cough", "fever", "rest", "rest", "water"],
    })
    exact = sequences_from_events(events, "patient", "time", "event")
    assert exact == [[{"cough"}, {"fever"}, {"rest", "water"}], [{"fever"}, {"rest"}]]
    binned = sequences_from_events(events, "patient", "time", "event", bin="1h")
    assert binned[0] == [{"cough", "fever"}, {"rest", "water"}]
    with pytest.raises(ValueError, match="columna"):
        sequences_from_events(events, "nobody", "time", "event")


def test_sequences_from_items_splits_by_group_and_skips_empty_rows():
    index = pd.date_range("2024-01-01 22:00", periods=6, freq="h")
    items = pd.DataFrame({"door": [True, False, False, True, False, False],
                          "light": [False, True, False, False, False, True]}, index=index)
    one = sequences_from_items(items)
    assert one == [[{"door"}, {"light"}, {"door"}, {"light"}]]
    byDay = sequences_from_items(items, by=lambda stamps: stamps.date)
    assert byDay == [[{"door"}, {"light"}], [{"door"}, {"light"}]]
    with pytest.raises(ValueError, match="una entrada por fila"):
        sequences_from_items(items, by=[1, 2])


# ------------------------------------------------------------------ Spark

def test_prefixspan_spark_matches_the_local_implementation(real_sc):
    database = random_database(count=40, seed=5)
    local = prefixspan(database, 0.25, max_len=4)
    distributed = prefixspan_spark(real_sc, database, 0.25, max_pattern_length=4)
    assert len(local) > 5
    assert as_dict(distributed).keys() == as_dict(local).keys()
    for pattern, support in as_dict(local).items():
        assert as_dict(distributed)[pattern] == pytest.approx(support)
    fromRdd = prefixspan_spark(real_sc, real_sc.parallelize(database, 3), 0.25, max_pattern_length=4)
    assert as_dict(fromRdd).keys() == as_dict(local).keys()


# --------------------------------------------------------- reglas secuenciales

def brute_force_rules(database, min_supp, min_conf, max_x, max_y):
    total = len(database)
    expected = {}
    for sizeX in range(1, max_x + 1):
        for antecedent in itertools.combinations(ITEMS, sizeX):
            for sizeY in range(1, max_y + 1):
                for consequent in itertools.combinations(ITEMS, sizeY):
                    if set(antecedent) & set(consequent):
                        continue
                    valid = sum(contains_rule(sequence, antecedent, consequent) for sequence in database)
                    present = sum(all(any(item in element for element in sequence) for item in antecedent)
                                  for sequence in database)
                    if present and valid / total >= min_supp - 1e-9 and valid / present >= min_conf - 1e-9:
                        expected[(frozenset(antecedent), frozenset(consequent))] = (valid / total, valid / present)
    return expected


@pytest.mark.parametrize("seed,min_supp,min_conf", [(0, 0.1, 0.5), (1, 0.2, 0.4), (2, 0.15, 0.6)])
def test_sequential_rules_match_exhaustive_enumeration(seed, min_supp, min_conf):
    database = random_database(count=60, seed=seed)
    rules = sequential_rules(database, min_supp, min_conf, max_antecedent=2, max_consequent=2)
    expected = brute_force_rules(database, min_supp, min_conf, 2, 2)
    assert len(expected) >= 3
    mined = {(rule["antecedents"], rule["consequents"]): (rule["support"], rule["confidence"])
             for _, rule in rules.iterrows()}
    assert mined.keys() == expected.keys()
    for key, (support, confidence) in expected.items():
        assert mined[key] == pytest.approx((support, confidence))


def test_sequential_rules_hand_computed_example_and_measures():
    database = [["a", "b"], ["a", "b"], ["b", "a"], [{"a", "b"}]]
    rules = sequential_rules(database, 0.25, 0.0, 1, 1)
    byRule = {(tuple(r["antecedents"]), tuple(r["consequents"])): r for _, r in rules.iterrows()}
    ab, ba = byRule[(("a",), ("b",))], byRule[(("b",), ("a",))]
    assert ab["support"] == pytest.approx(2 / 4) and ab["confidence"] == pytest.approx(2 / 4)   # a antes de b: 2 de 4
    assert ba["support"] == pytest.approx(1 / 4) and ba["confidence"] == pytest.approx(1 / 4)
    assert ab["antecedent support"] == pytest.approx(1.0) and ab["consequent support"] == pytest.approx(1.0)
    assert ab["lift"] == pytest.approx(0.5)
    # a y b en el mismo elemento no cuenta como "a antes de b"
    assert not contains_rule([{"a", "b"}], ["a"], ["b"])
    assert contains_rule([{"a"}, {"c"}, {"b", "a"}], ["a", "c"], ["a", "b"])        # a, c antes de (a, b)
    assert not contains_rule([{"a"}, {"c"}, {"b"}], ["a", "c"], ["a", "b"])
    assert list(rules.columns[:7]) == ["antecedents", "consequents", "antecedent support", "consequent support",
                                       "support", "confidence", "lift"]
    assert rules["confidence"].is_monotonic_decreasing


def test_sequential_rules_size_limits_and_metric_filter():
    database = random_database(count=60, seed=4)
    small = sequential_rules(database, 0.05, 0.3, max_antecedent=1, max_consequent=1)
    assert (small["antecedents"].apply(len) == 1).all() and (small["consequents"].apply(len) == 1).all()
    wide = sequential_rules(database, 0.05, 0.3, max_antecedent=2, max_consequent=2)
    assert len(wide) > len(small) and wide["antecedents"].apply(len).max() == 2
    threshold = float(wide["lift"].median())
    filtered = sequential_rules(database, 0.05, 0.3, max_antecedent=2, max_consequent=2, metric="lift",
                                min_threshold=threshold)
    assert 0 < len(filtered) < len(wide) and (filtered["lift"] >= threshold - 1e-12).all()
    assert len(filtered) == int((wide["lift"] >= threshold - 1e-12).sum())


def test_sequential_rules_validation_and_empty_cases():
    assert sequential_rules([], 0.1).empty
    with pytest.raises(ValueError, match="metric"):
        sequential_rules([["a", "b"]], 0.1, metric="magic")
    with pytest.raises(ValueError, match="max_antecedent"):
        sequential_rules([["a", "b"]], 0.1, max_antecedent=0)
    with pytest.raises(ValueError, match="'-'"):
        sequential_rules([["a-1", "b"]], 0.1)
    none = sequential_rules([["a"], ["b"]], 0.5, 0.9)
    assert none.empty and "confidence" in none.columns


def test_sequential_rules_from_a_planted_routine():
    """Cada dia: puerta, luego cocina, luego cafe (90 %), con ruido en medio."""
    rng = np.random.RandomState(0)
    days = []
    for _ in range(100):
        day = ["door"]
        day += [str(rng.choice(["tv", "phone"])) for _ in range(rng.randint(0, 3))]
        day.append("kitchen")
        if rng.rand() < 0.9:
            day.append("coffee")
        days.append(day)
    rules = sequential_rules(days, 0.5, 0.8, 2, 1)
    byRule = {(tuple(sorted(r["antecedents"])), tuple(r["consequents"])): r for _, r in rules.iterrows()}
    assert byRule[(("door",), ("kitchen",))]["confidence"] == pytest.approx(1.0)
    assert byRule[(("kitchen",), ("coffee",))]["confidence"] == pytest.approx(0.9, abs=0.07)
    assert (("coffee",), ("kitchen",)) not in byRule
