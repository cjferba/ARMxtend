# -*- coding: utf-8 -*-
"""
Tests of the fuzzy sequential patterns and rules (``FIM.fuzzy_sequential``,
``ARM.fuzzy_sequential_rules``): against brute-force enumerations, against the
crisp versions when the degrees are 0/1, and against hand-computed values.
"""
from itertools import combinations, product

import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import fuzzy_rule_degree, fuzzy_sequential_rules, sequential_rules
from ARMxtend.FIM import (
    fuzzy_pattern_degree, fuzzy_prefixspan, fuzzy_sequences_from_memberships, prefixspan,
)

ITEMS = ["a", "b", "c", "d"]


def tnorm_of(values, tnorm):
    return min(values) if tnorm == "min" else float(np.prod(values))


def random_sequences(seed, count=25, crisp=False, max_length=6):
    rng = np.random.RandomState(seed)
    sequences = []
    for _ in range(count):
        sequence = []
        for _ in range(rng.randint(1, max_length + 1)):
            element = {item: 1.0 if crisp else float(np.round(0.3 + 0.7 * rng.rand(), 2))
                       for item in ITEMS if rng.rand() < 0.4}
            if crisp or rng.rand() < 0.9:
                sequence.append(element)
        sequences.append(sequence)
    return sequences


def brute_degree(sequence, pattern, tnorm):
    """Best fit by enumerating all the strictly increasing positions."""
    best = 0.0
    for positions in combinations(range(len(sequence)), len(pattern)):
        degrees = [sequence[p].get(item, 0.0) for p, item in zip(positions, pattern)]
        best = max(best, tnorm_of(degrees, tnorm))
    return best


@pytest.mark.parametrize("tnorm", ["min", "product"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pattern_degree_matches_brute_force(seed, tnorm):
    for sequence in random_sequences(seed):
        for length in (1, 2, 3):
            for pattern in product(ITEMS, repeat=length):
                assert fuzzy_pattern_degree(sequence, pattern, tnorm) == pytest.approx(
                    brute_degree(sequence, pattern, tnorm))


def test_hand_computed_supports():
    sequences = [[{"a": 0.8}, {"b": 0.6}, {"c": 1.0}], [{"a": 1.0}, {"c": 0.5}], [{"b": 0.9}, {"a": 0.4}]]
    found = fuzzy_prefixspan(sequences, 0.1)
    support = {tuple(next(iter(e)) for e in row["sequence"]): row["support"] for _, row in found.iterrows()}
    assert support[("a",)] == pytest.approx((0.8 + 1.0 + 0.4) / 3)
    assert support[("a", "b")] == pytest.approx(0.6 / 3)                 # only in the first: min(0.8, 0.6)
    assert support[("a", "c")] == pytest.approx((0.8 + 0.5) / 3)
    assert support[("b", "a")] == pytest.approx(0.4 / 3)                 # tercera: b (0.9) y despues a (0.4)
    assert support[("a", "b", "c")] == pytest.approx(0.6 / 3)
    assert ("c", "a") not in support
    product_support = fuzzy_prefixspan(sequences, 0.1, tnorm="product")
    byPattern = {tuple(next(iter(e)) for e in row["sequence"]): row["support"] for _, row in product_support.iterrows()}
    assert byPattern[("a", "b", "c")] == pytest.approx(0.8 * 0.6 * 1.0 / 3)


@pytest.mark.parametrize("tnorm", ["min", "product"])
def test_prefixspan_equals_the_brute_force_enumeration(tnorm):
    sequences = random_sequences(5)
    min_supp = 0.05
    found = fuzzy_prefixspan(sequences, min_supp, max_len=3, tnorm=tnorm)
    result = {tuple(next(iter(e)) for e in row["sequence"]): row["support"] for _, row in found.iterrows()}
    expected = {}
    for length in (1, 2, 3):
        for pattern in product(ITEMS, repeat=length):
            support = np.mean([brute_degree(s, pattern, tnorm) for s in sequences])
            if support >= min_supp - 1e-9:
                expected[pattern] = support
    assert len(expected) > 10 and set(result) == set(expected)
    for pattern, support in expected.items():
        assert result[pattern] == pytest.approx(support)


def test_crisp_degrees_give_the_crisp_support_of_the_single_item_patterns():
    sequences = random_sequences(9, crisp=True)
    fuzzy = fuzzy_prefixspan(sequences, 0.2)
    crisp = prefixspan([[frozenset(e) for e in s if e] for s in sequences], 0.2)
    crispSingles = {row["sequence"]: row["support"] for _, row in crisp.iterrows()
                    if all(len(element) == 1 for element in row["sequence"])}
    assert len(crispSingles) > 5
    assert {row["sequence"]: row["support"] for _, row in fuzzy.iterrows()} == pytest.approx(crispSingles)


def test_output_format_ordering_and_limits():
    sequences = random_sequences(3)
    found = fuzzy_prefixspan(sequences, 0.1)
    assert list(found.columns) == ["support", "sequence", "length"]
    assert found["support"].is_monotonic_decreasing
    assert all(len(row["sequence"]) == row["length"] and all(len(e) == 1 for e in row["sequence"])
               for _, row in found.iterrows())
    assert found["length"].max() > 2
    assert fuzzy_prefixspan(sequences, 0.1, max_len=2)["length"].max() == 2
    assert fuzzy_prefixspan(sequences, 0.1, min_len=2)["length"].min() == 2
    assert fuzzy_prefixspan([], 0.5).empty
    # anti-monotonicity: the support of a pattern never exceeds that of its prefix
    bySequence = {row["sequence"]: row["support"] for _, row in found.iterrows()}
    assert all(support <= bySequence[pattern[:-1]] + 1e-12 for pattern, support in bySequence.items()
               if len(pattern) > 1)


def test_input_forms_and_errors():
    mixed = [[{"a": 0.5, "b": 1.0}, [("c", 0.25)], "d", ["a", "b"], {}], [["a"], "b"]]
    found = fuzzy_prefixspan(mixed, 0.05, max_len=1)
    support = {next(iter(row["sequence"][0])): row["support"] for _, row in found.iterrows()}
    assert support == pytest.approx({"a": (1.0 + 1.0) / 2, "b": 1.0, "c": 0.125, "d": 0.5})
    with pytest.raises(ValueError, match="min_supp"):
        fuzzy_prefixspan(mixed, 0.0)
    with pytest.raises(ValueError, match="tnorm"):
        fuzzy_prefixspan(mixed, 0.5, tnorm="lukasiewicz")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fuzzy_prefixspan([[{"a": 1.5}]], 0.5)
    with pytest.raises(ValueError, match="'-'"):
        fuzzy_prefixspan([[{"a-b": 0.5}]], 0.5)


def test_sequences_from_memberships_groups_rows_and_thresholds_degrees():
    memberships = pd.DataFrame({"hot": [0.9, 0.0, 0.2, 0.7], "on": [0.0, 0.5, 0.0, 1.0]})
    assert fuzzy_sequences_from_memberships(memberships) == [[{"hot": 0.9}, {"on": 0.5}, {"hot": 0.2}, {"hot": 0.7, "on": 1.0}]]
    byDay = fuzzy_sequences_from_memberships(memberships, by=["d2", "d2", "d1", "d1"], min_degree=0.3)
    assert byDay == [[{"hot": 0.9}, {"on": 0.5}], [{"hot": 0.7, "on": 1.0}]]       # row 2 (hot 0.2) gives no element
    with pytest.raises(ValueError, match="one entry per row"):
        fuzzy_sequences_from_memberships(memberships, by=["x"])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fuzzy_sequences_from_memberships(memberships * 2)


# --- Reglas -----------------------------------------------------------------------------------

def brute_rule_degree(sequence, antecedent, consequent, tnorm):
    best = 0.0
    for k in range(len(sequence)):
        before = [max([sequence[p].get(x, 0.0) for p in range(k + 1)]) for x in antecedent]
        after = [max([sequence[p].get(y, 0.0) for p in range(k + 1, len(sequence))] or [0.0]) for y in consequent]
        best = max(best, tnorm_of(before + after, tnorm))
    return best


def test_hand_computed_rule_measures():
    sequences = [[{"a": 0.8}, {"b": 0.6}], [{"a": 1.0}, {"b": 0.5}, {"a": 0.7}]]
    rules = fuzzy_sequential_rules(sequences, 0.2, min_conf=0.0, max_antecedent=1, max_consequent=1)
    byRule = {(next(iter(r["antecedents"])), next(iter(r["consequents"]))): r for _, r in rules.iterrows()}
    ab = byRule[("a", "b")]
    assert ab["support"] == pytest.approx((0.6 + 0.5) / 2)
    assert ab["antecedent support"] == pytest.approx((0.8 + 1.0) / 2)
    assert ab["consequent support"] == pytest.approx((0.6 + 0.5) / 2)
    assert ab["confidence"] == pytest.approx(0.55 / 0.9) and ab["lift"] == pytest.approx(0.55 / 0.9 / 0.55)
    ba = byRule[("b", "a")]
    assert ba["support"] == pytest.approx(0.5 / 2) and ba["confidence"] == pytest.approx(0.25 / 0.55)
    assert fuzzy_rule_degree(sequences[1], ["b"], ["a"]) == pytest.approx(0.5)
    assert fuzzy_rule_degree(sequences[0], ["b"], ["a"]) == 0.0
    assert fuzzy_rule_degree(sequences[0], ["a"], ["zzz"]) == 0.0
    product_rules = fuzzy_sequential_rules(sequences, 0.1, min_conf=0.0, max_antecedent=1, max_consequent=1, tnorm="product")
    assert product_rules[(product_rules["antecedents"] == {"a"}) & (product_rules["consequents"] == {"b"})][
        "support"].iloc[0] == pytest.approx((0.8 * 0.6 + 1.0 * 0.5) / 2)


@pytest.mark.parametrize("tnorm", ["min", "product"])
def test_rules_equal_the_brute_force_enumeration(tnorm):
    sequences = random_sequences(11)
    min_supp, min_conf = 0.04, 0.2
    rules = fuzzy_sequential_rules(sequences, min_supp, min_conf, max_antecedent=2, max_consequent=1, tnorm=tnorm)
    found = {(r["antecedents"], r["consequents"]): r for _, r in rules.iterrows()}

    def basket(items):
        return np.mean([tnorm_of([max([e.get(i, 0.0) for e in s] or [0.0]) for i in items], tnorm) for s in sequences])

    expected = {}
    for size in (1, 2):
        for antecedent in combinations(ITEMS, size):
            for consequent in (c for c in ITEMS if c not in antecedent):
                support = np.mean([brute_rule_degree(s, antecedent, [consequent], tnorm) for s in sequences])
                confidence = support / basket(antecedent) if basket(antecedent) else 0.0
                if support >= min_supp - 1e-9 and confidence >= min_conf - 1e-9:
                    expected[(frozenset(antecedent), frozenset([consequent]))] = (support, confidence)
    assert len(expected) > 5 and set(found) == set(expected)
    for key, (support, confidence) in expected.items():
        assert found[key]["support"] == pytest.approx(support) and found[key]["confidence"] == pytest.approx(confidence)
        assert fuzzy_rule_degree(sequences[0], key[0], key[1], tnorm) == pytest.approx(
            brute_rule_degree(sequences[0], sorted(key[0]), sorted(key[1]), tnorm))


def test_crisp_degrees_reproduce_the_sequential_rules():
    sequences = random_sequences(4, crisp=True)
    crisp = sequential_rules([[frozenset(e) for e in s if e] for s in sequences], 0.15, min_conf=0.3,
                             max_antecedent=2, max_consequent=2)
    fuzzy = fuzzy_sequential_rules(sequences, 0.15, min_conf=0.3, max_antecedent=2, max_consequent=2)
    assert len(crisp) > 10 and len(fuzzy) == len(crisp)
    key = lambda frame: frame.set_index(frame["antecedents"].map(sorted).map(tuple).str.join(",") + "=>" +
                                        frame["consequents"].map(sorted).map(tuple).str.join(",")).sort_index()
    left, right = key(crisp), key(fuzzy)
    assert list(left.index) == list(right.index)
    for column in ["support", "confidence", "lift", "leverage", "antecedent support", "consequent support"]:
        assert right[column].to_numpy(dtype=float) == pytest.approx(left[column].to_numpy(dtype=float))


def test_rule_filters_and_errors():
    sequences = random_sequences(2)
    every = fuzzy_sequential_rules(sequences, 0.1, min_conf=0.0)
    assert (every["confidence"].diff().dropna() <= 1e-12).all()
    median = float(every["lift"].median())
    lifted = fuzzy_sequential_rules(sequences, 0.1, min_conf=0.0, metric="lift", min_threshold=median)
    assert 0 < len(lifted) < len(every) and (lifted["lift"] >= median - 1e-9).all()
    assert fuzzy_sequential_rules([], 0.1).empty
    assert fuzzy_sequential_rules(sequences, 0.1, min_conf=0.0, max_antecedent=1, max_consequent=1)["antecedents"].map(len).max() == 1
    with pytest.raises(ValueError, match="metric"):
        fuzzy_sequential_rules(sequences, 0.1, metric="magic")
    with pytest.raises(ValueError, match="min_supp"):
        fuzzy_sequential_rules(sequences, 1.5)
    with pytest.raises(ValueError, match="max_antecedent"):
        fuzzy_sequential_rules(sequences, 0.1, max_antecedent=0)
    with pytest.raises(ValueError, match="tnorm"):
        fuzzy_sequential_rules(sequences, 0.1, tnorm="x")


def test_fuzzy_rules_recover_a_planted_routine_with_noisy_degrees():
    rng = np.random.RandomState(0)
    sequences = []
    for _ in range(200):
        high = rng.rand() * 0.5 + 0.5                              # temperatura alta con grado 0.5-1
        sequence = [{"temp_high": high, "noise": rng.rand()}, {"noise": rng.rand()}]
        if rng.rand() < 0.9:
            sequence.append({"heater_on": float(np.clip(high + rng.normal(0, 0.1), 0.05, 1))})
        sequence.append({"noise": rng.rand()})
        sequences.append(sequence)
    rules = fuzzy_sequential_rules(sequences, 0.3, min_conf=0.6, max_antecedent=1, max_consequent=1)
    planted = rules[(rules["antecedents"] == {"temp_high"}) & (rules["consequents"] == {"heater_on"})].iloc[0]
    assert 0.78 < planted["confidence"] < 0.92 and planted["lift"] > 1.0       # 90 % of the days, with somewhat lower degrees
    reverse = rules[(rules["antecedents"] == {"heater_on"}) & (rules["consequents"] == {"temp_high"})]
    assert reverse.empty                                                          # el orden importa


def brute_itemset_degree(sequence, pattern, tnorm):
    """Best fit of a pattern of itemset elements, enumerating all the strictly increasing positions."""
    best = 0.0
    for positions in combinations(range(len(sequence)), len(pattern)):
        degrees = [sequence[p].get(item, 0.0) for p, element in zip(positions, pattern) for item in element]
        best = max(best, tnorm_of(degrees, tnorm))
    return best


def all_itemset_patterns(max_items):
    elements = [frozenset(c) for size in range(1, max_items + 1) for c in combinations(ITEMS, size)]
    patterns = [(e,) for e in elements if len(e) <= max_items]
    frontier = patterns
    while frontier:
        grown = [p + (e,) for p in frontier for e in elements if sum(map(len, p)) + len(e) <= max_items]
        patterns += grown
        frontier = grown
    return patterns


@pytest.mark.parametrize("tnorm", ["min", "product"])
def test_itemset_elements_equal_the_brute_force_enumeration(tnorm):
    sequences = random_sequences(5, count=30)
    min_supp = 0.08
    found = fuzzy_prefixspan(sequences, min_supp, max_len=3, tnorm=tnorm, itemset_elements=True)
    got = {row["sequence"]: row["support"] for _, row in found.iterrows()}
    expected = {}
    for pattern in all_itemset_patterns(3):
        support = np.mean([brute_itemset_degree(s, pattern, tnorm) for s in sequences])
        if support >= min_supp - 1e-9:
            expected[tuple(pattern)] = support
    assert len(expected) > 20 and any(len(element) > 1 for pattern in expected for element in pattern)
    assert set(got) == set(expected)
    assert got == pytest.approx(expected)
    assert (found["length"] == [sum(map(len, p)) for p in found["sequence"]]).all()


def test_itemset_elements_with_crisp_degrees_reproduce_prefixspan():
    sequences = random_sequences(11, crisp=True, count=40)
    fuzzy = fuzzy_prefixspan(sequences, 0.15, itemset_elements=True)
    crisp = prefixspan([[frozenset(e) for e in s if e] for s in sequences], 0.15)
    assert {r["sequence"]: r["support"] for _, r in fuzzy.iterrows()} == pytest.approx(
        {r["sequence"]: r["support"] for _, r in crisp.iterrows()})


def test_pattern_degree_accepts_itemset_elements():
    sequence = [{"a": 0.9, "b": 0.5}, {"c": 0.7}]
    assert fuzzy_pattern_degree(sequence, [{"a", "b"}, "c"]) == pytest.approx(0.5)
    assert fuzzy_pattern_degree(sequence, [{"a", "b"}, "c"], tnorm="product") == pytest.approx(0.9 * 0.5 * 0.7)
    assert fuzzy_pattern_degree(sequence, ["c", {"a", "b"}]) == 0.0
