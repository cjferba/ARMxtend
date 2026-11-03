# -*- coding: utf-8 -*-
"""Tests of FIM.topk / ARM.topk_rules, FIM.rare / ARM.rare_rules and FIM.utility / ARM.utility_rules."""
from itertools import chain, combinations

import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import high_utility_rules, rare_rules, significant_rules, top_k_rules
from ARMxtend.FIM import high_utility_itemsets, minimal_rare_itemsets, top_k_itemsets
from ARMxtend.FIM._vertical import Vertical, count_bits
from ARMxtend.FIM.topk import iter_itemsets_by_support

ITEMS = list("abcdefgh")


def random_baskets(seed, rows=60, items=ITEMS, density=(0.1, 0.6)):
    rng = np.random.RandomState(seed)
    probability = rng.uniform(*density, len(items))
    return [[item for item, hit in zip(items, rng.rand(len(items)) < probability) if hit] for _ in range(rows)]


def all_itemsets(baskets, max_len=None):
    """{frozenset: number of transactions} of all the non-empty itemsets with some support."""
    items = sorted({item for basket in baskets for item in basket})
    counts = {}
    for size in range(1, (max_len or len(items)) + 1):
        for itemset in combinations(items, size):
            count = sum(set(itemset) <= set(basket) for basket in baskets)
            if count:
                counts[frozenset(itemset)] = count
    return counts


def subsets(itemset, minimum=1):
    return chain.from_iterable(combinations(sorted(itemset), size) for size in range(minimum, len(itemset)))


# -------------------------------------------------------------------------------- vertical

def test_vertical_representation_counts_transactions():
    baskets = [["a", "b"], ["a"], ["b", "c"], [], ["a", "b", "c"]]
    vertical = Vertical(baskets)
    assert vertical.items == ["a", "b", "c"] and vertical.total == 5 and vertical.counts == [3, 3, 2]
    assert vertical.count_of([0, 1]) == 2 and vertical.count_of([1, 0]) == 2 and vertical.count_of([0, 1, 2]) == 1
    assert count_bits(vertical.mask_of([0, 2])) == 1 and Vertical([]).total == 0


# ------------------------------------------------------------------------------------ top-k

def brute_top_k(baskets, k, min_len=1, max_len=None, min_supp=0.0):
    counts = all_itemsets(baskets, max_len)
    floor = max(1, int(np.ceil(min_supp * len(baskets) - 1e-9)))
    ranked = sorted(((count, sorted(itemset)) for itemset, count in counts.items() if count >= floor and len(itemset) >= min_len),
                    key=lambda entry: (-entry[0], len(entry[1]), entry[1]))
    return [(count / len(baskets), frozenset(itemset)) for count, itemset in ranked[:k]]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k,min_len,max_len,min_supp", [(1, 1, None, 0.0), (12, 1, None, 0.0), (30, 2, None, 0.0), (15, 2, 3, 0.1), (500, 1, 2, 0.0)])
def test_top_k_itemsets_equal_the_brute_force_ranking(seed, k, min_len, max_len, min_supp):
    baskets = random_baskets(seed)
    found = top_k_itemsets(baskets, k, min_len=min_len, max_len=max_len, min_supp=min_supp)
    expected = brute_top_k(baskets, k, min_len, max_len, min_supp)
    assert found["support"].tolist() == pytest.approx([s for s, _ in expected])
    assert list(found["itemsets"]) == [items for _, items in expected]
    assert found.attrs["n_transactions"] == len(baskets)


def test_the_generator_is_lazy_and_orders_by_decreasing_support():
    items = ["i{}".format(n) for n in range(40)]                       # 2^40 itemsets: imposible de enumerar
    rng = np.random.RandomState(0)
    baskets = [[item for item in items if rng.rand() < 0.7] for _ in range(200)]
    vertical = Vertical(baskets)
    first = [count for _, (count, _) in zip(range(300), iter_itemsets_by_support(vertical))]
    assert first == sorted(first, reverse=True) and len(first) == 300
    assert len(top_k_itemsets(baskets, 50, min_len=2)) == 50


def test_top_k_itemsets_edge_cases_and_errors():
    assert len(top_k_itemsets([], 3)) == 0 and top_k_itemsets([], 3).attrs["n_transactions"] == 0
    assert len(top_k_itemsets([["a"], ["b"]], 10)) == 2                                   # menos de k itemsets
    assert len(top_k_itemsets([["a"], ["b"]], 10, min_supp=0.9)) == 0
    for call in (lambda: top_k_itemsets([["a"]], 0), lambda: top_k_itemsets([["a"]], 1, min_supp=2)):
        with pytest.raises(ValueError):
            call()


def brute_top_k_rules(baskets, k, min_conf, min_lift=None, max_antecedent=None, max_consequent=None, max_len=None):
    counts, total = all_itemsets(baskets, max_len), len(baskets)
    rules = []
    for itemset, count in counts.items():
        for antecedent in subsets(itemset):
            consequent = itemset - frozenset(antecedent)
            if (max_antecedent and len(antecedent) > max_antecedent) or (max_consequent and len(consequent) > max_consequent):
                continue
            confidence = count / counts[frozenset(antecedent)]
            lift = confidence / (counts[consequent] / total)
            if confidence >= min_conf - 1e-9 and (min_lift is None or lift >= min_lift - 1e-9):
                rules.append((count / total, confidence, lift, sorted(antecedent), sorted(consequent)))
    rules.sort(key=lambda rule: (-rule[0], -rule[1], rule[3], rule[4]))
    return rules[:k]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k,min_conf,options", [(5, 0.6, {}), (20, 0.4, {"max_antecedent": 1, "max_consequent": 2}),
                                                (10, 0.5, {"min_lift": 1.2, "max_len": 3}), (3, 0.9, {}), (400, 0.3, {"max_len": 3})])
def test_top_k_rules_equal_the_brute_force_ranking(seed, k, min_conf, options):
    baskets = random_baskets(seed, rows=70)
    found = top_k_rules(baskets, k, min_conf=min_conf, **options)
    expected = brute_top_k_rules(baskets, k, min_conf, **{key: value for key, value in options.items()})
    assert len(found) == len(expected) > 0 or k == 3
    assert [(sorted(a), sorted(c)) for a, c in zip(found["antecedents"], found["consequents"])] == [(a, c) for *_, a, c in expected]
    assert found["support"].tolist() == pytest.approx([r[0] for r in expected])
    assert found["confidence"].tolist() == pytest.approx([r[1] for r in expected])
    assert found["lift"].tolist() == pytest.approx([r[2] for r in expected])
    assert found.attrs["n_transactions"] == len(baskets) and found.attrs["n_candidates"] >= len(found)


def test_top_k_rules_default_length_cap_downstream_and_errors():
    baskets = random_baskets(3, rows=300, density=(0.3, 0.8))
    rules = top_k_rules(baskets, 25, min_conf=0.5)
    assert (rules["antecedents"].map(len) + rules["consequents"].map(len)).max() <= 6
    assert rules["support"].is_monotonic_decreasing and {"lift", "leverage", "certainty_factor"} <= set(rules.columns)
    assert len(significant_rules(rules, 0.5, n_transactions=300)) <= len(rules)
    assert len(top_k_rules([], 5)) == 0 and len(top_k_rules([["a"], ["b"]], 5)) == 0
    for call in (lambda: top_k_rules(baskets, 0), lambda: top_k_rules(baskets, 5, min_conf=2),
                 lambda: top_k_rules(baskets, 5, max_antecedent=0)):
        with pytest.raises(ValueError):
            call()


# ------------------------------------------------------------------------------------ raros

def brute_minimal_rare(baskets, max_supp, min_supp=0.0, min_count=1, min_len=1, max_len=None):
    total = len(baskets)
    counts = all_itemsets(baskets)
    frequent = max(1, int(np.ceil(max_supp * total - 1e-9)))
    rare = max(min_count, int(np.ceil(min_supp * total - 1e-9)), 1)
    found = {}
    for itemset, count in counts.items():
        if rare <= count < frequent and len(itemset) >= min_len and (max_len is None or len(itemset) <= max_len) \
                and all(counts.get(frozenset(sub), 0) >= frequent for sub in subsets(itemset)):
            found[itemset] = count / total
    return found


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("max_supp,options", [(0.4, {}), (0.25, {"min_supp": 0.05}), (0.3, {"min_count": 3, "min_len": 2}), (0.5, {"max_len": 2})])
def test_minimal_rare_itemsets_equal_the_brute_force_definition(seed, max_supp, options):
    baskets = random_baskets(seed, rows=50)
    found = minimal_rare_itemsets(baskets, max_supp, **options)
    expected = brute_minimal_rare(baskets, max_supp, **options)
    assert len(expected) > 0 and dict(zip(found["itemsets"], found["support"])) == pytest.approx(expected)
    assert found["support"].is_monotonic_decreasing and found.attrs["n_transactions"] == 50 and found.attrs["n_frequent"] >= 0


def test_minimal_rare_itemsets_are_a_compact_basis_hand_checked():
    baskets = [["a", "b"]] * 4 + [["a", "c"]] * 3 + [["b", "c"]] * 3 + [["a", "b", "c"]] + [["d"]]
    found = minimal_rare_itemsets(baskets, 0.3)
    # a, b, c are frequent (8, 8, 7 of 12); {a,b} 5, {a,c} 4, {b,c} 4 also (>= 3.6); {a,b,c} = 1 is rare and minimal; d = 1 is rare
    assert set(found["itemsets"]) == {frozenset("abc"), frozenset("d")}
    assert minimal_rare_itemsets(baskets, 0.3, min_len=2)["itemsets"].tolist() == [frozenset("abc")]
    assert len(minimal_rare_itemsets(baskets, 0.3, min_count=2)) == 0
    assert len(minimal_rare_itemsets([], 0.5)) == 0
    for call in (lambda: minimal_rare_itemsets(baskets, 0), lambda: minimal_rare_itemsets(baskets, 0.3, min_supp=0.5),
                 lambda: minimal_rare_itemsets(baskets, 0.3, min_count=0)):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("seed", range(4))
def test_rare_rules_equal_the_brute_force_splits(seed):
    baskets = random_baskets(seed, rows=80)
    total = len(baskets)
    counts = all_itemsets(baskets)
    found = rare_rules(baskets, 0.3, min_count=2, min_conf=0.5)
    expected = {}
    for itemset in brute_minimal_rare(baskets, 0.3, min_count=2, min_len=2):
        for antecedent in subsets(itemset):
            consequent = itemset - frozenset(antecedent)
            confidence = counts[itemset] / counts[frozenset(antecedent)]
            if confidence >= 0.5 - 1e-9:
                expected[(frozenset(antecedent), consequent)] = (counts[itemset] / total, confidence)
    assert len(expected) > 0 and set(zip(found["antecedents"], found["consequents"])) == set(expected)
    for antecedent, consequent, support, confidence in zip(found["antecedents"], found["consequents"], found["support"], found["confidence"]):
        assert (support, confidence) == pytest.approx(expected[(antecedent, consequent)])
    assert (found["support"] < 0.3).all() and found["confidence"].is_monotonic_decreasing
    assert found["antecedent support"].min() >= 0.3 - 1e-9 and found["consequent support"].min() >= 0.3 - 1e-9
    assert found.attrs["n_candidates"] >= len(found)


def test_rare_rules_options_and_errors():
    # a and d are frequent (0.30) and together are not (0.15): the pair is rare and minimal, and two rules come out of it
    baskets = [["a", "d"]] * 15 + [["a"]] * 15 + [["d"]] * 15 + [["b"]] * 55
    rules = rare_rules(baskets, 0.2, min_count=2, min_conf=0.5)
    assert set(zip(rules["antecedents"], rules["consequents"])) == {(frozenset("a"), frozenset("d")), (frozenset("d"), frozenset("a"))}
    assert rules["support"].tolist() == pytest.approx([0.15, 0.15]) and rules["lift"].tolist() == pytest.approx([0.5 / 0.3] * 2)
    assert len(rare_rules(baskets, 0.2, min_conf=0.5, min_lift=1.8)) == 0
    assert len(rare_rules(baskets, 0.2, min_conf=0.5, min_lift=1.5)) == 2
    assert len(rare_rules(baskets, 0.2, min_conf=0.6)) == 0
    assert len(rare_rules(baskets, 0.2, min_conf=0.5, max_antecedent=1, max_consequent=1)) == 2
    assert len(rare_rules(baskets, 0.2, min_conf=0.5, min_count=16)) == 0
    assert len(rare_rules([], 0.2)) == 0
    for call in (lambda: rare_rules(baskets, 0.2, min_conf=2), lambda: rare_rules(baskets, 0.2, max_consequent=0),
                 lambda: rare_rules(baskets, 0.2, max_antecedent=0)):
        with pytest.raises(ValueError):
            call()


# ------------------------------------------------------------------------------ alta utilidad

PROFITS = {"a": 5, "b": 2, "c": 1, "d": 2, "e": 3, "f": 1, "g": 1}
QUANTITIES = [{"a": 1, "c": 1, "d": 1}, {"a": 2, "c": 6, "e": 2, "g": 5}, {"a": 1, "b": 2, "c": 1, "d": 6, "e": 1, "f": 5},
              {"b": 4, "c": 3, "d": 3, "f": 1}, {"b": 2, "c": 2, "e": 1, "g": 2}]


def brute_utilities(table, profits):
    """{itemset: utility} of all the itemsets with some transaction, by definition."""
    items = sorted({item for row in table for item in row})
    result = {}
    for size in range(1, len(items) + 1):
        for itemset in combinations(items, size):
            contained = [row for row in table if all(row.get(i, 0) > 0 for i in itemset)]
            if contained:
                result[frozenset(itemset)] = sum(row[i] * profits[i] for row in contained for i in itemset)
    return result


def test_high_utility_itemsets_hand_checked():
    utilities = brute_utilities(QUANTITIES, PROFITS)
    assert utilities[frozenset("bd")] == 30 and utilities[frozenset("bdfc")] == 40
    found = high_utility_itemsets(QUANTITIES, PROFITS, 30, absolute=True)
    assert dict(zip(found["itemsets"], found["utility"])) == pytest.approx({s: u for s, u in utilities.items() if u >= 30})
    assert found.attrs["total_utility"] == 94 and found.attrs["min_utility"] == 30 and found.attrs["n_transactions"] == 5
    assert found.iloc[0]["utility"] == 40 and found.iloc[0]["itemsets"] == frozenset("bcdf") and found.iloc[0]["support"] == pytest.approx(0.4)
    relative = high_utility_itemsets(QUANTITIES, PROFITS, 30 / 94)
    assert set(relative["itemsets"]) == set(found["itemsets"])


def random_quantities(seed, rows=40, items=ITEMS[:7]):
    rng = np.random.RandomState(seed)
    table = pd.DataFrame(rng.randint(0, 5, (rows, len(items))) * (rng.rand(rows, len(items)) < 0.45), columns=items)
    return table, {item: float(rng.randint(0, 9)) for item in items}


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("fraction", [0.05, 0.12])
def test_high_utility_itemsets_equal_the_brute_force_enumeration(seed, fraction):
    table, profits = random_quantities(seed)
    rows = [{item: float(v) for item, v in row.items() if v > 0} for row in table.to_dict("records")]
    utilities = brute_utilities(rows, profits)
    total = sum(row[i] * profits[i] for row in rows for i in row)
    found = high_utility_itemsets(table, profits, fraction)
    expected = {s: u for s, u in utilities.items() if u >= fraction * total - 1e-9}
    assert len(expected) > 0 and dict(zip(found["itemsets"], found["utility"])) == pytest.approx(expected)
    assert found["utility"].is_monotonic_decreasing and (found["length"] == found["itemsets"].map(len)).all()
    limited = high_utility_itemsets(table, profits, fraction, min_len=2, max_len=3)
    assert set(limited["itemsets"]) == {s for s in expected if 2 <= len(s) <= 3}


def test_utility_is_not_frequency():
    """The expensive pair is high-utility without being frequent; the cheap and frequent pair is not."""
    table = pd.DataFrame([{"gem": 1, "gold": 1}] * 3 + [{"gem": 0, "gold": 0, "pen": 2, "cap": 1}] * 40, columns=["gem", "gold", "pen", "cap"])
    profits = {"gem": 500, "gold": 400, "pen": 1, "cap": 1}
    found = high_utility_itemsets(table, profits, 0.5)       # total utility 2820: the pair is worth 2700 and the gem alone 1500
    assert set(found["itemsets"]) == {frozenset({"gem", "gold"}), frozenset({"gem"})}
    assert found.attrs["total_utility"] == 2820 and found.iloc[0]["utility"] == 2700
    assert dict(zip(found["itemsets"], found["support"]))[frozenset({"gem", "gold"})] == pytest.approx(3 / 43)


def test_high_utility_input_forms_and_errors():
    as_frame = pd.DataFrame(QUANTITIES)
    assert len(high_utility_itemsets(as_frame, PROFITS, 30, absolute=True)) == len(high_utility_itemsets(QUANTITIES, PROFITS, 30, absolute=True))
    assert len(high_utility_itemsets(as_frame, None, 0.3)) > 0                              # profit 1 for all
    empty = high_utility_itemsets(as_frame.iloc[:0], PROFITS, 0.5)                          # sin transacciones
    assert len(empty) == 0 and empty.attrs["total_utility"] == 0 and empty.attrs["n_transactions"] == 0
    assert len(high_utility_itemsets(as_frame, {item: 0 for item in PROFITS}, 0.5)) == 0   # without profit there is no utility
    assert len(high_utility_itemsets(as_frame, {item: 0 for item in PROFITS}, 1, absolute=True)) == 0
    for call in (lambda: high_utility_itemsets(as_frame, PROFITS, 0), lambda: high_utility_itemsets(as_frame, PROFITS, 2),
                 lambda: high_utility_itemsets(as_frame, PROFITS, 0, absolute=True),
                 lambda: high_utility_itemsets(as_frame, {"a": 1}, 0.5), lambda: high_utility_itemsets(as_frame, dict(PROFITS, a=-1), 0.5),
                 lambda: high_utility_itemsets(-as_frame.fillna(0), PROFITS, 0.5), lambda: high_utility_itemsets(as_frame, PROFITS, 0.5, max_len=0),
                 lambda: high_utility_itemsets(as_frame.astype(str), PROFITS, 0.5)):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("seed", range(4))
def test_high_utility_rules_equal_the_brute_force_splits(seed):
    table, profits = random_quantities(seed, rows=60)
    rows = [{item: float(v) for item, v in row.items() if v > 0} for row in table.to_dict("records")]
    utilities = brute_utilities(rows, profits)
    total = sum(row[i] * profits[i] for row in rows for i in row)
    counts = all_itemsets([list(row) for row in rows])
    found = high_utility_rules(table, profits, 0.12, min_conf=0.4)
    expected = {}
    for itemset, utility in utilities.items():
        if utility >= 0.12 * total - 1e-9 and len(itemset) >= 2:
            for antecedent in subsets(itemset):
                confidence = counts[itemset] / counts[frozenset(antecedent)]
                if confidence >= 0.4 - 1e-9:
                    expected[(frozenset(antecedent), itemset - frozenset(antecedent))] = (utility, confidence)
    assert len(expected) > 0 and set(zip(found["antecedents"], found["consequents"])) == set(expected)
    for antecedent, consequent, utility, confidence, share in zip(found["antecedents"], found["consequents"], found["utility"],
                                                                  found["confidence"], found["utility_share"]):
        assert (utility, confidence) == pytest.approx(expected[(antecedent, consequent)]) and share == pytest.approx(utility / total)
    assert found["utility"].is_monotonic_decreasing and found.attrs["total_utility"] == pytest.approx(total)
    assert len(significant_rules(found, 0.5, n_transactions=60)) <= len(found)
    assert len(high_utility_rules(table, profits, 0.12, min_conf=0.4, max_antecedent=1, max_consequent=1)) <= len(found)
    with pytest.raises(ValueError):
        high_utility_rules(table, profits, 0.12, min_conf=2)
    with pytest.raises(ValueError):
        high_utility_rules(table, profits, 0.12, max_antecedent=0)
