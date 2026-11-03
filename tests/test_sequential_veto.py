# -*- coding: utf-8 -*-
"""Tests of ARM.sequential_veto_rules: rules "once X has occurred, Y does not occur afterwards"."""
import itertools
import random

import pytest

from ARMxtend.ARM import contains_veto_rule, sequential_veto_rules

ITEMS = "abcd"


def random_database(count=60, seed=0):
    rng = random.Random(seed)
    return [[rng.sample(ITEMS, rng.randint(1, 2)) for _ in range(rng.randint(2, 6))] for _ in range(count)]


def brute(database, min_supp, min_conf, max_antecedent, min_antecedent_supp):
    total = len(database)
    found = {}
    for size in range(1, max_antecedent + 1):
        for x in itertools.combinations(ITEMS, size):
            with_x = [s for s in database if set(x) <= {i for e in s for i in e}]
            if len(with_x) / total < min_antecedent_supp - 1e-9:
                continue
            for y in ITEMS:
                if y in x:
                    continue
                valid = sum(contains_veto_rule(s, x, y) for s in database)
                if valid / total >= min_supp - 1e-9 and valid / len(with_x) >= min_conf - 1e-9:
                    found[(frozenset(x), y)] = (valid / total, valid / len(with_x))
    return found


@pytest.mark.parametrize("seed", range(4))
def test_equals_the_definition_on_random_databases(seed):
    database = random_database(seed=seed)
    rules = sequential_veto_rules(database, 0.1, 0.5, 2, 0.15)
    expected = brute(database, 0.1, 0.5, 2, 0.15)
    assert len(expected) > 0
    got = {(a, next(iter(c))): (s, f) for a, c, s, f in
           zip(rules["antecedents"], rules["consequents"], rules["support"], rules["confidence"])}
    assert set(got) == set(expected)
    for key, value in expected.items():
        assert got[key] == pytest.approx(value)


def test_hand_computed_example():
    # after "arm", "open" never happens in 3 of 4 sequences that contain arm
    database = [["arm", "sleep", "wake"], ["arm", "sleep"], ["arm", "open"], ["open", "arm"], ["wake"]]
    rules = sequential_veto_rules(database, 0.2, 0.7, 1, 0.2)
    row = rules[(rules["antecedents"] == frozenset({"arm"})) & (rules["consequents"] == frozenset({"open"}))].iloc[0]
    assert row["support"] == pytest.approx(3 / 5) and row["confidence"] == pytest.approx(3 / 4)
    assert row["consequent support"] == pytest.approx(3 / 5)       # sequences without "open"
    assert row["lift"] == pytest.approx((3 / 4) / (3 / 5))
    assert contains_veto_rule(["a", "b"], ["a"], "c") and not contains_veto_rule(["a", "c"], ["a"], "c")
    assert contains_veto_rule([["a", "c"]], ["a"], "c")            # simultaneous: not "afterwards"


def test_validation_and_empty():
    with pytest.raises(ValueError, match="min_supp"):
        sequential_veto_rules([["a"]], 0)
    with pytest.raises(ValueError, match="min_conf"):
        sequential_veto_rules([["a"]], 0.1, min_conf=2)
    with pytest.raises(ValueError, match="max_antecedent"):
        sequential_veto_rules([["a"]], 0.1, max_antecedent=0)
    with pytest.raises(ValueError, match="metric"):
        sequential_veto_rules([["a"]], 0.1, metric="nope")
    assert len(sequential_veto_rules([], 0.1)) == 0
