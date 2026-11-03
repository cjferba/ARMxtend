# -*- coding: utf-8 -*-
"""Tests of ARM.sequential_significance: permutation test of the order of the elements."""
import random

import pytest

from ARMxtend.ARM import sequential_rule_significance, sequential_rules, significant_sequential_rules


def database(count=120, seed=0):
    """a is always followed by b (planted); c and d are in many sequences but in random order."""
    rng = random.Random(seed)
    sequences = []
    for _ in range(count):
        events = ["a", "b"] if rng.random() < 0.6 else []
        noise = [x for x in "cd" if rng.random() < 0.7]
        rng.shuffle(noise)
        if events:
            position = rng.randint(0, len(noise))
            noise[position:position] = events
        else:
            noise += [x for x in "ab" if rng.random() < 0.2]
        sequences.append(noise or ["e"])
    return sequences


def test_planted_order_is_significant_and_random_order_is_not():
    sequences = database()
    rules = sequential_rules(sequences, min_supp=0.1, min_conf=0.3)
    table = sequential_rule_significance(sequences, rules, n_permutations=100, random_state=1)
    pairs = {(tuple(sorted(a)), tuple(sorted(c))): p for a, c, p in
             zip(table["antecedents"], table["consequents"], table["p_adjusted"])}
    assert pairs[(("a",), ("b",))] < 0.05
    assert pairs[(("c",), ("d",))] > 0.05
    assert (table["p_permutation"] >= 1 / 101).all() and (table["valid_sequences"] >= 0).all()
    kept = significant_sequential_rules(sequences, rules, alpha=0.05, n_permutations=100, random_state=1)
    assert ("a",) in {tuple(sorted(a)) for a in kept["antecedents"]}


def test_is_reproducible_and_does_not_modify_the_input():
    sequences = database()
    rules = sequential_rules(sequences, min_supp=0.1, min_conf=0.3)
    before = rules.copy()
    one = sequential_rule_significance(sequences, rules, n_permutations=20, random_state=3)
    two = sequential_rule_significance(sequences, rules, n_permutations=20, random_state=3)
    assert one["p_permutation"].tolist() == two["p_permutation"].tolist()
    assert rules.equals(before)


def test_validation_and_empty():
    sequences = database(30)
    rules = sequential_rules(sequences, min_supp=0.1, min_conf=0.3)
    with pytest.raises(ValueError, match="n_permutations"):
        sequential_rule_significance(sequences, rules, n_permutations=0)
    with pytest.raises(ValueError, match="alpha"):
        significant_sequential_rules(sequences, rules, alpha=1.5)
    with pytest.raises(ValueError, match="not in the sequences"):
        sequential_rule_significance([["x", "y"]], rules, n_permutations=2)
    assert len(sequential_rule_significance(sequences, rules.iloc[:0], n_permutations=2)) == 0
