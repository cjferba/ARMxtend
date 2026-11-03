# -*- coding: utf-8 -*-
"""
Tests of ``ARM.significance``: the p-values against an exact enumeration with integers, the
multiple-testing corrections against hand-computed values (and those of R), the control
of false discoveries on noise, the intervals and the permutation tests.
"""
import itertools
import math
from fractions import Fraction
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import (
    adjust_pvalues, association_rules, candidate_rule_count, confidence_intervals, crisp_rule_miner, permutation_null,
    permutation_p_values, rule_significance, shuffled_items, significant_miner, significant_rules,
    temporal_association_rules, window_rules,
)
from ARMxtend.ARM import significance
from ARMxtend.FFIM import fpgrowth
from ARMxtend.FIM._shared import key_to_itemset
from ARMxtend.preprocessing.timeseries import lag_items, split_lag


def rules_from_counts(tables, n):
    """Synthetic rules with the given frequencies (n_ac, n_a, n_c), over n transactions."""
    rules = pd.DataFrame({
        "antecedents": [frozenset({"a{}".format(i)}) for i in range(len(tables))],
        "consequents": [frozenset({"c{}".format(i)}) for i in range(len(tables))],
        "support": [ac / n for ac, _, _ in tables], "antecedent support": [a / n for _, a, _ in tables],
        "consequent support": [c / n for _, _, c in tables],
    })
    rules.attrs["n_transactions"] = n
    return rules


def exact_greater(n_ac, n_a, n_c, n):
    """P(X >= n_ac) of the hypergeometric, with integers and fractions (no logarithms or recurrences)."""
    total = math.comb(n, n_a)
    hit = sum(math.comb(n_c, x) * math.comb(n - n_c, n_a - x) for x in range(n_ac, min(n_a, n_c) + 1)
              if 0 <= n_a - x <= n - n_c)
    return float(Fraction(hit, total))


def exact_less(n_ac, n_a, n_c, n):
    total = math.comb(n, n_a)
    hit = sum(math.comb(n_c, x) * math.comb(n - n_c, n_a - x) for x in range(0, n_ac + 1)
              if 0 <= n_a - x <= n - n_c and x <= n_c)
    return float(Fraction(hit, total))


def valid_tables(n, count, seed):
    rng = np.random.RandomState(seed)
    tables = []
    while len(tables) < count:
        n_a, n_c = int(rng.randint(1, n)), int(rng.randint(1, n))
        n_ac = int(rng.randint(max(0, n_a + n_c - n), min(n_a, n_c) + 1))
        tables.append((n_ac, n_a, n_c))
    return tables


# ----------------------------------------------------------------- tests 2x2

def test_fisher_matches_the_known_tea_tasting_value():
    # Fisher's "lady tasting tea": 3 hits out of 4 with 4 cups of each type -> p = 17/70 (one tail)
    rules = rules_from_counts([(3, 4, 4)], 8)
    assert rule_significance(rules)["p_value"].iloc[0] == pytest.approx(17 / 70)
    assert rule_significance(rules, alternative="less")["p_value"].iloc[0] == pytest.approx(69 / 70)   # P(X <= 3) = 1 - P(X = 4)
    assert rule_significance(rules, alternative="two-sided")["p_value"].iloc[0] == pytest.approx(min(1.0, 2 * 17 / 70))


@pytest.mark.parametrize("n", [12, 40, 97])
def test_fisher_matches_exact_integer_enumeration(n):
    tables = valid_tables(n, 60, seed=n)
    rules = rules_from_counts(tables, n)
    greater = rule_significance(rules, n_transactions=n)["p_value"].to_numpy()
    less = rule_significance(rules, n_transactions=n, alternative="less")["p_value"].to_numpy()
    two = rule_significance(rules, n_transactions=n, alternative="two-sided")["p_value"].to_numpy()
    for (ac, a, c), p_greater, p_less, p_two in zip(tables, greater, less, two):
        assert p_greater == pytest.approx(exact_greater(ac, a, c, n), rel=1e-9, abs=1e-300)
        assert p_less == pytest.approx(exact_less(ac, a, c, n), rel=1e-9, abs=1e-300)
        assert p_two == pytest.approx(min(1.0, 2 * min(exact_greater(ac, a, c, n), exact_less(ac, a, c, n))), rel=1e-9)


def test_fisher_on_large_tables_and_with_small_chunks(monkeypatch):
    """The cut of negligible terms and the work in blocks do not change the exact result."""
    n = 6000
    tables = [(300, 900, 1800), (101, 500, 1200), (480, 900, 1800), (5, 20, 3000), (900, 900, 1800)]
    rules = rules_from_counts(tables, n)
    expected = [exact_greater(*table, n) for table in tables]
    for chunk in (4096, 50, 7):
        monkeypatch.setattr(significance, "_CHUNK", chunk)
        got = rule_significance(rules, n_transactions=n)["p_value"].to_numpy()
        assert got == pytest.approx(expected, rel=1e-8, abs=1e-300)
    assert expected[0] < 0.05 and 0.3 < expected[1] <= 1.0           # there is one significant case and another that is not


def test_chi2_matches_an_independent_formula():
    n = 200
    tables = [(40, 80, 90), (30, 80, 90), (60, 100, 100), (10, 20, 150)]
    got = rule_significance(rules_from_counts(tables, n), method="chi2")["p_value"].to_numpy()
    two = rule_significance(rules_from_counts(tables, n), method="chi2", alternative="two-sided")["p_value"].to_numpy()
    for (ac, a, c), p_one, p_two in zip(tables, got, two):
        cells = np.array([[ac, a - ac], [c - ac, n - a - c + ac]], dtype=float)
        expected = np.outer(cells.sum(axis=1), cells.sum(axis=0)) / n
        statistic = ((cells - expected) ** 2 / expected).sum()
        reference = 2 * (1 - NormalDist().cdf(math.sqrt(statistic)))
        assert p_two == pytest.approx(reference, rel=1e-9)
        positive = ac > a * c / n
        assert p_one == pytest.approx(reference / 2 if positive else 1 - reference / 2, rel=1e-9)


def test_chi2_and_fisher_agree_on_large_counts_and_degenerate_tables_give_one():
    n = 5000
    rules = rules_from_counts([(700, 1000, 3000), (560, 1000, 3000)], n)
    fisher = rule_significance(rules, method="fisher")["p_value"].to_numpy()
    chi2 = rule_significance(rules, method="chi2")["p_value"].to_numpy()
    assert np.log10(chi2[0]) == pytest.approx(np.log10(fisher[0]), abs=0.6)
    assert chi2[1] == pytest.approx(fisher[1], abs=0.02)
    constant = rules_from_counts([(50, 50, 100)], 100)               # the consequent is in all the transactions
    assert rule_significance(constant, method="chi2")["p_value"].iloc[0] == 1.0
    assert rule_significance(constant, method="fisher")["p_value"].iloc[0] == pytest.approx(1.0)


def test_rule_significance_counts_from_real_rules_and_validation():
    transactions = [{"a", "b"}] * 30 + [{"a"}] * 10 + [{"b"}] * 10 + [{"c"}] * 50
    frequent = fpgrowth(transactions, 0.1)
    itemsets = pd.DataFrame({"support": list(frequent.values()), "itemsets": [key_to_itemset(k) for k in frequent]})
    rules = association_rules(itemsets, metric="confidence", min_threshold=0.5)
    scored = rule_significance(rules, n_transactions=100)
    row = scored[(scored["antecedents"] == frozenset({"a"})) & (scored["consequents"] == frozenset({"b"}))].iloc[0]
    assert row["p_value"] == pytest.approx(exact_greater(30, 40, 40, 100))
    with pytest.raises(ValueError, match="n_transactions"):
        rule_significance(rules)
    with pytest.raises(ValueError, match="multiples"):
        rule_significance(rules, n_transactions=77)
    with pytest.raises(ValueError, match="method"):
        rule_significance(rules, 100, method="magic")
    with pytest.raises(ValueError, match="alternative"):
        rule_significance(rules, 100, alternative="up")
    with pytest.raises(ValueError, match="crisp"):
        rule_significance(rules.drop(columns="antecedent support"), 100)
    assert "p_value" not in rules.columns                              # does not modify the input


# ------------------------------------------------------------ correcciones

P = [0.01, 0.04, 0.03, 0.005]


def test_adjust_pvalues_hand_computed_and_r_values():
    assert adjust_pvalues(P, "bonferroni") == pytest.approx([0.04, 0.16, 0.12, 0.02])
    assert adjust_pvalues(P, "holm") == pytest.approx([0.03, 0.06, 0.06, 0.02])           # p.adjust(.., "holm") de R
    assert adjust_pvalues(P, "fdr_bh") == pytest.approx([0.02, 0.04, 0.04, 0.02])         # p.adjust(.., "BH") de R
    harmonic = 1 + 1 / 2 + 1 / 3 + 1 / 4
    assert adjust_pvalues(P, "fdr_by") == pytest.approx(np.array([0.02, 0.04, 0.04, 0.02]) * harmonic)
    assert adjust_pvalues([0.5, 0.9], "bonferroni") == pytest.approx([1.0, 1.0])            # nunca pasa de 1


@pytest.mark.parametrize("method", ["bonferroni", "holm", "fdr_bh", "fdr_by"])
def test_adjust_pvalues_properties(method):
    rng = np.random.RandomState(0)
    p = rng.rand(50) ** 3
    adjusted = adjust_pvalues(p, method)
    assert np.all(adjusted >= p - 1e-15) and np.all(adjusted <= 1.0)
    order = np.argsort(p)
    assert np.all(np.diff(adjusted[order]) >= -1e-15)                  # keeps the order of the p-values
    bigger = adjust_pvalues(p, method, n_tests=500)
    assert np.all(bigger >= adjusted - 1e-15) and bigger.min() > adjusted.min()
    assert adjust_pvalues(p, method, n_tests=len(p)) == pytest.approx(adjusted)
    assert len(adjust_pvalues([], method)) == 0


def test_adjust_pvalues_ordering_of_methods_and_validation():
    p = np.random.RandomState(1).rand(30) ** 4
    bonferroni, holm, bh = (adjust_pvalues(p, m) for m in ("bonferroni", "holm", "fdr_bh"))
    assert np.all(holm <= bonferroni + 1e-15) and np.all(bh <= holm + 1e-15)               # BH <= Holm <= Bonferroni
    with pytest.raises(ValueError, match="method"):
        adjust_pvalues(P, "sidak")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        adjust_pvalues([0.5, 1.5])
    with pytest.raises(ValueError, match="n_tests"):
        adjust_pvalues(P, n_tests=2)


# ------------------------------------------------- mined rules and families

def mine(table, min_supp, min_conf):
    transactions = [set(table.columns[row]) for row in table.to_numpy()]
    frequent = fpgrowth(transactions, min_supp)
    itemsets = pd.DataFrame({"support": list(frequent.values()), "itemsets": [key_to_itemset(k) for k in frequent]})
    return association_rules(itemsets, metric="confidence", min_threshold=min_conf), itemsets


def noise(seed, rows=300, items=12, p=0.3):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(rng.rand(rows, items) < p, columns=["i{:02d}".format(i) for i in range(items)])


def test_association_rules_annotates_the_number_of_candidates():
    table = noise(0)
    rules, itemsets = mine(table, 0.03, 0.45)
    assert rules.attrs["n_candidates"] == candidate_rule_count(itemsets) > len(rules)
    brute = sum(len(items) * 0 + (2 ** len(items) - 2) for items in itemsets["itemsets"] if len(items) >= 2)
    assert candidate_rule_count(itemsets) == brute
    nothing, its = mine(table, 0.03, 2.0)
    assert nothing.empty and nothing.attrs["n_candidates"] == candidate_rule_count(its)


def test_noise_produces_many_naive_rules_but_almost_no_significant_ones():
    naive, runs_with_discoveries, discoveries = 0, 0, 0
    for seed in range(20):
        rules, _ = mine(noise(seed), 0.03, 0.45)
        found = significant_rules(rules, 0.05, "fdr_bh", n_transactions=300)
        naive += len(rules)
        discoveries += len(found)
        runs_with_discoveries += int(len(found) > 0)
    assert naive > 300                                                    # the confidence threshold only lets noise through
    assert runs_with_discoveries <= 4 and discoveries <= 25               # with alpha = 0.05, ~1 run out of 20 is expected


def test_planted_association_survives_the_correction():
    rng = np.random.RandomState(1)
    table = noise(1)
    table["i01"] = table["i01"] | (table["i00"].to_numpy() & (rng.rand(300) < 0.6))
    rules, itemsets = mine(table, 0.03, 0.45)
    found = significant_rules(rules, 0.05, "fdr_bh", n_transactions=300)
    pairs = {(tuple(sorted(a)), tuple(sorted(c))) for a, c in zip(found["antecedents"], found["consequents"])}
    assert (("i00",), ("i01",)) in pairs and (("i01",), ("i00",)) in pairs
    top = found.iloc[0]
    assert top["p_adjusted"] < 1e-6 and found["p_adjusted"].is_monotonic_increasing
    assert (found["p_adjusted"] <= 0.05).all() and len(found) < len(rules)
    # the p-value of a rule is that of the exact test on the frequencies counted in the data
    row = found[(found["antecedents"] == frozenset({"i00"})) & (found["consequents"] == frozenset({"i01"}))].iloc[0]
    n_a, n_c = int(table["i00"].sum()), int(table["i01"].sum())
    n_ac = int((table["i00"] & table["i01"]).sum())
    assert row["p_value"] == pytest.approx(exact_greater(n_ac, n_a, n_c, 300), rel=1e-9)
    assert found.attrs["n_tests"] == rules.attrs["n_candidates"]
    stricter = significant_rules(rules, 0.05, "bonferroni", n_transactions=300)
    assert len(stricter) <= len(found)
    explicit = significant_rules(rules, 0.05, "fdr_bh", n_transactions=300, n_tests=10 * rules.attrs["n_candidates"])
    assert len(explicit) <= len(found)
    with pytest.raises(ValueError, match="alpha"):
        significant_rules(rules, 1.5, n_transactions=300)


def test_temporal_rules_carry_their_transactions_and_candidates():
    rng = np.random.RandomState(0)
    rows = 400
    door = rng.rand(rows) < 0.3
    heater = np.zeros(rows, dtype=bool)
    heater[1:] = door[:-1] & (rng.rand(rows - 1) < 0.9)
    items = pd.DataFrame({"door": door, "heater": heater, "noise": rng.rand(rows) < 0.5})
    rules = temporal_association_rules(items, max_lag=2, min_supp=0.03, min_threshold=0.5, targets=["heater"],
                                       exclude_persistence=True)
    assert rules.attrs["n_transactions"] == rows - 2
    # the family is the valid partitions of the frequent itemsets, counted here with another implementation
    lagged = lag_items(items, [0, 1, 2])
    frequent = fpgrowth([set(lagged.columns[row]) for row in lagged.to_numpy()], 0.03)
    brute = 0
    for key in frequent:
        itemset = key_to_itemset(key)
        for size in range(1, len(itemset)):
            for consequent in itertools.combinations(sorted(itemset), size):
                antecedent = itemset - set(consequent)
                parsed = [split_lag(item) for item in consequent]
                if all(lag == 0 and base == "heater" for base, lag in parsed) and any(split_lag(i)[1] > 0 for i in antecedent) \
                        and not any(split_lag(i)[0] == "heater" and split_lag(i)[1] > 0 for i in antecedent):
                    brute += 1
    assert rules.attrs["n_candidates"] == brute > len(rules) > 0
    scored = rule_significance(rules)                                   # n_transactions sale de attrs
    row = scored[(scored["antecedents"] == frozenset({"door@lag1"})) & (scored["consequents"] == frozenset({"heater"}))].iloc[0]
    n_ac = int((lagged["door@lag1"] & lagged["heater"]).sum())
    assert row["p_value"] == pytest.approx(exact_greater(n_ac, int(lagged["door@lag1"].sum()), int(lagged["heater"].sum()),
                                                         len(lagged)), rel=1e-9)
    kept = significant_rules(rules)
    assert any(a == frozenset({"door@lag1"}) for a in kept["antecedents"])


# --------------------------------------------------------------- intervalos

def test_wilson_intervals_match_a_hand_computed_value():
    rules = rules_from_counts([(8, 10, 12)], 20)                     # confianza 8/10, soporte 8/20
    out = confidence_intervals(rules, level=0.95, method="wilson", n_transactions=20)
    assert out["confidence_low"].iloc[0] == pytest.approx(0.4902, abs=2e-3)
    assert out["confidence_high"].iloc[0] == pytest.approx(0.9433, abs=2e-3)
    assert np.isnan(out["lift_low"].iloc[0])
    wide = confidence_intervals(rules, level=0.99, n_transactions=20)
    assert wide["confidence_low"].iloc[0] < out["confidence_low"].iloc[0] and wide["confidence_high"].iloc[0] > out["confidence_high"].iloc[0]


def test_bootstrap_intervals_contain_the_estimate_and_follow_the_data():
    rules = rules_from_counts([(300, 500, 600), (60, 400, 100)], 2000)
    rules["confidence"] = rules["support"] / rules["antecedent support"]
    rules["lift"] = rules["confidence"] / rules["consequent support"]
    out = confidence_intervals(rules, method="bootstrap", n_resamples=2000, random_state=3, n_transactions=2000)
    for _, row in out.iterrows():
        assert row["support_low"] <= row["support"] <= row["support_high"]
        assert row["confidence_low"] <= row["confidence"] <= row["confidence_high"]
        assert row["lift_low"] <= row["lift"] <= row["lift_high"]
    wilson = confidence_intervals(rules, method="wilson", n_transactions=2000)
    assert out["confidence_low"].to_numpy() == pytest.approx(wilson["confidence_low"].to_numpy(), abs=0.02)
    assert out["support_high"].to_numpy() == pytest.approx(wilson["support_high"].to_numpy(), abs=0.01)
    again = confidence_intervals(rules, method="bootstrap", n_resamples=2000, random_state=3, n_transactions=2000)
    pd.testing.assert_frame_equal(out, again)


def test_multinomial_bootstrap_equals_resampling_rows():
    """For one rule it is enough to resample its four cells: it matches resampling the rows for real."""
    rng = np.random.RandomState(5)
    n = 400
    antecedent = rng.rand(n) < 0.4
    consequent = (rng.rand(n) < 0.3) | (antecedent & (rng.rand(n) < 0.4))
    n_ac, n_a, n_c = int((antecedent & consequent).sum()), int(antecedent.sum()), int(consequent.sum())
    ours = confidence_intervals(rules_from_counts([(n_ac, n_a, n_c)], n), method="bootstrap", n_resamples=4000,
                                random_state=0, n_transactions=n).iloc[0]
    confidences, lifts = [], []
    for _ in range(4000):
        index = rng.randint(0, n, n)
        a, c = antecedent[index], consequent[index]
        if a.sum() and c.sum():
            confidences.append((a & c).sum() / a.sum())
            lifts.append((a & c).sum() / a.sum() / c.mean())
    low, high = np.quantile(confidences, [0.025, 0.975])
    assert ours["confidence_low"] == pytest.approx(low, abs=0.015) and ours["confidence_high"] == pytest.approx(high, abs=0.015)
    low, high = np.quantile(lifts, [0.025, 0.975])
    assert ours["lift_low"] == pytest.approx(low, abs=0.05) and ours["lift_high"] == pytest.approx(high, abs=0.05)


def test_interval_validation():
    rules = rules_from_counts([(8, 10, 12)], 20)
    with pytest.raises(ValueError, match="level"):
        confidence_intervals(rules, level=1.5, n_transactions=20)
    with pytest.raises(ValueError, match="method"):
        confidence_intervals(rules, method="magic", n_transactions=20)


# -------------------------------------------------------------- permutaciones

def test_shuffled_items_keep_each_column_but_break_the_alignment():
    rng = np.random.RandomState(0)
    index = pd.date_range("2024-01-01", periods=200, freq="h")
    items = pd.DataFrame({"a": rng.rand(200) < 0.3, "b": rng.rand(200) < 0.5, "c": np.arange(200) % 7 == 0}, index=index)
    for null in ("permute", "shift"):
        shuffled = shuffled_items(items, null, random_state=1)
        assert (shuffled.index == items.index).all() and list(shuffled.columns) == list(items.columns)
        assert (shuffled.sum() == items.sum()).all()                    # same frequency of each item
        assert not shuffled.equals(items)
    shifted = shuffled_items(items, "shift", random_state=1)
    for name in items.columns:                                          # each column is a rotation of the original
        assert any(np.array_equal(np.roll(items[name].to_numpy(), k), shifted[name].to_numpy()) for k in range(1, 200))
    same = lambda x: int((x[1:] == x[:-1]).sum())                       # noqa: E731 -- repeticiones consecutivas
    assert [same(shifted[n].to_numpy()) for n in items] in ([same(items[n].to_numpy()) + d for n, d in zip(items, ds)]
                                                            for ds in itertools.product([-1, 0, 1], repeat=3))
    permuted = shuffled_items(items, "permute", random_state=1)
    assert same(permuted["c"].to_numpy()) != same(items["c"].to_numpy())  # permuting does destroy the autocorrelation
    pd.testing.assert_frame_equal(shuffled_items(items, "shift", random_state=7), shuffled_items(items, "shift", random_state=7))
    with pytest.raises(ValueError, match="null"):
        shuffled_items(items, "zigzag")
    floats = pd.DataFrame(rng.rand(30, 2), columns=["x", "y"])
    assert shuffled_items(floats, "permute", 0).to_numpy().dtype == float


def test_permutation_p_value_arithmetic_with_a_deterministic_miner():
    """p = (1 + #{nulls at least as good}) / (1 + B), checked by recounting the null distribution by hand."""
    rng = np.random.RandomState(2)
    items = pd.DataFrame({"a": rng.rand(300) < 0.4, "b": rng.rand(300) < 0.4})
    items["b"] = items["b"] | (items["a"] & (rng.rand(300) < 0.5))

    def miner(table):
        joint = float((table["a"] & table["b"]).mean())
        lift = joint / (table["a"].mean() * table["b"].mean())
        return pd.DataFrame({"antecedents": [frozenset({"a"})], "consequents": [frozenset({"b"})], "lift": [lift]})

    null = permutation_null(items, miner, measure="lift", n_permutations=60, null="permute", random_state=4)
    assert list(null.columns) == ["permutation", "n_rules", "max_lift"] and len(null) == 60 and (null["n_rules"] == 1).all()
    real = miner(items)
    out = permutation_p_values(real, null)
    expected = (1 + int((null["max_lift"] >= real["lift"].iloc[0] - 1e-12).sum())) / 61
    assert out["p_permutation"].iloc[0] == pytest.approx(expected) and expected == pytest.approx(1 / 61)
    independent = pd.DataFrame({"a": rng.rand(300) < 0.4, "b": rng.rand(300) < 0.4})
    weak = permutation_p_values(miner(independent), null)
    assert weak["p_permutation"].iloc[0] > 0.05
    assert "p_permutation" not in real.columns
    with pytest.raises(ValueError, match="measure"):
        permutation_null(items, miner, measure="nope", n_permutations=2)
    with pytest.raises(ValueError, match="n_permutations"):
        permutation_null(items, miner, n_permutations=0)


def test_minp_permutation_separates_planted_from_noise_rules():
    rng = np.random.RandomState(3)
    items = pd.DataFrame(rng.rand(500, 7) < 0.3, columns=list("abcdefg"))
    items["b"] = items["b"] | (items["a"].to_numpy() & (rng.rand(500) < 0.5))
    miner = crisp_rule_miner(0.05, "confidence", 0.3)
    rules = miner(items)
    null = permutation_null(items, miner, n_permutations=40, null="permute", random_state=0)
    out = permutation_p_values(rules, null, n_transactions=500)
    assert null.attrs["measure"] == "p_value" and "min_p_value" in null.columns
    planted = out[(out["antecedents"] == frozenset({"a"})) & (out["consequents"] == frozenset({"b"}))].iloc[0]
    assert planted["p_permutation"] == pytest.approx(1 / 41)
    noise_rules = out[[not ({"a", "b"} & (set(x) | set(y))) for x, y in zip(out["antecedents"], out["consequents"])]]
    assert len(noise_rules) > 0 and noise_rules["p_permutation"].min() > 0.05
    assert (out["p_permutation"] >= planted["p_permutation"]).all()


def test_shift_null_exposes_false_alarms_that_the_iid_test_gives_on_autocorrelated_series():
    """Two independent series with long regimes look associated to Fisher; the circular shift refutes it."""
    def regimes(rng, rows, length):
        blocks = rng.rand(rows // length + 1) < 0.4
        return np.repeat(blocks, length)[:rows]

    miner = crisp_rule_miner(0.05, "confidence", 0.0)
    fisher_alarms, shift_alarms = 0, 0
    for seed in range(12):
        rng = np.random.RandomState(100 + seed)
        items = pd.DataFrame({"a": regimes(rng, 2000, 150), "b": regimes(rng, 2000, 150)})
        rules = miner(items)
        rules = rules[(rules["antecedents"] == frozenset({"a"})) & (rules["consequents"] == frozenset({"b"}))]
        if rules.empty:
            continue
        fisher_alarms += int(rule_significance(rules, 2000)["p_value"].iloc[0] < 0.01)
        null = permutation_null(items, miner, n_permutations=60, null="shift", random_state=seed)
        shift_alarms += int(permutation_p_values(rules, null, n_transactions=2000)["p_permutation"].iloc[0] < 0.05)
    assert fisher_alarms >= 6          # the test that assumes independence often gives false alarms
    assert shift_alarms <= 3           # the circular permutation, close to the nominal 5 %


# ------------------------------------------------------ minero y ventanas

def test_significant_miner_in_windows_drops_the_trivial_rules():
    def rows(window):
        out = []
        for position in range(20):
            row = {"D", "E"}
            if position % 5 in (0, 1):
                row |= {"A", "B"}
            if position % 5 in (2, 3):
                row |= {"C"}
            out.append(row)
        return out

    table = pd.DataFrame([{item: True for item in row} for window in range(4) for row in rows(window)]).fillna(False).astype(bool)
    table.index = pd.date_range("2024-01-01", periods=len(table), freq="h")
    plain = window_rules(table, 20, min_supp=0.3, min_threshold=0.9)
    filtered = window_rules(table, 20, miner=significant_miner(crisp_rule_miner(0.3, "confidence", 0.9), alpha=0.05))
    keys = lambda frame: {(tuple(sorted(a)), tuple(sorted(c))) for a, c in zip(frame["antecedents"], frame["consequents"])}   # noqa: E731
    assert (("D",), ("E",)) in keys(plain) and (("D",), ("E",)) not in keys(filtered)      # E is always present: nothing to test
    assert (("A",), ("B",)) in keys(filtered) and "p_adjusted" in filtered.columns
    assert filtered["p_adjusted"].max() <= 0.05
    empty = significant_miner(lambda t: pd.DataFrame(columns=["antecedents", "consequents"]))(table)
    assert empty.empty
