# -*- coding: utf-8 -*-
"""
Example: statistical significance of rules -- how many of the rules that
pass the thresholds are just noise, how to keep those that are not
(Fisher's exact test + multiple-testing correction, and permutation
tests) and why with time series one has to shuffle with
circular shifts.
"""
import numpy as np
import pandas as pd

from ARMxtend.ARM import (
    association_rules, confidence_intervals, crisp_rule_miner, permutation_null, permutation_p_values, rule_significance,
    significant_rules,
)
from ARMxtend.FFIM import fpgrowth
from ARMxtend.FIM._shared import key_to_itemset


def basket_data(rows=300, items=12, seed=1):
    """Random, independent baskets (each item with probability 0.3) with a single real association: i00 -> i01."""
    rng = np.random.RandomState(seed)
    table = pd.DataFrame(rng.rand(rows, items) < 0.3, columns=["i{:02d}".format(i) for i in range(items)])
    table["i01"] = table["i01"] | (table["i00"].to_numpy() & (rng.rand(rows) < 0.6))
    return table


def mine(table, min_supp=0.03, min_conf=0.45):
    frequent = fpgrowth([set(table.columns[row]) for row in table.to_numpy()], min_supp)
    itemsets = pd.DataFrame({"support": list(frequent.values()), "itemsets": [key_to_itemset(k) for k in frequent]})
    return association_rules(itemsets, metric="confidence", min_threshold=min_conf)


def regimes(rng, rows, length):
    """Boolean series with long stretches of constant value (highly autocorrelated)."""
    return np.repeat(rng.rand(rows // length + 1) < 0.4, length)[:rows]


def _spurious(rules):
    """Rules that do not relate i00 with i01: the only real thing in the data."""
    return sum(not {"i00", "i01"} <= rule["antecedents"] | rule["consequents"] for _, rule in rules.iterrows())


def main():
    # 1. Independent baskets with one real association: the confidence threshold lets a lot of noise through -----
    table = basket_data()
    rules = mine(table)
    print("Baskets:", len(table), "| rules with confidence >= 0.45:", len(rules), "| without i00 and i01 together (noise):", _spurious(rules))

    fisher = significant_rules(rules, alpha=0.05, correction="fdr_bh", n_transactions=len(table))
    print("With Fisher's test and FDR (alpha=0.05): {} rules, {} of them noise".format(len(fisher), _spurious(fisher)))

    miner = crisp_rule_miner(0.03, "confidence", 0.45)
    null = permutation_null(table, miner, n_permutations=50, null="permute", random_state=0)
    permuted = permutation_p_values(rules, null, n_transactions=len(table))
    best = permuted[permuted["p_permutation"] <= 0.05]
    print("With permutation (50 shuffled copies): {} rules, {} of them noise; in the shuffled data {:.0f} rules come out on average".format(
        len(best), _spurious(best), null["n_rules"].mean()))

    planted = confidence_intervals(rule_significance(rules, len(table)), n_transactions=len(table))
    planted = planted[(planted["antecedents"] == {"i00"}) & (planted["consequents"] == {"i01"})].iloc[0]
    print("\ni00 -> i01: support={:.3f} confidence={:.2f} [{:.2f}, {:.2f}] lift={:.2f} p={:.1e}".format(
        planted["support"], planted["confidence"], planted["confidence_low"], planted["confidence_high"],
        planted["lift"], planted["p_value"]))

    # 2. Time series: two INDEPENDENT series with long regimes look associated -------------
    fisher_alarms, shift_alarms, pairs = 0, 0, 0
    miner = crisp_rule_miner(0.05, "confidence", 0.0)
    for seed in range(12):
        rng = np.random.RandomState(100 + seed)
        series = pd.DataFrame({"a": regimes(rng, 2000, 150), "b": regimes(rng, 2000, 150)})
        found = miner(series)
        found = found[(found["antecedents"] == {"a"}) & (found["consequents"] == {"b"})]
        if found.empty:
            continue
        pairs += 1
        fisher_alarms += int(rule_significance(found, len(series))["p_value"].iloc[0] < 0.01)
        shifted = permutation_null(series, miner, n_permutations=60, null="shift", random_state=seed)
        shift_alarms += int(permutation_p_values(found, shifted, n_transactions=len(series))["p_permutation"].iloc[0] < 0.05)
    print("\n{} pairs of independent series: Fisher (assumes independent rows) gives {} false alarms (p < 0.01); "
          "the circular permutation test, {} (p < 0.05)".format(pairs, fisher_alarms, shift_alarms))

    return rules, fisher, permuted, (pairs, fisher_alarms, shift_alarms)


if __name__ == "__main__":
    main()
