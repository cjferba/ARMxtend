# -*- coding: utf-8 -*-
"""
Example: three ways of looking beyond the minimum support. A shop has 2000 baskets with
quantities. (1) Without knowing which support threshold to set, the ``k`` rules with the highest support.
(2) Pairs that are frequent separately and rare together -- diapers and wipes, wine and
cheese -- which a global threshold either does not see or drowns in noise. (3) What is worth the MOST is not what
sells the most: wine and cheese give more profit than bread and milk, even if they are bought less.
"""
import numpy as np
import pandas as pd

from ARMxtend.ARM import association_rules, high_utility_rules, rare_rules, top_k_rules
from ARMxtend.FFIM import fpgrowth
from ARMxtend.FIM import high_utility_itemsets, minimal_rare_itemsets, top_k_itemsets
from ARMxtend.FIM._shared import key_to_itemset

PROFITS = {"bread": 0.3, "milk": 0.4, "eggs": 0.6, "fruit": 0.5, "diapers": 1.5, "wipes": 1.0, "wine": 9.0, "cheese": 5.0}


def shop(rows=2000, seed=0):
    """
    Quantities bought in ``rows`` baskets:

    * bread (50 %), milk (45 %), eggs (35 %) and fruit (40 %), at random and each on its own;
    * 9 % are families with a baby and buy diapers and wipes almost always (95 % each); the rest buy them
      at random with 6 %: each reaches 14 %, but together only 8 %;
    * wine in 12 % of the baskets and cheese in 30 %, and 70 % of those who buy wine buy cheese.
    """
    rng = np.random.RandomState(seed)
    baby = rng.rand(rows) < 0.09
    wine = rng.rand(rows) < 0.12
    present = {
        "bread": rng.rand(rows) < 0.50, "milk": rng.rand(rows) < 0.45, "eggs": rng.rand(rows) < 0.35,
        "fruit": rng.rand(rows) < 0.40,
        "diapers": np.where(baby, rng.rand(rows) < 0.95, rng.rand(rows) < 0.06),
        "wipes": np.where(baby, rng.rand(rows) < 0.95, rng.rand(rows) < 0.06),
        "wine": wine, "cheese": np.where(wine, rng.rand(rows) < 0.70, rng.rand(rows) < 0.2455),     # 0.12 * 0.70 + 0.88 * 0.2455 = 0.30
    }
    top = {"bread": 3, "milk": 2, "eggs": 2, "fruit": 5, "diapers": 3, "wipes": 1, "wine": 2, "cheese": 2}
    return pd.DataFrame({item: np.where(present[item], rng.randint(1, top[item] + 1, rows), 0) for item in present})


def baskets(quantities):
    return [[item for item in quantities.columns if row[item] > 0] for _, row in quantities.iterrows()]


def flat_rules(transactions, min_supp, min_conf=0.5):
    """The usual rules: frequent itemsets with FP-Growth and rules with `association_rules`."""
    frequent = fpgrowth(transactions, min_supp)
    itemsets = pd.DataFrame({"support": list(frequent.values()), "itemsets": [key_to_itemset(key) for key in frequent]})
    return association_rules(itemsets, min_threshold=min_conf) if len(itemsets) > 1 else pd.DataFrame(columns=["lift"])


def describe(rules, extra=()):
    return ["{} => {}  support={:.3f} confidence={:.2f} lift={:.2f}".format(
        ", ".join(sorted(rule["antecedents"])), ", ".join(sorted(rule["consequents"])), rule["support"], rule["confidence"],
        rule["lift"]) + "".join("  {}={:.2f}".format(name, rule[name]) for name in extra) for _, rule in rules.iterrows()]


def main():
    quantities = shop()
    transactions = baskets(quantities)
    print("{} baskets, {} products".format(len(transactions), quantities.shape[1]))

    # 1. Top-k: the k rules with the highest support, without a threshold --------------------------------------------------------
    top = top_k_rules(transactions, 5, min_conf=0.5)
    print("\nThe 5 rules with the highest support with confidence >= 0.5 (without fixing a minimum support):")
    for line in describe(top):
        print("  " + line)
    informative = top_k_rules(transactions, 5, min_conf=0.5, min_lift=1.5)
    print("The 5 with the highest support that also have lift >= 1.5 (the popular stops covering up what associates):")
    for line in describe(informative):
        print("  " + line)
    for min_supp in (0.3, 0.01):
        print("With a threshold of {}: {} rules".format(min_supp, len(flat_rules(transactions, min_supp))))
    pairs = top_k_itemsets(transactions, 5, min_len=2)

    # 2. Raras: frecuentes por separado, raras juntas --------------------------------------------------------------
    mri = minimal_rare_itemsets(transactions, 0.1, min_len=2)
    rare = rare_rules(transactions, 0.1, min_conf=0.5, min_lift=2)
    flat = flat_rules(transactions, 0.01)
    print("\nMinimal rare itemsets (support < 0.1, all their subsets with support >= 0.1): {}".format(len(mri)))
    print("Rare rules with confidence >= 0.5 and lift >= 2: {}. With a threshold of 0.01, which also reaches them, "
          "{} rules come out with that confidence, {} with that lift:".format(len(rare), len(flat), int((flat["lift"] >= 2).sum())))
    for line in describe(rare.sort_values("lift", ascending=False)):
        print("  " + line)

    # 3. High utility: what is worth the most is not what sells the most -------------------------------------------------------
    by_support = top_k_itemsets(transactions, 3)
    useful = high_utility_itemsets(quantities, PROFITS, 0.1)
    print("\nMost frequent itemsets: " + "; ".join("{{{}}} {:.2f}".format(", ".join(sorted(s)), v)
                                                    for v, s in zip(by_support["support"], by_support["itemsets"])))
    print("High-utility itemsets (>= 10 % of the total utility, which is {:.0f}):".format(useful.attrs["total_utility"]))
    for _, row in useful.head(5).iterrows():
        print("  {{{}}}  utility={:.0f}  support={:.3f}".format(", ".join(sorted(row["itemsets"])), row["utility"], row["support"]))
    value_rules = high_utility_rules(quantities, PROFITS, 0.1, min_conf=0.5, min_lift=1.5)
    print("High-utility rules with confidence >= 0.5 and lift >= 1.5 (without the lift, 'cheese => bread' is worth a lot only "
          "because the cheese is worth a lot):")
    for line in describe(value_rules.head(3), extra=("utility", "utility_share")):
        print("  " + line)

    return quantities, top, informative, pairs, mri, rare, flat, by_support, useful, value_rules


if __name__ == "__main__":
    main()
