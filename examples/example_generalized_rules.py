# -*- coding: utf-8 -*-
"""
Example: multilevel rules with a product taxonomy. A supermarket sells
12 products grouped in categories (``Food/Dairy/milk``, ``Drinks/Juice/apple_juice``...).
Whoever buys bread (of any kind) usually buys some dairy product, but it is spread over
many products and none reaches a high support on its own; and the croissant, in
particular, goes with juice much more than the rest of the bread. Generalized rules see the
categories; R-interesting ones discard those that only repeat, with a specific product,
what the category already says.
"""
import numpy as np
import pandas as pd

from ARMxtend.ARM import association_rules, generalized_rules, interesting_rules
from ARMxtend.FFIM import fpgrowth
from ARMxtend.FIM import Taxonomy
from ARMxtend.FIM._shared import key_to_itemset

PATHS = [
    "Food/Dairy/milk", "Food/Dairy/yogurt", "Food/Dairy/cheese",
    "Food/Bakery/white_bread", "Food/Bakery/brown_bread", "Food/Bakery/croissant",
    "Food/Meat/beef", "Food/Meat/chicken",
    "Drinks/Soda/cola", "Drinks/Soda/lemonade",
    "Drinks/Juice/orange_juice", "Drinks/Juice/apple_juice",
]


def supermarket(rows=3000, seed=0):
    """
    Baskets of ``rows`` customers:

    * 50 % buy a bread (white, whole-wheat or croissant, in equal parts);
    * dairy (one of the three): 80 % of those who bought bread and 25 % of the rest;
    * juice (orange or apple): 90 % of those who bought a croissant and 20 % of the rest;
    * meat (40 %) and soda (30 %) at random.

    Returns the taxonomy and the non-empty baskets.
    """
    rng = np.random.RandomState(seed)
    taxonomy = Taxonomy.from_paths(PATHS)
    leaves = {category: sorted(taxonomy.descendants(category) & set(taxonomy.leaves()))
              for category in ("Dairy", "Bakery", "Meat", "Soda", "Juice")}
    baskets = []
    for _ in range(rows):
        basket = []
        bakery = rng.rand() < 0.5
        if bakery:
            basket.append(rng.choice(leaves["Bakery"]))
        if rng.rand() < (0.8 if bakery else 0.25):
            basket.append(rng.choice(leaves["Dairy"]))
        if rng.rand() < (0.9 if "croissant" in basket else 0.2):
            basket.append(rng.choice(leaves["Juice"]))
        if rng.rand() < 0.4:
            basket.append(rng.choice(leaves["Meat"]))
        if rng.rand() < 0.3:
            basket.append(rng.choice(leaves["Soda"]))
        if basket:
            baskets.append(basket)
    return taxonomy, baskets


def describe(rules):
    return ["{} => {}  support={:.2f} confidence={:.2f} lift={:.2f}".format(
        ", ".join(sorted(rule["antecedents"])), ", ".join(sorted(rule["consequents"])), rule["support"], rule["confidence"],
        rule["lift"]) for _, rule in rules.iterrows()]


def main():
    taxonomy, baskets = supermarket()
    print("{} cestas; taxonomia: {} items en {} niveles ({})".format(len(baskets), len(taxonomy), taxonomy.depth,
                                                                      ", ".join(taxonomy.roots())))

    # 1. Without a taxonomy: only individual products, with the support spread out -------------------------------------
    frequent = fpgrowth(baskets, 0.04)
    flat = association_rules(pd.DataFrame({"support": list(frequent.values()), "itemsets": [key_to_itemset(k) for k in frequent]}),
                             min_threshold=0.4)
    flat = flat[flat["lift"] >= 1.2]
    print("\nWithout a taxonomy (support 0.04, confidence 0.4, lift 1.2): {} rules, all between specific products".format(len(flat)))

    # 2. With a taxonomy: reduced support at the deep levels (Han and Fu) ----------------------------------
    rules = generalized_rules(baskets, taxonomy, {1: 0.2, 2: 0.1, 3: 0.04}, min_threshold=0.4)
    rules = rules[rules["lift"] >= 1.2]
    print("With taxonomy (support 0.2 / 0.1 / 0.04 per level): {} rules, {} of them between categories (level 1 or 2)".format(
        len(rules), int((rules["level"] <= 2).sum())))

    # 3. R-interesting: out go those that only repeat what the category says ---------------------------------
    kept = interesting_rules(generalized_rules(baskets, taxonomy, {1: 0.2, 2: 0.1, 3: 0.04}, min_threshold=0.4), interest=1.2)
    kept = kept[kept["lift"] >= 1.2].sort_values("lift", ascending=False)
    dropped = rules.loc[~rules.index.isin(kept.index)]
    print("\nR-interesting (R=1.2): {} of {} remain. By lift:".format(len(kept), len(rules)))
    for line in describe(kept.head(6)):
        print("  " + line)
    print("\nDiscarded for repeating what their ancestors already say, for example:")
    for line in describe(dropped[dropped["level"] == 3].sort_values("lift", ascending=False).head(4)):
        print("  " + line)

    return taxonomy, flat, rules, kept, dropped


if __name__ == "__main__":
    main()
