# -*- coding: utf-8 -*-
"""
Example: fuzzy frequent itemset mining (FFIM.fuzzy_fpgrowth) +
fuzzy association rules (FIM.FARE.fuzzy_association_rules), using the
example fuzzy database (Table 2) of:

    Delgado, M., Ruiz, M.D., Sanchez, D., Serrano, J.M. (2011). "A formal
    model for mining fuzzy rules using the RL representation theory".
    Information Sciences, 181, 5194-5213.

The rule {i1,i3} -> {i4} has, according to the paper (Section 4.4), FSupp =
0.266, FConf = 0.5 and FCF = 0.33 with 5 alpha-cuts {1, 0.8, 0.6, 0.4, 0.2}.
This example reproduces those values exactly, and therefore serves as a
check (see tests/test_examples.py) that the implementation is faithful
to the formal model of the paper.
"""
from ARMxtend.FFIM import fuzzy_fpgrowth
from ARMxtend.FIM.FARE import fuzzy_association_rules

# t1..t6, membership degrees of i1..i5 (Table 2 of the paper)
FUZZY_TRANSACTIONS = [
    [("i1", 1.0), ("i2", 0.2), ("i3", 1.0), ("i4", 0.8), ("i5", 0.9)],
    [("i1", 1.0), ("i2", 1.0), ("i3", 0.8)],
    [("i1", 0.4), ("i2", 0.1), ("i3", 0.7), ("i4", 0.6)],
    [("i1", 0.6), ("i3", 0.4), ("i4", 0.4), ("i5", 0.5)],
    [("i1", 0.4), ("i2", 0.1)],
    [("i2", 1.0)],
]

NUM_ALPHA = 5  # alpha-cuts {1, 0.8, 0.6, 0.4, 0.2}, as in the paper


def main():
    freqItemsets = fuzzy_fpgrowth(FUZZY_TRANSACTIONS, min_supp=0.1, num_alpha=NUM_ALPHA)
    print("Frequent fuzzy itemsets (min FSupp=0.1, {} alpha-cuts):".format(NUM_ALPHA))
    for itemsetKey in sorted(freqItemsets):
        print("  {}: bit-list por alpha-corte = {}".format(itemsetKey, freqItemsets[itemsetKey]))

    rulesDf = fuzzy_association_rules(freqItemsets, NUM_ALPHA, metric="support", min_threshold=0.0)

    print("\nFuzzy association rules:")
    target = None
    for _, rule in rulesDf.iterrows():
        antecedent = ",".join(sorted(rule["antecedents"]))
        consequent = ",".join(sorted(rule["consequents"]))
        print("  {} -> {:<5} FSupp={:.3f} FConf={:.3f} FCF={:.3f}".format(
            antecedent, consequent, rule["support"], rule["confidence"], rule["certainty_factor"]))
        if set(rule["antecedents"]) == {"i1", "i3"} and set(rule["consequents"]) == {"i4"}:
            target = rule

    print("\nCheck against the paper for {i1,i3} -> {i4}:")
    print("  expected  FSupp=0.266 FConf=0.5 FCF=0.33")
    print("  obtenido  FSupp={:.3f} FConf={:.3f} FCF={:.3f}".format(
        target["support"], target["confidence"], target["certainty_factor"]))

    return freqItemsets, rulesDf


if __name__ == "__main__":
    main()
