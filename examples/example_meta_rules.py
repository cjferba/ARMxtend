# -*- coding: utf-8 -*-
"""
Example: association meta-rules (ARM.meta_rules), crisp and fuzzy, on
three synthetic datasets that share (with different strength) the rule
A -> B, following:

    Ruiz, M.D., Gomez-Romero, J., Molina-Solana, M., Campana, J.R.,
    Martin-Bautista, M.J. (2016). "Meta-association rules for mining
    interesting associations in multiple datasets". Applied Soft Computing,
    49, 212-223.
"""
from ARMxtend.ARM.meta_rules import (
    mine_primary_rule_measures,
    crisp_meta_association_rules,
    fuzzy_meta_association_rules,
)

# Three datasets D1, D2, D3 in which A -> B is a strong and
# consistent rule (it appears whenever A appears), and C -> D only appears,
# debilmente, en D3.
DATASETS = [
    [["A", "B"], ["A", "B"], ["A", "B"], ["C"]],
    [["A", "B"], ["A", "B"], ["B"], ["C"]],
    [["A", "B"], ["A", "B", "C", "D"], ["C", "D"], ["B"]],
]


def main():
    ruleMeasuresPerDataset = mine_primary_rule_measures(DATASETS, min_supp=0.4, min_conf=0.5)

    print("Primary rules mined in each dataset:")
    for datasetIndex, ruleMeasures in enumerate(ruleMeasuresPerDataset):
        print("  D{}: {}".format(datasetIndex + 1, ruleMeasures))

    crispMetaRules = crisp_meta_association_rules(
        ruleMeasuresPerDataset, min_supp=0.6, min_conf=0.6)
    print("\nCrisp meta-rules (co-occurrence of rules between datasets):")
    for _, rule in crispMetaRules.iterrows():
        print("  {} -> {}  support={:.2f} confidence={:.2f}".format(
            set(rule["antecedents"]), set(rule["consequents"]), rule["support"], rule["confidence"]))

    fuzzyMetaRules = fuzzy_meta_association_rules(
        ruleMeasuresPerDataset, num_alpha=5, min_supp=0.3, min_conf=0.5)
    print("\nFuzzy meta-rules (weighted by the strength of each primary rule):")
    for _, rule in fuzzyMetaRules.iterrows():
        print("  {} -> {}  FSupp={:.2f} FConf={:.2f}".format(
            set(rule["antecedents"]), set(rule["consequents"]), rule["support"], rule["confidence"]))

    return ruleMeasuresPerDataset, crispMetaRules, fuzzyMetaRules


if __name__ == "__main__":
    main()
