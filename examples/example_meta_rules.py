# -*- coding: utf-8 -*-
"""
Ejemplo: meta-reglas de asociacion (ARM.meta_rules), crisp y difusas, sobre
tres datasets sinteticos que comparten (con distinta fuerza) la regla
A -> B, siguiendo:

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

# Tres datasets D1, D2, D3 en los que A -> B es una regla fuerte y
# consistente (aparece siempre que aparece A), y C -> D solo aparece,
# debilmente, en D3.
DATASETS = [
    [["A", "B"], ["A", "B"], ["A", "B"], ["C"]],
    [["A", "B"], ["A", "B"], ["B"], ["C"]],
    [["A", "B"], ["A", "B", "C", "D"], ["C", "D"], ["B"]],
]


def main():
    ruleMeasuresPerDataset = mine_primary_rule_measures(DATASETS, min_supp=0.4, min_conf=0.5)

    print("Reglas primarias minadas en cada dataset:")
    for datasetIndex, ruleMeasures in enumerate(ruleMeasuresPerDataset):
        print("  D{}: {}".format(datasetIndex + 1, ruleMeasures))

    crispMetaRules = crisp_meta_association_rules(
        ruleMeasuresPerDataset, min_supp=0.6, min_conf=0.6)
    print("\nMeta-reglas crisp (co-ocurrencia de reglas entre datasets):")
    for _, rule in crispMetaRules.iterrows():
        print("  {} -> {}  support={:.2f} confidence={:.2f}".format(
            set(rule["antecedents"]), set(rule["consequents"]), rule["support"], rule["confidence"]))

    fuzzyMetaRules = fuzzy_meta_association_rules(
        ruleMeasuresPerDataset, num_alpha=5, min_supp=0.3, min_conf=0.5)
    print("\nMeta-reglas difusas (ponderadas por la fuerza de cada regla primaria):")
    for _, rule in fuzzyMetaRules.iterrows():
        print("  {} -> {}  FSupp={:.2f} FConf={:.2f}".format(
            set(rule["antecedents"]), set(rule["consequents"]), rule["support"], rule["confidence"]))

    return ruleMeasuresPerDataset, crispMetaRules, fuzzyMetaRules


if __name__ == "__main__":
    main()
