# -*- coding: utf-8 -*-
"""
Ejemplo: mineria de itemsets frecuentes (FFIM.fpgrowth) + reglas de
asociacion crisp (ARM.association_rules) sobre el dataset de ejemplo
(Tabla 2) usado en:

    Fernandez-Basso, C., Ruiz, M.D., Martin-Bautista, M.J. (2024). "New
    Spark solutions for distributed frequent itemset and association rule
    mining algorithms". Cluster Computing, 27, 1217-1234.

Se ejecuta automaticamente en CI (ver tests/test_examples.py) para
comprobar que sigue funcionando con cada cambio en la libreria.
"""
import pandas as pd

from ARMxtend.ARM.association_rules import association_rules
from ARMxtend.FFIM import fpgrowth
from ARMxtend.FIM._shared import key_to_itemset

TRANSACTIONS = [
    "A,B,C", "B,D", "A,C,D", "B,C", "A,C",
    "B,D", "A,B,C", "B,C,D", "A,B,D", "B,C,D",
]


def main():
    transactions = [line.split(",") for line in TRANSACTIONS]

    freqItemsets = fpgrowth(transactions, min_supp=0.3)
    print("Itemsets frecuentes (min_supp=0.3):")
    for itemsetKey, support in sorted(freqItemsets.items()):
        print("  {:<10} support={:.2f}".format(itemsetKey, support))

    itemsetsDf = pd.DataFrame({
        "support": list(freqItemsets.values()),
        "itemsets": [key_to_itemset(key) for key in freqItemsets],
    })
    rulesDf = association_rules(itemsetsDf, metric="confidence", min_threshold=0.7)

    print("\nReglas de asociacion (min_confidence=0.7):")
    for _, rule in rulesDf.iterrows():
        antecedent = ",".join(sorted(rule["antecedents"]))
        consequent = ",".join(sorted(rule["consequents"]))
        print("  {} -> {:<6} support={:.2f} confidence={:.2f} lift={:.2f}".format(
            antecedent, consequent, rule["support"], rule["confidence"], rule["lift"]))

    return freqItemsets, rulesDf


if __name__ == "__main__":
    main()
