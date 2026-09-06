# -*- coding: utf-8 -*-
"""
Ejemplo: mineria de itemsets frecuentes difusos (FFIM.fuzzy_fpgrowth) +
reglas de asociacion difusas (FIM.FARE.fuzzy_association_rules), usando la
base de datos difusa de ejemplo (Tabla 2) de:

    Delgado, M., Ruiz, M.D., Sanchez, D., Serrano, J.M. (2011). "A formal
    model for mining fuzzy rules using the RL representation theory".
    Information Sciences, 181, 5194-5213.

La regla {i1,i3} -> {i4} tiene, segun el articulo (Seccion 4.4), FSupp =
0.266, FConf = 0.5 y FCF = 0.33 con 5 alpha-cortes {1, 0.8, 0.6, 0.4, 0.2}.
Este ejemplo reproduce esos valores exactamente, y sirve por tanto como
comprobacion (ver tests/test_examples.py) de que la implementacion es fiel
al modelo formal del articulo.
"""
from ARMxtend.FFIM import fuzzy_fpgrowth
from ARMxtend.FIM.FARE import fuzzy_association_rules

# t1..t6, grados de pertenencia de i1..i5 (Tabla 2 del articulo)
FUZZY_TRANSACTIONS = [
    [("i1", 1.0), ("i2", 0.2), ("i3", 1.0), ("i4", 0.8), ("i5", 0.9)],
    [("i1", 1.0), ("i2", 1.0), ("i3", 0.8)],
    [("i1", 0.4), ("i2", 0.1), ("i3", 0.7), ("i4", 0.6)],
    [("i1", 0.6), ("i3", 0.4), ("i4", 0.4), ("i5", 0.5)],
    [("i1", 0.4), ("i2", 0.1), ("i3", 0.6)],
    [("i2", 1.0)],
]

NUM_ALPHA = 5  # alpha-cortes {1, 0.8, 0.6, 0.4, 0.2}, como en el articulo


def main():
    freqItemsets = fuzzy_fpgrowth(FUZZY_TRANSACTIONS, min_supp=0.1, num_alpha=NUM_ALPHA)
    print("Itemsets difusos frecuentes (min FSupp=0.1, {} alpha-cortes):".format(NUM_ALPHA))
    for itemsetKey in sorted(freqItemsets):
        print("  {}: bit-list por alpha-corte = {}".format(itemsetKey, freqItemsets[itemsetKey]))

    rulesDf = fuzzy_association_rules(freqItemsets, NUM_ALPHA, metric="support", min_threshold=0.0)

    print("\nReglas de asociacion difusas:")
    target = None
    for _, rule in rulesDf.iterrows():
        antecedent = ",".join(sorted(rule["antecedents"]))
        consequent = ",".join(sorted(rule["consequents"]))
        print("  {} -> {:<5} FSupp={:.3f} FConf={:.3f} FCF={:.3f}".format(
            antecedent, consequent, rule["support"], rule["confidence"], rule["certainty_factor"]))
        if set(rule["antecedents"]) == {"i1", "i3"} and set(rule["consequents"]) == {"i4"}:
            target = rule

    print("\nComprobacion contra el articulo para {i1,i3} -> {i4}:")
    print("  esperado  FSupp=0.266 FConf=0.5 FCF=0.33")
    print("  obtenido  FSupp={:.3f} FConf={:.3f} FCF={:.3f}".format(
        target["support"], target["confidence"], target["certainty_factor"]))

    return freqItemsets, rulesDf


if __name__ == "__main__":
    main()
