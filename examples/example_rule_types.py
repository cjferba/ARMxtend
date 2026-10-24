# -*- coding: utf-8 -*-
"""
Ejemplo: otros tipos de reglas y herramientas -- reglas con items ausentes,
reglas de excepcion y anomalas, reglas graduales, itemsets cerrados y
maximales, poda de reglas redundantes, y fuzzificacion automatica seguida de
reglas difusas con medidas adicionales.
"""
import numpy as np
import pandas as pd

from ARMxtend.ARM import (
    anomalous_rules, association_rules, exception_rules, format_gradual_itemset, gradual_rules,
    negative_association_rules, prune_redundant_rules,
)
from ARMxtend.FFIM import fpgrowth, fuzzy_fpgrowth
from ARMxtend.FIM import closed_itemsets, maximal_itemsets
from ARMxtend.FIM._shared import key_to_itemset
from ARMxtend.FIM.FARE import fuzzy_association_rules
from ARMxtend.preprocessing import FuzzyLib


def _rep(items, times):
    return [set(items) for _ in range(times)]


def _fmt(items):
    return "{" + ", ".join(sorted(items)) + "}"


def main():
    # 1. Items ausentes: quien toma cafe no toma te (y viceversa) ---------------
    drinks = _rep({"coffee", "sugar"}, 30) + _rep({"tea"}, 25) + _rep({"tea", "sugar"}, 5) + _rep({"coffee"}, 10)
    negative = negative_association_rules(drinks, min_supp=0.25, min_threshold=0.95)
    print("Reglas con items ausentes:")
    for _, r in negative.iterrows():
        print("  {} -> {}  support={:.2f} confidence={:.2f}".format(
            _fmt(r["antecedents"]), _fmt(r["consequents"]), r["support"], r["confidence"]))

    # 2. Excepciones y anomalias de la regla "fiebre -> reposo" -------------------
    sickness = (_rep({"fever", "rest"}, 60) + _rep({"fever", "must_work"}, 5) + _rep({"must_work", "rest"}, 35)
                + _rep({"fever", "hospital"}, 10))
    exceptions = exception_rules(sickness, min_supp=0.3, min_conf=0.8, min_exception_supp=0.03)
    print("\nExcepciones a reglas comunes fuertes:")
    for _, r in exceptions.iterrows():
        print("  {} -> {} (conf {:.2f}), pero con {} ocurre ~{}: confidence={:.2f} (referencia {:.2f})".format(
            _fmt(r["antecedents"]), r["consequent"], r["common_confidence"], _fmt(r["condition"]),
            r["consequent"], r["exception_confidence"], r["reference_confidence"]))
    anomalies = anomalous_rules(sickness, min_supp=0.3, min_conf=0.7, min_anomalous_supp=0.05,
                                min_anomalous_conf=0.6)
    print("Anomalias (A y no b -> C):")
    for _, r in anomalies.iterrows():
        print("  {} -> {}: en {} casos con ~{}, aparece {} (confidence={:.2f})".format(
            _fmt(r["antecedents"]), r["consequent"], r["anomalous_cases"], r["consequent"],
            _fmt(r["anomaly"]), r["anomalous_confidence"]))

    # 3. Reglas graduales sobre datos numericos, sin discretizar ----------------------
    rng = np.random.RandomState(0)
    age = rng.uniform(20, 80, 150)
    exercise = rng.uniform(0, 10, 150)
    health = pd.DataFrame({
        "age": age,
        "exercise": exercise,
        "cholesterol": 120 + 1.5 * age - 6 * exercise + rng.normal(0, 8, 150),
    })
    gradual = gradual_rules(health, min_supp=0.25, min_conf=0.8)
    print("\nReglas graduales:")
    for _, r in gradual.sort_values("confidence", ascending=False).head(3).iterrows():
        print("  cuanto {} -> {}  support={:.2f} confidence={:.2f}".format(
            format_gradual_itemset(r["antecedents"]), format_gradual_itemset(r["consequents"]),
            r["support"], r["confidence"]))

    # 4. Itemsets cerrados/maximales y poda de reglas redundantes -------------------------
    baskets = _rep({"a", "b", "c"}, 3) + _rep({"a", "b"}, 1) + _rep({"d"}, 1)
    frequent = fpgrowth(baskets, min_supp=0.2)
    closed, maximal = closed_itemsets(frequent), maximal_itemsets(frequent)
    allRules = association_rules(pd.DataFrame({"support": list(frequent.values()),
                                               "itemsets": [key_to_itemset(k) for k in frequent]}),
                                 min_threshold=0.0)
    pruned = prune_redundant_rules(allRules)
    print("\nItemsets frecuentes: {}, cerrados: {}, maximales: {}".format(
        len(frequent), sorted(closed), sorted(maximal)))
    print("Reglas: {} -> {} tras eliminar las redundantes".format(len(allRules), len(pruned)))

    # 5. Fuzzificacion automatica + reglas difusas con medidas adicionales ---------------------
    fuzzy = FuzzyLib()
    fuzzy.data = health[["age", "cholesterol"]].copy()
    fuzzy.AutoFuzzification(n_labels=3, method="quantile")
    transactions = fuzzy.GetFuzzyTransactions()
    itemsets = fuzzy_fpgrowth(transactions, min_supp=0.1, num_alpha=10)
    fuzzyRules = fuzzy_association_rules(itemsets, 10, metric="zhangs_metric", min_threshold=0.4,
                                         measures=["zhangs_metric", "kulczynski"])
    print("\nReglas difusas con Zhang >= 0.4:")
    for _, r in fuzzyRules.sort_values("zhangs_metric", ascending=False).head(3).iterrows():
        print("  {} -> {}  FSupp={:.2f} FConf={:.2f} zhang={:.2f} kulczynski={:.2f}".format(
            _fmt(r["antecedents"]), _fmt(r["consequents"]), r["support"], r["confidence"],
            r["zhangs_metric"], r["kulczynski"]))

    return negative, exceptions, anomalies, gradual, (closed, maximal, pruned), fuzzyRules


if __name__ == "__main__":
    main()
