# -*- coding: utf-8 -*-
"""
Ejemplo: significacion estadistica de las reglas -- cuantas de las reglas que
superan los umbrales son solo ruido, como quedarse con las que no lo son
(test exacto de Fisher + correccion por pruebas multiples, y tests de
permutacion) y por que con series temporales hay que barajar con
desplazamientos circulares.
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
    """Cestas aleatorias e independientes (cada item con probabilidad 0.3) con una unica asociacion real: i00 -> i01."""
    rng = np.random.RandomState(seed)
    table = pd.DataFrame(rng.rand(rows, items) < 0.3, columns=["i{:02d}".format(i) for i in range(items)])
    table["i01"] = table["i01"] | (table["i00"].to_numpy() & (rng.rand(rows) < 0.6))
    return table


def mine(table, min_supp=0.03, min_conf=0.45):
    frequent = fpgrowth([set(table.columns[row]) for row in table.to_numpy()], min_supp)
    itemsets = pd.DataFrame({"support": list(frequent.values()), "itemsets": [key_to_itemset(k) for k in frequent]})
    return association_rules(itemsets, metric="confidence", min_threshold=min_conf)


def regimes(rng, rows, length):
    """Serie booleana con tramos largos de valor constante (muy autocorrelacionada)."""
    return np.repeat(rng.rand(rows // length + 1) < 0.4, length)[:rows]


def _spurious(rules):
    """Reglas que no relacionan i00 con i01: lo unico real que hay en los datos."""
    return sum(not {"i00", "i01"} <= rule["antecedents"] | rule["consequents"] for _, rule in rules.iterrows())


def main():
    # 1. Cestas independientes con una asociacion real: el umbral de confianza deja pasar mucho ruido -----
    table = basket_data()
    rules = mine(table)
    print("Cestas:", len(table), "| reglas con confianza >= 0.45:", len(rules), "| sin i00 e i01 juntos (ruido):", _spurious(rules))

    fisher = significant_rules(rules, alpha=0.05, correction="fdr_bh", n_transactions=len(table))
    print("Con test de Fisher y FDR (alpha=0.05): {} reglas, {} de ruido".format(len(fisher), _spurious(fisher)))

    miner = crisp_rule_miner(0.03, "confidence", 0.45)
    null = permutation_null(table, miner, n_permutations=50, null="permute", random_state=0)
    permuted = permutation_p_values(rules, null, n_transactions=len(table))
    best = permuted[permuted["p_permutation"] <= 0.05]
    print("Con permutacion (50 copias barajadas): {} reglas, {} de ruido; en los datos barajados salen {:.0f} reglas de media".format(
        len(best), _spurious(best), null["n_rules"].mean()))

    planted = confidence_intervals(rule_significance(rules, len(table)), n_transactions=len(table))
    planted = planted[(planted["antecedents"] == {"i00"}) & (planted["consequents"] == {"i01"})].iloc[0]
    print("\ni00 -> i01: support={:.3f} confidence={:.2f} [{:.2f}, {:.2f}] lift={:.2f} p={:.1e}".format(
        planted["support"], planted["confidence"], planted["confidence_low"], planted["confidence_high"],
        planted["lift"], planted["p_value"]))

    # 2. Series temporales: dos series INDEPENDIENTES con regimenes largos parecen asociadas -------------
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
    print("\n{} pares de series independientes: Fisher (supone filas independientes) da {} falsas alarmas (p < 0.01); "
          "el test de permutacion circular, {} (p < 0.05)".format(pairs, fisher_alarms, shift_alarms))

    return rules, fisher, permuted, (pairs, fisher_alarms, shift_alarms)


if __name__ == "__main__":
    main()
