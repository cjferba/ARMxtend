# -*- coding: utf-8 -*-
"""
Ejemplo: tres formas de mirar mas alla del soporte minimo. Una tienda tiene 2000 cestas con
cantidades. (1) Sin saber que umbral de soporte poner, las ``k`` reglas de mayor soporte.
(2) Los pares que son frecuentes por separado y raros juntos -- panales y toallitas, vino y
queso -- que un umbral global o no ve o ahoga entre ruido. (3) Lo que mas VALE no es lo que mas
se vende: el vino y el queso dan mas beneficio que el pan y la leche, aunque se compren menos.
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
    Cantidades compradas en ``rows`` cestas:

    * pan (50 %), leche (45 %), huevos (35 %) y fruta (40 %), al azar y cada uno por su cuenta;
    * el 9 % son familias con bebe y compran panales y toallitas casi siempre (95 % cada uno); el resto los compra
      al azar con un 6 %: cada uno llega al 14 %, pero juntos solo al 8 %;
    * vino en el 12 % de las cestas y queso en el 30 %, y el 70 % de quien compra vino compra queso.
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
    """Las reglas de siempre: itemsets frecuentes con FP-Growth y reglas con `association_rules`."""
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
    print("{} cestas, {} productos".format(len(transactions), quantities.shape[1]))

    # 1. Top-k: las k reglas de mayor soporte, sin umbral --------------------------------------------------------
    top = top_k_rules(transactions, 5, min_conf=0.5)
    print("\nLas 5 reglas de mayor soporte con confianza >= 0.5 (sin fijar un soporte minimo):")
    for line in describe(top):
        print("  " + line)
    informative = top_k_rules(transactions, 5, min_conf=0.5, min_lift=1.5)
    print("Las 5 de mayor soporte que ademas tienen lift >= 1.5 (lo popular deja de tapar lo que asocia):")
    for line in describe(informative):
        print("  " + line)
    for min_supp in (0.3, 0.01):
        print("Con un umbral de {}: {} reglas".format(min_supp, len(flat_rules(transactions, min_supp))))
    pairs = top_k_itemsets(transactions, 5, min_len=2)

    # 2. Raras: frecuentes por separado, raras juntas --------------------------------------------------------------
    mri = minimal_rare_itemsets(transactions, 0.1, min_len=2)
    rare = rare_rules(transactions, 0.1, min_conf=0.5, min_lift=2)
    flat = flat_rules(transactions, 0.01)
    print("\nItemsets raros minimales (soporte < 0.1, todos sus subconjuntos con soporte >= 0.1): {}".format(len(mri)))
    print("Reglas raras con confianza >= 0.5 y lift >= 2: {}. Con un umbral de 0.01, que tambien las alcanza, "
          "salen {} reglas con esa confianza, {} con ese lift:".format(len(rare), len(flat), int((flat["lift"] >= 2).sum())))
    for line in describe(rare.sort_values("lift", ascending=False)):
        print("  " + line)

    # 3. Alta utilidad: lo que vale no es lo que mas se vende -------------------------------------------------------
    by_support = top_k_itemsets(transactions, 3)
    useful = high_utility_itemsets(quantities, PROFITS, 0.1)
    print("\nItemsets mas frecuentes: " + "; ".join("{{{}}} {:.2f}".format(", ".join(sorted(s)), v)
                                                    for v, s in zip(by_support["support"], by_support["itemsets"])))
    print("Itemsets de alta utilidad (>= 10 % de la utilidad total, que es {:.0f}):".format(useful.attrs["total_utility"]))
    for _, row in useful.head(5).iterrows():
        print("  {{{}}}  utilidad={:.0f}  soporte={:.3f}".format(", ".join(sorted(row["itemsets"])), row["utility"], row["support"]))
    value_rules = high_utility_rules(quantities, PROFITS, 0.1, min_conf=0.5, min_lift=1.5)
    print("Reglas de alta utilidad con confianza >= 0.5 y lift >= 1.5 (sin el lift, 'queso => pan' vale mucho solo "
          "porque el queso vale mucho):")
    for line in describe(value_rules.head(3), extra=("utility", "utility_share")):
        print("  " + line)

    return quantities, top, informative, pairs, mri, rare, flat, by_support, useful, value_rules


if __name__ == "__main__":
    main()
