# -*- coding: utf-8 -*-
"""
Ejemplo: reglas multinivel con una taxonomia de productos. Un supermercado vende
12 productos agrupados en categorias (``Food/Dairy/milk``, ``Drinks/Juice/apple_juice``...).
Quien compra pan (de cualquier tipo) suele comprar algun lacteo, pero se reparte entre
muchos productos y ninguno alcanza por separado un soporte alto; y el croissant, en
concreto, va con zumo mucho mas que el resto del pan. Las reglas generalizadas ven las
categorias; las R-interesantes descartan las que solo repiten, con un producto concreto,
lo que ya dice la categoria.
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
    Cestas de ``rows`` clientes:

    * el 50 % compra un pan (blanco, integral o croissant, a partes iguales);
    * lacteo (uno de los tres): el 80 % de quien compro pan y el 25 % del resto;
    * zumo (naranja o manzana): el 90 % de quien compro croissant y el 20 % del resto;
    * carne (40 %) y refresco (30 %) al azar.

    Devuelve la taxonomia y las cestas no vacias.
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

    # 1. Sin taxonomia: solo productos sueltos, con el soporte repartido -------------------------------------
    frequent = fpgrowth(baskets, 0.04)
    flat = association_rules(pd.DataFrame({"support": list(frequent.values()), "itemsets": [key_to_itemset(k) for k in frequent]}),
                             min_threshold=0.4)
    flat = flat[flat["lift"] >= 1.2]
    print("\nSin taxonomia (soporte 0.04, confianza 0.4, lift 1.2): {} reglas, todas entre productos concretos".format(len(flat)))

    # 2. Con taxonomia: soporte reducido en los niveles profundos (Han y Fu) ----------------------------------
    rules = generalized_rules(baskets, taxonomy, {1: 0.2, 2: 0.1, 3: 0.04}, min_threshold=0.4)
    rules = rules[rules["lift"] >= 1.2]
    print("Con taxonomia (soporte 0.2 / 0.1 / 0.04 por nivel): {} reglas, {} de ellas entre categorias (nivel 1 o 2)".format(
        len(rules), int((rules["level"] <= 2).sum())))

    # 3. R-interesantes: fuera las que solo repiten lo que dice la categoria ---------------------------------
    kept = interesting_rules(generalized_rules(baskets, taxonomy, {1: 0.2, 2: 0.1, 3: 0.04}, min_threshold=0.4), interest=1.2)
    kept = kept[kept["lift"] >= 1.2].sort_values("lift", ascending=False)
    dropped = rules.loc[~rules.index.isin(kept.index)]
    print("\nR-interesantes (R=1.2): quedan {} de {}. Por lift:".format(len(kept), len(rules)))
    for line in describe(kept.head(6)):
        print("  " + line)
    print("\nDescartadas por repetir lo que ya dicen sus antecesoras, por ejemplo:")
    for line in describe(dropped[dropped["level"] == 3].sort_values("lift", ascending=False).head(4)):
        print("  " + line)

    return taxonomy, flat, rules, kept, dropped


if __name__ == "__main__":
    main()
