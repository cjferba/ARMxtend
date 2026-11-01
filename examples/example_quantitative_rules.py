# -*- coding: utf-8 -*-
"""
Ejemplo: reglas cuantitativas por intervalos (Srikant y Agrawal, 1996) frente a las
reglas difusas. Una tienda tiene la edad, los ingresos y el sexo de sus clientes, y si
compraron un producto. Los ingresos son altos solo entre los 30 y los 45 anos, y las
mujeres con ingresos altos compran casi siempre. Los intervalos base de igual frecuencia
se funden en rangos hasta cierto soporte maximo, y la poda R-interesante descarta los
rangos que se pueden ensanchar o estrechar sin cambiar la regla.
"""
import numpy as np
import pandas as pd

from ARMxtend.ARM import IntervalPartition, partial_completeness_intervals, quantitative_rules
from ARMxtend.FFIM import fuzzy_fpgrowth
from ARMxtend.FIM.FARE import fuzzy_association_rules
from ARMxtend.preprocessing import FuzzyLib


def customers(rows=3000, seed=0):
    """
    Edad uniforme en [18, 70); ingresos N(70, 8) si la edad esta en [30, 45) y N(35, 10) en otro caso (minimo 5);
    sexo al azar; compra el 90 % de las mujeres con ingresos > 60 y el 10 % del resto.
    """
    rng = np.random.RandomState(seed)
    age = rng.uniform(18, 70, rows)
    income = np.where((age >= 30) & (age < 45), rng.normal(70, 8, rows), rng.normal(35, 10, rows)).clip(5)
    sex = rng.choice(["F", "M"], rows)
    buys = np.where((income > 60) & (sex == "F"), rng.rand(rows) < 0.9, rng.rand(rows) < 0.1)
    return pd.DataFrame({"age": age, "income": income, "sex": sex, "buys": np.where(buys, "yes", "no")})


def describe(rules):
    return ["{} => {}  support={:.2f} confidence={:.2f} lift={:.2f}".format(
        " & ".join(sorted(rule["antecedents"])), " & ".join(sorted(rule["consequents"])), rule["support"], rule["confidence"],
        rule["lift"]) for _, rule in rules.iterrows()]


def one_to_one(rules, antecedent_prefix, consequent_prefix):
    """Reglas de un item a un item entre dos atributos."""
    first = rules["antecedents"].map(lambda items: len(items) == 1 and next(iter(items)).startswith(antecedent_prefix))
    second = rules["consequents"].map(lambda items: len(items) == 1 and next(iter(items)).startswith(consequent_prefix))
    return rules[first & second]


def main():
    table = customers()
    print("{} clientes. Con completitud parcial K=1.5 y soporte 0.05 el articulo pide {} intervalos por atributo; "
          "aqui bastan 10 y rangos de hasta el 40 %.".format(len(table), partial_completeness_intervals(2, 0.05, 1.5)))

    # 1. Intervalos base y rangos --------------------------------------------------------------------------------
    partition = IntervalPartition(n_intervals=10, max_supp=0.4).fit(table)
    ranges = partition.intervals_
    print("Particion: {} items de rango ({} intervalos base + {} rangos de varios); edad en {} intervalos de ~{:.1f} anos".format(
        len(ranges), int((ranges["n_base"] == 1).sum()), int((ranges["n_base"] > 1).sum()),
        len(partition.edges_["age"]) - 1, float(np.diff(partition.edges_["age"]).mean())))

    # 2. Reglas, y las R-interesantes -----------------------------------------------------------------------------
    rules = quantitative_rules(table, min_supp=0.05, partition=partition, min_threshold=0.7, max_len=3)
    kept = quantitative_rules(table, min_supp=0.05, partition=partition, min_threshold=0.7, max_len=3, interest=1.3)
    print("\n{} reglas (soporte 0.05, confianza 0.7); R-interesantes con R=1.3: {}".format(len(rules), len(kept)))

    age_income = one_to_one(rules, "age=", "income=")
    region = age_income[age_income["confidence"] >= 0.85].sort_values("support", ascending=False).head(1)
    print("\nLa region plantada [30, 45), con el mayor soporte con confianza >= 0.85:")
    for line in describe(region):
        print("  " + line)

    buyers = rules[(rules["consequents"] == frozenset({"buys=yes"})) & rules["antecedents"].map(lambda items: "sex=F" in items)]
    print("\nQuien compra (mujeres, por lift):")
    for line in describe(buyers.sort_values("lift", ascending=False).head(3)):
        print("  " + line)

    # 3. Frente a las reglas difusas con etiquetas fijas (5 por atributo, de igual frecuencia) -------------------
    library = FuzzyLib()
    library.data = table[["age", "income"]].copy()
    library.AutoFuzzification(Atri=["age", "income"], n_labels=5, method="quantile")
    frequent = fuzzy_fpgrowth(library.GetFuzzyTransactions(), 0.02, 10)
    fuzzy = fuzzy_association_rules(frequent, 10, metric="confidence", min_threshold=0.4, measures=["lift"])
    fuzzy_age_income = one_to_one(fuzzy, "age_", "income_").sort_values("confidence", ascending=False)
    print("\nReglas difusas edad => ingresos (5 etiquetas fijas), por confianza:")
    for line in describe(fuzzy_age_income.head(3)):
        print("  " + line)
    print("La region [30, 45) cae a caballo de dos etiquetas y ninguna llega a la confianza del rango fundido.")

    return table, partition, rules, kept, region, fuzzy_age_income


if __name__ == "__main__":
    main()
