# -*- coding: utf-8 -*-
"""
Ejemplo: patrones y reglas secuenciales DIFUSOS sobre lecturas numericas. Cada dia
de una vivienda tiene 8 lecturas (cada 3 horas) de temperatura y de potencia de
calefaccion; cuanto mas frio es el amanecer, mas potencia se consume por la tarde.
Con umbrales fijos el resultado depende de donde se ponga el corte ("frio" = menos
de 8, 10, 12 o 14 grados) y cada dia cuenta entero o nada; con conjuntos difusos no
hay corte y cada dia cumple la regla EN SU GRADO.
"""
import numpy as np
import pandas as pd

from ARMxtend.ARM import fuzzy_rule_degree, fuzzy_sequential_rules, sequential_rules
from ARMxtend.FIM import fuzzy_prefixspan, fuzzy_sequences_from_memberships
from ARMxtend.preprocessing import FuzzyLib


def daily_readings(days=80, seed=4):
    """
    Lecturas de ``days`` dias, 8 por dia. Cada dia tiene una intensidad de frio ``cold`` en [0, 1]:

    * temperatura: 18 grados, menos ``12 * cold`` en las tres primeras lecturas (la manana);
    * potencia: 2 kW, mas ``6 * cold`` en las tres ultimas (la tarde-noche).

    Devuelve la tabla de lecturas y la serie de intensidades de frio por dia.
    """
    rng = np.random.RandomState(seed)
    cold = rng.rand(days)
    rows = []
    for day in range(days):
        for step in range(8):
            temp = 18 - (12 * cold[day] if step < 3 else 0) + rng.normal(0, 1)
            power = 2 + (6 * cold[day] if step >= 5 else 0) + rng.normal(0, 0.4)
            rows.append((day, step, temp, power))
    return pd.DataFrame(rows, columns=["day", "step", "temp", "power"]), pd.Series(cold, name="cold")


def main():
    readings, cold = daily_readings()

    # 1. Fuzzificar las lecturas y construir una secuencia difusa por dia --------------------------------------
    library = FuzzyLib()
    library.data = readings.copy()
    names = library.AutoFuzzification(Atri=["temp", "power"], n_labels=3, method="uniform")
    sequences = fuzzy_sequences_from_memberships(library.data[names], by=library.data["day"], min_degree=0.2)
    print("{} dias; etiquetas difusas: {}".format(len(sequences), ", ".join(names)))

    # 2. Patrones: el orden importa -----------------------------------------------------------------------------
    patterns = fuzzy_prefixspan(sequences, 0.2, max_len=2, min_len=2)
    support = {tuple(next(iter(element)) for element in pattern): value
               for pattern, value in zip(patterns["sequence"], patterns["support"])}
    print("\nPatrones de dos eventos (soporte difuso, {} en total):".format(len(patterns)))
    for pair in [("temp_low", "power_high"), ("power_high", "temp_low")]:
        print("  <{} -> {}>  support={}".format(pair[0], pair[1], "{:.2f}".format(support[pair]) if pair in support
                                                else "< 0.20 (no frecuente)"))

    # 3. Reglas: la que une el frio de la manana con el consumo de la tarde destaca por su lift -----------------
    rules = fuzzy_sequential_rules(sequences, 0.1, min_conf=0.5, max_antecedent=1, max_consequent=1)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    print("\nReglas difusas mas llamativas (por lift):")
    for _, rule in rules.head(3).iterrows():
        print("  {} => {}  support={:.2f} confidence={:.2f} lift={:.2f}".format(
            ", ".join(sorted(rule["antecedents"])), ", ".join(sorted(rule["consequents"])), rule["support"],
            rule["confidence"], rule["lift"]))

    # 4. Cada dia cumple la regla en su grado, que sigue a la intensidad del frio -------------------------------
    degrees = pd.Series([fuzzy_rule_degree(sequence, ["temp_low"], ["power_high"]) for sequence in sequences], name="degree")
    correlation = float(np.corrcoef(cold, degrees)[0, 1])
    print("\ntemp_low => power_high por dia: grado 0 en {} dias suaves, hasta {:.2f} los mas frios; correlacion con la "
          "intensidad del frio {:.2f}".format(int((degrees == 0).sum()), degrees.max(), correlation))

    # 5. Comparacion con umbrales fijos: cold = temp < corte, peak = power > 5 -------------------------------------
    fuzzy_rule = rules[(rules["antecedents"] == frozenset({"temp_low"})) & (rules["consequents"] == frozenset({"power_high"}))].iloc[0]
    print("\nCon conjuntos difusos: temp_low => power_high  support={:.2f} confidence={:.2f} lift={:.2f}".format(
        fuzzy_rule["support"], fuzzy_rule["confidence"], fuzzy_rule["lift"]))
    crisp_rules = {}
    for cut in (8, 10, 12, 14):
        days = [[{label for label, holds in (("cold", row.temp < cut), ("peak", row.power > 5)) if holds}
                 for row in day.itertuples()] for _, day in readings.groupby("day")]
        found = sequential_rules(days, 0.05, min_conf=0.0, max_antecedent=1, max_consequent=1)
        crisp_rules[cut] = found[(found["antecedents"] == frozenset({"cold"})) & (found["consequents"] == frozenset({"peak"}))].iloc[0]
        print("Con umbral temp < {:>2}:      cold => peak           support={:.2f} confidence={:.2f} lift={:.2f}".format(
            cut, crisp_rules[cut]["support"], crisp_rules[cut]["confidence"], crisp_rules[cut]["lift"]))

    return patterns, rules, degrees, cold, (crisp_rules, fuzzy_rule)


if __name__ == "__main__":
    main()
