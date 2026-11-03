# -*- coding: utf-8 -*-
"""
Example: FUZZY sequential patterns and rules over numeric readings. Each day
of a home has 8 readings (every 3 hours) of temperature and of heating
power; the colder the dawn, the more power is consumed in the afternoon.
With fixed thresholds the result depends on where the cut is placed ("cold" = less
than 8, 10, 12 or 14 degrees) and each day counts entirely or not at all; with fuzzy sets there
is no cut and each day satisfies the rule TO ITS DEGREE.
"""
import numpy as np
import pandas as pd

from ARMxtend.ARM import fuzzy_rule_degree, fuzzy_sequential_rules, sequential_rules
from ARMxtend.FIM import fuzzy_prefixspan, fuzzy_sequences_from_memberships
from ARMxtend.preprocessing import FuzzyLib


def daily_readings(days=80, seed=4):
    """
    Readings of ``days`` days, 8 per day. Each day has a cold intensity ``cold`` in [0, 1]:

    * temperature: 18 degrees, minus ``12 * cold`` in the first three readings (the morning);
    * power: 2 kW, plus ``6 * cold`` in the last three (the afternoon-evening).

    Returns the table of readings and the series of cold intensities per day.
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

    # 1. Fuzzify the readings and build a fuzzy sequence per day --------------------------------------
    library = FuzzyLib()
    library.data = readings.copy()
    names = library.AutoFuzzification(Atri=["temp", "power"], n_labels=3, method="uniform")
    sequences = fuzzy_sequences_from_memberships(library.data[names], by=library.data["day"], min_degree=0.2)
    print("{} days; fuzzy labels: {}".format(len(sequences), ", ".join(names)))

    # 2. Patrones: el orden importa -----------------------------------------------------------------------------
    patterns = fuzzy_prefixspan(sequences, 0.2, max_len=2, min_len=2)
    support = {tuple(next(iter(element)) for element in pattern): value
               for pattern, value in zip(patterns["sequence"], patterns["support"])}
    print("\nPatterns of two events (fuzzy support, {} in total):".format(len(patterns)))
    for pair in [("temp_low", "power_high"), ("power_high", "temp_low")]:
        print("  <{} -> {}>  support={}".format(pair[0], pair[1], "{:.2f}".format(support[pair]) if pair in support
                                                else "< 0.20 (no frecuente)"))

    # 3. Rules: the one linking the morning cold with the afternoon consumption stands out by its lift -----------------
    rules = fuzzy_sequential_rules(sequences, 0.1, min_conf=0.5, max_antecedent=1, max_consequent=1)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    print("\nMost striking fuzzy rules (by lift):")
    for _, rule in rules.head(3).iterrows():
        print("  {} => {}  support={:.2f} confidence={:.2f} lift={:.2f}".format(
            ", ".join(sorted(rule["antecedents"])), ", ".join(sorted(rule["consequents"])), rule["support"],
            rule["confidence"], rule["lift"]))

    # 4. Each day satisfies the rule to its degree, which follows the cold intensity -------------------------------
    degrees = pd.Series([fuzzy_rule_degree(sequence, ["temp_low"], ["power_high"]) for sequence in sequences], name="degree")
    correlation = float(np.corrcoef(cold, degrees)[0, 1])
    print("\ntemp_low => power_high per day: degree 0 on {} mild days, up to {:.2f} on the coldest; correlation with the "
          "cold intensity {:.2f}".format(int((degrees == 0).sum()), degrees.max(), correlation))

    # 5. Comparacion con umbrales fijos: cold = temp < corte, peak = power > 5 -------------------------------------
    fuzzy_rule = rules[(rules["antecedents"] == frozenset({"temp_low"})) & (rules["consequents"] == frozenset({"power_high"}))].iloc[0]
    print("\nWith fuzzy sets: temp_low => power_high  support={:.2f} confidence={:.2f} lift={:.2f}".format(
        fuzzy_rule["support"], fuzzy_rule["confidence"], fuzzy_rule["lift"]))
    crisp_rules = {}
    for cut in (8, 10, 12, 14):
        days = [[{label for label, holds in (("cold", row.temp < cut), ("peak", row.power > 5)) if holds}
                 for row in day.itertuples()] for _, day in readings.groupby("day")]
        found = sequential_rules(days, 0.05, min_conf=0.0, max_antecedent=1, max_consequent=1)
        crisp_rules[cut] = found[(found["antecedents"] == frozenset({"cold"})) & (found["consequents"] == frozenset({"peak"}))].iloc[0]
        print("With threshold temp < {:>2}:      cold => peak           support={:.2f} confidence={:.2f} lift={:.2f}".format(
            cut, crisp_rules[cut]["support"], crisp_rules[cut]["confidence"], crisp_rules[cut]["lift"]))

    return patterns, rules, degrees, cold, (crisp_rules, fuzzy_rule)


if __name__ == "__main__":
    main()
