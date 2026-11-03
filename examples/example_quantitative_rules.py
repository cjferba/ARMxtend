# -*- coding: utf-8 -*-
"""
Example: quantitative rules by intervals (Srikant and Agrawal, 1996) versus
fuzzy rules. A shop has the age, income and sex of its customers, and whether
they bought a product. Income is high only between 30 and 45 years old, and
women with high income almost always buy. The equal-frequency base intervals
are merged into ranges up to a maximum support, and R-interesting pruning discards the
ranges that can be widened or narrowed without changing the rule.
"""
import numpy as np
import pandas as pd

from ARMxtend.ARM import IntervalPartition, partial_completeness_intervals, quantitative_rules
from ARMxtend.FFIM import fuzzy_fpgrowth
from ARMxtend.FIM.FARE import fuzzy_association_rules
from ARMxtend.preprocessing import FuzzyLib


def customers(rows=3000, seed=0):
    """
    Age uniform in [18, 70); income N(70, 8) if the age is in [30, 45) and N(35, 10) otherwise (minimum 5);
    sex at random; 90 % of the women with income > 60 buy and 10 % of the rest.
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
    """Rules from one item to one item between two attributes."""
    first = rules["antecedents"].map(lambda items: len(items) == 1 and next(iter(items)).startswith(antecedent_prefix))
    second = rules["consequents"].map(lambda items: len(items) == 1 and next(iter(items)).startswith(consequent_prefix))
    return rules[first & second]


def main():
    table = customers()
    print("{} customers. With partial completeness K=1.5 and support 0.05 the paper asks for {} intervals per attribute; "
          "here 10 and ranges of up to 40 % are enough.".format(len(table), partial_completeness_intervals(2, 0.05, 1.5)))

    # 1. Intervalos base y rangos --------------------------------------------------------------------------------
    partition = IntervalPartition(n_intervals=10, max_supp=0.4).fit(table)
    ranges = partition.intervals_
    print("Partition: {} range items ({} base intervals + {} ranges of several); age in {} intervals of ~{:.1f} years".format(
        len(ranges), int((ranges["n_base"] == 1).sum()), int((ranges["n_base"] > 1).sum()),
        len(partition.edges_["age"]) - 1, float(np.diff(partition.edges_["age"]).mean())))

    # 2. Rules, and the R-interesting ones -----------------------------------------------------------------------------
    rules = quantitative_rules(table, min_supp=0.05, partition=partition, min_threshold=0.7, max_len=3)
    kept = quantitative_rules(table, min_supp=0.05, partition=partition, min_threshold=0.7, max_len=3, interest=1.3)
    print("\n{} rules (support 0.05, confidence 0.7); R-interesting with R=1.3: {}".format(len(rules), len(kept)))

    age_income = one_to_one(rules, "age=", "income=")
    region = age_income[age_income["confidence"] >= 0.85].sort_values("support", ascending=False).head(1)
    print("\nThe planted region [30, 45), with the highest support with confidence >= 0.85:")
    for line in describe(region):
        print("  " + line)

    buyers = rules[(rules["consequents"] == frozenset({"buys=yes"})) & rules["antecedents"].map(lambda items: "sex=F" in items)]
    print("\nQuien compra (mujeres, por lift):")
    for line in describe(buyers.sort_values("lift", ascending=False).head(3)):
        print("  " + line)

    # 3. Versus fuzzy rules with fixed labels (5 per attribute, equal frequency) -------------------
    library = FuzzyLib()
    library.data = table[["age", "income"]].copy()
    library.AutoFuzzification(Atri=["age", "income"], n_labels=5, method="quantile")
    frequent = fuzzy_fpgrowth(library.GetFuzzyTransactions(), 0.02, 10)
    fuzzy = fuzzy_association_rules(frequent, 10, metric="confidence", min_threshold=0.4, measures=["lift"])
    fuzzy_age_income = one_to_one(fuzzy, "age_", "income_").sort_values("confidence", ascending=False)
    print("\nFuzzy rules age => income (5 fixed labels), by confidence:")
    for line in describe(fuzzy_age_income.head(3)):
        print("  " + line)
    print("The region [30, 45) straddles two labels and none reaches the confidence of the merged range.")

    return table, partition, rules, kept, region, fuzzy_age_income


if __name__ == "__main__":
    main()
