# -*- coding: utf-8 -*-
"""Tests of ARM.quantitative_rules: partition into intervals and ranges, and quantitative rules."""
from itertools import combinations, product

import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import IntervalPartition, interesting_rules, partial_completeness_intervals, quantitative_rules, significant_rules


def random_table(seed, rows=150):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, 100, rows)
    return pd.DataFrame({"x": x, "y": x * 0.5 + rng.normal(0, 15, rows), "color": rng.choice(["red", "blue", "green"], rows),
                         "flag": rng.randint(0, 2, rows)})


def planted_table(rows=3000, seed=0):
    """Income is high only between 30 and 45 years; women with high income almost always buy."""
    rng = np.random.RandomState(seed)
    age = rng.uniform(18, 70, rows)
    income = np.where((age >= 30) & (age < 45), rng.normal(70, 8, rows), rng.normal(35, 10, rows)).clip(5)
    sex = rng.choice(["F", "M"], rows)
    buys = np.where((income > 60) & (sex == "F"), rng.rand(rows) < 0.9, rng.rand(rows) < 0.1)
    return pd.DataFrame({"age": age, "income": income, "sex": sex, "buys": np.where(buys, "yes", "no")})


# ------------------------------------------------------------------------------- particion

def test_partition_builds_equal_frequency_intervals_and_ranges():
    table = random_table(0)
    part = IntervalPartition(n_intervals=5, max_supp=0.5).fit(table)
    assert part.categorical_ == ["color", "flag"]                           # 'flag' tiene 2 valores: categorica
    assert set(part.edges_) == {"x", "y"} and len(part.edges_["x"]) == 6
    bases = part.intervals_[part.intervals_["n_base"] == 1]
    assert len(bases) == 10 and bases["support"].between(0.15, 0.25).all()        # 5 intervalos de ~20 %
    ranges = part.intervals_
    assert (ranges["support"][ranges["n_base"] > 1] <= 0.5 + 1e-9).all() and ranges["n_base"].max() == 2   # 3 intervalos pasan del 50 %
    assert len(ranges) == 2 * (5 + 4)
    first = ranges[(ranges["column"] == "x") & (ranges["n_base"] == 1)].iloc[0]
    assert first["item"].startswith("x=[") and first["item"].endswith(")") and first["low"] == pytest.approx(table["x"].min())
    last = ranges[(ranges["column"] == "x") & (ranges["n_base"] == 1)].iloc[-1]
    assert last["item"].endswith("]") and last["high"] == pytest.approx(table["x"].max())


def test_partition_taxonomy_nests_ranges_and_groups_them_by_column():
    part = IntervalPartition(n_intervals=4, max_supp=1.0, columns=["x"]).fit(random_table(1))
    items = part.intervals_.set_index(["n_base", "low"])["item"]
    base = part.intervals_[part.intervals_["n_base"] == 1]["item"].tolist()
    assert len(part.intervals_) == 10 and part.taxonomy_.depth == 4          # from one interval up to the full range
    whole = part.intervals_[part.intervals_["n_base"] == 4]["item"].iloc[0]
    assert part.taxonomy_.roots() == [whole] and part.taxonomy_.descendants(whole) == set(part.intervals_["item"]) - {whole}
    assert part.taxonomy_.ancestors(base[1]) >= {whole}
    wide = part.intervals_[part.intervals_["n_base"] == 2]
    for item in wide["item"]:
        assert len(part.taxonomy_.parents(item)) >= 1 and all(part.intervals_.set_index("item").loc[p, "n_base"] == 3 for p in part.taxonomy_.parents(item))
    assert set(part.groups_.values()) == {"x"} and set(part.groups_) == set(part.intervals_["item"])
    assert items is not None


def test_transform_maps_rows_to_base_intervals_categories_and_missing_values():
    table = pd.DataFrame({"v": np.arange(100.0), "c": ["a", "b"] * 50})
    part = IntervalPartition(n_intervals=4).fit(table)
    new = pd.DataFrame({"v": [0.0, 24.0, 25.0, 99.0, 500.0, -9.0, np.nan], "c": ["a", None, "b", "a", "b", "a", "a"]})
    baskets = part.transform(new)
    labels = part.intervals_[part.intervals_["n_base"] == 1]["item"].tolist()
    assert [b[0] for b in baskets[:6]] == [labels[0], labels[0], labels[1], labels[3], labels[3], labels[0]]   # those outside, to the end
    assert baskets[1] == [labels[0]] and baskets[6] == ["c=a"] and baskets[0] == [labels[0], "c=a"]
    assert part.transform(table.iloc[:0]) == []
    with pytest.raises(ValueError, match="fit"):
        IntervalPartition().transform(new)
    with pytest.raises(ValueError, match="missing"):
        part.transform(new[["c"]])


def test_partition_edge_cases_and_errors():
    constant = IntervalPartition(n_intervals=3, quantitative=["v"]).fit(pd.DataFrame({"v": [2.0] * 10}))
    assert constant.intervals_["item"].tolist() == ["v=[2,2]"] and constant.transform(pd.DataFrame({"v": [2.0]})) == [["v=[2,2]"]]
    ties = IntervalPartition(n_intervals=8).fit(pd.DataFrame({"v": [0.0] * 50 + list(np.arange(1.0, 51.0))}))
    assert len(ties.edges_["v"]) < 9                                         # repeated edges are merged
    negative = IntervalPartition(n_intervals=2).fit(pd.DataFrame({"t": np.arange(-10.0, 10.0)}))
    assert all("-" not in item for item in negative.intervals_["item"]) and any("−" in item for item in negative.intervals_["item"])
    uniform = IntervalPartition(n_intervals=4, method="uniform", quantitative=["v"]).fit(pd.DataFrame({"v": [0.0, 1.0, 1.0, 1.0, 10.0]}))
    assert uniform.edges_["v"].tolist() == [0.0, 2.5, 5.0, 7.5, 10.0]
    for bad in ({"n_intervals": 0}, {"method": "x"}, {"max_supp": 0}):
        with pytest.raises(ValueError):
            IntervalPartition(**bad)
    with pytest.raises(ValueError, match="nonexistent"):
        IntervalPartition(columns=["nope"]).fit(random_table(0))
    with pytest.raises(ValueError, match="numeric"):
        IntervalPartition(quantitative=["color"]).fit(random_table(0))
    with pytest.raises(ValueError, match="there are no columns"):
        IntervalPartition(columns=[]).fit(random_table(0))
    with pytest.raises(ValueError, match="has no values"):
        IntervalPartition(quantitative=["v"]).fit(pd.DataFrame({"v": [np.nan, np.nan]}))


def test_partial_completeness_intervals_follow_the_formula():
    assert partial_completeness_intervals(2, 0.05, 1.5) == 160                # 2 n / (m (K - 1))
    assert partial_completeness_intervals(1, 0.1, 2.0) == 20 and partial_completeness_intervals(3, 0.3, 1.5) == 40
    for bad in ((0, 0.1, 1.5), (1, 0, 1.5), (1, 0.1, 1.0)):
        with pytest.raises(ValueError):
            partial_completeness_intervals(*bad)


# ----------------------------------------------------------------------------------- reglas

def brute_force_itemsets(table, part, min_supp, max_len):
    """Support of all the sets of (one range per numeric attribute | one value per categorical) with support >= min_supp."""
    rows = len(table)
    candidates = {}
    for name, edges in part.edges_.items():
        interior = edges[1:-1]
        index = np.array([int((interior <= v).sum()) for v in table[name]])
        options = []
        for _, item in part.intervals_[part.intervals_["column"] == name].iterrows():
            first = int(np.searchsorted(edges, item["low"]))
            last = int(np.searchsorted(edges, item["high"])) - 1
            options.append((item["item"], (index >= first) & (index <= last)))
        candidates[name] = options
    for name in part.categorical_:
        candidates[name] = [("{}={}".format(name, value), (table[name] == value).to_numpy()) for value in sorted(table[name].unique())]
    found = {}
    for size in range(1, max_len + 1):
        for names in combinations(sorted(candidates), size):
            for choice in product(*(candidates[name] for name in names)):
                mask = np.logical_and.reduce([m for _, m in choice])
                if mask.sum() / rows >= min_supp - 1e-9:
                    found[frozenset(item for item, _ in choice)] = mask.sum() / rows
    return found


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n_intervals,max_supp", [(4, 0.5), (6, 0.35)])
def test_frequent_itemsets_equal_the_brute_force_enumeration(seed, n_intervals, max_supp):
    table = random_table(seed)
    part = IntervalPartition(n_intervals, max_supp=max_supp).fit(table)
    rules = quantitative_rules(table, min_supp=0.12, partition=part, min_threshold=0.0, max_len=3)
    expected = brute_force_itemsets(table, part, 0.12, 3)
    assert len(expected) > 20 and rules.attrs["supports"] == pytest.approx(expected)
    # no rule mentions two ranges of the same attribute
    for antecedent, consequent in zip(rules["antecedents"], rules["consequents"]):
        columns = [part.groups_[i] for i in antecedent | consequent if i in part.groups_]
        assert len(columns) == len(set(columns))


def test_planted_region_is_recovered_and_the_interest_pruning_keeps_it():
    table = planted_table()
    rules = quantitative_rules(table, min_supp=0.05, n_intervals=10, max_supp=0.4, min_threshold=0.7, max_len=3)
    part = rules.attrs["partition"]
    assert isinstance(part, IntervalPartition) and rules.attrs["n_transactions"] == len(table)

    def bounds(item):
        row = part.intervals_.set_index("item").loc[item]
        return row["low"], row["high"]

    one_to_one = rules[(rules["antecedents"].map(len) == 1) & (rules["consequents"].map(len) == 1)]
    age_income = one_to_one[one_to_one["antecedents"].map(lambda a: next(iter(a)).startswith("age="))
                            & one_to_one["consequents"].map(lambda c: next(iter(c)).startswith("income="))]
    assert len(age_income) > 0
    low, high = bounds(next(iter(age_income.sort_values("lift", ascending=False).iloc[0]["antecedents"])))
    assert 27 <= low and high <= 50                                          # the one with the highest lift falls inside the planted region [30, 45)
    widest = age_income[age_income["confidence"] >= 0.85].sort_values("support", ascending=False)
    low, high = bounds(next(iter(widest.iloc[0]["antecedents"])))
    assert 27 <= low <= 36 and 40 <= high <= 50                              # the one with the most support covers it almost entirely
    women = rules[rules["antecedents"].map(lambda a: "sex=F" in a) & (rules["consequents"] == frozenset({"buys=yes"}))]
    assert len(women) > 0 and women["confidence"].max() > 0.85

    kept = quantitative_rules(table, min_supp=0.05, n_intervals=10, max_supp=0.4, min_threshold=0.7, max_len=3, interest=1.3)
    assert 0 < len(kept) < len(rules) / 2 and "interest" in kept.columns
    assert set(zip(kept["antecedents"], kept["consequents"])) <= set(zip(rules["antecedents"], rules["consequents"]))
    assert len(interesting_rules(rules, interest=1.3)) == len(kept)


def test_quantitative_rules_use_the_partition_on_new_data_and_downstream_tools():
    table = planted_table()
    part = IntervalPartition(8, max_supp=0.4).fit(table.iloc[:2000])
    rules = quantitative_rules(table.iloc[2000:], min_supp=0.05, partition=part, min_threshold=0.7, max_len=2)
    assert rules.attrs["partition"] is part and rules.attrs["n_transactions"] == 1000 and len(rules) > 5
    assert len(significant_rules(rules, 0.05, n_transactions=1000)) <= len(rules)
    assert len(quantitative_rules(table, min_supp=0.99)) == 0
    with pytest.raises(ValueError):
        quantitative_rules(table, min_supp=0.1, interest=0.5)
    with pytest.raises(ValueError):
        quantitative_rules(table, min_supp=0.1, columns=["nope"])
