# -*- coding: utf-8 -*-
"""
Runs each script in ``examples/`` as part of the test suite, so that
continuous integration (see .github/workflows/ci.yml) detects whether a
change in the library breaks any of the examples documented in
docs/sources/examples.md.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

EXAMPLE_MODULES = [
    "example_association_rules",
    "example_fuzzy_association_rules",
    "example_meta_rules",
    "example_preprocessing",
    "example_vizarm",
    "example_rule_types",
    "example_temporal_rules",
    "example_sequences",
    "example_significance",
    "example_rules_in_action",
    "example_fuzzy_sequences",
    "example_generalized_rules",
    "example_quantitative_rules",
    "example_special_rules",
]


@pytest.fixture(autouse=True, scope="module")
def _examples_on_path():
    sys.path.insert(0, str(EXAMPLES_DIR))
    yield
    sys.path.remove(str(EXAMPLES_DIR))


@pytest.mark.parametrize("module_name", EXAMPLE_MODULES)
def test_example_runs_without_error(module_name, capsys):
    module = importlib.import_module(module_name)
    result = module.main()

    assert result is not None
    captured = capsys.readouterr()
    assert len(captured.out.strip()) > 0


def test_fuzzy_example_reproduces_paper_values(capsys):
    """
    example_fuzzy_association_rules.py must reproduce exactly the
    FSupp/FConf/FCF values published for {i1,i3} -> {i4} in Delgado,
    Ruiz, Sanchez & Serrano (2011) / Ruiz et al. (2016, Section 4.4).
    """
    module = importlib.import_module("example_fuzzy_association_rules")
    _, rulesDf = module.main()

    rule = next(
        row for _, row in rulesDf.iterrows()
        if set(row["antecedents"]) == {"i1", "i3"} and set(row["consequents"]) == {"i4"}
    )
    assert rule["support"] == pytest.approx(0.266, abs=0.002)
    assert rule["confidence"] == pytest.approx(0.5, abs=0.002)
    assert rule["certainty_factor"] == pytest.approx(0.33, abs=0.002)


def test_rule_types_example_finds_the_planted_patterns():
    module = importlib.import_module("example_rule_types")
    negative, exceptions, anomalies, gradual, (closed, maximal, pruned), fuzzyRules = module.main()

    # quien toma cafe no toma te
    assert ((negative["antecedents"] == frozenset(["coffee"])) & (negative["consequents"] == frozenset(["~tea"]))).any()
    # la excepcion plantada a "fever -> rest"
    assert ((exceptions["antecedents"] == frozenset(["fever"])) & (exceptions["condition"] == frozenset(["must_work"]))).any()
    assert frozenset(["hospital"]) in set(anomalies["anomaly"])
    # the more age and the less exercise, the more cholesterol
    planted = (gradual["antecedents"] == frozenset([("age", "+"), ("exercise", "-")])) \
        & (gradual["consequents"] == frozenset([("cholesterol", "+")]))
    assert planted.any()
    assert set(maximal) == {"a-b-c", "d"} and len(pruned) == 5
    assert ((fuzzyRules["antecedents"] == frozenset(["age_low"]))
            & (fuzzyRules["consequents"] == frozenset(["cholesterol_low"]))).any()


def test_temporal_example_recovers_the_planted_behaviour(capsys):
    """The simulated house has an evening rule (door -> heating) and a rule that emerges in the second half."""
    module = importlib.import_module("example_temporal_rules")
    _, rules, profile, drift = module.main()

    evening = rules[rules["antecedents"].apply(lambda a: {"door_open@lag1", "part_of_day=evening"} <= set(a))]
    assert len(evening) > 0 and evening["confidence"].max() > 0.9
    assert profile.loc[10, "confidence"] == 0.0 and profile.loc[17:23, "confidence"].min() > 0.7
    emerging = drift[(drift["antecedents"] == frozenset({"temp=low@lag1"}))
                     & (drift["consequents"] == frozenset({"heater_on"}))]
    assert len(emerging) == 1 and emerging.iloc[0]["status"] == "emerging" and emerging.iloc[0]["first_window"] == 2


def test_sequences_example_recovers_the_planted_routine(capsys):
    """Quiet days have coffee and then television; the phone is noise and generates no informative rules."""
    module = importlib.import_module("example_sequences")
    sequences, patterns, rules = module.main()

    assert len(sequences) == 120
    kitchen = frozenset(["alarm"]), frozenset(["bathroom"]), frozenset(["kitchen"])
    assert any(tuple(pattern[:3]) == kitchen for pattern in patterns["sequence"])
    rule = rules[(rules["antecedents"] == frozenset(["coffee"])) & (rules["consequents"] == frozenset(["tv"]))].iloc[0]
    assert rule["lift"] > 2 and rule["confidence"] > 0.7
    assert rules[(rules["antecedents"] == frozenset(["tv"])) & (rules["consequents"] == frozenset(["coffee"]))].empty
    informative = rules[(rules["confidence"] >= 0.7) & (rules["lift"] > 1.1)]
    assert set(informative["consequents"].apply(lambda y: next(iter(y)))) == {"tv"}


def test_significance_example_removes_the_noise_rules(capsys):
    """Almost all the rules that pass the thresholds are noise; after correcting, those of the real association remain."""
    module = importlib.import_module("example_significance")
    rules, fisher, permuted, (pairs, fisher_alarms, shift_alarms) = module.main()

    union = lambda frame: [a | c for a, c in zip(frame["antecedents"], frame["consequents"])]   # noqa: E731
    noise = lambda frame: sum(not {"i00", "i01"} <= items for items in union(frame))              # noqa: E731
    assert len(rules) == 100 and noise(rules) > 40
    assert 0 < len(fisher) < len(rules) / 2 and noise(fisher) <= 3
    best = permuted[permuted["p_permutation"] <= 0.05]
    assert len(best) > 0 and noise(best) == 0
    assert pairs >= 10 and fisher_alarms >= 5 and shift_alarms <= 2


def test_rules_in_action_example_finds_predicts_and_classifies(capsys):
    """The rules mined from normal operation find the planted faults, predict and classify."""
    module = importlib.import_module("example_rules_in_action")
    rules, (flagged, fault), (quality, upcoming), (hits, after_coffee), (classifier, accuracy, majority) = module.main()

    assert {frozenset(a) for a in rules["antecedents"]} == {frozenset({"motion@lag1"}), frozenset({"door@lag1"})}
    assert int(fault.loc[flagged.index].sum()) == 30                                  # the 30 faults, among 39 instants
    assert len(flagged) < 45
    assert quality.loc["light", "precision"] > 0.9 and quality.loc["chime", "precision"] == pytest.approx(0.8, abs=0.08)
    assert "light" in upcoming.index
    assert after_coffee.index[0] == "tv" and hits["hit_rate"] > hits["baseline_hit_rate"] + 0.05
    assert accuracy > majority + 0.3 and len(classifier.rules_) <= 4


def test_fuzzy_sequences_example_orders_events_and_grades_the_rule(capsys):
    """The fuzzy rule 'cold in the morning => consumption in the afternoon' stands out, only in that order, and each day satisfies it to its degree."""
    module = importlib.import_module("example_fuzzy_sequences")
    patterns, rules, degrees, cold, (crisp_rules, fuzzy_rule) = module.main()

    pairs = {tuple(next(iter(element)) for element in pattern): value for pattern, value in zip(patterns["sequence"], patterns["support"])}
    assert pairs[("temp_low", "power_high")] > 0.2 and ("power_high", "temp_low") not in pairs
    assert (rules.iloc[0]["antecedents"], rules.iloc[0]["consequents"]) == (frozenset({"temp_low"}), frozenset({"power_high"}))
    assert rules.iloc[0]["lift"] > 2 > rules.iloc[1]["lift"]
    assert fuzzy_rule["lift"] == pytest.approx(rules.iloc[0]["lift"])
    assert float(np.corrcoef(cold, degrees)[0, 1]) > 0.8 and 0 < degrees.max() < 1 and (degrees == 0).sum() > 20
    supports = [rule["support"] for rule in crisp_rules.values()]                       # the cut changes the support a lot
    assert max(supports) > 2.5 * min(supports)


def test_generalized_rules_example_sees_categories_and_drops_their_echoes(capsys):
    """Without a taxonomy only product rules come out; with it, the category ones, and the R-interesting ones remove those that repeat them."""
    module = importlib.import_module("example_generalized_rules")
    taxonomy, flat, rules, kept, dropped = module.main()

    categories = {item for item in taxonomy.items if taxonomy.descendants(item)}
    assert not any((a | c) & categories for a, c in zip(flat["antecedents"], flat["consequents"]))
    keys = lambda frame: {(tuple(sorted(a)), tuple(sorted(c))) for a, c in zip(frame["antecedents"], frame["consequents"])}   # noqa: E731
    assert (("Bakery",), ("Dairy",)) in keys(rules) and (("Bakery",), ("Dairy",)) in keys(kept)
    assert (("croissant",), ("Juice",)) in keys(kept) and (("croissant",), ("orange_juice",)) in keys(dropped)
    assert (("white_bread",), ("Dairy",)) in keys(dropped) and (("milk",), ("Bakery",)) in keys(dropped)
    assert len(kept) + len(dropped) == len(rules) and 0 < len(kept) < len(rules)


def test_quantitative_rules_example_recovers_the_region_and_beats_fixed_labels(capsys):
    """The merged range recovers the planted region [30, 45); fixed fuzzy labels split it in two and lose confidence."""
    module = importlib.import_module("example_quantitative_rules")
    table, partition, rules, kept, region, fuzzy = module.main()

    bounds = partition.intervals_.set_index("item").loc[next(iter(region.iloc[0]["antecedents"])), ["low", "high"]]
    assert 27 <= bounds["low"] <= 36 and 40 <= bounds["high"] <= 50 and region.iloc[0]["confidence"] >= 0.85
    assert 0 < len(kept) < len(rules) / 2
    women = rules[(rules["consequents"] == frozenset({"buys=yes"})) & rules["antecedents"].map(lambda items: "sex=F" in items)]
    assert women["confidence"].max() > 0.85 and women["lift"].max() > 3.5
    assert fuzzy.iloc[0]["confidence"] < 0.6 < region.iloc[0]["confidence"]


def test_special_rules_example_finds_rare_pairs_and_what_is_worth_more(capsys):
    """Top-k without a threshold, the rare pairs that a global threshold drowns in noise, and utility versus frequency."""
    module = importlib.import_module("example_special_rules")
    quantities, top, informative, pairs, mri, rare, flat, by_support, useful, value_rules = module.main()

    key = lambda frame: [(tuple(sorted(a)), tuple(sorted(c))) for a, c in zip(frame["antecedents"], frame["consequents"])]   # noqa: E731
    assert len(top) == 5 and (top["lift"] < 1.1).sum() >= 2 and top["support"].is_monotonic_decreasing
    assert (informative["lift"] >= 1.5).all() and key(informative)[:3] == [(("diapers",), ("wipes",)), (("wipes",), ("diapers",)),
                                                                           (("wine",), ("cheese",))]
    assert len(pairs) == 5 and pairs["itemsets"].map(len).min() == 2

    assert set(key(rare)) == {(("diapers",), ("wipes",)), (("wipes",), ("diapers",)), (("wine",), ("cheese",))}
    assert (rare["support"] < 0.1).all() and len(mri) > len(rare) and int((flat["lift"] >= 2).sum()) > 10 * len(rare)

    assert [next(iter(s)) for s in by_support["itemsets"]] == ["bread", "milk", "fruit"]
    assert useful.iloc[0]["itemsets"] == frozenset({"cheese"}) and frozenset({"cheese", "wine"}) in set(useful["itemsets"])
    assert not set(by_support["itemsets"]) & set(useful.head(3)["itemsets"])
    assert useful.iloc[0]["utility"] > 0.1 * useful.attrs["total_utility"] and (useful["utility"] >= 0.1 * useful.attrs["total_utility"] - 1e-9).all()
    assert key(value_rules)[0] == (("wine",), ("cheese",)) and (value_rules["lift"] >= 1.5).all()
    assert value_rules.iloc[0]["utility_share"] == pytest.approx(0.27, abs=0.03)


def test_estimators_example_recovers_patterns_and_classifies():
    """Rules with RuleMiner on synthetic data, rule attributes and a classifier."""
    module = importlib.import_module("example_estimators")
    patterns, recovered, miner, features, top, accuracy, tennis, weather = module.main()

    assert 0 < len(recovered) <= len(patterns) and features.shape == (2000, len(miner.rules_))
    assert len(top) == 5 and top["support"].is_monotonic_decreasing and accuracy > 0.9
    assert tennis.score(weather.data, weather.target) >= 0.9
