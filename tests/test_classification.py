# -*- coding: utf-8 -*-
"""
Tests of ``ARM.classification``: the class rules against a brute-force enumeration, the
CBA selection against hand-computed runs, and the classifier on data with planted structure.
"""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import RuleClassifier, class_association_rules, select_rules, significant_rules


def brute_force(items, labels, min_supp, min_conf, max_len, per_class=True):
    """All the rules A => class with enough support and confidence, counting with integers."""
    n = len(items)
    matrix = items.to_numpy(bool)
    found, candidates = {}, 0
    for label in sorted(set(labels)):
        inClass = np.asarray(labels) == label
        size = int(inClass.sum())
        needed = (min_supp * size if per_class else min_supp * n) - 1e-9
        for length in range(1, max_len + 1):
            for columns in combinations(range(items.shape[1]), length):
                holds = matrix[:, columns].all(axis=1)
                joint, antecedent = int((holds & inClass).sum()), int(holds.sum())
                if joint >= max(needed, 1e-9):
                    candidates += 1
                    if joint / antecedent >= min_conf - 1e-9:
                        found[(frozenset(items.columns[list(columns)]), label)] = (joint, antecedent, size)
    return found, candidates


def random_cases(seed, rows=120, columns=6):
    rng = np.random.RandomState(seed)
    items = pd.DataFrame(rng.rand(rows, columns) < 0.45, columns=list("abcdef")[:columns])
    labels = np.where(items["a"] & items["b"], "x", np.where(rng.rand(rows) < 0.3, "y", "z"))
    return items, pd.Series(labels, name="kind")


@pytest.mark.parametrize("support", ["class", "global"])
@pytest.mark.parametrize("seed", [0, 1])
def test_class_rules_equal_the_brute_force_enumeration(seed, support):
    items, labels = random_cases(seed)
    rules = class_association_rules(items, labels, min_supp=0.15, min_conf=0.5, max_len=3, support=support)
    expected, candidates = brute_force(items, labels, 0.15, 0.5, 3, per_class=support == "class")
    assert len(expected) > 5
    assert len(rules) == len(expected)
    n = len(items)
    for _, rule in rules.iterrows():
        joint, antecedent, size = expected[(rule["antecedents"], rule["class"])]
        assert rule["consequents"] == frozenset(["kind={}".format(rule["class"])])
        assert rule["support"] == pytest.approx(joint / n)
        assert rule["antecedent support"] == pytest.approx(antecedent / n)
        assert rule["consequent support"] == pytest.approx(size / n)
        assert rule["confidence"] == pytest.approx(joint / antecedent)
        assert rule["lift"] == pytest.approx((joint / antecedent) / (size / n))
    assert rules.attrs["n_transactions"] == n and rules.attrs["n_candidates"] == candidates


def test_class_rules_are_sorted_by_precedence():
    items, labels = random_cases(3)
    rules = class_association_rules(items, labels, min_supp=0.1, min_conf=0.4, max_len=3)
    key = list(zip(-rules["confidence"].round(12), -rules["support"].round(12), rules["antecedents"].map(len)))
    assert key == sorted(key)


def test_class_support_keeps_the_rules_of_a_rare_class():
    rows = 1000
    rare = np.arange(rows) < 20
    items = pd.DataFrame({"r": rare, "noise": np.arange(rows) % 3 == 0})
    labels = np.where(rare, "rare", "common")
    assert any(rule == frozenset(["r"]) for rule in class_association_rules(
        items, labels, min_supp=0.5, min_conf=0.99, support="class")["antecedents"])
    assert class_association_rules(items, labels, min_supp=0.05, min_conf=0.99, support="global").empty


def test_class_rules_input_errors():
    items, labels = random_cases(0)
    with pytest.raises(ValueError, match="support"):
        class_association_rules(items, labels, support="both")
    with pytest.raises(ValueError, match="max_len"):
        class_association_rules(items, labels, max_len=0)
    with pytest.raises(ValueError, match="rows"):
        class_association_rules(items, labels[:-1])
    with pytest.raises(ValueError, match="'-'"):
        class_association_rules(items.rename(columns={"a": "a-1"}), labels)
    with pytest.raises(ValueError, match="null values"):
        class_association_rules(items, [None] + list(labels[1:]))
    assert class_association_rules(items, labels, class_name="y")["consequents"].map(
        lambda c: next(iter(c)).startswith("y=")).all()


def rule_list(*rules):
    """(antecedent, class, confidence, support) as a DataFrame with the format of class_association_rules."""
    return pd.DataFrame({"antecedents": [frozenset(a) for a, *_ in rules], "class": [c for _, c, *_ in rules],
                         "confidence": [p for *_, p, _ in rules], "support": [s for *_, s in rules]})


def table(rows, columns=("a", "b")):
    return pd.DataFrame([[item in row for item in columns] for row, _ in rows], columns=list(columns)), \
        [label for _, label in rows]


def test_select_rules_hand_computed_cba_run():
    # 8 cases; R1 {a,b} => X covers 0 and 1; R2 {a} => X covers 2 (X) and 3 (Y); R3 {b} => Y covers 4 and 5
    items, labels = table([({"a", "b"}, "X"), ({"a", "b"}, "X"), ({"a"}, "X"), ({"a"}, "Y"), ({"b"}, "Y"),
                           ({"b"}, "Y"), (set(), "Y"), (set(), "Y")])
    rules = rule_list(({"a", "b"}, "X", 1.0, 0.25), ({"a"}, "X", 0.75, 0.375), ({"b"}, "Y", 0.5, 0.25))
    kept, default = select_rules(rules, items, labels)
    # total errors after R1, R2 and R3: 1, 1 and 1 -> cut at the first (the shortest list)
    assert [set(a) for a in kept["antecedents"]] == [{"a", "b"}] and default == "Y"


def test_select_rules_skips_rules_that_cover_nothing_new_or_only_wrong_cases():
    # X: 0-4, 9, 10; Y: 5-8. After R1, 9 cases remain and the default (X) fails 4 (total 4); after R4, 3
    items, labels = table([({"a", "b"}, "X")] * 2 + [({"a"}, "X")] * 3 + [({"a"}, "Y")] + [({"b"}, "Y")] * 3 +
                          [(set(), "X")] * 2)
    rules = rule_list(({"a", "b"}, "X", 1.0, 0.18),
                      ({"a", "b"}, "Y", 0.9, 0.1),      # R1 already covered its cases
                      ({"b"}, "X", 0.85, 0.2),           # covers cases 6-8, all Y: none right
                      ({"a"}, "X", 0.8, 0.4))
    kept, default = select_rules(rules, items, labels)
    assert [set(a) for a in kept["antecedents"]] == [{"a", "b"}, {"a"}] and default == "Y"
    assert kept["class"].tolist() == ["X", "X"]


def test_select_rules_cuts_where_the_total_error_is_lowest():
    # X: cases 0,1,2,7,8,9; Y: 3,4,5,6. R1 {a} => X (0-3); R2 {b} => Y (4-7)
    items, labels = table([({"a"}, "X")] * 3 + [({"a"}, "Y")] + [({"b"}, "Y")] * 3 + [({"b"}, "X")] +
                          [(set(), "X")] * 2)
    rules = rule_list(({"a"}, "X", 0.75, 0.3), ({"b"}, "Y", 0.75, 0.3))
    kept, default = select_rules(rules, items, labels)
    # after R1: 1 rule error + 3 from the default = 4; after R2: 2 errors and a perfect default = 2 -> both
    assert [set(a) for a in kept["antecedents"]] == [{"a"}, {"b"}] and default == "X"
    # if the second were X (worse), the total error would rise from 2 to 3 and it is cut after the first
    items2, labels2 = table([({"a"}, "X")] * 4 + [({"a"}, "Y")] + [({"c"}, "Y")] * 2 + [({"c"}, "X")],
                            columns=("a", "c"))
    rules2 = rule_list(({"a"}, "X", 0.8, 0.5), ({"c"}, "X", 0.4, 0.125))
    kept2, default2 = select_rules(rules2, items2, labels2)
    assert [set(a) for a in kept2["antecedents"]] == [{"a"}] and default2 == "Y"


def test_select_rules_without_useful_rules_uses_the_majority_class():
    items, labels = table([({"a"}, "X"), ({"a"}, "Y"), (set(), "Y")])
    kept, default = select_rules(rule_list(({"a"}, "Z", 0.9, 0.3)).iloc[:0], items, labels)
    assert kept.empty and default == "Y"
    assert select_rules(rule_list(({"a"}, "Z", 0.9, 0.3)), items, labels)[0].empty      # class Z does not exist


def planted(rows, seed):
    rng = np.random.RandomState(seed)
    items = pd.DataFrame(rng.rand(rows, 8) < 0.4, columns=["x{}".format(i) for i in range(8)])
    labels = np.where(items["x0"] & items["x1"], "A", np.where(items["x2"] & items["x3"], "B", "C"))
    flip = rng.rand(rows) < 0.05
    return items, pd.Series(np.where(flip, rng.choice(["A", "B", "C"], rows), labels), name="kind")


def test_classifier_recovers_the_planted_rules_and_beats_the_majority_class():
    train_items, train_labels = planted(1500, 1)
    test_items, test_labels = planted(1000, 2)
    model = RuleClassifier(min_supp=0.05, min_conf=0.7, max_len=2).fit(train_items, train_labels)
    antecedents = [set(a) for a in model.rules_["antecedents"]]
    assert {"x0", "x1"} in antecedents and {"x2", "x3"} in antecedents
    assert model.default_class_ == "C" and model.classes_ == ["A", "B", "C"]
    assert model.score(test_items, test_labels) > 0.9 > (test_labels == "C").mean() + 0.2
    assert len(model.rules_) <= len(model.candidate_rules_)


def test_explain_reports_the_deciding_rule_and_preserves_the_index():
    items, labels = planted(600, 3)
    model = RuleClassifier(min_supp=0.05, min_conf=0.7, max_len=2).fit(items, labels)
    shifted = items.set_axis(range(100, 100 + len(items)))
    explanation = model.explain(shifted)
    assert explanation.index.equals(shifted.index) and model.predict(shifted).equals(explanation["prediction"].rename("class"))
    fired = explanation[explanation["rule"] >= 0]
    assert fired["confidence"].notna().all() and explanation[explanation["rule"] < 0]["confidence"].isna().all()
    assert (explanation[explanation["rule"] < 0]["prediction"] == model.default_class_).all()
    first = fired.iloc[0]
    rule = model.rules_.iloc[int(first["rule"])]
    assert first["prediction"] == rule["class"] and first["confidence"] == rule["confidence"]
    assert shifted.loc[fired.index[0], sorted(rule["antecedents"])].all()
    # an earlier rule of the list does not hold in that row: the first one that holds is applied
    for earlier in model.rules_.iloc[:int(first["rule"])]["antecedents"]:
        assert not shifted.loc[fired.index[0], sorted(earlier)].all()


def test_numeric_labels_keep_their_type_and_unfitted_or_bad_inputs_are_rejected():
    items, labels = planted(400, 4)
    numeric = labels.map({"A": 1, "B": 2, "C": 3})
    model = RuleClassifier(min_supp=0.05, min_conf=0.7, max_len=2).fit(items, numeric)
    assert model.predict(items).dtype.kind == "i" and model.score(items, numeric) > 0.85
    with pytest.raises(ValueError, match="not trained"):
        RuleClassifier().predict(items)
    with pytest.raises(ValueError, match="no cases to train"):
        RuleClassifier().fit(items.iloc[:0], [])
    with pytest.raises(ValueError, match="is not a column"):
        model.predict(items.drop(columns=["x0", "x1", "x2", "x3"]))
    with pytest.raises(ValueError, match="rows"):
        model.score(items, numeric[:-1])


def test_pruning_unproductive_rules_gives_a_shorter_list_with_similar_accuracy():
    train_items, train_labels = planted(1500, 5)
    test_items, test_labels = planted(1000, 6)
    plain = RuleClassifier(min_supp=0.03, min_conf=0.7, max_len=3).fit(train_items, train_labels)
    pruned = RuleClassifier(min_supp=0.03, min_conf=0.7, max_len=3, min_improvement=0.05).fit(train_items, train_labels)
    assert len(pruned.rules_) < len(plain.rules_)
    assert pruned.score(test_items, test_labels) >= plain.score(test_items, test_labels) - 0.02


def test_class_rules_work_with_significance_testing():
    rng = np.random.RandomState(5)
    items = pd.DataFrame(rng.rand(300, 10) < 0.4, columns=["i{}".format(i) for i in range(10)])
    noise = class_association_rules(items, rng.choice(["p", "q"], 300), min_supp=0.05, min_conf=0.52, max_len=2)
    assert len(noise) > 20 and significant_rules(noise, alpha=0.05).empty                  # ninguna es real
    signal = np.where(items["i0"] & items["i1"] & (rng.rand(300) < 0.95), "p", "q")
    real = significant_rules(class_association_rules(items, signal, min_supp=0.05, min_conf=0.8, max_len=2), alpha=0.05)
    assert {"i0", "i1"} in [set(a) for a in real["antecedents"]]


def test_per_class_thresholds_and_min_count_equal_the_brute_force():
    items, labels = random_cases(3)
    supp = {"x": 0.30, "y": 0.10, "z": 0.20}
    conf = {"x": 0.7, "y": 0.4, "z": 0.5}
    rules = class_association_rules(items, labels, min_supp=supp, min_conf=conf, max_len=3, min_count=4)
    matrix, array = items.to_numpy(bool), np.asarray(labels)
    expected = set()
    for label in supp:
        inClass = array == label
        for length in (1, 2, 3):
            for columns in combinations(range(items.shape[1]), length):
                holds = matrix[:, columns].all(axis=1)
                joint = int((holds & inClass).sum())
                if joint >= max(supp[label] * inClass.sum() - 1e-9, 4) and joint / holds.sum() >= conf[label] - 1e-9:
                    expected.add((frozenset(items.columns[list(columns)]), label))
    assert len(expected) > 3
    assert set(zip(rules["antecedents"], rules["class"])) == expected
    classifier = RuleClassifier(min_supp=supp, min_conf=conf, max_len=3, min_count=4).fit(items, labels)
    assert classifier.score(items, labels) > 0.5
    assert classifier.get_params()["min_count"] == 4


def test_per_class_thresholds_validation():
    items, labels = random_cases(0)
    with pytest.raises(ValueError, match="no value for the classes"):
        class_association_rules(items, labels, min_supp={"x": 0.1})
    with pytest.raises(ValueError, match="min_conf must be in"):
        class_association_rules(items, labels, min_conf={"x": 0.5, "y": 2, "z": 0.5})
    with pytest.raises(ValueError, match="min_count"):
        class_association_rules(items, labels, min_count=0)
