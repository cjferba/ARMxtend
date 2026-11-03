# -*- coding: utf-8 -*-
"""Tests of ARM.fuzzy_classification: fuzzy class rules against the definition, and the classifier."""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import FuzzyRuleClassifier, fuzzy_class_rules

NAMES = ["a", "b", "c", "d", "e"]


def random_cases(seed, rows=150):
    rng = np.random.RandomState(seed)
    degrees = pd.DataFrame(np.round(rng.rand(rows, len(NAMES)), 2), columns=NAMES)
    degrees = degrees.where(rng.rand(rows, len(NAMES)) < 0.7, 0.0)
    a, b = degrees["a"].to_numpy(), degrees["b"].to_numpy()
    labels = np.where(a > 0.6, "x", np.where(b > 0.5, "y", "z"))
    return degrees, pd.Series(labels, name="kind")


def definition(degrees, labels, min_supp, min_conf, max_len, tnorm):
    values = degrees.to_numpy()
    array = np.asarray(labels)
    classes = sorted(set(array))
    found = {}
    for length in range(1, max_len + 1):
        for columns in combinations(range(values.shape[1]), length):
            mu = values[:, columns].min(axis=1) if tnorm == "min" else values[:, columns].prod(axis=1)
            whole = mu.sum()
            if whole <= 0:
                continue
            conf = {c: mu[array == c].sum() / whole for c in classes}
            for c in classes:
                supp = mu[array == c].sum() / (array == c).sum()
                others = [conf[o] for o in classes if o != c]
                weight = conf[c] - np.mean(others)
                if supp >= min_supp - 1e-9 and conf[c] >= min_conf - 1e-9 and weight > 1e-9:
                    found[(frozenset(degrees.columns[list(columns)]), c)] = (mu[array == c].sum() / len(array), conf[c], weight)
    return found


@pytest.mark.parametrize("tnorm", ["min", "product"])
@pytest.mark.parametrize("seed", [0, 1])
def test_rules_equal_the_definition(seed, tnorm):
    degrees, labels = random_cases(seed)
    rules = fuzzy_class_rules(degrees, labels, min_supp=0.15, min_conf=0.45, max_len=3, tnorm=tnorm)
    expected = definition(degrees, labels, 0.15, 0.45, 3, tnorm)
    assert len(expected) > 3
    got = {(a, r["class"]): (r["support"], r["confidence"], r["weight"])
           for (_, r), a in zip(rules.iterrows(), rules["antecedents"])}
    assert set(got) == set(expected)
    for key, value in expected.items():
        assert got[key] == pytest.approx(value)
    assert rules.attrs["n_transactions"] == len(degrees)
    assert (rules["weight"] > 0).all() and rules["consequents"].iloc[0] <= {"kind=x", "kind=y", "kind=z"}


def test_hand_computed_rule_and_prediction():
    degrees = pd.DataFrame({"hot": [1.0, 0.8, 0.2, 0.0], "dry": [1.0, 0.5, 0.9, 0.1]})
    labels = ["fire", "fire", "calm", "calm"]
    rules = fuzzy_class_rules(degrees, labels, min_supp=0.3, min_conf=0.5, max_len=2)
    row = rules[(rules["antecedents"] == frozenset({"hot"})) & (rules["class"] == "fire")].iloc[0]
    assert row["confidence"] == pytest.approx(1.8 / 2.0) and row["support"] == pytest.approx(1.8 / 4)
    assert row["weight"] == pytest.approx(0.9 - 0.1)
    model = FuzzyRuleClassifier(min_supp=0.3, min_conf=0.5, max_len=2).fit(degrees, labels)
    assert model.predict(pd.DataFrame({"hot": [0.9], "dry": [0.8]})).tolist() == ["fire"]
    assert model.predict(pd.DataFrame({"hot": [0.0], "dry": [0.0]})).tolist() == [model.default_class_]


@pytest.mark.parametrize("inference", ["winner", "sum"])
def test_classifier_recovers_planted_structure(inference):
    degrees, labels = random_cases(2, rows=300)
    train, test = slice(0, 200), slice(200, 300)
    model = FuzzyRuleClassifier(min_supp=0.05, min_conf=0.4, max_len=2, inference=inference, max_rules_per_class=10)
    model.fit(degrees.iloc[train], labels.iloc[train])
    majority = labels.iloc[test].value_counts(normalize=True).max()
    assert model.score(degrees.iloc[test], labels.iloc[test]) > majority + 0.1
    assert (model.rules_.groupby("class").size() <= 10).all()
    assert list(model.decision_scores(degrees.iloc[test]).columns) == model.classes_
    assert model.get_params()["inference"] == inference


def test_validation():
    degrees, labels = random_cases(0, 30)
    with pytest.raises(ValueError, match="tnorm"):
        fuzzy_class_rules(degrees, labels, tnorm="nope")
    with pytest.raises(ValueError, match="min_supp"):
        fuzzy_class_rules(degrees, labels, min_supp=0)
    with pytest.raises(ValueError, match="rows"):
        fuzzy_class_rules(degrees, labels[:5])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fuzzy_class_rules(degrees * 3, labels)
    with pytest.raises(ValueError, match="not trained"):
        FuzzyRuleClassifier().predict(degrees)
    with pytest.raises(ValueError, match="inference"):
        FuzzyRuleClassifier(inference="nope").fit(degrees, labels)
    model = FuzzyRuleClassifier().fit(degrees, labels)
    with pytest.raises(ValueError, match="missing the columns"):
        model.predict(degrees[["a", "b"]])
