# -*- coding: utf-8 -*-
"""Tests de los estimadores estilo scikit-learn (ARMxtend.estimators, ARMxtend.base) y de ARMxtend.datasets."""
import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import RuleClassifier, rare_rules, top_k_rules
from ARMxtend.base import ParamsMixin
from ARMxtend.datasets import (list_datasets, load_dataset, make_labeled_items, make_quantities, make_sequences,
                               make_transactions)
from ARMxtend.estimators import RuleMiner, TransactionEncoder, as_transactions
from ARMxtend.FIM import high_utility_itemsets, prefixspan


def keys(rules):
    return sorted((tuple(sorted(a)), tuple(sorted(c)), round(s, 9)) for a, c, s in
                  zip(rules["antecedents"], rules["consequents"], rules["support"]))


# ----- ParamsMixin -----

class Toy(ParamsMixin):
    def __init__(self, a=1, b="x"):
        self.a = a
        self.b = b


def test_params_get_set_repr_and_clone():
    toy = Toy(a=3)
    assert toy.get_params() == {"a": 3, "b": "x"} and repr(toy) == "Toy(a=3)" and repr(Toy()) == "Toy()"
    assert toy.set_params(b="y") is toy and toy.b == "y"
    clone = Toy(**toy.get_params())
    assert clone.get_params() == toy.get_params() and clone is not toy
    with pytest.raises(ValueError, match="parametro no valido"):
        toy.set_params(c=1)


def test_every_estimator_stores_its_parameters_untouched():
    for estimator in (TransactionEncoder(), RuleMiner(), RuleClassifier()):
        params = estimator.get_params()
        assert set(params) == set(type(estimator)._param_names())
        assert type(estimator)(**params).get_params() == params


def test_sklearn_clone_pipeline_and_grid_search():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV

    assert clone(RuleMiner(min_supp=0.3)).min_supp == 0.3
    items, labels = make_labeled_items(300, random_state=0)
    search = GridSearchCV(RuleClassifier(), {"min_supp": [0.05, 0.2]}, cv=3).fit(items, labels)
    assert search.best_score_ > 0.8 and sklearn.__version__


# ----- TransactionEncoder -----

def test_encoder_round_trip_and_unknown_items():
    baskets = [["b", "a"], ["c"], [], ["a", "c"]]
    encoder = TransactionEncoder()
    table = encoder.fit_transform(baskets)
    assert encoder.columns_ == ["a", "b", "c"] and table.dtypes.eq(bool).all()
    assert table.to_numpy().tolist() == [[True, True, False], [False, False, True], [False] * 3, [True, False, True]]
    assert encoder.inverse_transform(table) == [["a", "b"], ["c"], [], ["a", "c"]]
    assert encoder.transform([["a", "zzz"]]).to_numpy().tolist() == [[True, False, False]]
    with pytest.raises(ValueError, match="no visto"):
        TransactionEncoder(handle_unknown="error").fit(baskets).transform([["zzz"]])


def test_encoder_errors_and_mixed_item_types():
    with pytest.raises(ValueError, match="no esta ajustado"):
        TransactionEncoder().transform([["a"]])
    with pytest.raises(ValueError, match="handle_unknown"):
        TransactionEncoder(handle_unknown="boom").fit([["a"]])
    assert TransactionEncoder().fit([[1, "a"], [2]]).columns_ == [1, 2, "a"]


def test_encoder_keeps_the_index_of_a_table_and_rejects_non_boolean():
    frame = pd.DataFrame({"a": [1, 0], "b": [0, 1]}, index=["x", "y"])
    assert as_transactions(frame) == [["a"], ["b"]]
    assert list(TransactionEncoder().fit_transform(frame).index) == ["x", "y"]
    with pytest.raises(ValueError, match="booleana"):
        as_transactions(pd.DataFrame({"a": [0.5, 1.0]}))


# ----- RuleMiner -----

BASKETS = make_transactions(400, 25, avg_length=6, random_state=11)


def test_fpgrowth_method_matches_the_functional_pipeline():
    from ARMxtend.ARM import association_rules
    miner = RuleMiner(min_supp=0.1, min_threshold=0.7).fit(BASKETS)
    expected = association_rules(miner.itemsets_, min_threshold=0.7)
    assert keys(miner.rules_) == keys(expected) and len(miner.rules_) > 0
    assert miner.n_transactions_ == 400 and len(miner.itemsets_) >= len(miner.rules_) / 10


def test_top_k_and_rare_methods_match_the_functions():
    top = RuleMiner(method="top_k", k=7, min_threshold=0.5, min_lift=1.0, max_len=3).fit(BASKETS).rules_
    assert keys(top) == keys(top_k_rules(BASKETS, 7, min_conf=0.5, min_lift=1.0, max_len=3)) and len(top) == 7
    pairs = [["a", "d"]] * 15 + [["a"]] * 15 + [["d"]] * 15 + [["b"]] * 55
    rare = RuleMiner(method="rare", max_supp=0.2, min_threshold=0.5).fit(pairs).rules_
    assert keys(rare) == keys(rare_rules(pairs, 0.2, min_conf=0.5)) and len(rare) == 2


def test_filters_lift_alpha_and_prune():
    base = RuleMiner(min_supp=0.1, min_threshold=0.5).fit(BASKETS).rules_
    lifted = RuleMiner(min_supp=0.1, min_threshold=0.5, min_lift=1.2).fit(BASKETS).rules_
    assert 0 < len(lifted) < len(base) and (lifted["lift"] >= 1.2).all()
    significant = RuleMiner(min_supp=0.1, min_threshold=0.5, alpha=0.01).fit(BASKETS).rules_
    assert len(significant) < len(base) and "p_adjusted" in significant.columns
    pruned = RuleMiner(min_supp=0.1, min_threshold=0.5, prune=True).fit(BASKETS).rules_
    assert len(pruned) <= len(base)


def test_accepts_a_boolean_table_and_gives_the_same_rules():
    table = TransactionEncoder().fit_transform(BASKETS)
    assert keys(RuleMiner(min_supp=0.1).fit(table).rules_) == keys(RuleMiner(min_supp=0.1).fit(BASKETS).rules_)


def test_miner_errors():
    with pytest.raises(ValueError, match="method"):
        RuleMiner(method="boom").fit(BASKETS)
    with pytest.raises(ValueError, match="necesita el parametro k"):
        RuleMiner(method="top_k").fit(BASKETS)
    with pytest.raises(ValueError, match="necesita el parametro max_supp"):
        RuleMiner(method="rare").fit(BASKETS)
    with pytest.raises(ValueError, match="confianza"):
        RuleMiner(method="top_k", k=3, metric="lift").fit(BASKETS)
    with pytest.raises(ValueError, match="no esta ajustado"):
        RuleMiner().transform(BASKETS)


def test_transform_marks_where_the_antecedent_holds():
    miner = RuleMiner(min_supp=0.1, min_threshold=0.6).fit(BASKETS)
    features = miner.transform(BASKETS)
    assert features.shape == (400, len(miner.rules_)) and features.dtypes.eq(bool).all()
    names = miner.get_feature_names_out()
    assert len(names) == len(miner.rules_) and " => " in names[0]
    sets = [set(basket) for basket in BASKETS]
    for position in (0, len(miner.rules_) // 2, len(miner.rules_) - 1):
        antecedent = set(miner.rules_.iloc[position]["antecedents"])
        assert features[position].tolist() == [antecedent <= basket for basket in sets]
    assert np.allclose(features[0].mean(), miner.rules_.iloc[0]["antecedent support"], atol=1e-9)
    assert miner.fit_transform(BASKETS).shape == features.shape


def test_transform_with_unseen_items_and_without_rules():
    miner = RuleMiner(min_supp=0.1, min_threshold=0.6).fit(BASKETS)
    assert not miner.transform([["never_seen"]]).to_numpy().any()
    empty = RuleMiner(min_supp=0.99).fit(BASKETS)
    assert len(empty.rules_) == 0 and empty.transform(BASKETS).shape == (400, 0)


def test_classifier_follows_the_same_protocol():
    items, labels = make_labeled_items(600, noise=0.05, random_state=5)
    model = RuleClassifier(min_supp=0.05, min_conf=0.7).set_params(max_len=3)
    assert model.max_len == 3 and model.fit(items[:400], labels[:400]).score(items[400:], labels[400:]) > 0.9


# ----- datasets -----

def test_transactions_are_reproducible_and_plant_patterns():
    first, patterns = make_transactions(800, 40, avg_length=6, n_patterns=5, pattern_length=3, random_state=3,
                                        return_patterns=True)
    assert first == make_transactions(800, 40, avg_length=6, n_patterns=5, pattern_length=3, random_state=3)
    assert first != make_transactions(800, 40, avg_length=6, n_patterns=5, pattern_length=3, random_state=4)
    assert len(first) == 800 and all(len(set(basket)) == len(basket) and basket == sorted(basket) for basket in first)
    assert abs(np.mean([len(basket) for basket in first]) - 6) < 1.5
    commonest = max(patterns, key=lambda pattern: sum(set(pattern) <= set(basket) for basket in first))
    assert sum(set(commonest) <= set(basket) for basket in first) / 800 > 0.05
    noisy = make_transactions(800, 40, corruption=0.0, random_state=1)
    assert sum(len(basket) for basket in noisy) > 0


def test_transactions_can_be_mined_for_their_planted_patterns():
    baskets, patterns = make_transactions(1500, 40, avg_length=6, n_patterns=3, pattern_length=3, corruption=0.0,
                                          random_state=0, return_patterns=True)
    found = RuleMiner(min_supp=0.05, min_threshold=0.5).fit(baskets).itemsets_["itemsets"].tolist()
    assert any(frozenset(pattern) in found for pattern in patterns)


def test_sequences_plant_ordered_patterns():
    sequences, patterns = make_sequences(400, 20, avg_elements=5, n_patterns=2, pattern_length=3, random_state=2,
                                         return_patterns=True)
    assert sequences == make_sequences(400, 20, avg_elements=5, n_patterns=2, pattern_length=3, random_state=2)
    frequent = prefixspan(sequences, 0.1)
    found = {tuple(next(iter(element)) for element in sequence) for sequence in frequent["sequence"]
             if all(len(element) == 1 for element in sequence)}
    assert any(tuple(item for (item,) in pattern) in found for pattern in patterns)
    assert all(isinstance(element, list) and element for sequence in sequences for element in sequence)


def test_quantities_work_with_the_utility_miner():
    quantities, profits = make_quantities(300, 12, random_state=1)
    assert quantities.shape == (300, 12) and (quantities.to_numpy() >= 0).all() and set(profits) == set(quantities.columns)
    assert (np.array(list(profits.values())) > 0).all()
    found = high_utility_itemsets(quantities, profits, min_utility=0.1)
    assert len(found) > 0 and found["utility"].max() >= 0.1 * found.attrs["total_utility"]


def test_labeled_items_depend_on_the_planted_rules():
    items, labels = make_labeled_items(2000, n_features=10, n_classes=3, n_rules=3, noise=0.0, random_state=0)
    assert items.shape == (2000, 10) and items.dtypes.eq(bool).all() and set(labels) <= {0, 1, 2} and labels.nunique() >= 2
    model = RuleClassifier(min_supp=0.02, min_conf=0.8, max_len=4).fit(items[:1500], labels[:1500])
    assert model.score(items[1500:], labels[1500:]) > 0.95
    noisy_items, noisy_labels = make_labeled_items(2000, noise=0.3, random_state=0)
    assert (noisy_labels != make_labeled_items(2000, noise=0.0, random_state=0)[1]).mean() > 0.05 and len(noisy_items) == 2000


@pytest.mark.parametrize("call", [
    lambda: make_transactions(0), lambda: make_transactions(10, corruption=1.0), lambda: make_sequences(n_items=0),
    lambda: make_quantities(max_quantity=0), lambda: make_labeled_items(n_classes=1),
    lambda: make_labeled_items(noise=1.0), lambda: make_labeled_items(n_rules=0), lambda: load_dataset("boom")])
def test_generator_validation(call):
    with pytest.raises(ValueError):
        call()


def test_generators_accept_a_numpy_generator():
    rng = np.random.default_rng(5)
    assert make_transactions(20, 10, random_state=rng) != make_transactions(20, 10, random_state=rng)


def test_packaged_datasets():
    assert list_datasets() == ["han", "weather", "sales", "routines"]
    han = load_dataset("han")
    assert len(han.data) == 9 and han.description and han["name"] == "han"
    frequent = RuleMiner(min_supp=2 / 9).fit(han.data).itemsets_
    assert frozenset({"I1", "I2", "I5"}) in set(frequent["itemsets"])
    han.data.append(["x"])
    assert len(load_dataset("han").data) == 9

    weather = load_dataset("weather")
    assert weather.data.shape == (14, 10) and weather.target.value_counts().to_dict() == {"yes": 9, "no": 5}
    assert "outlook=sunny" in weather.data.columns and len(weather.table) == 14
    assert RuleClassifier(min_supp=0.1, min_conf=0.8).fit(weather.data, weather.target).score(weather.data, weather.target) >= 0.9

    sales = load_dataset("sales")
    top = high_utility_itemsets(sales.data, sales.profits, min_utility=30, absolute=True).iloc[0]
    assert top["utility"] == 40.0 and top["itemsets"] == frozenset("bcdf")

    routines = load_dataset("routines")
    assert len(routines.data) == 6 and prefixspan(routines.data, 0.8).shape[0] > 0
