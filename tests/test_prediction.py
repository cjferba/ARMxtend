# -*- coding: utf-8 -*-
"""
Tests of ``ARM.prediction``: scores of the temporal rules, forecast of the next instant,
forecast metrics and event prediction with sequential rules, all against
hand-computed values.
"""
import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import (
    evaluate_forecast, evaluate_next_events, forecast_next, predict_next_events, rule_scores, sequential_rules,
)


def make_rules(*rules):
    return pd.DataFrame({"antecedents": [frozenset(a) for a, _, _ in rules],
                         "consequents": [frozenset(c) for _, c, _ in rules],
                         "confidence": [p for _, _, p in rules]})


ITEMS = pd.DataFrame({"a": [1, 0, 1, 1, 0], "c": [0, 0, 1, 1, 1]}).astype(bool)
RULES = make_rules(({"a@lag1"}, {"b"}, 0.8), ({"c@lag1"}, {"b"}, 0.5), ({"a@lag1", "c@lag1"}, {"b"}, 0.9))


def test_scores_take_the_best_rule_or_combine_them():
    # row 1: only a made one step (0.8). Row 3: a and c made one step (0.8, 0.5, 0.9). Row 4: same
    best = rule_scores(ITEMS, RULES)
    assert list(best.columns) == ["b"] and best.index.equals(ITEMS.index)
    assert best["b"].tolist() == pytest.approx([0, 0.8, 0, 0.9, 0.9])
    # row 2: a and c absent one step earlier. Row 3: 1 - 0.2 * 0.5 * 0.1
    assert rule_scores(ITEMS, RULES, combine="noisy_or")["b"].tolist() == pytest.approx([0, 0.8, 0, 0.99, 0.99])


def test_rules_with_an_antecedent_in_the_present_are_ignored_unless_asked():
    rules = make_rules(({"a"}, {"b"}, 0.95), ({"a@lag1"}, {"b"}, 0.8))
    assert rule_scores(ITEMS, rules)["b"].tolist() == pytest.approx([0, 0.8, 0, 0.8, 0.8])
    assert rule_scores(ITEMS, rules, past_only=False)["b"].tolist() == pytest.approx([0.95, 0.8, 0.95, 0.95, 0.8])


def test_a_consequent_in_the_past_is_rejected_and_no_rules_give_no_columns():
    with pytest.raises(ValueError, match="current instant"):
        rule_scores(ITEMS, make_rules(({"a"}, {"c@lag1"}, 0.9)))
    with pytest.raises(ValueError, match="combine"):
        rule_scores(ITEMS, RULES, combine="mean")
    assert rule_scores(ITEMS, RULES.iloc[:0]).shape == (5, 0)


def test_forecast_next_applies_the_rules_to_the_last_rows():
    last_a = pd.DataFrame({"a": [0, 1], "c": [1, 0]}).astype(bool)           # ahora: a=1, c=0
    forecast = forecast_next(last_a, RULES)
    assert forecast.to_dict() == pytest.approx({"b": 0.8}) and forecast.name == "score" and forecast.index.name == "item"
    both = pd.DataFrame({"a": [0, 1], "c": [1, 1]}).astype(bool)
    assert forecast_next(both, RULES)["b"] == pytest.approx(0.9)
    assert forecast_next(both, RULES, combine="noisy_or")["b"] == pytest.approx(0.99)
    neither = pd.DataFrame({"a": [1, 0], "c": [1, 0]}).astype(bool)
    assert forecast_next(neither, RULES).empty
    short = pd.DataFrame({"a": [True], "c": [True]})                           # a rule with 2 steps asks for more history
    assert forecast_next(short, make_rules(({"a@lag2"}, {"b"}, 0.9))).empty


def test_evaluate_forecast_hand_computed():
    items = pd.DataFrame({"a": [1, 0, 1, 1, 0, 1], "b": [0, 1, 0, 0, 1, 0]}).astype(bool)
    quality = evaluate_forecast(items, make_rules(({"a@lag1"}, {"b"}, 0.8)))
    # rows 1..5: b is predicted in rows 1, 3 and 4 (a one row earlier); b occurs in 1 and 4
    row = quality.loc["b"]
    assert (row["n"], row["positives"], row["predicted"], row["hits"]) == (5, 2, 3, 2)
    assert row["precision"] == pytest.approx(2 / 3) and row["recall"] == pytest.approx(1.0)
    assert row["f1"] == pytest.approx(0.8) and row["base_rate"] == pytest.approx(0.4)
    # with a threshold higher than the confidence nothing is predicted
    nothing = evaluate_forecast(items, make_rules(({"a@lag1"}, {"b"}, 0.8)), threshold=0.9).loc["b"]
    assert nothing["predicted"] == 0 and np.isnan(nothing["precision"]) and nothing["recall"] == 0.0
    assert np.isnan(nothing["f1"])


def test_evaluate_forecast_on_unseen_data_recovers_the_generating_probabilities():
    rng = np.random.RandomState(3)
    rows = 20000
    a = rng.rand(rows) < 0.3
    b = np.where(np.r_[False, a[:-1]], rng.rand(rows) < 0.9, rng.rand(rows) < 0.05)
    row = evaluate_forecast(pd.DataFrame({"a": a, "b": b}), make_rules(({"a@lag1"}, {"b"}, 0.9))).loc["b"]
    assert row["precision"] == pytest.approx(0.9, abs=0.02)
    assert row["recall"] == pytest.approx(0.3 * 0.9 / (0.3 * 0.9 + 0.7 * 0.05), abs=0.02)


# --- Secuencias ----------------------------------------------------------------------------------

SEQUENCE_RULES = make_rules(({"alarm"}, {"bathroom"}, 1.0), ({"alarm", "bathroom"}, {"coffee"}, 0.6),
                            ({"kitchen"}, {"tv"}, 0.5), ({"coffee"}, {"tv"}, 0.8))


def test_predict_next_events_fires_rules_whose_antecedent_has_happened():
    assert predict_next_events([["alarm"]], SEQUENCE_RULES).to_dict() == {"bathroom": 1.0}
    # bathroom has already occurred after alarm: the rule alarm => bathroom is fulfilled
    assert predict_next_events([["alarm"], ["bathroom"]], SEQUENCE_RULES).to_dict() == {"coffee": 0.6}
    after_coffee = predict_next_events(["alarm", "bathroom", "coffee"], SEQUENCE_RULES)
    assert after_coffee.to_dict() == {"tv": 0.8}
    assert predict_next_events([], SEQUENCE_RULES).empty


def test_an_event_seen_before_the_antecedent_completed_is_still_pending():
    # coffee occurred before alarm and bathroom were complete
    prediction = predict_next_events(["coffee", "alarm", "bathroom"], SEQUENCE_RULES)
    assert prediction.to_dict() == {"coffee": 0.6, "tv": 0.8}


def test_predictions_are_ranked_by_score_then_name_and_can_be_combined():
    history = ["kitchen", "coffee"]
    assert predict_next_events(history, SEQUENCE_RULES).to_dict() == {"tv": 0.8}
    assert predict_next_events(history, SEQUENCE_RULES, combine="noisy_or")["tv"] == pytest.approx(0.9)
    ties = make_rules(({"x"}, {"b"}, 0.5), ({"x"}, {"a"}, 0.5), ({"x"}, {"c"}, 0.9))
    assert predict_next_events(["x"], ties).index.tolist() == ["c", "a", "b"]
    with pytest.raises(ValueError, match="combine"):
        predict_next_events(history, SEQUENCE_RULES, combine="min")


def test_evaluate_next_events_hand_computed():
    sequences = [["a", "b", "c"], ["a", "c", "b"]]
    rules = make_rules(({"a"}, {"b"}, 0.9), ({"a"}, {"c"}, 0.8))
    future = evaluate_next_events(sequences, rules, k=1, train=sequences)
    # cortes: [a] -> b; [a,b] -> c; [a] -> b; [a,c] -> b. Todo ocurre despues
    assert future == {"cases": 4, "coverage": 1.0, "hit_rate": 1.0, "precision": 1.0,
                      "baseline_hit_rate": 1.0, "baseline_precision": 1.0}
    following = evaluate_next_events(sequences, rules, k=1, train=sequences, target="next")
    # the next element: b, c, c, b; b, c, b, b is predicted -> 3 of 4 (the baseline predicts b, c, b, b)
    assert following["hit_rate"] == pytest.approx(0.75) and following["baseline_hit_rate"] == pytest.approx(0.75)
    assert evaluate_next_events(sequences, rules, k=2, min_prefix=2)["cases"] == 2


def test_evaluate_next_events_without_predictions_counts_misses():
    result = evaluate_next_events([["x", "y"]], make_rules(({"q"}, {"r"}, 0.9)))
    assert result["cases"] == 1 and result["coverage"] == 0.0 and result["hit_rate"] == 0.0
    assert np.isnan(result["precision"]) and "baseline_hit_rate" not in result
    for bad in ({"k": 0}, {"min_prefix": 0}, {"target": "soon"}, {"combine": "mean"}):
        with pytest.raises(ValueError):
            evaluate_next_events([["x", "y"]], SEQUENCE_RULES, **bad)


def test_rules_beat_popularity_when_the_history_decides_what_comes_next():
    rng = np.random.RandomState(0)
    # after 'start' there are two branches: gym -> shake or walk -> coffee (50 %), and then 'work' always
    def day():
        branch = ["gym", "shake"] if rng.rand() < 0.5 else ["walk", "coffee"]
        return ["start"] + branch + ["work"]

    train, test = [day() for _ in range(300)], [day() for _ in range(300)]
    rules = sequential_rules(train, min_supp=0.2, min_conf=0.5, max_antecedent=1, max_consequent=1)
    result = evaluate_next_events(test, rules, k=1, min_prefix=2, train=train, target="next")
    assert result["hit_rate"] > 0.95 and result["baseline_hit_rate"] < 0.6
