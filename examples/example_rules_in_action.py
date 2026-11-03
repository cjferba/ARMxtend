# -*- coding: utf-8 -*-
"""
Example: rules in action -- use already mined rules to detect anomalies
(faults of an actuator), predict what will happen (next instant of some
sensors and next events of a routine) and classify with class
association rules (CBA).
"""
import numpy as np
import pandas as pd

from ARMxtend.ARM import (
    RuleClassifier, detect_anomalies, evaluate_forecast, evaluate_next_events, forecast_next,
    anomaly_scores, predict_next_events, prune_unproductive_rules, sequential_rules, significant_rules,
    temporal_association_rules,
)
from ARMxtend.FIM import sequences_from_events

from example_sequences import simulated_events


def sensor_log(steps, faults=0, seed=0):
    """
    Four boolean sensors, one reading every 10 minutes:

    * ``motion`` (30 % of the instants) and ``door`` (20 %) at random;
    * ``light``: if there was motion in the previous instant it turns on 99 % of the time,
      and otherwise 10 %;
    * ``chime`` (doorbell): if the door was opened in the previous instant it rings 80 %, and otherwise 5 %;
    * with ``faults`` > 0, in that number of instants after a motion the light does NOT turn on (fault).

    Returns the items and a boolean series marking the instants with a fault.
    """
    rng = np.random.RandomState(seed)
    motion, door = rng.rand(steps) < 0.3, rng.rand(steps) < 0.2
    light, chime = rng.rand(steps) < 0.1, rng.rand(steps) < 0.05
    light[1:] = np.where(motion[:-1], rng.rand(steps - 1) < 0.99, light[1:])
    chime[1:] = np.where(door[:-1], rng.rand(steps - 1) < 0.8, chime[1:])
    fault = np.zeros(steps, dtype=bool)
    if faults:
        candidates = np.flatnonzero(np.r_[False, motion[:-1]])
        fault[rng.choice(candidates, faults, replace=False)] = True
        light[fault] = False
    index = pd.date_range("2024-03-01", periods=steps, freq="10min")
    items = pd.DataFrame({"motion": motion, "door": door, "light": light, "chime": chime}, index=index)
    return items, pd.Series(fault, index=index)


def auc(score, positive):
    """Area under the ROC curve: probability that an anomalous row scores higher than a normal one (ties count half)."""
    ranks = pd.Series(np.asarray(score, dtype=float)).rank().to_numpy()
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def patients(rows=1200, seed=2):
    """
    Ten boolean symptoms and a class: 'flu' if there is fever and cough (95 %), otherwise 'allergy' if there is itching
    and sneezing (90 %), and otherwise 'healthy' (93 % of the time; the rest, at random among the three).
    """
    rng = np.random.RandomState(seed)
    names = ["fever", "cough", "itchy", "sneezing", "fatigue", "headache", "nausea", "rash", "dizzy", "sore"]
    items = pd.DataFrame(rng.rand(rows, len(names)) < [0.5, 0.5, 0.5, 0.5] + [0.3] * 6, columns=names)
    label = np.where(items["fever"] & items["cough"], "flu", np.where(items["itchy"] & items["sneezing"], "allergy", "healthy"))
    noisy = rng.rand(rows) < np.where(label == "healthy", 0.07, np.where(label == "flu", 0.05, 0.10))
    label = np.where(noisy, rng.choice(["flu", "allergy", "healthy"], rows), label)
    return items, pd.Series(label, name="diagnosis")


def main():
    # 1. Anomalies: rules learned from normal operation, applied to new data with faults ---
    train, _ = sensor_log(3000, seed=1)
    test, fault = sensor_log(2000, faults=30, seed=2)
    rules = temporal_association_rules(train, 1, 0.02, min_threshold=0.75, targets=["light", "chime"],
                                       exclude_persistence=True, same_time_antecedent=False)
    rules = significant_rules(rules, alpha=0.05)
    rules = prune_unproductive_rules(rules[rules["consequents"].map(len) == 1], min_improvement=0.05)
    print("Rules from normal operation ({} instants):".format(len(train)))
    for _, rule in rules.iterrows():
        print("  {} (t-1) => {} (t)  confidence={:.2f}".format(
            ", ".join(sorted(i.split("@")[0] for i in rule["antecedents"])), ", ".join(sorted(rule["consequents"])),
            rule["confidence"]))

    flagged = detect_anomalies(test, rules, min_score=6.0)
    found = int(fault.loc[flagged.index].sum())
    surprise = anomaly_scores(test, rules)["score"]
    counted = anomaly_scores(test, rules, weight="count")["score"]
    print("\nNew data: {} instants with {} light faults. {} instants exceed the 6 bits of surprise and "
          "contain {} of the faults".format(len(test), int(fault.sum()), len(flagged), found))
    print("AUC with weight 'surprise': {:.3f}; with weight 'count': {:.3f}".format(auc(surprise, fault), auc(counted, fault)))
    first = flagged.iloc[0]
    print("The most anomalous, {}: score={:.1f} for {}".format(flagged.index[0], first["score"], "; ".join(first["rules"])))

    # 2. Prediction of the next instant with the same rules ----------------------------------------
    quality = evaluate_forecast(test, rules, threshold=0.7)
    print("\nPrediction of each sensor with the rules (threshold 0.7), on new data:")
    for item, row in quality.iterrows():
        print("  {:<6} precision={:.2f} recall={:.2f} (the sensor activates {:.0%} of the time)".format(
            item, row["precision"], row["recall"], row["base_rate"]))
    upcoming = forecast_next(test, rules)
    print("Forecast for the instant after the last reading (motion: {}, door: {}): {}".format(
        "yes" if test["motion"].iloc[-1] else "no", "yes" if test["door"].iloc[-1] else "no",
        ", ".join("{} ({:.2f})".format(item, score) for item, score in upcoming.items()) or "nothing forecast"))

    # 3. Next event of a routine, with sequential rules ------------------------------------------
    events = simulated_events(days=120, seed=0)
    sequences = sequences_from_events(events, "day", "time", "event")
    learned, unseen = sequences[:80], sequences[80:]
    routine = sequential_rules(learned, min_supp=0.2, min_conf=0.3, max_antecedent=2, max_consequent=1)
    after_coffee = predict_next_events([["alarm"], ["bathroom"], ["kitchen"], ["coffee"]], routine)
    print("\nAfter alarm, bathroom, kitchen, coffee the expected next is: {}".format(
        ", ".join("{} ({:.2f})".format(item, score) for item, score in after_coffee.head(2).items())))
    hits = evaluate_next_events(unseen, routine, k=2, min_prefix=3, train=learned, target="next")
    print("On {} cuts of {} new days (with 3 or more events already occurred), the 2 predicted events include the "
          "next one {:.0%} of the time (predicting the 2 most frequent: {:.0%})".format(
              hits["cases"], len(unseen), hits["hit_rate"], hits["baseline_hit_rate"]))

    # 4. Classification with class association rules (CBA) ---------------------------------------------
    cases, diagnosis = patients()
    cut = 800
    classifier = RuleClassifier(min_supp=0.05, min_conf=0.7, max_len=3, min_improvement=0.1)
    classifier.fit(cases.iloc[:cut], diagnosis.iloc[:cut])
    majority = float((diagnosis.iloc[cut:] == diagnosis.iloc[:cut].mode()[0]).mean())
    accuracy = classifier.score(cases.iloc[cut:], diagnosis.iloc[cut:])
    print("\nCBA classifier: {} candidate class rules, {} selected; accuracy {:.2f} on {} new cases "
          "(majority class: {:.2f})".format(len(classifier.candidate_rules_), len(classifier.rules_), accuracy,
                                                len(cases) - cut, majority))
    for _, rule in classifier.rules_.iterrows():
        print("  si {} => {}  confidence={:.2f} support={:.2f}".format(
            " y ".join(sorted(rule["antecedents"])), rule["class"], rule["confidence"], rule["support"]))
    print("  en otro caso => {}".format(classifier.default_class_))

    return rules, (flagged, fault), (quality, upcoming), (hits, after_coffee), (classifier, accuracy, majority)


if __name__ == "__main__":
    main()
