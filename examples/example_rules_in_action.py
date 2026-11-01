# -*- coding: utf-8 -*-
"""
Ejemplo: reglas en accion -- usar reglas ya minadas para detectar anomalias
(fallos de un actuador), predecir lo que ocurrira (siguiente instante de unos
sensores y siguientes eventos de una rutina) y clasificar con reglas de
asociacion de clase (CBA).
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
    Cuatro sensores booleanos, un dato cada 10 minutos:

    * ``motion`` (movimiento, 30 % de los instantes) y ``door`` (puerta, 20 %) al azar;
    * ``light``: si hubo movimiento en el instante anterior se enciende el 99 % de las veces,
      y si no, el 10 %;
    * ``chime`` (timbre): si se abrio la puerta en el instante anterior suena el 80 %, y si no, el 5 %;
    * con ``faults`` > 0, en ese numero de instantes tras un movimiento la luz NO se enciende (fallo).

    Devuelve los items y una serie booleana que marca los instantes con fallo.
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
    """Area bajo la curva ROC: probabilidad de que una fila anomala puntue mas que una normal (los empates, medio punto)."""
    ranks = pd.Series(np.asarray(score, dtype=float)).rank().to_numpy()
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def patients(rows=1200, seed=2):
    """
    Diez sintomas booleanos y una clase: 'flu' si hay fiebre y tos (95 %), si no 'allergy' si hay picor
    y estornudos (90 %), y si no 'healthy' (el 93 % de las veces; el resto, al azar entre las tres).
    """
    rng = np.random.RandomState(seed)
    names = ["fever", "cough", "itchy", "sneezing", "fatigue", "headache", "nausea", "rash", "dizzy", "sore"]
    items = pd.DataFrame(rng.rand(rows, len(names)) < [0.5, 0.5, 0.5, 0.5] + [0.3] * 6, columns=names)
    label = np.where(items["fever"] & items["cough"], "flu", np.where(items["itchy"] & items["sneezing"], "allergy", "healthy"))
    noisy = rng.rand(rows) < np.where(label == "healthy", 0.07, np.where(label == "flu", 0.05, 0.10))
    label = np.where(noisy, rng.choice(["flu", "allergy", "healthy"], rows), label)
    return items, pd.Series(label, name="diagnosis")


def main():
    # 1. Anomalias: reglas aprendidas con el funcionamiento normal, aplicadas a datos nuevos con fallos ---
    train, _ = sensor_log(3000, seed=1)
    test, fault = sensor_log(2000, faults=30, seed=2)
    rules = temporal_association_rules(train, 1, 0.02, min_threshold=0.75, targets=["light", "chime"],
                                       exclude_persistence=True, same_time_antecedent=False)
    rules = significant_rules(rules, alpha=0.05)
    rules = prune_unproductive_rules(rules[rules["consequents"].map(len) == 1], min_improvement=0.05)
    print("Reglas del funcionamiento normal ({} instantes):".format(len(train)))
    for _, rule in rules.iterrows():
        print("  {} (t-1) => {} (t)  confidence={:.2f}".format(
            ", ".join(sorted(i.split("@")[0] for i in rule["antecedents"])), ", ".join(sorted(rule["consequents"])),
            rule["confidence"]))

    flagged = detect_anomalies(test, rules, min_score=6.0)
    found = int(fault.loc[flagged.index].sum())
    surprise = anomaly_scores(test, rules)["score"]
    counted = anomaly_scores(test, rules, weight="count")["score"]
    print("\nDatos nuevos: {} instantes con {} fallos de la luz. {} instantes superan los 6 bits de sorpresa y "
          "contienen {} de los fallos".format(len(test), int(fault.sum()), len(flagged), found))
    print("AUC con peso 'surprise': {:.3f}; con peso 'count': {:.3f}".format(auc(surprise, fault), auc(counted, fault)))
    first = flagged.iloc[0]
    print("El mas anomalo, {}: score={:.1f} por {}".format(flagged.index[0], first["score"], "; ".join(first["rules"])))

    # 2. Prediccion del siguiente instante con las mismas reglas ----------------------------------------
    quality = evaluate_forecast(test, rules, threshold=0.7)
    print("\nPrediccion de cada sensor con las reglas (umbral 0.7), sobre datos nuevos:")
    for item, row in quality.iterrows():
        print("  {:<6} precision={:.2f} recall={:.2f} (el sensor se activa el {:.0%} de las veces)".format(
            item, row["precision"], row["recall"], row["base_rate"]))
    upcoming = forecast_next(test, rules)
    print("Prevision para el instante siguiente al ultimo dato (hubo movimiento: {}, puerta: {}): {}".format(
        "si" if test["motion"].iloc[-1] else "no", "si" if test["door"].iloc[-1] else "no",
        ", ".join("{} ({:.2f})".format(item, score) for item, score in upcoming.items()) or "nada previsto"))

    # 3. Siguiente evento de una rutina, con reglas secuenciales ------------------------------------------
    events = simulated_events(days=120, seed=0)
    sequences = sequences_from_events(events, "day", "time", "event")
    learned, unseen = sequences[:80], sequences[80:]
    routine = sequential_rules(learned, min_supp=0.2, min_conf=0.3, max_antecedent=2, max_consequent=1)
    after_coffee = predict_next_events([["alarm"], ["bathroom"], ["kitchen"], ["coffee"]], routine)
    print("\nTras alarm, bathroom, kitchen, coffee se espera: {}".format(
        ", ".join("{} ({:.2f})".format(item, score) for item, score in after_coffee.head(2).items())))
    hits = evaluate_next_events(unseen, routine, k=2, min_prefix=3, train=learned, target="next")
    print("En {} cortes de {} dias nuevos (con 3 o mas eventos ya ocurridos), los 2 eventos previstos incluyen el "
          "siguiente el {:.0%} de las veces (prediciendo los 2 mas frecuentes: {:.0%})".format(
              hits["cases"], len(unseen), hits["hit_rate"], hits["baseline_hit_rate"]))

    # 4. Clasificacion con reglas de asociacion de clase (CBA) ---------------------------------------------
    cases, diagnosis = patients()
    cut = 800
    classifier = RuleClassifier(min_supp=0.05, min_conf=0.7, max_len=3, min_improvement=0.1)
    classifier.fit(cases.iloc[:cut], diagnosis.iloc[:cut])
    majority = float((diagnosis.iloc[cut:] == diagnosis.iloc[:cut].mode()[0]).mean())
    accuracy = classifier.score(cases.iloc[cut:], diagnosis.iloc[cut:])
    print("\nClasificador CBA: {} reglas de clase candidatas, {} seleccionadas; exactitud {:.2f} en {} casos nuevos "
          "(clase mayoritaria: {:.2f})".format(len(classifier.candidate_rules_), len(classifier.rules_), accuracy,
                                                len(cases) - cut, majority))
    for _, rule in classifier.rules_.iterrows():
        print("  si {} => {}  confidence={:.2f} support={:.2f}".format(
            " y ".join(sorted(rule["antecedents"])), rule["class"], rule["confidence"], rule["support"]))
    print("  en otro caso => {}".format(classifier.default_class_))

    return rules, (flagged, fault), (quality, upcoming), (hits, after_coffee), (classifier, accuracy, majority)


if __name__ == "__main__":
    main()
