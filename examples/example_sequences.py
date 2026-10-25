# -*- coding: utf-8 -*-
"""
Ejemplo: patrones y reglas secuenciales sobre el registro de eventos de una
casa sensorizada -- que hace habitualmente la persona, en que orden, y que
suele ocurrir despues de un evento.
"""
import numpy as np
import pandas as pd

from ARMxtend.ARM import sequential_rules
from ARMxtend.FIM import format_sequence, prefixspan, sequences_from_events


def simulated_events(days=120, seed=0):
    """
    Registro de eventos (una fila por evento) de una persona durante `days` dias:

    * Todas las mananas: suena la alarma, va al bano y pasa a la cocina.
    * Dia con prisa (el 75 % de los dias laborables y el 15 % del fin de semana):
      sale de casa justo despues (puerta), sin tomar cafe.
    * Dia tranquilo: toma cafe (95 %) y despues ve la television (85 %).
    * Cualquier dia puede haber llamadas de telefono a cualquier hora (ruido).
    """
    rng = np.random.RandomState(seed)
    rows = []
    for day in range(days):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
        rushed = rng.rand() < (0.15 if date.day_of_week >= 5 else 0.75)
        clock = 7 * 60 + int(rng.randint(0, 45))                       # minutos desde medianoche

        def add(event, gap):
            nonlocal clock
            clock += gap
            rows.append((day, date + pd.Timedelta(minutes=clock), event))

        add("alarm", 0)
        add("bathroom", int(rng.randint(5, 12)))
        add("kitchen", int(rng.randint(10, 20)))
        if rushed:
            add("door", int(rng.randint(3, 10)))
        else:
            if rng.rand() < 0.95:
                add("coffee", int(rng.randint(3, 8)))
            if rng.rand() < 0.85:
                add("tv", int(rng.randint(15, 40)))
        for _ in range(rng.randint(0, 3)):
            clock += int(rng.randint(0, 120))
            rows.append((day, date + pd.Timedelta(minutes=clock), "phone"))
    return pd.DataFrame(rows, columns=["day", "time", "event"])


def _fmt(items):
    return ", ".join(sorted(items))


def main():
    events = simulated_events()

    # 1. Una secuencia por dia, con sus eventos ordenados por hora ---------------------------
    sequences = sequences_from_events(events, "day", "time", "event")
    print("Secuencias:", len(sequences), "| primer dia:", " ".join("{" + _fmt(e) + "}" for e in sequences[0]))

    # 2. Patrones secuenciales frecuentes (PrefixSpan) --------------------------------------
    patterns = prefixspan(sequences, min_supp=0.3, min_len=4)
    print("\nPatrones frecuentes de 4 o mas eventos (soporte >= 0.3):")
    for _, pattern in patterns.head(4).iterrows():
        print("  {}  support={:.2f}".format(format_sequence(pattern["sequence"]), pattern["support"]))

    # 3. Reglas secuenciales: "si X, despues Y" con su confianza; el lift separa lo informativo -
    every = sequential_rules(sequences, min_supp=0.2, min_conf=0.0, max_antecedent=1, max_consequent=1)
    rules = every[every["confidence"] >= 0.7].sort_values("lift", ascending=False, kind="mergesort")
    print("\nReglas secuenciales con confianza >= 0.7, por lift decreciente:")
    for _, rule in rules.head(4).iterrows():
        print("  {} => {}  support={:.2f} confidence={:.2f} lift={:.2f}".format(
            _fmt(rule["antecedents"]), _fmt(rule["consequents"]), rule["support"], rule["confidence"], rule["lift"]))

    # 4. El orden importa: coffee => tv se cumple, tv => coffee no ----------------------------
    byRule = {(_fmt(r["antecedents"]), _fmt(r["consequents"])): r["confidence"] for _, r in every.iterrows()}
    print("\nconfianza de coffee => tv: {:.2f}; de tv => coffee: {:.2f}".format(
        byRule.get(("coffee", "tv"), 0.0), byRule.get(("tv", "coffee"), 0.0)))

    return sequences, patterns, every


if __name__ == "__main__":
    main()
