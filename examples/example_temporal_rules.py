# -*- coding: utf-8 -*-
"""
Example: temporal rules over sensor data of a house -- from an
hourly series to items (symbols, events, calendar), rules with lag,
rules that only hold at certain hours, and drift of the rules over time.
"""
import numpy as np
import pandas as pd

from ARMxtend.ARM import (
    compare_rule_sets, format_lagged_item, prune_redundant_rules, rule_drift, rule_profile, temporal_association_rules,
    window_rules,
)
from ARMxtend.preprocessing.timeseries import calendar_items, discretize, to_items


def simulated_house(days=60, seed=0):
    """
    Hourly sensors of a house: the outdoor temperature (C), the door
    (open or closed) and the heating.

    * The whole period: the heating turns on one hour after the door is opened,
      but only in the afternoon and evening (from 17 h on).
    * Second half: in addition, it always turns on one hour after a cold
      hour (the temperature was in the lowest third).
    """
    rng = np.random.RandomState(seed)
    index = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    temperature = 8 + 7 * np.sin((index.hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 2, len(index))
    door_open = rng.rand(len(index)) < 0.25
    evening = np.asarray(index.hour >= 17)
    cold = temperature < np.quantile(temperature, 0.33)
    second_half = np.arange(len(index)) >= len(index) // 2

    heater = np.zeros(len(index), dtype=bool)
    heater[1:] = door_open[:-1] & evening[1:] & (rng.rand(len(index) - 1) < 0.95)
    heater[1:] |= second_half[1:] & cold[:-1]
    return pd.DataFrame({"temp": temperature, "door_open": door_open, "heater_on": heater}, index=index)


def _fmt(items):
    return ", ".join(sorted(format_lagged_item(item) for item in items))


def main():
    house = simulated_house()

    # 1. From the series to items: temperature symbols, events and calendar -----
    items = to_items(discretize(house, ["temp"], n_symbols=3, method="quantile"))
    with_calendar = items.join(calendar_items(items.index, ["part_of_day"]))
    print("Items:", ", ".join(items.columns))

    # 2. Rules with lag + calendar: the door only matters in the afternoon --------
    rules = temporal_association_rules(with_calendar, max_lag=1, min_supp=0.02, min_threshold=0.9,
                                       targets=["heater_on"], exclude_persistence=True)
    rules = prune_redundant_rules(rules).sort_values("confidence", ascending=False)
    print("\nRules with lag (consequent: heater_on):")
    for _, rule in rules.head(4).iterrows():
        print("  {} -> {}  support={:.3f} confidence={:.2f} lift={:.1f}".format(
            _fmt(rule["antecedents"]), _fmt(rule["consequents"]), rule["support"], rule["confidence"], rule["lift"]))

    # 3. The same rule without calendar, hour by hour: it only holds from 17 h on ---
    profile = rule_profile(items, ["door_open@lag1"], ["heater_on"], lambda stamps: stamps.hour)
    print("\nConfidence of 'door open 1 hour ago -> heating on' by hour:")
    for hour in (4, 10, 16, 17, 20, 23):
        print("  {:02d}h: {:.2f}".format(hour, profile.loc[hour, "confidence"]))

    # 4. Drift: rules with lag of the heating in windows of 15 days ----------
    def miner(window):
        return temporal_association_rules(window, max_lag=1, min_supp=0.03, min_threshold=0.85,
                                          targets=["heater_on"], exclude_persistence=True)

    windows = window_rules(with_calendar, "15D", miner=miner)
    drift = rule_drift(windows)
    print("\nVentanas:", windows.attrs["n_windows"], "| rule states:", drift["status"].value_counts().to_dict())
    new_rule = drift[(drift["antecedents"] == frozenset({"temp=low@lag1"})) & (drift["consequents"] == frozenset({"heater_on"}))]
    for _, rule in new_rule.iterrows():
        print("  {}: {} -> {} (since window {}, mean confidence {:.2f})".format(
            rule["status"], _fmt(rule["antecedents"]), _fmt(rule["consequents"]), rule["first_window"],
            rule["mean_confidence"]))
    changes = compare_rule_sets(windows[windows["window"] == 0], windows[windows["window"] == windows["window"].max()])
    print("Primera frente a ultima ventana:", changes["status"].value_counts().to_dict())

    return items, rules, profile, drift


if __name__ == "__main__":
    main()
