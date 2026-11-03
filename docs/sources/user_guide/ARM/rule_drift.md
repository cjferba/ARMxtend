# `ARM.rule_drift` -- how rules change over time

Rules mined once describe the past on average. In sensor data, logs and streams the interesting question is
often *when* a rule holds: did it appear last month, is it fading, does it come and go? `rule_drift` mines
the rules in consecutive **time windows** and follows each rule from one window to the next.

It is the batch, single-machine counterpart of the trends that FIMoTS ([`SARE`/`SFIM`](../SARE_SFIM.md)) keeps
incrementally on a Spark stream, and it works with any miner: crisp, fuzzy or [temporal](temporal_rules.md).

```python
from ARMxtend.ARM import window_rules, rule_drift, compare_rule_sets

windows = window_rules(items, "15D", min_supp=0.1, min_threshold=0.8)   # one row per (window, rule)
drift = rule_drift(windows, measure="confidence")                       # one row per rule
drift[drift["status"] == "emerging"]
```

## Pipeline

1. **`window_rules(data, width, step=None, ...)`** splits the table into windows and mines each one.
   `width` (and `step`) is a number of rows or, with a `DatetimeIndex`, a duration (`"7D"`, `"12h"`); the default
   `step` equals `width` (no overlap), a smaller `step` gives sliding windows. The result is the union of the
   rules of all windows with the columns `window`, `start`, `end` and `n_rows` added; the number of windows
   (including those without any rule) is in `result.attrs["n_windows"]`. Windows with fewer than `min_rows`
   rows (typically the incomplete last one) are skipped.
2. **`rule_drift(windowed_rules, measure="confidence", stable_fraction=0.8)`** summarises each rule: in how many
   windows it is present (`windows`, `persistence`), `first_window`, `last_window`, `mean_/min_/max_<measure>`,
   the `slope` of the measure per window (`NaN` if it appears in fewer than 3 windows) and a `status`.
3. **`compare_rule_sets(before, after, measure, tolerance)`** compares two sets of rules (two periods, or the
   first and the last window): `new`, `lost`, `changed` (the measure moved by more than `tolerance`) and `stable`.

## Rule status

With `W` windows:

| Status | Meaning |
|---|---|
| `stable` | present in the first and the last window and in at least `stable_fraction` of all of them |
| `emerging` | absent in the first window, present in the last: a new behaviour |
| `vanishing` | present in the first window, absent in the last: a behaviour that ended |
| `intermittent` | everything else: comes and goes, appears only in the middle, or sits at both ends with many gaps |

The table is sorted by status and persistence.

## The miner

By default each window is mined with FP-Growth and [`association_rules`](association_rules.md)
(`crisp_rule_miner(min_supp, metric, min_threshold)`). Pass `miner=` any function `window -> rules DataFrame`
with `antecedents` and `consequents` columns, for instance:

```python
from ARMxtend.ARM import fuzzy_rule_miner, temporal_association_rules

fuzzy = window_rules(memberships, 20, miner=fuzzy_rule_miner(min_supp=0.3, num_alpha=4))
lagged = window_rules(items, "15D", miner=lambda w: temporal_association_rules(
    w, max_lag=1, min_supp=0.03, min_threshold=0.85, targets=["heater_on"], exclude_persistence=True))
```

Example (simulated house, 60 days, 15-day windows): the rule *"temperature low one hour ago -> heater on"* is
planted only in the second half of the period, and `rule_drift` reports it as `emerging` from window 2.

## On a stream: `RuleDriftMonitor`

`window_rules` re-mines every window of a table you already have. For a **stream** that arrives in chunks use
`RuleDriftMonitor`, which mines only the windows each chunk completes and drops the rows no later window needs, so memory stays
bounded:

```python
from ARMxtend.ARM import RuleDriftMonitor

monitor = RuleDriftMonitor(width=20, step=10, min_supp=0.3, min_threshold=0.9, history=50)
for chunk in stream:                      # DataFrames of new rows, in time order
    new_rules = monitor.update(chunk)     # the rules of the windows this chunk completed (possibly empty)
    if monitor.n_windows >= 2:
        print(monitor.latest_changes())   # compare_rule_sets between the last two windows
monitor.drift()                           # rule_drift over the windows kept (renumbered from 0)
```

Same arguments as `window_rules` (`width`, `step`, `miner`, `min_rows`; windows by rows or by duration) plus `history`, which keeps
only the rules of the last windows (window numbers keep growing). Only **complete** windows are mined: by rows, when all their
rows have arrived; by duration, when a row at or after the end has arrived. After feeding all the data the windows and rules are
those of `window_rules` (the tests compare them for several chunk sizes), except that a final window by duration that ends
exactly at the last row waits for a later row.

## Notes

- Each window needs enough rows for the support threshold to mean something; very short windows produce noisy
  rules that look `intermittent`.
- `slope` is a least-squares line over the windows in which the rule is present; use it to spot rules that are
  gaining or losing confidence, not to extrapolate.
- Windows are independent: rules are re-mined from scratch, so the cost is `W` times the cost of one window. `RuleDriftMonitor` does not make a window cheaper, it avoids repeating the ones already mined.

Command line: [`armxtend rule-drift`](../cli.md#temporal-rules-and-rule-drift-sensor-csv-files).
