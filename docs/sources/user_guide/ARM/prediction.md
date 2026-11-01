# `ARM.prediction` -- predicting with rules

Rules that were mined on past data can anticipate what will happen. Two settings are covered.

## Series of items: the next time step

[Lagged rules](temporal_rules.md) `A@lag1, B@lag2 => C` say that `C` will hold now if `A` held one step ago and `B` two
steps ago.

```python
from ARMxtend.ARM import rule_scores, forecast_next, evaluate_forecast

rules = temporal_association_rules(train, 2, 0.02, min_threshold=0.75, same_time_antecedent=False)

scores = rule_scores(test, rules)                  # confidence with which the rules predict each item, per row
forecast_next(test, rules)                         # the same for the step after the last row
evaluate_forecast(test, rules, threshold=0.7)      # precision / recall on real data
```

| Function | Returns |
|---|---|
| `rule_scores(items, rules, combine="max", past_only=True)` | a table with the index of `items` and a column for every item that is the consequent of some rule; the value is the confidence of the best rule that fires (`combine="noisy_or"`: `1 - prod(1 - confidence)`, which assumes independent rules and overcounts if there are redundant ones) |
| `forecast_next(items, rules, combine="max")` | a `Series` `score` indexed by item with the prediction for the step after the last row (only items some rule predicts) |
| `evaluate_forecast(items, rules, threshold=0.5, ...)` | per item: `n`, `positives`, `predicted`, `hits`, `precision`, `recall`, `f1` and `base_rate` (the precision of predicting always) |

With `past_only=True` (default) the rules whose antecedent has an item at the current step are ignored, because that
information is not available when forecasting; mine with `same_time_antecedent=False` to avoid them. For an
honest evaluation, mine the rules on one period and evaluate on a later one. The first rows, without history for the
lags, are not evaluated.

## Sequences of events: what is still to come

[Sequential rules](sequential_rules.md) `X => Y` say that if all of `X` has happened, `Y` happens afterwards.

```python
from ARMxtend.ARM import predict_next_events, evaluate_next_events

rules = sequential_rules(learned, min_supp=0.2, min_conf=0.3, max_antecedent=2, max_consequent=1)

predict_next_events([["alarm"], ["bathroom"], ["kitchen"], ["coffee"]], rules)
# item
# tv       0.77
# phone    0.65
evaluate_next_events(unseen, rules, k=2, min_prefix=3, train=learned, target="next")
```

A rule fires when all of `X` has already happened in the history, and predicts every item of `Y` that is still
**pending**: that has not happened again after `X` was completed (otherwise the rule has already been fulfilled for that
item). `predict_next_events` returns a `Series` ordered from the highest to the lowest score (ties: alphabetically).

`evaluate_next_events` cuts every sequence after each element (from `min_prefix` elements of history) and checks whether the
`k` best predictions come true. `target="future"` counts a hit if the event happens at any later point, which is
what sequential rules claim; `target="next"` only in the very next element, which is stricter -- sequential rules say
*eventually*, not *immediately*, so they rank a certain-but-distant event above an uncertain-but-imminent one.
The result is a dictionary with `cases`, `coverage` (share of cuts with some prediction), `hit_rate` (a cut without
predictions counts as a miss) and `precision`; with `train=`, also `baseline_hit_rate` and `baseline_precision` of
predicting the `k` most frequent events not yet seen.

In the example routine (80 days to learn, 40 to test, at least 3 events of history, strict "next" criterion, `k=2`)
the rules include the next event in 63 % of the cuts, against 49 % for the popularity baseline. When the history does
not matter (nothing in it changes what happens next) the rules cannot beat the baseline, which is a useful check
of whether sequential rules are worth using on your data.

**A limitation.** Rules fire independently and cannot veto one another: after `coffee` the ranking of the example still
lists `door` (59 %) because `kitchen => door` holds, although in this routine a coffee means the person is not in a
hurry. A rule such as `kitchen, coffee => door` has confidence 0, so it is never mined. The same happens with
`rule_scores`. Rules cannot replace a model that conditions on the whole history; they are a transparent, cheap
predictor whose every prediction can be traced to the rules that produced it.

Command line: [`armxtend forecast`](../cli.md#rules-in-action-anomalies-forecast-next-events-and-classify) and
`armxtend next-events`.
