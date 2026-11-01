# `ARM.anomaly` -- anomalies as rule violations

Rules mined on a period of normal behaviour describe what usually happens. A row that breaks a reliable rule --
its antecedent holds and its consequent does not -- is a candidate anomaly, the more so the more reliable the
rule. This is the idea of [exception rules](exception_rules.md), applied to **new data** with rules that have
**already been mined**, including [lagged rules](temporal_rules.md): *"the door opened one step ago and the heater
has not switched on"*.

```python
from ARMxtend.ARM import (
    temporal_association_rules, significant_rules, prune_unproductive_rules, detect_anomalies,
)

rules = temporal_association_rules(train, 1, 0.02, min_threshold=0.75, targets=["light", "chime"],
                                   exclude_persistence=True, same_time_antecedent=False)
rules = significant_rules(rules, alpha=0.05)                                    # drop rules that are noise
rules = prune_unproductive_rules(rules[rules["consequents"].map(len) == 1], min_improvement=0.05)

flagged = detect_anomalies(test, rules, min_score=6.0)
flagged.iloc[0]["rules"]      # ['motion (t-1) => light (t)', 'door (t-1) => chime (t)']
```

A full run is in the [example](../../examples.md#rules-in-action-armanomaly-armprediction-armclassification).

## Violations and scores

A row **violates** `A => C` if all the items of `A` hold and not all the items of `C` do. Items with a lag
(`x@lag2`, see [`ARM.temporal_rules`](temporal_rules.md)) are read from earlier rows; the first rows, without enough
history for a rule, never violate it (to analyse a new period with history, include the previous rows in
`items` and select the period afterwards).

| Function | Returns |
|---|---|
| `rule_violations(items, rules)` | boolean table, one row per row of `items` and one column per rule (with the index of `rules`) |
| `anomaly_scores(items, rules, weight="surprise", floor=1e-3)` | `score`, `violations` and `applicable` (rules whose antecedent holds) for each row |
| `detect_anomalies(items, rules, min_score=0.0, top=None, ...)` | the rows with a score above `min_score`, highest first, with the violated rules written out |

The score of a row is the **sum of the weights of the rules it violates**:

| `weight` | Weight of a violated rule | Meaning |
|---|---|---|
| `surprise` (default) | `-log2(1 - confidence)` | bits of surprise of a violation if the rule held with probability equal to its confidence: breaking a 99 % rule weighs 6.6 bits, an 80 % rule 2.3 |
| `confidence` | the confidence | |
| `count` | 1 | number of rules violated |

A rule of confidence 1 would weigh infinity: `floor` bounds the violation probability from below (default `1e-3`, so
at most 10 bits).

In the example, a light that fails to switch on after motion is injected 30 times in 2000 steps, while a weak rule
(`door => chime`, 77 %) is violated by chance in about 4 % of the rows. With `surprise` the AUC is 0.998 and the 30 faults
are all among the 39 rows above 6 bits (the other 9 are the 1 % of times the light fails by itself); counting violated
rules gives 0.980 and cannot tell the faults apart from the harmless violations of the weak rule.

## Getting good rules

The quality of the alarms is the quality of the rules:

- **Mine on normal data.** If faults are common in the training period they become part of the "normal" rules.
- **Keep only real rules** with [`significant_rules`](significance.md): a rule that is just noise raises false alarms.
- **Prune** with [`prune_redundant_rules`](redundancy.md) and
  [`prune_unproductive_rules`](redundancy.md#unproductive-rules): rules that differ in an irrelevant extra condition are
  violated together and their weights add up. Restricting to rules with a single item as consequent also avoids
  counting one failure twice.
- **Use `same_time_antecedent=False`** (and `exclude_persistence=True`) when mining, so that the rules say what *should*
  happen after what already happened.
- Natural violations of a very reliable rule are indistinguishable from faults: with a 99 % rule, 1 in 100 triggers
  will look like a fault. The score ranks; deciding the threshold is up to the application.

Command line: [`armxtend anomalies`](../cli.md#rules-in-action-anomalies-forecast-next-events-and-classify).
