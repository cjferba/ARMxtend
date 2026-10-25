# `ARM.temporal_rules` -- rules with a delay

Rules such as *"if the door was open one hour ago and it is evening, the heater is on now"* or *"high
temperature two steps ago and low humidity one step ago -> the pump starts"*. They are the **inter-transaction
association rules** of Lu, Han & Feng (1998) and Tung, Lu, Han & Feng (2003): each time step becomes a
transaction that contains the items of the present *and* of the previous `max_lag` steps.

```python
from ARMxtend.ARM import temporal_association_rules, format_lagged_item

rules = temporal_association_rules(items, max_lag=2, min_supp=0.02, min_threshold=0.9,
                                   targets=["heater_on"], exclude_persistence=True)
for _, r in rules.iterrows():
    print(", ".join(map(format_lagged_item, r["antecedents"])), "->", ", ".join(map(format_lagged_item, r["consequents"])))
# door_open (t-1), part_of_day=evening (t) -> heater_on (t)      support=0.051 confidence=0.95 lift=4.1
```

`items` is an item table (see [`preprocessing.timeseries`](../preprocessing/timeseries.md) to build one from
numeric columns, events and the calendar).

## Model

- With lags `L = {0, 1, ..., max_lag}`, every row `t >= max_lag` becomes the transaction
  `{ x : x in row t } u { x@lagk : x in row t-k, k in 1..max_lag }`.
- Frequent itemsets are mined on these transactions with FP-Growth and turned into rules with
  [`ARM.association_rules`](association_rules.md), keeping only the rules whose **consequent is at lag 0** (the
  present) and, by default, whose **antecedent contains at least one lagged item**.
- **Support is measured over the `N - max_lag` rows that have all the lags available**, so support, confidence,
  lift and the other measures share the same denominator.
- The `lags` and `max_lag` columns of the result give the distinct lags that appear in the antecedent and the
  largest one.

## Signatures

```python
temporal_association_rules(items, max_lag, min_supp, metric="confidence", min_threshold=0.8,
                           lags=None, targets=None, require_lagged_antecedent=True,
                           same_time_antecedent=True, exclude_persistence=False)
fuzzy_temporal_association_rules(memberships, max_lag, min_supp, num_alpha=10, metric="confidence",
                                 min_threshold=0.8, lags=None, targets=None, require_lagged_antecedent=True,
                                 same_time_antecedent=True, exclude_persistence=False, measures=None)
rule_profile(items, antecedent, consequent, by)
format_lagged_item(item)      # 'temp=high@lag2' -> 'temp=high (t-2)'; 'temp=high' -> 'temp=high (t)'
```

| Argument | Meaning |
|---|---|
| `max_lag` | how many steps back the antecedent can look (>= 1). |
| `lags` | only these lags, within `1..max_lag` (e.g. `[1, 24]` for "one step ago" and "one day ago" in hourly data). |
| `targets` | restrict the consequent to these items (`'heater_on'`, `'temp=high'`) or variables (`'temp'`). |
| `require_lagged_antecedent` | `True`: the antecedent must contain a past item; `False`: also plain same-instant co-occurrence rules. |
| `same_time_antecedent` | `False`: the antecedent may contain only past items. |
| `exclude_persistence` | drop rules that only say "the state persists" (`temp=high@lag1 -> temp=high`). |

The fuzzy version takes **membership degrees** (e.g. the columns `FuzzyLib.AutoFuzzification` adds) and computes
support, confidence and certainty factor with the alpha-cut decomposition of [`FIM.FARE`](../FIM/FARE.md); for
degrees in `{0, 1}` it returns exactly the crisp result.

## Rules that only hold at some times: `rule_profile`

A rule that is weak on average may be very reliable at certain hours or days. `rule_profile` measures one rule
inside each group of time steps:

```python
profile = rule_profile(items, ["door_open@lag1"], ["heater_on"], by=lambda stamps: stamps.hour)
profile.loc[[10, 17, 20], "confidence"]    # 10 h: 0.00 | 17 h: 1.00 | 20 h: 0.86
```

`by` is a sequence with one group per row of `items`, or a function of the index. The result has the group
size `n`, `antecedent_count`, `joint_count`, `support`, `confidence` and `lift` per group (`NaN` where undefined).
Adding `calendar_items` to the item table lets the miner find the same pattern as a rule
(`door_open (t-1), part_of_day=evening (t) -> heater_on (t)`).

## Practical notes

- **Cost.** The number of items multiplies by the number of lags. Use few symbols, a sensible `min_supp`, `lags`
  to skip useless delays and `targets` to focus the output; [`prune_redundant_rules`](redundancy.md) removes
  the many variants of a rule that differ only in irrelevant extra items.
- **Rows must be regularly spaced** (or you must accept that "lag 1" means "previous row"): resample sensor
  data to a fixed frequency first (`DataFrame.resample("1h")`).
- **Association is not causation**: a lagged rule says the pattern is frequent and reliable, not that the
  antecedent produces the consequent.

Command line: [`armxtend temporal-rules`](../cli.md#temporal-rules-and-rule-drift-sensor-csv-files).

## References

- H. Lu, J. Han, L. Feng (1998). *Stock movement prediction and N-dimensional inter-transaction association
  rules.* ACM SIGMOD Workshop on Research Issues in Data Mining and Knowledge Discovery.
- A. K. H. Tung, H. Lu, J. Han, L. Feng (2003). *Efficient mining of intertransaction association rules.* IEEE
  Transactions on Knowledge and Data Engineering 15(1), 43-56.
