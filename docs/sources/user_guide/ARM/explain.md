# `ARM.explain` -- why a rule fires, and how it compares with the alternatives

Rules are only useful if someone can check them. These functions explain rules from **any miner** against the boolean table of
items they apply to (items with a lag, `item@lag1`, work as in [`ARM.anomaly`](anomaly_detection.md)).

```python
from ARMxtend.ARM import rule_evidence, explain_rule, explain_row

rule_evidence(items, {"a", "b"}, {"c"})            # contingency table of A => C on the data
explain_rule(items, rules, position=0)             # evidence + contribution of each item + alternatives
explain_row(items, rules, row=3)                   # what every rule says about one row
```

## `rule_evidence(items, antecedent, consequent)`

A dictionary with the counts `both` (A and C), `only_antecedent` (A without C: the violations), `only_consequent` and `neither`;
`n_rows` (rows with enough history for the lags), `support`, `confidence`, `lift`, and the index labels `supporting` (rows where the
rule holds) and `violating` (rows where `A` holds and `C` does not), so you can look at the actual cases.

## `explain_rule(items, rules, position, top=5)`

For the rule with index label `position` in `rules`:

- `rule`: readable text (`a (t), b (t) => c (t)`).
- `evidence`: the dictionary above.
- `contributions`: for each item of the antecedent, the **confidence of the rule without it** and the `gain` the item adds
  (`confidence - confidence_without`). A negative gain means the item does not help: the shorter rule is at least as good.
  With a one-item antecedent the comparison is the base rate of the consequent.
- `alternatives`: the other rules of `rules` that are **more general** (a proper subset of the antecedent, same consequent), **more
  specific** (a proper superset) or have the **same antecedent** and another consequent, each with its `confidence`, `support` and
  `delta_confidence` against the explained rule, measured on `items`.

## `explain_row(items, rules, row)`

One line per rule, with the `status` on that row: `confirmed` (antecedent and consequent hold), `violated` (the antecedent holds and
the consequent does not), `not fired` (the antecedent does not hold) or `no history` (the lags reach before the first row);
`missing` lists the items of the antecedent that do not hold, i.e. **why it did not fire**; `confidence` is the rule's, if the
frame has that column.

## Notes

- Everything is recomputed from `items`: pass the data the rules apply to, not necessarily the data they were mined from, to see
  how they behave on a new period.
- It explains rules, it does not choose between them; for removing the redundant ones see [redundancy](redundancy.md) and for
  telling real rules from noise, [significance](significance.md).
