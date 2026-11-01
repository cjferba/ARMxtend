# `ARM.redundancy` -- pruning redundant rules

```python
from ARMxtend.ARM import prune_redundant_rules

pruned = prune_redundant_rules(rules, tol=1e-9)
```

A rule `X2 -> Y2` is **redundant** if another rule `X1 -> Y1` is *more general* -- smaller or equal
antecedent (`X1 ⊆ X2`) and larger or equal consequent (`Y1 ⊇ Y2`) -- with **exactly the same support
and confidence** (Bastide, Taouil, Pasquier, Stumme & Lakhal, 2000, *Mining minimal non-redundant
association rules using frequent closed itemsets*). The general rule asks for fewer conditions and concludes at least as much, so
it carries all the information of the redundant one. Duplicated rules are reduced to one.

On a typical rule set the reduction is large: for the 5-transaction example of
`examples/example_rule_types.py`, 12 rules are reduced to 5.

- **rules**: any `DataFrame` with `antecedents`/`consequents` (frozensets), `support` and
  `confidence` columns: the output of [`ARM.association_rules`](association_rules.md),
  [`FIM.FARE`](../FIM/FARE.md), [`ARM.negative_rules`](negative_rules.md) or
  [`ARM.gradual_rules`](gradual_rules.md).
- **tol**: absolute tolerance to consider two support/confidence values equal (use a larger value for
  fuzzy rules computed from many alpha-cuts if you want to merge near-ties).
- `support_column`, `confidence_column`: to prune by other column names.

It preserves the structure and relative order of `rules`.

## Unproductive rules

```python
from ARMxtend.ARM import prune_unproductive_rules

productive = prune_unproductive_rules(rules, min_improvement=0.05)
```

Redundancy requires *exactly* equal support and confidence. A looser and often more useful notion is the *improvement*
of Bayardo, Agrawal and Gunopulos (2000, *Constraint-based rule mining in large, dense databases*): the rule
`X => Y` is **unproductive** if some more general rule `X' => Y` (with `X'` a proper subset of `X`) is present in
`rules` and `conf(X => Y) - conf(X' => Y) <= min_improvement`; that is, the extra conditions of `X` do not improve the
confidence by more than `min_improvement`. With `0` (the default), the rules that do not improve on a more general
rule at all are removed; with `0.05`, the extra conditions have to earn at least five points of confidence.

The comparison is with **every** more general rule in `rules`, kept or not, so the result does not depend on the order;
the most general rule of each family is always kept. The rules must therefore be mined with a threshold low enough for the
general rules to be in the set. It works on any `DataFrame` with `antecedents`, `consequents` and `confidence`, and keeps
the order of `rules`. Typical use: before [detecting anomalies](anomaly_detection.md) with the rules, or before building
a [classifier](classification.md).

