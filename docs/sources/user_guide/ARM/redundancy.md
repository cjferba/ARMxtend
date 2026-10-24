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
