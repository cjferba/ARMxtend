# `estimators` -- scikit-learn style API

The functions of ARMxtend take lists of baskets and return `DataFrame`s. For code that already uses scikit-learn (or
mlxtend) there are also **estimators** with `fit` / `transform` / `get_params`. They do not need scikit-learn: it is only
used, if installed, to let `clone`, `Pipeline` and `GridSearchCV` accept them.

```python
from ARMxtend.estimators import RuleMiner, TransactionEncoder
from ARMxtend.datasets import load_dataset

baskets = load_dataset("han").data                 # list of baskets
table = TransactionEncoder().fit_transform(baskets)   # boolean DataFrame, one column per item

miner = RuleMiner(min_supp=2 / 9, min_threshold=0.7, min_lift=1.2).fit(table)
miner.rules_                         # the usual rule frame (support, confidence, lift...)
miner.transform(baskets)             # boolean DataFrame: which rule antecedents each basket satisfies
miner.get_feature_names_out()        # one text per rule, 'I5 => I1'
```

## `TransactionEncoder`

Baskets <-> boolean table, as in `mlxtend.preprocessing.TransactionEncoder`. `fit` learns `columns_` (sorted items);
`transform` builds the table, ignoring items not seen in `fit` (`handle_unknown="error"` raises instead);
`inverse_transform` goes back. A boolean (or 0/1) `DataFrame` is accepted as input and keeps its index.

## `RuleMiner`

One estimator for the three ways of asking for rules:

| `method` | what it returns | main parameters |
|---|---|---|
| `"fpgrowth"` (default) | all rules with support >= `min_supp` | `min_supp`, `metric`, `min_threshold` |
| `"top_k"` | the `k` rules with the highest support ([`ARM.top_k_rules`](ARM/topk_rules.md)) | `k`, `min_threshold` (confidence), `max_len` |
| `"rare"` | rules among frequent items that are rare together ([`ARM.rare_rules`](ARM/rare_rules.md)) | `max_supp`, `min_count`, `min_threshold` |

`algorithm="dense"` (only with `"fpgrowth"`) searches the itemsets with [`FIM.dense_itemsets`](comparison.md), a bit-vector Apriori that is faster on tables with few items; the rules are the same.

Common to all: `min_lift`, `alpha` (keep only the [significant](ARM/significance.md) rules, with `correction` and `test`)
and `prune` (drop [redundant](ARM/redundancy.md) rules). After `fit`: `rules_`, `n_transactions_` and, for
`"fpgrowth"`, `itemsets_`. With `"top_k"` and `"rare"` the measure is the confidence and `min_supp` is only an optional floor.

`transform(X)` gives one boolean column per rule, true where the basket satisfies its **antecedent**. Those columns are
features for another model in a `Pipeline`; whether the consequent also holds is what a classifier would learn.
With no rules the result has the rows of `X` and no columns.

## `RuleClassifier` and `ParamsMixin`

[`ARM.RuleClassifier`](ARM/classification.md) (CBA) follows the same protocol, so
`GridSearchCV(RuleClassifier(), {"min_supp": [0.02, 0.05]}, cv=3)` works. Any class that stores its constructor arguments
unchanged and inherits `ARMxtend.base.ParamsMixin` gets `get_params`, `set_params` and a short `repr`.

## Things to know

- **No scikit-learn dependency.** `RuleMiner` is a transformer in spirit but does not inherit scikit-learn's classes; the
  tests run `clone` and `GridSearchCV` only if scikit-learn is installed (tested with scikit-learn 1.x).
- **`transform` is not a probability.** It tells that a rule applies, not that it is right; use the rule's `confidence`.
- **`fit` ignores `y`**; it is accepted so that pipelines can call it.
