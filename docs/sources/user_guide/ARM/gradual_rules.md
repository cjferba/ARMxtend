# `ARM.gradual_rules` -- gradual rules

Dependencies such as *"the more age and the less exercise, the higher the cholesterol"*, mined
directly from numeric columns -- no discretization or fuzzification. Based on Hullermeier (2002),
"Association rules for expressing gradual dependencies" (PKDD 2002).

```python
import pandas as pd
from ARMxtend.ARM import gradual_rules, format_gradual_itemset

rules = gradual_rules(health_df, min_supp=0.25, min_conf=0.8)
for _, r in rules.iterrows():
    print(format_gradual_itemset(r["antecedents"]), "->", format_gradual_itemset(r["consequents"]))
# age↑, exercise↓ -> cholesterol↑     support=0.51 confidence=0.98
```

## Model

- A **gradual item** is `(attribute, direction)`, with direction `'+'` ("the more") or `'-'`
  ("the less"); a gradual itemset has items of distinct attributes.
- A pair of rows `(s, t)` **satisfies** an itemset if, going from `s` to `t`, every attribute moves in
  its indicated direction (strictly: ties do not count).
- **Support** = fraction of the `n(n-1)/2` unordered row pairs that satisfy the itemset (in either
  orientation); **confidence** of `A -> B` = `supp(A u B) / supp(A)`. Support is anti-monotone, so the
  search is a level-wise Apriori with pruning.
- An itemset and its **mirror** (all directions flipped: "the more A, the more B" = "the less A, the
  less B") have the same support, so only the *canonical* form (first attribute `'+'`) is generated;
  every returned rule has an equivalent mirror rule that is not listed.

## Signatures

```python
gradual_itemsets(data, min_supp, attributes=None, max_len=None, max_rows=None, random_state=0)
gradual_rules(data, min_supp, min_conf, attributes=None, max_len=None, max_rows=None, random_state=0)
```

- **data**: a `DataFrame` (or 2-D array) of numeric values; rows with NaN in the used attributes are
  dropped. `attributes` defaults to all numeric columns.
- `gradual_itemsets` returns `{canonical itemset (tuple of (attr, dir)): support}`; `gradual_rules`
  a `DataFrame` with `antecedents`, `consequents` (frozensets of `(attr, dir)`), `antecedent support`,
  `consequent support`, `support`, `confidence`.
- **Cost**: `O(n^2 m)` memory (an `n x n` boolean matrix per attribute). For big tables pass
  `max_rows` to work on a reproducible random sample (`random_state`).
