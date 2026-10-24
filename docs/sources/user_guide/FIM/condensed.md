# `FIM.condensed` -- closed and maximal itemsets

```python
from ARMxtend.FIM import closed_itemsets, maximal_itemsets, support_from_closed
```

Condensed representations of a frequent-itemset result (`{'A-B-C': support}`), valid for the output
of **any** ARMxtend miner, crisp (scalar support: `FFIM.fpgrowth`, `FIM.apriori`, `FIM.Eclat`, ...) or
fuzzy (support = bit-list per alpha-cut: `FFIM.fuzzy_fpgrowth`, `FIM.Eclat.FuzzyDECLAT`, ...).

- **`closed_itemsets(freq_itemsets, tol=1e-12)`** -- itemsets with *no* proper superset of exactly the
  same support (Pasquier et al., 1999). They preserve the support of **every** frequent itemset with
  fewer entries.
- **`maximal_itemsets(freq_itemsets)`** -- itemsets with no frequent proper superset (Bayardo, 1998).
  Even smaller, but the supports of the subsets are lost.
- **`support_from_closed(itemset, closed)`** -- recovers the support of any frequent itemset as the
  maximum support among the closed itemsets that contain it (`None` if it is not frequent).

```python
from ARMxtend.FFIM import fpgrowth

baskets = [{"a", "b", "c"}] * 3 + [{"a", "b"}] + [{"d"}]
frequent = fpgrowth(baskets, min_supp=0.2)       # 8 itemsets
closed_itemsets(frequent)                        # {'a-b': 0.8, 'a-b-c': 0.6, 'd': 0.2}
maximal_itemsets(frequent)                       # {'a-b-c': 0.6, 'd': 0.2}
support_from_closed("a-c", closed_itemsets(frequent))   # 0.6
```

In the **fuzzy** case, two itemsets have "the same support" only if their bit-lists coincide at *every*
alpha-cut, i.e. closure is defined level by level, consistently with the RL-theory.
