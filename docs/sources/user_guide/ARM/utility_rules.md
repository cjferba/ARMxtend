# `ARM.high_utility_rules` and `FIM.high_utility_itemsets` -- what is worth most, not what is repeated most

Frequency counts transactions; **utility counts value**. A pair of expensive products bought in 3 of 2000 baskets can bring
more than bread and milk together. Each transaction carries **quantities** and each item a **unit profit**.

```python
from ARMxtend.ARM import high_utility_rules
from ARMxtend.FIM import high_utility_itemsets

profits = {"a": 5, "b": 2, "c": 1, "d": 2, "e": 3, "f": 1, "g": 1}
sales = [{"a": 1, "c": 1, "d": 1},
         {"a": 2, "c": 6, "e": 2, "g": 5},
         {"a": 1, "b": 2, "c": 1, "d": 6, "e": 1, "f": 5},
         {"b": 4, "c": 3, "d": 3, "f": 1},
         {"b": 2, "c": 2, "e": 1, "g": 2}]

high_utility_itemsets(sales, profits, min_utility=30, absolute=True).head(4)
#  utility      itemsets  support  length
#     40.0  {b, c, d, f}      0.4       4
#     36.0     {b, d, f}      0.4       3
#     34.0     {b, c, d}      0.4       3
#     31.0     {a, c, e}      0.4       3
# attrs: n_transactions 5, total_utility 94.0, min_utility 30.0

high_utility_rules(sales, profits, 30, absolute=True, min_conf=0.8, min_lift=2.5, max_antecedent=1)
# antecedents consequents  support  confidence  lift  utility  utility_share
#           f       b,c,d      0.4         1.0   2.5     40.0       0.425532
#           f         b,d      0.4         1.0   2.5     36.0       0.382979
```

## Definitions

The **utility of an item in a transaction** is `quantity x profit`; the utility of an itemset in a transaction that contains
all its items is the sum over its items, and its **utility** is the sum over the transactions. An itemset is **high-utility**
(an HUI) if its utility reaches `min_utility`, a fraction of the total utility of the database or, with `absolute=True`, a
value. In the example `{b, d}` is in transactions 3 and 4, where it is worth 2*2 + 6*2 = 16 and 4*2 + 3*2 = 14, so its
utility is 30; `{b, c, d, f}` brings 40 of the 94 the whole database is worth.

## How it works

Utility is **not** anti-monotone: adding an item can raise it, so it cannot prune on its own. The two-phase algorithm uses the
**transaction-weighted utility** (TWU) of an itemset, the sum of the *total* utilities of the transactions that contain it.
TWU is anti-monotone and bounds the utility of every superset, so a level-wise search keeps only itemsets with
`TWU >= min_utility` and computes the exact utility of those. The result is exact; the tests compare it with the definition
applied to every itemset.

## Rules

`high_utility_rules` splits every HUI of two or more items into `X => Y`. The measures are the **classical** ones
(support, confidence, lift... count transactions); to them it adds `utility` (the utility of `X u Y`) and `utility_share` (its
fraction of the total). Other definitions of "utility confidence" (utility of `X u Y` over utility of `X`) are not used here.
`attrs` keeps `n_transactions`, `total_utility` and `n_candidates`, so [`significant_rules`](significance.md) works as with
any other rule frame (it tests the association of `X` and `Y`, not their value).

## Signature

```python
high_utility_itemsets(quantities, profits=None, min_utility=0.1, absolute=False, min_len=1, max_len=None)
high_utility_rules(quantities, profits=None, min_utility=0.1, absolute=False, min_conf=0.6, min_lift=None,
                   max_antecedent=None, max_consequent=None, max_len=None)
```

`quantities` is a `DataFrame` (one row per transaction, one column per item, 0 or NaN if not bought) or a list of
`{item: quantity}` dictionaries. `profits` maps each item to its unit profit; without it every item is worth 1 and the utility
is the number of units bought.

## Things to know

- **Every split of one itemset has the same utility.** The 6 HUIs of the example give 52 rules with confidence 0.8, and they have
  only 5 different utilities; what tells `f => b,c,d` from `b,d,f => c` is the lift. Use `min_lift` and `max_antecedent`.
- **A rule can be valuable only because of one valuable item.** In the shop example `cheese => bread` is a high-utility rule
  (lift 0.99) simply because cheese is worth a lot; without `min_lift` such rules crowd the list.
- **Profits and quantities must be >= 0**: with negative profits the TWU bound no longer holds, so they are rejected.
- **It is the plain exact algorithm**, not the fastest one. It builds dense *transactions x items* matrices in memory and generates
  candidates level by level, which is fine for thousands of items and rows but can be slow with a low `min_utility` on many
  items. There is no Spark version.
- **A fraction of the total utility** depends on the whole database: add a very valuable transaction and the threshold moves.
  Use `absolute=True` to compare across data sets.

Command line: [`armxtend high-utility-rules`](../cli.md#high-utility-rules).

## References

- H. Yao, H. J. Hamilton, C. J. Butz (2004). *A foundational approach to mining itemset utilities from databases.* Proceedings of
  the 4th SIAM International Conference on Data Mining, 482-486 (the utility of an itemset and its mining as a problem).
- Y. Liu, W. Liao, A. Choudhary (2005). *A two-phase algorithm for fast discovery of high utility itemsets.* Advances in
  Knowledge Discovery and Data Mining (PAKDD 2005), 689-695 (transaction-weighted utility and the two-phase algorithm).
