# `FIM.BD_ARE` -- crisp association rule mining (Spark)

Requires the `spark` extra (`pip install -e ".[spark]"`, see [Installation](../../installation.md)).

Mines crisp association rules from a set of frequent itemsets already computed (e.g. with
[`FIM.apriori.DApriori`/`DAprioriTID`](apriori.md) or [`FIM.Eclat.DECLAT`](Eclat.md)), implementing
Algorithm 4 ("Spark procedure for association rule mining") of Fernandez-Basso, Ruiz &
Martin-Bautista (2024) -- see [Citing ARMxtend](../../cite.md). Rule generation is the part shared
by all three frequent-itemset-mining algorithms above.

## `generate_rules` (sequential)

```python
generate_rules(freq_itemsets, min_conf)
```

The plain-Python computation, useful for small itemset sets or for testing: for every frequent
itemset, generates all its possible antecedent/consequent splits and keeps those with confidence
`>= min_conf`.

## `association_rules_bd` (distributed)

```python
association_rules_bd(sc, freq_itemsets, min_conf)
```

Same computation as `generate_rules`, distributed across the cluster via a Spark `flatMap` +
`filter` over the candidate itemsets (broadcasting `freq_itemsets` so every partition can look up
antecedent/consequent supports).

Both take `freq_itemsets` (`dict {itemset_key: support}`, as returned by `DApriori`/`DAprioriTID`/
`DECLAT`) and `min_conf` (minimum confidence, in `(0, 1]`), and return
`dict {(antecedent_key, consequent_key): confidence}`.

```python
from ARMxtend.FIM.apriori import DApriori
from ARMxtend.FIM.BD_ARE import association_rules_bd

freq_itemsets = DApriori.run(sc, transactions, min_supp=0.5)
rules = association_rules_bd(sc, freq_itemsets, min_conf=0.7)
```
