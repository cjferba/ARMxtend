# `FIM.apriori` -- DApriori / DAprioriTID (Spark)

Requires the `spark` extra (`pip install -e ".[spark]"`, see [Installation](../../installation.md)).

Frequent itemset mining on Apache Spark, inspired by the sequential Apriori and Apriori-TID
algorithms, implementing Algorithms 1 and 2 of Fernandez-Basso, Ruiz & Martin-Bautista (2024) --
see [Citing ARMxtend](../../cite.md).

## `DApriori`

```python
DApriori.run(sc, transactions, min_supp)
```

- **sc** (`pyspark.SparkContext`): used to create broadcast variables with the candidate itemsets
  at each iteration.
- **transactions** (`RDD[str]`): one transaction per line, items separated by `,` (customize via
  `DApriori.item_sep`).
- **min_supp** (`float`): minimum relative support, in `(0, 1]`.

Returns `dict {itemset_key: support}` with every frequent itemset of any length (candidates are
generated with the classic apriori-gen procedure, pruning any candidate with an infrequent
subset -- see `FIM._shared.generate_candidates`).

## `DAprioriTID`

Same signature and result as `DApriori`. The difference is internal: after the first pass, items
are sorted by descending support and infrequent items are removed from every transaction, so later
iterations scan a smaller, pre-filtered dataset -- matching the memory-usage advantage reported for
Apriori-TID over plain Apriori on large datasets.

## Example

```python
from pyspark import SparkContext, SparkConf
from ARMxtend.FIM.apriori import DApriori, DAprioriTID

sc = SparkContext(conf=SparkConf().setAppName("armxtend").setMaster("local[*]"))
transactions = sc.parallelize([
    "bread,milk",
    "bread,diapers,beer,eggs",
    "milk,diapers,beer,cola",
    "bread,milk,diapers,beer",
    "bread,milk,diapers,cola",
])

freq_itemsets = DApriori.run(sc, transactions, min_supp=0.5)
# or: DAprioriTID.run(sc, transactions, min_supp=0.5)
```

Feed the result into [`FIM.BD_ARE.association_rules_bd`](BD_ARE.md) to mine association rules.
