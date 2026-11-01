# `FIM.Eclat` -- DECLAT / FuzzyDECLAT (Spark)

Requires the `spark` extra (`pip install -e ".[spark]"`, see [Installation](../../installation.md)).

Frequent itemset mining on Apache Spark inspired by the sequential ECLAT algorithm (Zaki, 2000),
which represents each item by its list of transaction identifiers (TID-list) and computes an
itemset's support by intersecting the TID-lists of its items.

## Input from DataFrames and Parquet

The Spark algorithms take RDDs. `FIM.spark_io` builds them from a DataFrame without collecting anything in the driver:

```python
from ARMxtend.FIM import read_transactions, transactions_from_dataframe, fuzzy_transactions_from_dataframe

rdd = read_transactions(spark, "baskets.parquet", column="items")             # array<string> column
rdd = transactions_from_dataframe(frame, "line", sep=",")                      # text column with a separator
fuzzy = read_transactions(spark, "memberships.parquet", fuzzy=True)            # array<struct<item,degree>>
fuzzy = fuzzy_transactions_from_dataframe(frame, "items", degrees_column="degrees")   # two parallel arrays
DECLAT.run(sc, rdd, 0.05)
```

Item names cannot contain `,` or `-`. Rows with a null items column become empty transactions (they still count in the total).

## Memory

The TID-lists are built in the executors as numpy arrays, collected in the driver and broadcast once; see
[Scalability](../scalability.md#memory-of-the-driver).

## `DECLAT` (crisp)

Implements Algorithm 3 of Fernandez-Basso, Ruiz & Martin-Bautista (2024) -- see
[Citing ARMxtend](../../cite.md).

```python
DECLAT.run(sc, transactions, min_supp)
```

Same arguments and return type as [`FIM.apriori.DApriori`](apriori.md) (`RDD[str]` transactions,
`,`-separated items, `dict {itemset_key: support}` result) -- the three algorithms
(`DApriori`/`DAprioriTID`/`DECLAT`) are interchangeable and produce identical results, they only
differ in how the counting is distributed internally.

```python
from ARMxtend.FIM.Eclat import DECLAT

freq_itemsets = DECLAT.run(sc, transactions, min_supp=0.5)
```

## `FuzzyDECLAT` (fuzzy)

The fuzzy, alpha-cut-based counterpart, consistent with
[`FIM.BD_FARE.FuzzyDAprioriTID`](BD_FARE.md) and [`FFIM.fuzzy_fpgrowth`](../FFIM.md) (same input
convention, same output format, interchangeable as input to
[`FIM.FARE.fuzzy_association_rules`](FARE.md)).

```python
FuzzyDECLAT.run(sc, transactions, min_supp, num_alpha)
```

- **transactions** (`RDD[Iterable[Tuple[str, float]]]`): each transaction is a list of
  `(item, membership degree in [0, 1])` pairs.
- **num_alpha** (`int`): number of alpha-cuts (10 is a reasonable default).

Returns `dict {itemset_key: numpy.ndarray(num_alpha)}`: the bit-list of relative support of each
frequent itemset at every alpha-cut. An itemset is frequent if its aggregated fuzzy support FSupp
(see [`FIM.FARE`](FARE.md)) reaches `min_supp`.

```python
from ARMxtend.FIM.Eclat import FuzzyDECLAT

fuzzy_transactions = sc.parallelize([
    [("cold", 1.0), ("low_humidity", 0.8)],
    [("cold", 0.9), ("low_humidity", 0.6)],
    [("warm", 0.7), ("low_humidity", 0.9)],
])
freq_itemsets = FuzzyDECLAT.run(sc, fuzzy_transactions, min_supp=0.3, num_alpha=10)
```

## Scalability

`DECLAT` and `FuzzyDECLAT` collect all the TID-lists in the driver, so the amount of data they can handle is set by
the driver's memory (`spark.driver.memory`), not by the size of the cluster. On a single machine `DECLAT` is the
fastest crisp algorithm but its speed-up flattens after 2 to 4 cores, and `FuzzyDECLAT` grows faster than linearly
with the data and does not speed up with more cores: for big fuzzy datasets prefer
[`FuzzyDAprioriTID`](BD_FARE.md). Measurements and how to reproduce them: [Scalability](../scalability.md).
