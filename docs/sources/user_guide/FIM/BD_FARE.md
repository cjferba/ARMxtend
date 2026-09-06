# `FIM.BD_FARE` -- FuzzyDAprioriTID (Spark)

Requires the `spark` extra (`pip install -e ".[spark]"`, see [Installation](../../installation.md)).

Fuzzy frequent itemset mining on Apache Spark, inspired by Apriori-TID, implementing Algorithm 1
("BDFARE-Apriori"/"BDFARE-Apriori-TID") of Fernandez-Basso, Ruiz & Martin-Bautista (2021) -- see
[Citing ARMxtend](../../cite.md).

```python
FuzzyDAprioriTID.run(sc, transactions, min_supp, num_alpha)
```

- **sc** (`pyspark.SparkContext`).
- **transactions** (`RDD[Iterable[Tuple[str, float]]]`): each transaction is a list of
  `(item, membership degree in [0, 1])` pairs.
- **min_supp** (`float`): minimum aggregated fuzzy support (FSupp), in `(0, 1]`.
- **num_alpha** (`int`): number of alpha-cuts used to decompose the fuzzy database (the paper
  experimentally shows 10 equidistributed alpha-cuts are enough to uncover every significant fuzzy
  association rule).

Returns `dict {itemset_key: numpy.ndarray(num_alpha)}`: the bit-list of relative support of each
frequent itemset at every alpha-cut (see `FIM._shared.alpha_cuts`) -- the same output convention as
[`FIM.Eclat.FuzzyDECLAT`](Eclat.md) and [`FFIM.fuzzy_fpgrowth`](../FFIM.md), so all three are
interchangeable as input to [`FIM.FARE.fuzzy_association_rules`](FARE.md).

```python
from ARMxtend.FIM.BD_FARE import FuzzyDAprioriTID

fuzzy_transactions = sc.parallelize([
    [("cold", 1.0), ("low_humidity", 0.8)],
    [("cold", 0.9), ("low_humidity", 0.6)],
    [("warm", 0.7), ("low_humidity", 0.9)],
])
freq_itemsets = FuzzyDAprioriTID.run(sc, fuzzy_transactions, min_supp=0.3, num_alpha=10)
```
