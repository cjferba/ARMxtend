# `FFIM` -- FP-Growth (crisp and fuzzy)

Single-machine frequent itemset mining via FP-Growth (Han, Pei & Yin, 2000): builds a frequent
pattern tree and mines it recursively without generating candidates, which is usually faster than
Apriori/Eclat-style approaches on a single machine.

## Crisp: `fpgrowth`

```python
fpgrowth(transactions, min_supp)
```

- **transactions** (`Sequence[Iterable[str]]`): one transaction per element.
- **min_supp** (`float`): minimum relative support, in `(0, 1]`.

Returns `dict {itemset_key: support}`, e.g. `{"bread": 0.8, "bread-milk": 0.6, ...}` -- convert a
key back to a `frozenset` with `FIM._shared.key_to_itemset` before feeding it to
[`ARM.association_rules`](ARM/association_rules.md).

```python
from ARMxtend.FFIM import fpgrowth

transactions = [
    ["bread", "milk"],
    ["bread", "diapers", "beer", "eggs"],
    ["milk", "diapers", "beer", "cola"],
    ["bread", "milk", "diapers", "beer"],
    ["bread", "milk", "diapers", "cola"],
]
fpgrowth(transactions, min_supp=0.5)
# {'beer': 0.6, 'beer-diapers': 0.6, 'milk': 0.8, 'diapers-milk': 0.6,
#  'bread-milk': 0.6, 'diapers': 0.8, 'bread-diapers': 0.6, 'bread': 0.8}
```

## Fuzzy: `fuzzy_fpgrowth`

```python
fuzzy_fpgrowth(transactions, min_supp, num_alpha)
```

- **transactions** (`Sequence[Iterable[Tuple[str, float]]]`): each transaction is a list of
  `(item, membership degree in [0, 1])` pairs.
- **num_alpha** (`int`): number of alpha-cuts used to decompose the fuzzy database (10 is a
  reasonable default, see [Citing ARMxtend](../cite.md)).

Since the FP-tree merges transaction paths that share the *same* items, it cannot directly
accommodate arbitrary real-valued membership degrees (two transactions with identical items but
different degrees would stop being "the same path"). `fuzzy_fpgrowth` works around this by
binarizing the fuzzy database independently at each of the `num_alpha` alpha-cuts (see
`FIM._shared.alpha_cuts`) to discover candidate itemsets, then recomputing each candidate's exact
support at every alpha-cut. It returns `dict {itemset_key: numpy.ndarray(num_alpha)}` -- the
bit-list of relative support of the itemset at each alpha-cut, ready to feed into
[`FIM.FARE.fuzzy_association_rules`](FIM/FARE.md) (exactly the same output convention as
[`FIM.Eclat.FuzzyDECLAT`](FIM/Eclat.md) and [`FIM.BD_FARE.FuzzyDAprioriTID`](FIM/BD_FARE.md), so the
three are interchangeable as `FARE`'s input).

An itemset is considered frequent if its **aggregated** fuzzy support FSupp (a weighted sum of its
support at every alpha-cut, see [`FIM.FARE`](FIM/FARE.md)) reaches `min_supp` -- not merely if it
is frequent in isolation at any single alpha-cut.

```python
from ARMxtend.FFIM import fuzzy_fpgrowth

fuzzy_transactions = [
    [("cold", 1.0), ("low_humidity", 0.8)],
    [("cold", 0.9), ("low_humidity", 0.6)],
    [("warm", 0.7), ("low_humidity", 0.9)],
    [("cold", 0.6), ("low_humidity", 0.7)],
    [("warm", 1.0), ("low_humidity", 0.2)],
]
fuzzy_fpgrowth(fuzzy_transactions, min_supp=0.3, num_alpha=10)
```

## Faster: `FIM.dense_fuzzy_itemsets`

`fuzzy_fpgrowth` runs FP-Growth once per alpha-cut and then re-scans the transactions for every candidate. When the number of
fuzzy items is moderate (tens to a few hundred) there is a much faster exact miner with the **same input and output**:

```python
from ARMxtend.FIM import dense_fuzzy_itemsets

dense_fuzzy_itemsets(fuzzy_transactions, min_supp=0.3, num_alpha=10)    # same dict as fuzzy_fpgrowth
```

It keeps one packed bit vector per item *and* alpha-cut, and counts the candidates of each Apriori level with a bitwise AND and a
bit count. The aggregated support FSupp is anti-monotone (it is a non-negative weighted sum of supports that are), so pruning on it
is exact. Measured with `benchmarks/fuzzy_dense.py` (10 alpha-cuts, `min_supp=0.03`, synthetic data, best of 2):

| transactions | items | itemsets | `fuzzy_fpgrowth` | `dense_fuzzy_itemsets` |
|---:|---:|---:|---:|---:|
| 2 000 | 30 | 353 | 2.317 s | 0.009 s |
| 2 000 | 200 | 135 | 0.524 s | 0.012 s |
| 5 000 | 30 | 354 | 5.735 s | 0.020 s |
| 5 000 | 200 | 138 | 1.338 s | 0.023 s |

The output is checked to be identical in every run. With thousands of very sparse items keep `fuzzy_fpgrowth`. From the command
line: `armxtend fuzzy-rules data.csv --algorithm dense`.
