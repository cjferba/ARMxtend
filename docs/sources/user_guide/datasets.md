# `datasets` -- synthetic data and small example data sets

To try an algorithm, or to measure it, you need data where the answer is known. `ARMxtend.datasets` has generators that
**plant patterns** (so you can check that they come out) and a few tiny data sets inside the package; nothing is downloaded.

```python
from ARMxtend.datasets import make_transactions, make_sequences, make_quantities, make_labeled_items, load_dataset

baskets, patterns = make_transactions(2000, n_items=60, n_patterns=8, pattern_length=3, random_state=7, return_patterns=True)
sequences = make_sequences(500, random_state=1)
quantities, profits = make_quantities(1000, random_state=1)     # for high_utility_rules
items, labels = make_labeled_items(1500, noise=0.05, random_state=3)   # for RuleClassifier
weather = load_dataset("weather")                             # weather.data, weather.target
```

## Generators

All take `random_state` (an integer or a `numpy.random.Generator`) and give the same data for the same seed.

- **`make_transactions`**: in the style of the IBM Quest generator (Agrawal and Srikant, 1994, but not a port of it). Each
  transaction is a mix of planted patterns, each item lost with probability `corruption`, and single items drawn with a
  Zipf-like popularity. The pattern weights are exponential, so a few are frequent and some are rare. With
  `return_patterns=True` you also get the patterns.
- **`make_sequences`**: sequences of itemsets; with probability 1/2 a planted pattern is interleaved in order, each of its
  items in its own element.
- **`make_quantities`**: baskets from `make_transactions` with quantities in `1..max_quantity` and log-normal unit profits.
- **`make_labeled_items`**: a boolean table and a class decided by `n_rules` two-condition rules (the first that applies);
  `noise` changes that fraction of the classes at random, so about `1 - noise` is the best accuracy you can expect.

## Example data sets: `load_dataset(name)`

| name | content |
|---|---|
| `"han"` | 9 transactions over 5 items (the AllElectronics example of Han and Kamber) |
| `"weather"` | 14 days, 4 categorical attributes as items (`outlook=sunny`) and the class `play` (Quinlan's tennis table) |
| `"sales"` | quantities of 7 items in 5 sales and their unit profits (the [high-utility](ARM/utility_rules.md) example) |
| `"routines"` | 6 days of events of a house, as sequences |

The result is a `Dataset` (a dictionary whose keys also read as attributes) with `data`, `description` and, as needed,
`target`, `profits` or `table`. Each call gives fresh copies. `list_datasets()` lists the names.

Command line: [`armxtend generate`](cli.md#generate) writes synthetic data in the formats the other commands read.

## Things to know

- **Synthetic data is easier than real data.** Planted patterns are cleanly separated from noise, which real data rarely is;
  use the generators to check and to time algorithms, not to compare their quality.
- **Some planted patterns will not be found** at a given `min_supp`: the exponential weights make some of them rare (that
  is deliberate). Compare with `return_patterns`, not with the number of patterns.
- **The small data sets were typed in from the usual textbook tables.** Check them against the sources if you publish
  numbers from them.

## References

- R. Agrawal, R. Srikant (1994). *Fast algorithms for mining association rules.* VLDB 1994, 487-499 (the synthetic data
  generator used to evaluate itemset miners).
- J. Han, M. Kamber (2006). *Data Mining: Concepts and Techniques*, 2nd ed. (the 9-transaction example).
- J. R. Quinlan (1986). *Induction of decision trees.* Machine Learning 1, 81-106 (the weather table).
