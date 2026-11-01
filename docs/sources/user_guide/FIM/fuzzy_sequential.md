# `FIM.fuzzy_sequential` -- fuzzy sequential patterns

[`FIM.sequential`](sequential.md) finds routines that repeat in a given order, but an event either happens or it
does not. Sensor data is rarely like that: the temperature was *high to degree 0.8*, the heating was *on to degree
0.6*. Fuzzy sequential patterns keep that **intensity**: a pattern holds in a sequence to a degree, and its support
is the average degree over all the sequences.

```python
from ARMxtend.FIM import format_sequence, fuzzy_prefixspan

days = [
    [{"temp_low": 0.9}, {"temp_medium": 0.6, "power_low": 0.4}, {"power_high": 0.8}],
    [{"temp_low": 0.3}, {"power_high": 0.2}],
    [{"temp_high": 1.0}, {"power_low": 1.0}],
]
patterns = fuzzy_prefixspan(days, min_supp=0.2, min_len=2)
for _, p in patterns.head(2).iterrows():
    print(format_sequence(p["sequence"]), round(p["support"], 2))
# <{temp_high} {power_low}> 0.33
# <{temp_low} {power_high}> 0.33
```

`<{temp_low} {power_high}>` holds to degree `min(0.9, 0.8) = 0.8` in the first day, `min(0.3, 0.2) = 0.2` in the
second and 0 in the third: its support is `(0.8 + 0.2 + 0)/3`.

## Data model

- A **fuzzy sequence** is a list of **events** ordered in time. An event is a dictionary `{item: degree}` with the
  degrees in `[0, 1]`; a list of `(item, degree)` pairs, a bare string or a collection of strings (all with degree 1)
  are accepted too. Items with degree 0 are ignored.
- A **pattern** `<x1 x2 ... xk>` is an ordered list of single items. Unlike in `FIM.sequential` there are no
  multi-item elements (`<{a, b} {c}>`): the events of a fuzzy sequence are already "baskets of items with degrees".
- Item names must not contain `-`.

## Definition

The degree of `<x1 ... xk>` in a sequence is the **best way it can be embedded**: the maximum, over the strictly
increasing positions `p1 < ... < pk`, of the t-norm of the degrees `d(p1, x1), ..., d(pk, xk)`. The fuzzy support is
the mean of that degree over the sequences, and a pattern is frequent if it reaches `min_supp`.

The t-norm is `min` (default) or the product (`tnorm="product"`, stricter: every extra event multiplies the degree
by a number at most 1). With either of them, extending a pattern never increases its degree, so a pattern whose
prefix is infrequent cannot be frequent: the **anti-monotone pruning of PrefixSpan stays valid** and the search is
exact. If every degree is 1, the fuzzy support equals the usual support of the patterns whose elements have a single
item (the test suite checks it against `prefixspan`).

## Functions

```python
fuzzy_prefixspan(sequences, min_supp, max_len=None, min_len=1, tnorm="min")
fuzzy_pattern_degree(sequence, pattern, tnorm="min")
fuzzy_sequences_from_memberships(memberships, by=None, min_degree=0.0)
```

- **`fuzzy_prefixspan`** returns a `DataFrame` with `support` (fuzzy), `sequence` (a tuple of one-item `frozenset`,
  as in `prefixspan`, so `format_sequence` prints it) and `length`, sorted by decreasing support. `max_len` and
  `min_len` bound the number of items. Memory is proportional to *sequences x longest sequence x distinct items*.
- **`fuzzy_pattern_degree`** gives the degree of one pattern (a list of items) in one sequence.
- **`fuzzy_sequences_from_memberships`** builds the sequences from a table of membership degrees with one row per
  time step, such as the one [`FuzzyLib.AutoFuzzification`](../preprocessing/FuzzyLib.md) produces from numeric
  series. Each row is an event holding the items whose degree is above `min_degree` (use it to drop the noise of
  tiny memberships, e.g. `0.2`); `by` says which sequence each row belongs to (a day, a patient; the order of the rows
  is kept inside each sequence).

## Example: from numeric readings to patterns

```python
from ARMxtend.FIM import fuzzy_prefixspan, fuzzy_sequences_from_memberships, format_sequence
from ARMxtend.preprocessing import FuzzyLib

library = FuzzyLib()
library.data = readings                      # columns: day, temp, power (rows already in time order)
labels = library.AutoFuzzification(Atri=["temp", "power"], n_labels=3, method="uniform")
days = fuzzy_sequences_from_memberships(library.data[labels], by=library.data["day"], min_degree=0.2)

patterns = fuzzy_prefixspan(days, min_supp=0.2, max_len=2, min_len=2)
```

`<{temp_low} {power_high}>` (a cold morning followed by a high consumption) is frequent, and its mirror image
`<{power_high} {temp_low}>` is not: the order matters. The full script, with data generated so that the answer is
known, is [`examples/example_fuzzy_sequences.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_fuzzy_sequences.py)
(see [Examples](../../examples.md#fuzzy-sequential-patterns-and-rules-fimfuzzy_sequential-armfuzzy_sequential_rules)).

Command line: [`armxtend fuzzy-sequences`](../cli.md#fuzzy-sequences-and-fuzzy-sequential-rules).

## Notes

- The patterns are *inspired by* the soft sequential patterns of Fiot, Laurent and Teisseire (2007), but the
  definition above (best embedding, t-norm, mean over sequences) is this library's own and the algorithms of that
  paper are not implemented.
- A high `min_supp` is the usual way to keep the output small; fuzzy supports are smoother than crisp ones, so
  many more patterns than in the crisp case reach a given value. Use `max_len` to bound the search.
- To turn the patterns into "if X, then later Y" statements with confidence and lift, use
  [`ARM.fuzzy_sequential_rules`](../ARM/fuzzy_sequential_rules.md).

## Reference

- C. Fiot, A. Laurent, M. Teisseire (2007). *From crispness to fuzziness: three algorithms for soft sequential
  pattern mining.* IEEE Transactions on Fuzzy Systems 15(6), 1263-1277.
- E. P. Klement, R. Mesiar, E. Pap (2000). *Triangular Norms.* Kluwer.
