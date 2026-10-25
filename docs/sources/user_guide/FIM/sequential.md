# `FIM.sequential` -- frequent sequential patterns (PrefixSpan)

Sequential patterns are routines that repeat **in a given order**: the daily activities of a person in a smart
home, the trajectory of a patient through a hospital, the pages of a web session, the alarms of a machine before a
failure. Where an association rule asks *what goes together*, a sequential pattern asks *what happens after
what*.

```python
from ARMxtend.FIM import prefixspan, format_sequence

sequences = [
    [{"alarm"}, {"bathroom"}, {"kitchen"}, {"coffee"}, {"tv"}],
    [{"alarm"}, {"bathroom"}, {"kitchen"}, {"door"}],
    ...
]
patterns = prefixspan(sequences, min_supp=0.3, min_len=3)
for _, p in patterns.head(3).iterrows():
    print(format_sequence(p["sequence"]), p["support"])
# <{alarm} {bathroom} {kitchen}> 1.0
```

## Data model

- A **sequence** is a list of **elements** ordered in time; an **element** is a set of items that happen at
  the same time (`{"door", "light"}`). A bare string is accepted as a one-item element (`"coffee"`).
- A **pattern** `<{a, b} {c}>` is *contained* in a sequence if there are two different elements, one before
  the other, containing `{a, b}` and `{c}` respectively. Other elements may lie in between.
- The **support** of a pattern is the fraction of sequences that contain it, counting each sequence once.
- Item names must not contain `-` (see the note in [`preprocessing.timeseries`](../preprocessing/timeseries.md)).

## Functions

```python
prefixspan(sequences, min_supp, max_len=None, min_len=1)
prefixspan_spark(spark, sequences, min_supp, max_pattern_length=10)
sequences_from_events(events, sequence_column, time_column, item_column, bin=None)
sequences_from_items(items, by=None)
contains_sequence(sequence, pattern)
format_sequence(pattern)       # ({'a','b'}, {'c'}) -> '<{a, b} {c}>'
```

- **`prefixspan`** is a pure-Python implementation of PrefixSpan (Pei et al., 2004): pattern growth over
  projected databases, with *s-extensions* (a new element after the prefix) and *i-extensions* (an item added to the
  last element). It returns a `DataFrame` with `support`, `sequence` (a tuple of `frozenset`) and `length` (number
  of items), sorted by decreasing support. `max_len` / `min_len` bound the number of items of a pattern.
- **`prefixspan_spark`** runs the same computation with Spark MLlib's `PrefixSpan` (`pyspark.ml.fpm`), for
  sequence databases that do not fit in a machine. It accepts a `SparkSession` or a `SparkContext`, and the
  sequences as a Python list or an RDD, and returns the same `DataFrame`. It needs the `spark` extra
  (`pip install "ARMxtend[spark]"`); `pyspark` is imported only when the function is called.
- **`sequences_from_events`** builds sequences from an **event log** (one row per event): one sequence per value of
  `sequence_column` (patient, user, day...), events ordered by `time_column`. Events with exactly the same time
  form one element; with `bin="5min"` (or `"1h"`) all the events of a sequence within the same interval do.
- **`sequences_from_items`** builds sequences from a boolean item table (one row per time step, e.g. the output
  of [`to_items`](../preprocessing/timeseries.md)): every non-empty row is an element. `by` splits the table into
  several sequences (a column, or a function of the index such as `lambda idx: idx.date` for one sequence per day).

## Example

An event log of a house (`day`, `time`, `event`), one sequence per day:

```python
from ARMxtend.FIM import prefixspan, sequences_from_events

sequences = sequences_from_events(events, "day", "time", "event")
prefixspan(sequences, min_supp=0.3, min_len=4).head(2)
#                                     sequence  support  length
# <{alarm} {bathroom} {kitchen} {phone}>   0.67       4
# <{alarm} {bathroom} {kitchen} {door}>    0.59       4
```

From patterns to predictions, see [`ARM.sequential_rules`](../ARM/sequential_rules.md): a frequent pattern says
*how often* the routine happens; a sequential rule says *how reliably* the second part follows the first.

Command line: [`armxtend sequences`](../cli.md#sequences-and-sequential-rules).

## Notes

- The search is exact. Cost depends mostly on `min_supp` and on how repetitive the sequences are; set `max_len`
  to bound it.
- The Spark version relies on MLlib's PrefixSpan, whose `minSupport` is relative as well; results match the
  local implementation (this is checked in the test suite).
- Repeated items inside a sequence do not increase the support (each sequence counts once), but they do allow
  patterns such as `<{a} {a}>`.

## Reference

- J. Pei, J. Han, B. Mortazavi-Asl, J. Wang, H. Pinto, Q. Chen, U. Dayal, M.-C. Hsu (2004). *Mining sequential
  patterns by pattern-growth: the PrefixSpan approach.* IEEE Transactions on Knowledge and Data Engineering
  16(11), 1424-1440.
