# `ARM.sequential_rules` -- "if X happens, Y happens afterwards"

A sequential rule `X => Y` says that, in a sequence of events, once all the items of `X` have occurred, all the
items of `Y` occur later. Unlike an association rule the **order matters** (`coffee => tv` is a different rule
from `tv => coffee`); unlike a sequential pattern, the order *inside* `X` and inside `Y` does not, and the rule
comes with a measure of how reliable the prediction is.

```python
from ARMxtend.ARM import sequential_rules

rules = sequential_rules(sequences, min_supp=0.2, min_conf=0.7, max_antecedent=2, max_consequent=1)
rules[["antecedents", "consequents", "support", "confidence", "lift"]]
# {coffee} => {tv}   support=0.28 confidence=0.77 lift=2.44
```

`sequences` has the same format as in [`FIM.sequential`](../FIM/sequential.md) (lists of elements, each a set
of simultaneous items), and can be built with `sequences_from_events` or `sequences_from_items`.

## Definition

The rule `X => Y`, with `X` and `Y` disjoint, is **valid in a sequence** if there is a position `k` such that all
the items of `X` have appeared in the elements up to `k` and all the items of `Y` appear in elements strictly
after `k`; equivalently, `max(first occurrence of each item of X) < min(last occurrence of each item of Y)`. This
is the notion of sequential rule of Fournier-Viger et al. (2015). Then:

- `support(X => Y)` = valid sequences / all sequences;
- `confidence(X => Y)` = valid sequences / sequences that contain all the items of `X` (in any order);
- `lift` = confidence / fraction of sequences that contain all the items of `Y`.

`antecedent support` and `consequent support` are the fractions of sequences containing all the items of `X`
(`Y`), in any order. The other measures of `ARM._measures` (leverage, conviction, certainty factor, ...) are
computed from these three supports and added as columns.

**Why lift matters here.** If every sequence contains `alarm`, `bathroom` and `kitchen` in this order, then
`alarm => kitchen` has confidence 1.0 but lift 1.0: it is true but tells nothing. Sort or filter by `lift` (or
`metric="lift", min_threshold=1.2`) to keep the rules that carry information.

## Signature

```python
sequential_rules(sequences, min_supp, min_conf=0.8, max_antecedent=2, max_consequent=2,
                 metric="confidence", min_threshold=0.0)
contains_rule(sequence, antecedent, consequent)    # is the rule valid in this sequence?
```

`metric` / `min_threshold` add a second filter on top of `min_conf` (any measure of `ARM._measures`, `"lift"`,
...). The result is sorted by decreasing confidence and support.

## How it is mined

A rule can only be valid in a sequence that contains `X`, `Y` and `X u Y`. The miner therefore runs FP-Growth
on the sequences seen as baskets of items (order ignored), and for every frequent itemset tries its splits
into `X` and `Y` (up to the size limits), computing the number of valid sequences exactly with the first and
last occurrence of each item. The result equals the definition above (the test suite checks it against an
exhaustive enumeration). This is **not** the RuleGrowth algorithm of Fournier-Viger et al. (which searches the
rules incrementally without enumerating all the itemsets), but it yields the same rules with the same support
and confidence.

## Notes

- Keep `max_antecedent` and `max_consequent` small (1-3): the number of splits grows quickly, and long rules are
  hard to read.
- With sensor data, choose the granularity of the elements (`bin="5min"` in `sequences_from_events`) so that
  "at the same time" means what you want.
- Item names must not contain `-`.

Command line: [`armxtend sequential-rules`](../cli.md#sequences-and-sequential-rules).

## Reference

- P. Fournier-Viger, C.-W. Wu, V. S. Tseng, L. Cao, R. Nkambou (2015). *Mining partially-ordered sequential rules
  common to multiple sequences.* IEEE Transactions on Knowledge and Data Engineering 27(8), 2203-2216.
