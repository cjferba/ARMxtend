# `ARM.quantitative_rules` -- rules over numeric and categorical columns with merged intervals

Rules for tables that mix numbers and categories, such as `age in [30, 45) => income in [45, 96)` or
`sex=F & income in [56, 96] => buys=yes`. Every numeric column is split into **equal-frequency base intervals**, and
adjacent intervals are **merged into ranges** as long as the range does not become too frequent (a range that is almost
always true says nothing). This is the approach of Srikant and Agrawal (1996); the fuzzy alternative, which keeps
the boundaries soft, is [`FIM.FARE`](../FIM/FARE.md).

```python
from ARMxtend.ARM import quantitative_rules

rules = quantitative_rules(customers, min_supp=0.05, n_intervals=10, max_supp=0.4, min_threshold=0.8, interest=1.3)
rules[["antecedents", "consequents", "support", "confidence", "lift"]]
# {income=[56.302,96.45], sex=F}  => {buys=yes}              support=0.12 confidence=0.83 lift=3.84
# {age=[33.654,44.04), sex=F}     => {buys=yes}              support=0.08 confidence=0.82 lift=3.79
# {age=[33.654,44.04), buys=yes}  => {income=[56.302,96.45]} support=0.09 confidence=1.00 lift=3.32
```

`customers` is a `DataFrame` with columns `age`, `income`, `sex` and `buys` (see
[`examples/example_quantitative_rules.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_quantitative_rules.py)).

## How it works

1. **Items.** A numeric column with more than `n_intervals` distinct values is *quantitative*; the others (text,
   booleans, numbers with few values) give categorical items `column=value`. Missing values give no item.
2. **Base intervals.** A quantitative column is split with `n_intervals` quantiles (`method="quantile"`; fewer if there
   are ties) or equal widths (`method="uniform"`). The items are named `column=[low,high)` (`]` in the last interval),
   with a true minus sign (U+2212) for negative bounds, because item names cannot contain `-`.
3. **Ranges.** Every run of adjacent base intervals whose support is at most `max_supp` becomes an item; a single base
   interval always does. A record has, for each column, its base interval *and* every range that contains it.
4. **Frequent itemsets and rules.** The ranges form a hierarchy by inclusion (`[30,40)` is inside `[30,50)`), so the
   machinery is that of [`ARM.generalized_rules`](generalized_rules.md), with a taxonomy of ranges and the restriction that a
   rule never mentions two ranges of the same column. Itemsets are counted with the exact support, and the test suite checks them
   against an exhaustive enumeration.
5. **Interest.** With `interest=R`, the rules whose range could be widened or narrowed without changing the rule are
   dropped: `interesting_rules` keeps those whose support or confidence is at least `R` times what the rule with the wider
   range predicts ([R-interesting rules](generalized_rules.md#rules-that-only-echo-their-category-interesting_rules)).

## Signature

```python
quantitative_rules(table, min_supp=0.1, n_intervals=8, max_supp=0.5, columns=None, quantitative=None,
                   method="quantile", metric="confidence", min_threshold=0.8, max_len=None, interest=None,
                   partition=None)

IntervalPartition(n_intervals=8, method="quantile", max_supp=0.5, columns=None, quantitative=None)
    .fit(table)            # intervals_, edges_, taxonomy_, groups_, categorical_
    .transform(table)      # the base items of each row (also for new rows)
partial_completeness_intervals(n_quantitative, min_supp, completeness=1.5)
```

- The output has the columns of [`association_rules`](association_rules.md) plus `level` and (with `interest`) `interest`;
  `attrs` keeps `n_transactions`, `n_candidates` (so [`significant_rules`](significance.md) works), `taxonomy`,
  `supports` and `partition`.
- **`IntervalPartition`** can be fitted on one table and applied to another (`partition=`): the rules of a new period use
  the *same* intervals as the old ones. New values outside the fitted range fall into the first or last interval.
  `intervals_` is a `DataFrame` with the `item`, `column`, `low`, `high`, `support` and `n_base` of every range.
- **`partial_completeness_intervals`** returns `2n / (m (K - 1))`, the number of base intervals per column that
  Srikant and Agrawal derive to lose at most a "partial completeness" `K` of the rules (`n` numeric columns per rule,
  `m` the minimum support). The result is often large (160 for `n=2`, `m=0.05`, `K=1.5`); in practice 8-20 intervals plus merging work well.

## Intervals or fuzzy sets?

| | Merged intervals | Fuzzy sets ([`FIM.FARE`](../FIM/FARE.md)) |
|---|---|---|
| Boundaries | sharp, learned from the data by merging | soft, fixed by the partition (`FuzzyLib`) |
| A region that does not match the labels | found (the range adapts) | split over two labels |
| Values near a boundary | all or nothing | partial membership |
| Reading | `age in [28.5, 44)` | `age is medium` |

In the example the region `[30, 45)` is recovered by one range with confidence 0.93, while the fuzzy rules `age_low => income_high`
and `age_medium => income_high` get 0.55 and 0.52 because each covers half of it. The fuzzy version wins when
the boundaries are not sharp and the labels have a meaning of their own (a *warm* day).

## Notes

- `max_supp` and the number of intervals are the main knobs: the larger the ranges allowed, the more redundant rules.
  Use `interest` to prune them.
- Rules like `age in [20, 30) => age in [30, 40)` are never generated: one column, one range per rule.
- Keep `max_len` small (2-4); the number of range combinations grows quickly.

Command line: [`armxtend quantitative-rules`](../cli.md#quantitative-rules).

## Reference

- R. Srikant, R. Agrawal (1996). *Mining quantitative association rules in large relational tables.* Proceedings of
  the 1996 ACM SIGMOD International Conference on Management of Data, 1-12.
