# `ARM.fuzzy_sequential_rules` -- fuzzy "if X happens, Y happens afterwards"

The fuzzy version of [`ARM.sequential_rules`](sequential_rules.md): the events carry a degree (*the temperature was
low to degree 0.9*), and a rule `X => Y` is valid in a sequence **to a degree**. Every day, patient or run
contributes what it really shows instead of being counted in full or not at all.

```python
from ARMxtend.ARM import fuzzy_sequential_rules

rules = fuzzy_sequential_rules(days, min_supp=0.1, min_conf=0.5, max_antecedent=1, max_consequent=1)
rules[["antecedents", "consequents", "support", "confidence", "lift"]]
# {temp_low} => {power_high}   support=0.22 confidence=0.63 lift=2.76
```

`days` has the format of [`FIM.fuzzy_sequential`](../FIM/fuzzy_sequential.md): lists of events `{item: degree}`,
built by hand or with `fuzzy_sequences_from_memberships` from a table of memberships.

## Definition

With `T` the t-norm (`min` or product), the degree of `X => Y` in a sequence is

```text
mu_X(k) = T over x in X of  max over p <= k of d(p, x)     -- to what degree all of X has appeared by event k
mu_Y(k) = T over y in Y of  max over p >  k of d(p, y)     -- to what degree all of Y appears after event k
degree(X => Y) = max over k of T(mu_X(k), mu_Y(k))
```

and the measures are the means over the sequences, with the same names as in the crisp case:

- `support(X => Y)` = mean degree of the rule;
- `antecedent support` (`consequent support`) = mean over the sequences of `T_x max_p d(p, x)`: to what degree all
  the items of `X` (of `Y`) appear, in any order;
- `confidence` = `support(X => Y)` / `antecedent support`; `lift` and the other measures of `ARM._measures` are
  computed from the three supports and added as columns, so the table is filtered, sorted and exported like any
  other rule table.

With degrees 0 and 1 every number equals the one of `sequential_rules` (checked in the test suite).
`fuzzy_rule_degree(sequence, antecedent, consequent, tnorm)` gives the degree of one rule in one sequence, which is
useful to see *which* days fulfil a rule and how well.

## Signature

```python
fuzzy_sequential_rules(sequences, min_supp, min_conf=0.8, max_antecedent=2, max_consequent=2,
                       metric="confidence", min_threshold=0.0, tnorm="min")
fuzzy_rule_degree(sequence, antecedent, consequent, tnorm="min")
```

The result is sorted by decreasing confidence and support, and `result.attrs["n_sequences"]` holds the number of
sequences. `metric` / `min_threshold` add a second filter, as in `sequential_rules`.

## How it is mined

A rule can only reach a support `s` if the itemset `X u Y` reaches `s` when the sequences are seen as baskets (the
degree of an itemset in a sequence is the t-norm of the best degrees of its items, which can only go down when items
are added). The miner therefore finds the frequent itemsets level by level and tests every split of each one into
`X` and `Y` within the size limits, evaluating the rule degree of all the sequences at once with cumulative
maxima. Memory is proportional to *sequences x longest sequence x distinct items*.

## Reading the numbers

- A **fuzzy confidence is not comparable one-to-one with a crisp one.** In the example of
  [`examples/example_fuzzy_sequences.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_fuzzy_sequences.py)
  the crisp rule `cold => peak` has confidence 0.96 with a cut at 12 degrees and 0.82 with a cut at 14, while the fuzzy
  `temp_low => power_high` has 0.63: a mildly cold morning is only partly "low" and the evening after it only partly
  "high", so it adds a small amount to both the numerator and the denominator. What the fuzzy rule does not need is the
  cut point (the crisp *support* of the same rule goes from 0.21 to 0.62 as the cut moves from 8 to 14 degrees), and it
  tells how well each day fulfils the rule (`fuzzy_rule_degree`).
- The statistical tests of [`ARM.significance`](significance.md) count transactions and do not apply to fuzzy
  supports; for fuzzy sequential rules use the supports and lift as a ranking and validate on held-out sequences.
- As with crisp rules, **sort by `lift`** to leave out the rules that are true but uninformative.
- `min_degree` in `fuzzy_sequences_from_memberships` matters: with triangular memberships almost every reading has
  some degree on its neighbouring labels, and the resulting tiny degrees add noise (and work). `0.1`-`0.3` is a
  reasonable range.

Command line: [`armxtend fuzzy-sequential-rules`](../cli.md#fuzzy-sequences-and-fuzzy-sequential-rules).

## Reference

- P. Fournier-Viger, C.-W. Wu, V. S. Tseng, L. Cao, R. Nkambou (2015). *Mining partially-ordered sequential rules
  common to multiple sequences.* IEEE Transactions on Knowledge and Data Engineering 27(8), 2203-2216 (the crisp rule
  notion this page fuzzifies).
