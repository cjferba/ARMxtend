# `ARM.significance` -- are the rules real?

Mining with thresholds on support and confidence finds the rules that *pass the thresholds*, not the rules that are
*stronger than chance*. The more rules are examined, the more of them appear by accident -- and in time series, where
consecutive rows are not independent, even more. This module attaches p-values to rules, corrects them for multiple
testing, and offers permutation tests that stay valid when the usual assumptions fail.

```python
from ARMxtend.ARM import association_rules, significant_rules, confidence_intervals

rules = association_rules(itemsets, metric="confidence", min_threshold=0.45)   # 100 rules in the demo data
real = significant_rules(rules, alpha=0.05, correction="fdr_bh", n_transactions=300)   # 22
real[["antecedents", "consequents", "confidence", "p_value", "p_adjusted"]]
```

In the [example](../../examples.md#statistical-significance-of-rules-armsignificance) 300 random baskets, with a single
planted association, give 100 rules with confidence >= 0.45, of which 54 do not involve the real association.

## 1. A test for each rule

For `A => C` over `n` transactions, the 2x2 table (A / not A) x (C / not C) is rebuilt from the supports, and the null
hypothesis is that A and C are independent.

```python
rule_significance(rules, n_transactions=None, method="fisher", alternative="greater")
```

- `method="fisher"` (default): exact test. The p-value is the tail of the hypergeometric distribution: the
  probability of seeing at least `n(A and C)` joint occurrences if A and C were independent. It is computed by
  summing the probabilities from the observed value outwards and stopping once the terms stop counting, so it is
  exact (checked against integer enumeration) and fast even with millions of transactions.
- `method="chi2"`: Pearson's chi-squared (without continuity correction); good when the expected counts are >= 5.
- `alternative="greater"` (default) tests *positive* association, which is what a rule claims; `"less"` and
  `"two-sided"` (twice the smaller one-sided p-value) are available.
- `n_transactions` is needed to turn supports into counts. Rules from
  [`temporal_association_rules`](temporal_rules.md) carry it in `rules.attrs["n_transactions"]`. If the supports are not
  multiples of `1/n` a `ValueError` is raised (wrong `n`, or fuzzy rules).

The result is a copy of the rules with a `p_value` column. These tests assume **independent transactions** (see
section 4 for time series) and need **crisp** rules.

## 2. Multiple testing

```python
adjust_pvalues(p_values, method="fdr_bh", n_tests=None)
significant_rules(rules, alpha=0.05, correction="fdr_bh", n_transactions=None, method="fisher", n_tests=None)
candidate_rule_count(itemsets)
```

| `correction` | Controls | Notes |
|---|---|---|
| `bonferroni` | probability of any false positive (FWER) | simplest, most conservative |
| `holm` | FWER | uniformly more powerful than Bonferroni (Holm, 1979) |
| `fdr_bh` | expected share of false discoveries (FDR) | Benjamini-Hochberg (1995); the usual choice when you accept a few false rules among many; independent or positively dependent tests |
| `fdr_by` | FDR | Benjamini-Yekutieli (2001); valid under any dependence, more conservative |

**How many hypotheses?** Correcting only for the rules you *see* understates the problem: they were selected for having
high confidence. The right family is every rule that was *tested*. `association_rules` and `temporal_association_rules`
record it in `rules.attrs["n_candidates"]` (all the ways of splitting the frequent itemsets into antecedent and
consequent; for temporal rules, the structurally valid ones), and `significant_rules` uses it by default. Pass
`n_tests=` to override it, or `candidate_rule_count(itemsets)` to compute it from the itemsets. `significant_rules`
returns the rules with `p_adjusted <= alpha`, sorted by `p_adjusted`.

Many of the surviving rules are usually variants of one real rule with irrelevant extra items; combine with
[`prune_redundant_rules`](redundancy.md).

## 3. Confidence intervals

```python
confidence_intervals(rules, n_transactions=None, level=0.95, method="wilson", n_resamples=1000, random_state=0)
```

Adds `support_low/high`, `confidence_low/high` and `lift_low/high`. `"wilson"` is the Wilson score interval of a
proportion (the support over `n`, the confidence over the transactions that satisfy the antecedent; no interval for
the lift). `"bootstrap"` resamples the transactions with replacement and takes percentiles; for a rule it is enough to
resample its four 2x2 cells (a multinomial draw), which is equivalent to resampling the rows and needs no transaction
table. Both assume independent transactions.

## 4. Permutation tests

A test of independence between rows does not hold for sensor data: two *independent* slowly-varying series (a heater
on for hours, a window open for days) overlap by chance far more than independent rows would, and Fisher reports absurd
p-values. Permutation tests compare the real rules with what the **same miner** finds in data where the relations
between items have been destroyed but each item keeps its frequency (and, optionally, its own autocorrelation):

```python
from ARMxtend.ARM import crisp_rule_miner, permutation_null, permutation_p_values

miner = crisp_rule_miner(min_supp=0.05, metric="confidence", min_threshold=0.5)   # or any `table -> rules` function
rules = miner(items)
null = permutation_null(items, miner, n_permutations=200, null="shift", random_state=0)
rules = permutation_p_values(rules, null)          # adds p_permutation
```

- `shuffled_items(items, null)`: `null="permute"` permutes every column independently (independent rows);
  `null="shift"` (default) rotates every column circularly by a random amount, which keeps each series' autocorrelation and
  cycles and only breaks the alignment between series. **Use `shift` for time series and lagged rules.**
- `permutation_null` re-mines each shuffled copy and records the *best* value of a statistic among its rules.
  With `measure="p_value"` (default) the statistic is the smallest Fisher p-value (Westfall and Young's minP: it accounts for
  rules with little support reaching extreme confidences by chance); with any other column of the rules (`"lift"`,
  `"confidence"`, `"zhangs_metric"`, ...) it is the maximum, which is the only option for **fuzzy** rules.
- `p_permutation = (1 + #{shuffled copies whose best rule is at least as good}) / (1 + B)`. Because each rule is
  compared with the best of *all* the rules of every copy, it already controls the family-wise error rate and the
  selection of rules by their own values; no further correction is needed. The smallest possible value is `1/(B+1)`.
- `significant_miner(miner, alpha, ...)` wraps a miner so that it returns only significant rules; use it as the `miner=`
  of [`window_rules`](rule_drift.md) to check which rules of each time window are real before following them over time.

In the [example](../../examples.md#statistical-significance-of-rules-armsignificance), 12 pairs of independent regime-switching series give 6 false alarms with
Fisher (p < 0.01, some with p ~ 1e-100) and 1 with the circular-shift test (p < 0.05, the nominal rate).

## 5. Fuzzy rules, rule by rule: `fuzzy_rule_significance`

Fisher and chi-squared need integer counts. For fuzzy rules there is a per-rule permutation test:

```python
from ARMxtend.ARM import fuzzy_significant_rules

rules = fuzzy_association_rules(frequent, 5, metric="confidence", min_threshold=0.5)   # has attrs['n_candidates']
fuzzy_significant_rules(rules, transactions, alpha=0.05, n_permutations=999, random_state=0)
```

For every rule `A => B` the membership of each side in each transaction is the minimum of its items' degrees; the statistic
is the **scalar-cardinality confidence** `sum(min(a, b)) / sum(a)` (not the alpha-cut aggregate `FConf`, but the same idea);
the null hypothesis shuffles `b` among the transactions, and the p-value is `(1 + #shuffles >= observed) / (1 + n_permutations)`.
Then `adjust_pvalues` corrects it, over `n_candidates` (the splits examined, now annotated by `fuzzy_association_rules`).
`transactions` is the list of `(item, degree)` lists used to mine, or a table of degrees.

- **The minimum p-value is `1 / (n_permutations + 1)`**: a Bonferroni correction over `m` tests needs at least `m / alpha`
  shuffles to reject anything. Use `fdr_bh` or raise `n_permutations`.
- **It tests one rule against shuffled data, not the maximum over all rules** (that is `permutation_null`): the correction takes
  care of the multiplicity, not of the selection by thresholds, as with the crisp tests.
- **Cost** is `n_permutations x transactions` per rule: 1000 shuffles of 10 000 transactions takes about 30 ms per rule.
- Command line: `armxtend fuzzy-rules --significance 0.05 --permutations 999 --seed 1`.

## Choosing

| Situation | Use |
|---|---|
| Independent transactions (baskets, records) and crisp rules | `significant_rules` (Fisher + `fdr_bh`) |
| You want few false rules at the price of power | `correction="holm"` or the permutation test |
| Time series, sensors, lagged rules | permutation test with `null="shift"` |
| Fuzzy rules | `fuzzy_significant_rules` (per rule), or the max-statistic permutation test with `measure="lift"` |
| Sequential rules | not covered (the contingency table is different); use a permutation test with your own miner |

Command line: `--alpha`, `--correction` and `--test` in [`armxtend rules`](../cli.md#significance-of-the-rules),
`temporal-rules` (also `--permutations`, `--null`, `--seed`) and `rule-drift`.

## References

- R. A. Fisher (1922, 1935) and K. Pearson (1900): the exact test and the chi-squared test of independence.
- S. Holm (1979). *A simple sequentially rejective multiple test procedure.* Scandinavian Journal of Statistics 6, 65-70.
- Y. Benjamini, Y. Hochberg (1995). *Controlling the false discovery rate.* J. Royal Statistical Society B 57, 289-300.
- Y. Benjamini, D. Yekutieli (2001). *The control of the false discovery rate in multiple testing under dependency.*
  Annals of Statistics 29, 1165-1188.
- P. H. Westfall, S. S. Young (1993). *Resampling-based multiple testing.* Wiley.
- G. I. Webb (2007). *Discovering significant patterns.* Machine Learning 68, 1-33.
- E. B. Wilson (1927). *Probable inference, the law of succession, and statistical inference.* JASA 22, 209-212.
