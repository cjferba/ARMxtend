# `ARM.association_rules`

Generates crisp association rules from a set of frequent itemsets already mined (e.g. with
[`FFIM.fpgrowth`](../FFIM.md) or [`FIM.apriori`](../FIM/apriori.md)/[`FIM.Eclat`](../FIM/Eclat.md)
on Spark).

## Signature

```python
association_rules(df, metric="confidence", min_threshold=0.8, support_only=False)
```

- **df** (`pandas.DataFrame`): frequent itemsets, with columns `support` (float) and `itemsets`
  (a `frozenset` of items).
- **metric** (`str`): one of `"support"`, `"confidence"`, `"lift"`, `"leverage"`, `"conviction"`,
  `"certainty_factor"` or any of the [additional measures](#additional-interest-measures) below.
- **min_threshold** (`float`): minimum value of `metric` for a rule to be returned.
- **support_only** (`bool`): if `True`, only computes `support` (useful when `df` does not contain
  the support of every antecedent/consequent needed for the other metrics).

Returns a `pandas.DataFrame` with columns `antecedents`, `consequents`, `antecedent support`,
`consequent support`, `support`, `confidence`, `lift`, `leverage`, `conviction`,
`certainty_factor`, plus one column per additional measure.

## Additional interest measures

All are functions of `(supp(AB), supp(A), supp(B))` (equivalently of the 4-fold table
`<a, b, c, d> = <sAB, sA-sAB, sB-sAB, 1-sA-sB+sAB>`), defined in `ARMxtend.ARM._measures`. They are
neutral (`0`) when undefined (`0/0`), never `NaN`.

| Measure (`metric=`) | Definition | Range | Symmetric |
|---|---|---|---|
| `jaccard` | `sAB / (sA + sB - sAB)` | [0, 1] | yes |
| `cosine` | `sAB / sqrt(sA sB)` | [0, 1] | yes |
| `kulczynski` | `(Conf(A->B) + Conf(B->A)) / 2` -- robust to null transactions (Wu, Chen & Han, 2010) | [0, 1] | yes |
| `all_confidence` | `sAB / max(sA, sB)` = `min(Conf(A->B), Conf(B->A))` | [0, 1] | yes |
| `max_confidence` | `max(Conf(A->B), Conf(B->A))` | [0, 1] | yes |
| `added_value` | `Conf(A->B) - supp(B)` (Tan, Kumar & Srivastava, 2004) | [-1, 1] | no |
| `zhangs_metric` | `(sAB - sA sB) / max(sAB(1-sA), sB(sA-sAB))` (Zhang, 2000) | [-1, 1] | no |
| `yules_q` | `(ad - bc) / (ad + bc)` | [-1, 1] | yes |

For asymmetric-direction questions use `added_value`, `zhangs_metric` or `certainty_factor`; for
"are A and B just co-occurring?" the symmetric ones. In the fuzzy case
([`FIM.FARE`](../FIM/FARE.md)) every one of them is evaluated at each alpha-cut and aggregated with
the RL-theory weights, exactly like `FConf` and `FCF`.

## Certainty factor

Besides the classic confidence/lift/conviction, `metric="certainty_factor"` computes the
Shortliffe & Buchanan certainty factor (Delgado, Ruiz & Sanchez):

```
CF(A -> B) = (Conf(A->B) - supp(B)) / (1 - supp(B))   if Conf(A->B) > supp(B)
           = (Conf(A->B) - supp(B)) / supp(B)          if Conf(A->B) < supp(B)
           = 0                                          if Conf(A->B) = supp(B)
```

Unlike confidence, `CF` is bounded in `[-1, 1]` and takes the *base rate* of the consequent into
account: a positive value means the antecedent genuinely increases belief in the consequent, a
negative value means it *decreases* it (a negative or independent association -- exactly the kind
of rule that plain confidence can misleadingly rank as "interesting"), and 0 means no change. See
[Citing ARMxtend](../../cite.md) for the reference.

## Example

```python
import pandas as pd
from ARMxtend.ARM import association_rules

df = pd.DataFrame({
    "support": [0.5, 0.8, 0.7, 0.3],
    "itemsets": [frozenset(["bread"]), frozenset(["milk"]), frozenset(["butter"]),
                frozenset(["bread", "milk"])],
})
rules = association_rules(df, metric="confidence", min_threshold=0.5)
print(rules[["antecedents", "consequents", "support", "confidence", "certainty_factor"]])
#   antecedents consequents  support  confidence  certainty_factor
# 0     (bread)      (milk)      0.3         0.6             -0.25
```

Here confidence alone (0.6 >= 0.5) would flag "bread -> milk" as interesting, but the negative
certainty factor reveals that milk is already present in 80% of all transactions, so bread's
presence actually makes milk *less* likely relative to its base rate.
