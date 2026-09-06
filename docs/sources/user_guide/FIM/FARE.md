# `FIM.FARE` -- fuzzy association rules (single machine)

FARE (Fuzzy Association Rule Extraction) generates association rules from fuzzy frequent itemsets,
using the alpha-cut decomposition of the Representation by Levels (RL) theory of Delgado, Ruiz,
Sanchez & Serrano (2011), as formalized for Big Data by Fernandez-Basso, Ruiz & Martin-Bautista
(2021) -- see [Citing ARMxtend](../../cite.md).

## Fuzzy transactions and itemsets

A fuzzy transaction `t` assigns every item `i` a membership degree `t(i)` in `[0, 1]`. An itemset
`A`'s membership degree in `t` is the minimum of its items' degrees: `t(A) = min_{i in A} t(i)`. A
crisp transaction is the special case where every degree is 0 or 1.

## Alpha-cuts and the FSupp / FConf / FCF measures

Rather than picking a single threshold, a fuzzy measure is computed by decomposing `[0, 1]` into a
set of `p` equidistant alpha-cuts `Lambda = {alpha_1 > alpha_2 > ... > alpha_p}` (with
`alpha_(p+1) = 0` by convention), and *integrating* the corresponding crisp measure at every cut,
weighted by the cut's width `(alpha_i - alpha_(i+1))`:

```
FSupp(A)      = sum_i (alpha_i - alpha_(i+1)) * |{t : t(A) >= alpha_i}| / |D|
FSupp(A -> B) = sum_i (alpha_i - alpha_(i+1)) * |{t : t(A) >= alpha_i and t(B) >= alpha_i}| / |D|
FConf(A -> B) = sum_i (alpha_i - alpha_(i+1)) * |{t : t(A) >= alpha_i and t(B) >= alpha_i}| / |{t : t(A) >= alpha_i}|
FCF(A -> B)   = sum_i (alpha_i - alpha_(i+1)) * CF_i(A, B)
```

where `CF_i` is the crisp certainty factor (see [`ARM.association_rules`](../ARM/association_rules.md))
computed from the 4-fold table at level `alpha_i`. Note `FSupp(A -> B) = FSupp(A U B)`, since
`t(A U B) = min(t(A), t(B)) >= alpha_i` exactly when both `t(A) >= alpha_i` and `t(B) >= alpha_i`.
When `|{t : t(A) >= alpha_i}| = 0` at some level (the antecedent never appears at that level), that
level's confidence term is taken to be 1, following the "0/0" convention in Ruiz et al. (2016).

For equidistant alpha-cuts, every weight equals `1/p`, so `FSupp`/`FConf`/`FCF` are simply the
*average* of the crisp measure across the `p` alpha-cuts.

## `fuzzy_association_rules`

```python
fuzzy_association_rules(freq_itemsets, num_alpha, metric="confidence", min_threshold=0.8)
```

- **freq_itemsets** (`dict[str, numpy.ndarray]`): fuzzy frequent itemsets, mapping `'A-B-C'` to its
  bit-list of relative support at each of `num_alpha` alpha-cuts, as returned by
  [`FIM.Eclat.FuzzyDECLAT`](Eclat.md), [`FIM.BD_FARE.FuzzyDAprioriTID`](BD_FARE.md) or
  [`FFIM.fuzzy_fpgrowth`](../FFIM.md) (all three are interchangeable). Must include the bit-list of
  every non-empty subset of each itemset -- guaranteed by the downward-closure property of those
  mining algorithms.
- **num_alpha** (`int`): number of alpha-cuts used to mine `freq_itemsets` (must match).
- **metric** (`str`): `"support"` (FSupp), `"confidence"` (FConf) or `"certainty_factor"` (FCF).
- **min_threshold** (`float`): minimum value of `metric` for a rule to be kept.

Returns a `pandas.DataFrame` with columns `antecedents`, `consequents`, `support` (FSupp),
`confidence` (FConf) and `certainty_factor` (FCF).

## Example

Reproducing the numerical example of Ruiz et al. (2016), Section 4.4 (fuzzy database in their
Table 2, itemset `A={i1,i3} -> B={i4}`, alpha-cuts `{1, 0.8, 0.6, 0.4, 0.2}`):

```python
from pyspark import SparkContext
from ARMxtend.FIM.Eclat import FuzzyDECLAT
from ARMxtend.FIM.FARE import fuzzy_association_rules

sc = SparkContext(master="local[*]")
degrees = {
    "i1": [1, 1, 0.4, 0.6, 0.4, 0],
    "i2": [0.2, 1, 0.1, 0, 0.1, 1],
    "i3": [1, 0.8, 0.7, 0.4, 0, 0],
    "i4": [0.8, 0, 0.6, 0.4, 0, 0],
    "i5": [0.9, 0, 0, 0.5, 0, 0],
}
transactions = sc.parallelize([[(item, degrees[item][t]) for item in degrees] for t in range(6)])

freq_itemsets = FuzzyDECLAT.run(sc, transactions, min_supp=0.0, num_alpha=5)
rules = fuzzy_association_rules(freq_itemsets, num_alpha=5, metric="support", min_threshold=0.0)

rule = rules[(rules["antecedents"] == frozenset(["i1", "i3"])) & (rules["consequents"] == frozenset(["i4"]))]
print(rule[["support", "confidence", "certainty_factor"]])
#    support  confidence  certainty_factor
# 0  0.266667         0.5              0.33
```

matching the paper's reported `FSupp=0.266`, `FConf=0.5`, `FCF=0.33` exactly (see
`tests/test_fim_fare.py`).
