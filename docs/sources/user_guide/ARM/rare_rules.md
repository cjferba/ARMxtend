# `ARM.rare_rules` and `FIM.minimal_rare_itemsets` -- rare combinations of frequent things

Some of the most interesting patterns are **infrequent**: two drugs that are each common but rarely given together and then
cause an event, a fault that follows a particular pair of alarms. To reach them with a minimum support you have to lower it
until nearly every itemset is "frequent", and millions come out. Rare-itemset mining looks only where it matters.

```python
from ARMxtend.ARM import rare_rules
from ARMxtend.FIM import minimal_rare_itemsets

# a and d are frequent (0.30 each) but together only 0.15; b is frequent and goes alone
baskets = [["a", "d"]] * 15 + [["a"]] * 15 + [["d"]] * 15 + [["b"]] * 55

minimal_rare_itemsets(baskets, max_supp=0.2)
#  support itemsets
#     0.15      a,d

rare_rules(baskets, max_supp=0.2, min_conf=0.5)
# antecedents consequents  support  confidence     lift
#           a           d     0.15         0.5 1.666667
#           d           a     0.15         0.5 1.666667
```

## Definitions

An itemset is **frequent** if its support is at least `max_supp` and **rare** if it is below. A rare itemset is
**minimal** (an MRI) if *all its proper subsets are frequent*: it is rare only because of the combination. The MRIs are a
compact basis: any rare itemset is an MRI or contains one. `min_supp` and `min_count` are floors against noise (what
happens once or twice is not a pattern); an itemset is rare when `min_supp <= support < max_supp` and it appears in at least
`min_count` transactions.

The MRIs are found level by level, as in Apriori: frequent itemsets of size `k-1` are joined; a frequent candidate keeps
growing, a rare one is an MRI and stops (its supersets would not be minimal). The result is exact; the tests compare it with
the definition applied to every itemset.

## Rare rules

`rare_rules` splits every MRI of two or more items into `X => Y`. Since all proper subsets of an MRI are frequent, both
`X` and `Y` are frequent: the rule is rare only through the combination, and its **confidence and lift tell whether that
combination is more than chance**. The output is the usual rule frame (all the [measures](association_rules.md)), sorted by
decreasing confidence and support, with `attrs` `n_transactions` and `n_candidates` (the splits examined) so that
[`significant_rules`](significance.md) and [`prune_redundant_rules`](redundancy.md) work as with any other rules.

In the [shop example](../../examples.md#top-k-rare-and-high-utility-rules-armtop_k_rules-armrare_rules-armhigh_utility_rules)
a flat miner with a threshold low enough to see the rare pairs (0.01) returns 47 rules with lift of 2 or more, many of them
variations of the same pair (`diapers, fruit => wipes`); `rare_rules` returns the 3 that matter.

## Signature

```python
minimal_rare_itemsets(transactions, max_supp, min_supp=0.0, min_count=1, min_len=1, max_len=None)
rare_rules(transactions, max_supp, min_supp=0.0, min_count=2, min_conf=0.6, min_lift=None,
           max_antecedent=None, max_consequent=None, max_len=None)
```

`minimal_rare_itemsets` returns `support` / `itemsets` sorted by decreasing support; `attrs['n_frequent']` is the number of
frequent itemsets visited (for comparison with ordinary mining).

## Things to know

- **`max_supp` is the knob.** It defines what is frequent, so it decides which combinations are minimal; changing it changes
  the rules, not just how many there are.
- **A rare single item has no rules.** It is an MRI of size one and needs two items to be split. Rules between a rare item
  and the rest are not produced because their antecedent or consequent would not be frequent.
- **Supersets of an MRI are not generated** (`bread, wine => cheese` when `wine => cheese` is already there). That is the
  point of a compact basis, but it means that a rare combination that needs a third item to appear is only found if its
  pairs are frequent.
- **Rare combinations of independent items have lift near 1.** In the example, every pair of a common item and a baby
  product is rare, with lift about 1; `min_lift` (or a significance test) is what separates them from the real ones.
- **`min_count=2` by default** in `rare_rules` (and 1 in `minimal_rare_itemsets`): a rule seen once has no weight as a rule.

Command line: [`armxtend rare-rules`](../cli.md#rare-rules).

## References

- L. Szathmary, A. Napoli, P. Valtchev (2007). *Towards rare itemset mining.* Proceedings of the 19th IEEE International
  Conference on Tools with Artificial Intelligence, 305-312 (minimal rare itemsets and the Apriori-Rare algorithm).
