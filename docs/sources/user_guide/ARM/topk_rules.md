# `ARM.top_k_rules` and `FIM.top_k_itemsets` -- the k best, with no support threshold

Choosing a minimum support is guesswork: too high and nothing comes out, too low and millions of itemsets do. Here you
ask for **the `k` itemsets or rules with the highest support** and the library finds exactly those.

```python
from ARMxtend.ARM import top_k_rules
from ARMxtend.FIM import top_k_itemsets

baskets = [["a", "b", "c"], ["a", "b"], ["a", "c"], ["a"], ["b", "c"], ["b"], ["a", "b", "c", "d"]]

top_k_itemsets(baskets, 3, min_len=2)
#    support itemsets
#   0.428571   {a, b}
#   0.428571   {a, c}
#   0.428571   {b, c}

top_k_rules(baskets, 2, min_conf=0.7)
# antecedents consequents  support  confidence  lift
#           c           a 0.428571        0.75  1.05
#           c           b 0.428571        0.75  1.05
```

In the [shop example](../../examples.md#top-k-rare-and-high-utility-rules-armtop_k_rules-armrare_rules-armhigh_utility_rules)
(2000 baskets, confidence 0.5) a support threshold of 0.3 gives 0 rules and 0.01 gives 93; `top_k_rules(baskets, 5,
min_conf=0.5)` gives the 5 you asked for.

## Signature

```python
top_k_itemsets(transactions, k, min_len=1, max_len=None, min_supp=0.0)
top_k_rules(transactions, k, min_conf=0.6, min_lift=None, max_antecedent=None, max_consequent=None, max_len=6, min_supp=0.0)
```

`transactions` is a list of baskets (any iterable of items), as in the rest of the library. `top_k_itemsets` returns the
usual `support` / `itemsets` frame (so [`association_rules`](association_rules.md) takes it); `top_k_rules` returns the
rule frame with all the [interest measures](association_rules.md), sorted by decreasing support. `attrs` keeps
`n_transactions` and, for the rules, `n_candidates` (the rules examined), which is the family
[`significant_rules`](significance.md) corrects over.

## How it works, and why the result is exact

Support is anti-monotone: an itemset is never more frequent than any of its subsets. The itemsets are therefore explored
**best-first**: a priority queue ordered by support starts with the single items, and each time an itemset is taken out
its extensions are put in. Itemsets come out in non-increasing support, so the first `k` are the `k` most frequent, and
the search stops without visiting the rest.

For rules, every split `X => Y` of an itemset has the support of that itemset. Once the next itemset has less support than
the `k`-th best rule found so far, nothing that follows can enter the top `k`, and the search stops. Ties in support are
broken deterministically: fewer items first, then lexicographic order for itemsets; higher confidence first, then
lexicographic order for rules. The tests compare both functions with the exhaustive enumeration.

## Things to know

- **`min_conf`, `min_lift`, `max_antecedent` and `max_consequent` are filters applied before the ranking**, so you get
  the `k` best rules *that satisfy them*, not the `k` best rules of which some are then discarded. If fewer than `k`
  rules satisfy them you get all of them.
- **The most frequent rules are the most obvious ones.** Without a lift filter the top of the list is full of popular items
  (`fruit => bread` with lift 1.00); `min_lift=1.5` or similar brings up the ones that associate. The shop example shows both.
- **`max_len=6` by default** in `top_k_rules`: the splits of an itemset grow exponentially with its size and long rules are
  hard to read. `max_len=None` removes the limit, which on dense data can be very slow.
- **`min_supp` is an optional floor**, not the main knob: with it, fewer than `k` results may come out.
- **Memory** is proportional to *transactions x items* (one bit per cell) plus the frontier of the search, which is largest
  on dense data with a large `k`.
- **The p-values of the selected rules are optimistic**: the rules were chosen *because* they are the most frequent. The
  correction over `n_candidates` covers the rules examined, but treat `significant_rules` after `top_k_rules` as a screen,
  not as a proof.

Command line: [`armxtend top-k-rules`](../cli.md#top-k-rules).

## References

- P. Fournier-Viger, V. S. Tseng (2012). *Mining top-k association rules.* Proceedings of the 25th Canadian Conference on
  Artificial Intelligence, 61-73 (the problem: the `k` best rules with no support threshold). The algorithm here is not
  theirs: it enumerates itemsets best-first and splits them, which is simpler and exact, but it does not reproduce their
  search over rules.
