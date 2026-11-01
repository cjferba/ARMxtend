# `ARM.generalized_rules` -- multilevel rules over a taxonomy

Association rules whose items can be **products or categories**: `Outerwear => Hiking Boots`, `Dairy => Bakery`,
`croissant => Juice`. Categories group products whose support is spread too thin to reach any threshold on its own, and
reveal what a whole family has in common.

```python
from ARMxtend.ARM import generalized_rules, interesting_rules
from ARMxtend.FIM import Taxonomy

taxonomy = Taxonomy.from_paths(["Clothes/Outerwear/Jacket", "Clothes/Outerwear/Ski Pants", "Clothes/Shirts",
                                "Footwear/Shoes", "Footwear/Hiking Boots"])
baskets = [["Shirts"], ["Jacket", "Hiking Boots"], ["Ski Pants", "Hiking Boots"], ["Shoes"], ["Shoes"], ["Jacket"]]

rules = generalized_rules(baskets, taxonomy, min_supp=0.3, min_threshold=0.6)
rules[["antecedents", "consequents", "support", "confidence", "lift", "level"]]
# {Outerwear} => {Hiking Boots}  0.33  0.67  2.0   2
# {Outerwear} => {Footwear}      0.33  0.67  1.0   2
# {Hiking Boots} => {Outerwear}  0.33  1.00  2.0   2
# {Hiking Boots} => {Clothes}    0.33  1.00  1.5   2
```

(The example of Srikant and Agrawal, 1995: no rule involves a single product, because `Jacket`, `Ski Pants` and `Hiking
Boots` are each in too few baskets.)

## Signature

```python
generalized_rules(transactions, taxonomy, min_supp, metric="confidence", min_threshold=0.8, max_len=None, interest=None)
interesting_rules(rules, taxonomy=None, interest=1.1, supports=None)
```

`generalized_rules` mines the [generalized itemsets](../FIM/taxonomy.md) (with an optional minimum support per level)
and turns them into rules with [`association_rules`](association_rules.md); `metric` / `min_threshold` are those of
that function. Since no itemset contains an item with its own ancestor, the trivial rules `Jacket => Clothes`
(confidence 1) never appear. The output has the usual columns plus `level`, the level of the deepest item of the rule
(`level <= 2` selects the rules about categories), and `attrs` keeps `n_transactions`, `n_candidates` (so
[`significant_rules`](significance.md) works), `taxonomy` and `supports` (the support of every frequent itemset).

## Rules that only echo their category: `interesting_rules`

Categories multiply the rules: if `Bakery => Dairy` holds, so will `white_bread => Dairy`, `brown_bread => Dairy`,
`white_bread => milk`... and each of them only repeats, with a product, what the category already says.
`interesting_rules` keeps the **R-interesting** rules of Srikant and Agrawal (1995).

An *ancestor* of a rule is another rule of the set obtained by replacing items with themselves or with their
ancestors (`Bakery => Dairy` is an ancestor of `white_bread => milk`); the *close* ancestors are those with no other
ancestor of the rule in between. If `X^ => Y^` is an ancestor of `X => Y` and `z^` is the ancestor that replaces the item
`z`, the support and confidence **expected** from it are

```text
expected support    = support(X^ u Y^) * prod over z in X u Y of support(z) / support(z^)
expected confidence = confidence(X^ => Y^) * prod over y in Y of support(y) / support(y^)
```

(what the rule would have if every product behaved like its category). A rule is **interesting** if it has no ancestors
or if, with respect to every close ancestor that is itself interesting, its actual support **or** its actual
confidence is at least `interest` (R) times the expected one. The rules come out in the same order, with a column
`interest`: the smallest, over the close ancestors, of `max(support / expected, confidence / expected)` (NaN if the rule has no ancestors).

```python
interesting_rules(rules, interest=1.1)
# {Hiking Boots} => {Clothes}    interest NaN     (no ancestors)
# {Hiking Boots} => {Outerwear}  interest 1.33    (confidence 1 against the 0.75 expected from Hiking Boots => Clothes)
# {Outerwear} => {Footwear}      interest NaN
# {Outerwear} => {Hiking Boots}  interest 2.00    (support 1/3 against the 1/6 expected from Outerwear => Footwear)
```

`generalized_rules(..., interest=R)` does both steps. Apply it to the **unpruned** set: the ancestors of a rule have to be
in the set to be compared with (with a high `min_threshold` they may be missing, and the rule then counts as having no
ancestors).

## Choosing the supports

- A single `min_supp` is the simplest and it favours categories; with `{level: support}` the deeper levels can use lower
  thresholds, which is what lets product-level rules appear. **An itemset needs the support of its most general item**,
  so a rule that mixes levels (`Jacket => Footwear`) needs the threshold of the higher one (see
  [`FIM.taxonomy`](../FIM/taxonomy.md#one-minimum-support-per-level)).
- Check the effect by comparing with mining the flat baskets (`fpgrowth` + `association_rules`) at the same threshold:
  the [supermarket example](../../examples.md#multilevel-rules-over-a-taxonomy-fimtaxonomy-armgeneralized_rules) finds 4 rules
  without the taxonomy and 31 with it.
- R = 1.1-1.5 are usual values. R = 1 removes only the rules that are worse than expected; larger values keep only
  the rules that clearly add something.

Command line: [`armxtend generalized-rules`](../cli.md#generalized-rules).

## References

- R. Srikant, R. Agrawal (1995). *Mining generalized association rules.* Proceedings of the 21st VLDB Conference, 407-419
  (generalized rules, extended baskets, *Cumulate*, R-interesting rules).
- J. Han, Y. Fu (1995). *Discovery of multiple-level association rules from large databases.* Proceedings of the 21st VLDB
  Conference, 420-431 (a support threshold per level).
