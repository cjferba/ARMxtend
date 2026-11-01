# `FIM.taxonomy` -- item hierarchies and generalized itemsets

Products come in categories (`Jacket -> Outerwear -> Clothes`), diagnoses in chapters, sensors in rooms. A
**taxonomy** states those "is-a" relations so that patterns can be found at **every level**: nobody buys *exactly* the
same jacket and boots, but "outerwear goes with footwear" can be a strong pattern. This page covers the hierarchy and
the frequent itemsets over it; the rules are built in [`ARM.generalized_rules`](../ARM/generalized_rules.md).

```python
from ARMxtend.FIM import Taxonomy, generalized_itemsets

taxonomy = Taxonomy.from_paths(["Clothes/Outerwear/Jacket", "Clothes/Outerwear/Ski Pants", "Clothes/Shirts",
                                "Footwear/Shoes", "Footwear/Hiking Boots"])
baskets = [["Shirts"], ["Jacket", "Hiking Boots"], ["Ski Pants", "Hiking Boots"], ["Shoes"], ["Shoes"], ["Jacket"]]

generalized_itemsets(baskets, taxonomy, min_supp=0.3)
# {Clothes} 0.67   {Footwear} 0.67   {Outerwear} 0.50   {Outerwear, Footwear} 0.33   {Outerwear, Hiking Boots} 0.33 ...
```

## `Taxonomy`

A directed acyclic graph child -> parent; an item may have **several parents**. It can be built from

```python
Taxonomy({"Jacket": "Outerwear", "Outerwear": "Clothes"})        # {child: parent} or {child: [parents]}
Taxonomy([("Jacket", "Outerwear"), ("Outerwear", "Clothes")])    # pairs (child, parent)
Taxonomy(frame)                                                  # DataFrame: first column child, second parent
Taxonomy.from_paths(["Clothes/Outerwear/Jacket"], sep="/")       # paths from the root to the leaf
```

Cycles and self-parents are rejected. Items that only appear as parents are roots, and **items that are not in the
taxonomy** are treated as roots with no descendants, so a taxonomy can cover just part of the catalogue.

| Method | Meaning |
|---|---|
| `parents(i)`, `children(i)` | direct relatives |
| `ancestors(i)`, `descendants(i)` | all of them, without the item itself |
| `is_ancestor(a, i)` | is `a` a (strict) ancestor of `i`? |
| `level(i)` | 1 for the roots; otherwise 1 + the highest level among its parents, so an ancestor always has a *lower* level |
| `at_level(i, L)` | the item seen from level `L`: itself if `level(i) <= L`, else its deepest ancestors of level `<= L` |
| `roots()`, `leaves()`, `items`, `depth`, `to_frame()` | the structure |

## Functions

```python
extend_transactions(transactions, taxonomy)                  # each basket + the ancestors of its items
generalize_transactions(transactions, taxonomy, level)       # each item replaced by its ancestor at `level`
generalized_itemsets(transactions, taxonomy, min_supp, max_len=None)
```

- **`extend_transactions`** is the base of generalized mining (Srikant and Agrawal, 1995): a basket with `Jacket` also
  "contains" `Outerwear` and `Clothes`.
- **`generalize_transactions`** is the one-level-at-a-time approach (Han and Fu, 1995): mine the baskets rolled up to
  level 2, then to level 3, with the usual tools (`fpgrowth`, `association_rules`).
- **`generalized_itemsets`** mines all the levels at once on the extended baskets and returns the usual
  `support` / `itemsets` frame (so `ARM.association_rules` takes it). It never builds an itemset that contains an item
  **and one of its ancestors** (`{Jacket, Outerwear}`): its support equals that of the itemset without the ancestor,
  so it only adds noise and an exponential number of useless itemsets (the optimization of Srikant and Agrawal's
  *Cumulate* algorithm). The candidates are counted with one boolean column per item; memory is proportional to
  *transactions x frequent itemsets of the widest level*.

## One minimum support per level

Categories are far more frequent than the products below them, so a single threshold either drowns you in
category-level itemsets or hides every product-level one. `min_supp` can be a dictionary **level -> support**:

```python
generalized_itemsets(baskets, taxonomy, min_supp={1: 0.6, 2: 0.3, 3: 0.1})
```

A level that is not listed uses the value of the deepest listed level above it (`{1: 0.6, 3: 0.1}` gives level 2 the
value 0.6). **An itemset must reach the highest threshold among the levels of its items**, i.e. the one of its most
general item: `{Jacket, Footwear}` needs the level-1 support even though `Jacket` is level 3. This keeps the
anti-monotone property (adding an item never lowers the threshold, so every subset of a frequent itemset is
frequent) and therefore the search is exact; it is this library's choice and not the only possible one (see the notes
in [`ARM.generalized_rules`](../ARM/generalized_rules.md#choosing-the-supports)).

## References

- R. Srikant, R. Agrawal (1995). *Mining generalized association rules.* Proceedings of the 21st VLDB Conference,
  407-419.
- J. Han, Y. Fu (1995). *Discovery of multiple-level association rules from large databases.* Proceedings of the 21st
  VLDB Conference, 420-431.
