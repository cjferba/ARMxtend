# `ARM.negative_rules` -- rules with absent items

Rules such as `A -> ~B` ("if A, then **not** B"), `~A -> B` or `~A -> ~B`, where `~i` means that item
`i` is *absent* from the transaction. Crisp and fuzzy versions, following Section 4 of Delgado,
Ruiz, Sanchez & Serrano (2011) -- see [Citing ARMxtend](../../cite.md).

```python
from ARMxtend.ARM import negative_association_rules

drinks = [{"coffee", "sugar"}] * 30 + [{"tea"}] * 25 + [{"tea", "sugar"}] * 5 + [{"coffee"}] * 10
rules = negative_association_rules(drinks, min_supp=0.25, min_threshold=0.95)
#  {coffee} -> {~tea}   support=0.57 confidence=1.00
#  {tea}    -> {~coffee} support=0.43 confidence=1.00
#  {coffee, sugar} -> {~tea} ...
```

## How absence is modelled

Absent items are added to the transactions as ordinary items named `~item` and mined with the same
algorithms (`FFIM.fpgrowth` / `FFIM.fuzzy_fpgrowth`), so every measure, threshold and downstream tool
works unchanged. An itemset never contains both `i` and `~i` (its support is 0), so trivial rules such
as `i -> ~i` cannot appear.

In the **fuzzy** case, the absence of an item is its **complement at every alpha-cut level** (Eqs.
(27)-(28) of the paper) -- *not* the fuzzy-set negation `1 - degree`. The 4-fold table of `(~A, B)` is
`<c, d, a, b>` obtained from the one of `(A, B)`, so no new frequencies are needed, and
`FSupp(~A) = 1 - FSupp(A)` holds exactly. E.g. a degree 0.6 with `Lambda = {1, 0.8, 0.6, 0.4, 0.2}`
holds at the levels `0.6, 0.4, 0.2` (`FSupp = 0.6`) and its absence at `1, 0.8` (`FSupp = 0.4`).

## The explosion of absent items

In sparse data the *absence* of an item is almost always frequent, and the number of itemsets
explodes (support of absence does not satisfy the upward closure used for pruning). Two controls,
both from the paper:

- `only_frequent_positive=True` (default): only items whose *positive* form is frequent are negated
  (the restriction of Han & Beheshti adopted in Section 6 of the paper).
- `negate_items=[...]`: explicitly restrict which items can appear as absent.

## Signatures

```python
negative_association_rules(transactions, min_supp, metric="confidence", min_threshold=0.8,
                           negate_items=None, only_frequent_positive=True, only_negative=True)

fuzzy_negative_association_rules(transactions, min_supp, num_alpha, metric="confidence",
                                 min_threshold=0.8, negate_items=None, only_frequent_positive=True,
                                 only_negative=True, measures=None)
```

- **transactions**: crisp: `Sequence[Iterable[str]]`; fuzzy: a list of `(item, degree)` pairs per
  transaction, as produced by [`FuzzyLib.GetFuzzyTransactions`](../preprocessing/FuzzyLib.md).
- **only_negative** (`True`): return only rules with at least one absent item; `False` also returns
  the purely positive ones.
- **metric / min_threshold / measures**: any measure of [`ARM.association_rules`](association_rules.md)
  (crisp) or [`FIM.FARE`](../FIM/FARE.md) (fuzzy).

Item names must not contain `-` (the itemset-key separator). Helpers: `negate("A") == "~A"`,
`is_negated("~A")`, `add_absent_items(transactions, negate_items=None)`.
