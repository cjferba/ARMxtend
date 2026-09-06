# `ARM.meta_rules`

Meta-association rules: mining associations over a set of association rules already extracted from
**multiple** datasets, following Ruiz, Gomez-Romero, Molina-Solana, Campana & Martin-Bautista
(2016) -- see [Citing ARMxtend](../../cite.md).

Given k datasets `D_1, ..., D_k` sharing a domain (e.g. one per store, sensor, or time window),
each yielding a set of primary association rules `R_1, ..., R_k`, a **meta-database** is built where
each row is a dataset `D_j` and each column (item) is a rule `r_i` (plus, optionally, extra
attributes of the dataset). Mining association rules over this meta-database yields **meta-rules**
such as `r1 ^ r2 -> r3` (rules that co-occur across a high proportion of datasets) or `r1 -> at2`
(a rule associated with a dataset attribute).

Two variants are provided, mirroring Algorithms 3 and 4 of the paper:

- **Crisp** (`crisp_meta_association_rules`): the meta-database is boolean -- 1 if the rule was
  found in that dataset, 0 otherwise. Only presence/absence matters, not how strong the rule was.
- **Fuzzy** (`fuzzy_meta_association_rules`): the meta-database uses the rule's own support or
  certainty factor in that dataset as its fuzzy degree, so a rule mined with confidence 0.95
  contributes more than one mined with confidence 0.51 -- the resulting meta-rules incorporate the
  quality of the primary rules, not just their presence.

!!! warning "Rule and item identifiers"
    Identifiers used as meta-database items (rule keys, attribute names) must not contain the `-`
    character, since it is used internally as the itemset-key separator. Use `rule_key()` to build
    safe rule identifiers.

## Mining the primary rules

```python
mine_primary_rule_measures(datasets, min_supp=0.1, min_conf=0.5, metric="confidence")
```

Mines crisp rules independently in each dataset of `datasets` (a list of transaction lists) using
[`FFIM.fpgrowth`](../FFIM.md) + [`ARM.association_rules`](association_rules.md), and returns one
`dict {rule_key: metric value}` per dataset -- the input expected by both meta-rule variants below.

## Crisp meta-rules

```python
build_crisp_meta_database(rule_sets, attributes=None)
crisp_meta_association_rules(rule_sets, attributes=None, min_supp=0.5, min_conf=0.5, metric="confidence")
```

- **rule_sets** (`Sequence[Set[str]]`): for each dataset, the set of rule keys found in it (or a
  `dict`, as returned by `mine_primary_rule_measures` -- only its keys are used).
- **attributes** (`Sequence[dict[str, int]]`, optional): extra boolean attributes per dataset.

## Fuzzy meta-rules

```python
build_fuzzy_meta_database(rule_measure_sets, attributes=None, normalize=False)
fuzzy_meta_association_rules(rule_measure_sets, attributes=None, num_alpha=10,
                             min_supp=0.1, min_conf=0.5, metric="confidence", normalize=False)
```

- **rule_measure_sets** (`Sequence[dict[str, float]]`): for each dataset, `{rule_key: measure in
  [0, 1]}` -- the output of `mine_primary_rule_measures`.
- **normalize** (`bool`): if `True`, scales each rule's column to `[0, 1]` by its maximum across
  datasets (useful when every primary rule's support is very low in absolute terms).

## Example

```python
from ARMxtend.ARM.meta_rules import mine_primary_rule_measures, crisp_meta_association_rules, \
    fuzzy_meta_association_rules

store1 = [["bread", "milk"]] * 6 + [["bread"]] * 2 + [["milk"]] * 2
store2 = [["bread", "milk"]] * 5 + [["bread"]] * 3 + [["milk"]] * 2
store3 = [["cola", "chips"]] * 7 + [["cola"]] * 3

rule_measures = mine_primary_rule_measures([store1, store2, store3], min_supp=0.3, min_conf=0.5)
# [{'milk=>bread': 0.75, 'bread=>milk': 0.75},
#  {'milk=>bread': 0.71, 'bread=>milk': 0.62},
#  {'chips=>cola': 1.0, 'cola=>chips': 0.7}]

crisp_meta = crisp_meta_association_rules([set(m) for m in rule_measures], min_supp=0.5, min_conf=0.9)
# bread=>milk <-> milk=>bread, support=0.67 (found in 2/3 stores);
# the chips<->cola rules are dropped (found in only 1/3 stores, below min_supp)

fuzzy_meta = fuzzy_meta_association_rules(rule_measures, num_alpha=10, min_supp=0.3, min_conf=0.5)
```
