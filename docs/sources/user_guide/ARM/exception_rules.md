# `ARM.exception_rules` -- exception and anomalous rules

Patterns describing **when a strong rule fails**. Both start from a strong "common sense" rule
`A -> b` and relate to the treatment of absence in Section 4 of Delgado et al. (2011), based on
Suzuki (1997) and Delgado, Ruiz & Sanchez (2011, *Int. J. Uncertainty, Fuzziness and KBS*, 19(2)).

```python
from ARMxtend.ARM import exception_rules, anomalous_rules
```

!!! note "Formulation"
    The conditions below are a direct formulation in terms of support and confidence. The original
    papers contain more general variants (other measures, "double" rules); check them if you need
    those exact definitions.

## Exception rules

For a strong rule `A -> b`, an **exception** is a set of items `Z` such that

- `A ^ Z -> ~b` is strong (`exception_confidence >= min_exception_conf`, with the much lower
  `min_exception_supp`: exceptions are rare), **and**
- the *reference* rule `Z -> ~b` is **not** strong (`reference_confidence <= max_reference_conf`),
  so the exception is due to the combination `A ^ Z` and not to `Z` alone.

```python
exception_rules(transactions, min_supp=0.1, min_conf=0.8, min_exception_supp=0.01,
                min_exception_conf=0.8, max_reference_conf=0.5, max_condition_size=1, rules=None)
```

```python
data = ([{"fever", "rest"}] * 60 + [{"fever", "must_work"}] * 5
        + [{"must_work", "rest"}] * 35 + [{"fever", "hospital"}] * 10)
exception_rules(data, min_supp=0.3, min_conf=0.8, min_exception_supp=0.03)
#   {fever} -> rest (conf 0.80), but with {must_work}, ~rest: confidence=1.00 (reference 0.12)
```

Returns a `DataFrame` with `antecedents` (A), `consequent` (b), `condition` (Z), `common_support`,
`common_confidence`, `exception_support`, `exception_confidence` and `reference_confidence`, sorted by
exception confidence. `rules=` accepts a precomputed `association_rules` result; otherwise the common
rules are mined with `min_supp`/`min_conf`. `max_condition_size` allows conditions with several items.

## Anomalous rules

For a strong rule `A -> b`, the **anomalies** are the cases where `A` holds but `b` does not
(`A ^ ~b`). An **anomalous rule** `A ^ ~b -> C` says that `C` characterizes them
(`anomalous_confidence >= min_anomalous_conf`), where `A -> C` is **not** strong
(`reference_confidence <= max_reference_conf`) -- `C` is not already usual when `A` holds.

```python
anomalous_rules(transactions, min_supp=0.1, min_conf=0.8, min_anomalous_supp=0.01,
                min_anomalous_conf=0.8, max_reference_conf=0.5, max_anomaly_size=1, rules=None)
```

Returns `antecedents`, `consequent`, `anomaly` (C), `common_support`, `common_confidence`,
`anomalous_cases` (number of transactions with `A ^ ~b`), `anomalous_support`,
`anomalous_confidence` and `reference_confidence`.

Only common rules with a **single-item consequent** are used: the negation of a multi-item
consequent is a disjunction (`~(b1 ^ b2) = ~b1 v ~b2`), which Delgado et al. warn may not be of
interest. Supports of `A ^ Z ^ ~b` are counted directly on the transactions (they are usually too
rare to be recovered from a frequent-itemset result).
