# `ARM.fuzzy_class_rules` and `FuzzyRuleClassifier` -- classification with fuzzy rules

[`RuleClassifier`](classification.md) needs boolean items, so numeric attributes are cut at thresholds. Here the attributes
are **fuzzy** ("temperature is HIGH to degree 0.7") and a rule `A => class=c` holds in each case to a degree, following the
fuzzy rule-based classification systems of Ishibuchi and Yamamoto (2005) in their simplest form.

```python
from ARMxtend.ARM import FuzzyRuleClassifier, fuzzy_class_rules

# degrees: one row per case, one column per fuzzy item (e.g. from preprocessing.FuzzyLib or timeseries.to_fuzzy_transactions)
rules = fuzzy_class_rules(degrees, labels, min_supp=0.05, min_conf=0.4, max_len=2)
model = FuzzyRuleClassifier(min_supp=0.05, min_conf=0.4, max_len=2, max_rules_per_class=10).fit(degrees, labels)
model.predict(new_degrees)           # class of each case (the majority class if no rule fires)
model.decision_scores(new_degrees)   # score of each class
model.score(new_degrees, new_labels)
```

## Definitions

- The degree to which the antecedent `A` holds in a case is the t-norm of the degrees of its items: `tnorm="min"` (default) or
  `"product"`.
- Supports and confidences use the **scalar cardinality** (sum of degrees). With `mu_A` the degrees of the antecedent:
  `support = sum(mu_A over the cases of class c) / N` and `confidence = sum(mu_A over the cases of class c) / sum(mu_A over all)`.
  Both are anti-monotone with the two t-norms, so the level-wise search is exact (the tests compare it with the definition
  applied to every antecedent).
- As in [`class_association_rules`](classification.md), `min_supp` is applied **within each class**, so rare classes get rules.
- The **rule weight** is the certainty grade of Ishibuchi and Yamamoto: the confidence of the rule minus the mean confidence of the
  other classes for the same antecedent. Rules with weight <= 0 are discarded.
- **Inference**: `inference="winner"` (default) predicts the class of the rule with the highest `degree x weight`;
  `"sum"` predicts the class with the highest sum of `degree x weight` over its rules. `max_rules_per_class` keeps only the best
  rules of each class (by weight x confidence).

The rules have the format of [`association_rules`](association_rules.md) (supports in scalar cardinality) plus `class` and
`weight`, and `attrs` holds `n_transactions` and `n_candidates`.

## Notes

- It does not fuzzify for you: build the degrees with `FuzzyLib.AutoFuzzification` or `to_fuzzy_transactions`. Item names must not
  contain `-`.
- Two labels of the same attribute (`temp_low`, `temp_high`) can appear in one antecedent; their conjunction has a low degree, so
  such rules are rare in practice.
- There is no command-line command for it yet; the crisp [`classify`](../cli.md) command is unchanged.
