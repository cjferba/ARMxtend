# `ARM.classification` -- classifying with association rules (CBA)

Association rules whose consequent is a value of the variable to predict -- *class association rules*,
`A => class=c` -- can be turned into a classifier that is easy to read: an ordered list of "if ... then ..." rules and a default
class. The construction follows CBA (*Classification Based on Associations*, Liu, Hsu and Ma, KDD 1998).

```python
from ARMxtend.ARM import RuleClassifier

model = RuleClassifier(min_supp=0.05, min_conf=0.7, max_len=3, min_improvement=0.1)
model.fit(train_items, train_labels)               # items: boolean table; labels: the class of every row

model.predict(test_items)                          # Series with the class of each row
model.score(test_items, test_labels)               # accuracy
model.explain(test_items)                          # prediction, deciding rule and its confidence
model.rules_                                       # the selected rules, in order of application
```

In the [example](../../examples.md#rules-in-action-armanomaly-armprediction-armclassification), 10 symptoms and three
diagnoses: out of 12 candidate rules the classifier keeps `cough, fever => flu` and `itchy, sneezing => allergy`, with
`healthy` otherwise, and it classifies 96 % of 400 new cases correctly (the majority class: 56 %).

## The steps

**1. Class association rules.**
`class_association_rules(items, labels, min_supp, min_conf, max_len, support="class")` returns rules in the format of
[`association_rules`](association_rules.md) (so [`significant_rules`](significance.md) and
[`prune_redundant_rules`](redundancy.md) work on them), with the item `<name>=<value>` as consequent and the value of
the class in the column `class`. The items must be boolean columns; `preprocessing.timeseries.to_items` converts numeric
(by discretising) and categorical columns. `support="class"` (default) applies `min_supp` inside each class, so that
rare classes get their rules; `"global"` applies it to all rows. The supports in the result are always relative to all rows.

**2. Optional pruning.**
`prune_unproductive_rules(rules, min_improvement)` (see [redundancy](redundancy.md#unproductive-rules)) removes the
rules whose extra conditions do not improve the confidence of a more general rule. In the `RuleClassifier`, pass
`min_improvement` (default `None`: no pruning, as in the original CBA).

**3. Selecting the rules.**
`select_rules(rules, items, labels)` is CBA-CB (M1): the rules are ordered by precedence (higher confidence, then higher
support, then shorter antecedent); each rule that classifies correctly at least one still-uncovered training row is kept,
and removes all the rows that match its antecedent. After each kept rule the default class (the majority among the
rows not covered) and the total error (errors of the rules plus rows left over that are not of the default class) are
recorded, and the list is cut where the total error is lowest (the shortest list in case of a tie).

**4. Predicting.** The first rule of the list that holds decides; if none does, the default class.

## Notes

- Only presence of items is expressed: there are no rules like "not fever". Add negated items with
  [`ARM.negative_rules`](negative_rules.md) if you need them.
- The precedence puts the rules of 100 % confidence first, whatever their class, so the list often starts with rules
  for the majority class: that is normal in CBA.
- It is a transparent classifier, not necessarily the most accurate: use it when you need to read and audit why a case
  was classified in a way, and compare it with a baseline (the majority class) as the example does.

Command line: [`armxtend classify`](../cli.md#rules-in-action-anomalies-forecast-next-events-and-classify).

## Reference

- B. Liu, W. Hsu, Y. Ma (1998). *Integrating classification and association rule mining.* Proc. KDD-98, 80-86.
- B. Liu, Y. Ma, C.-K. Wong (2000). *Improving an association rule based classifier.* PKDD 2000 (minimum support per
  class).
