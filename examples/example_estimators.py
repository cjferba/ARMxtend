# -*- coding: utf-8 -*-
"""
Scikit-learn style estimators and synthetic data: encode transactions, extract rules with ``RuleMiner``,
use the rules as features and check that the planted patterns are recovered.

Run:  python examples/example_estimators.py
"""
from ARMxtend.ARM import RuleClassifier
from ARMxtend.datasets import load_dataset, make_labeled_items, make_transactions
from ARMxtend.estimators import RuleMiner, TransactionEncoder


def main():
    baskets, patterns = make_transactions(2000, 60, avg_length=7, n_patterns=8, pattern_length=3, random_state=7,
                                          return_patterns=True)
    table = TransactionEncoder().fit_transform(baskets)

    miner = RuleMiner(min_supp=0.1, min_threshold=0.8, min_lift=1.5, alpha=0.01).fit(table)
    found = {frozenset(a) | frozenset(c) for a, c in zip(miner.rules_["antecedents"], miner.rules_["consequents"])}
    recovered = [pattern for pattern in patterns if frozenset(pattern) in found]
    features = miner.transform(baskets)

    top = RuleMiner(method="top_k", k=5, min_threshold=0.7).fit(baskets).rules_

    items, labels = make_labeled_items(1500, noise=0.05, random_state=3)
    classifier = RuleClassifier(min_supp=0.05, min_conf=0.7).fit(items[:1000], labels[:1000])
    accuracy = classifier.score(items[1000:], labels[1000:])

    weather = load_dataset("weather")
    tennis = RuleClassifier(min_supp=0.1, min_conf=0.8).fit(weather.data, weather.target)

    print("Planted patterns:", len(patterns), "- recovered as a whole rule itemset:", len(recovered))
    print("Rules:", len(miner.rules_), "- rule features:", features.shape)
    print("Top-5 rules:\n", top[["antecedents", "consequents", "support", "confidence"]].to_string(index=False))
    print("Classifier accuracy on held-out rows: {:.3f}".format(accuracy))
    print("Tennis rules in the list:", len(tennis.rules_), "- training accuracy:",
          round(tennis.score(weather.data, weather.target), 3))
    return patterns, recovered, miner, features, top, accuracy, tennis, weather


if __name__ == "__main__":
    main()
