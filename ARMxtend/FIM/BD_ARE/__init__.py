# -*- coding: utf-8 -*-
"""
Crisp association rule mining on Big Data (Apache Spark), from an already computed set of frequent itemsets (e.g.
with ``ARMxtend.FIM.apriori.DApriori/DAprioriTID`` or ``ARMxtend.FIM.Eclat.DECLAT``).

It implements Algorithm 4 ("Spark procedure for association rule mining") of:
    Fernandez-Basso, C., Ruiz, M.D., Martin-Bautista, M.J. (2024). "New Spark
    solutions for distributed frequent itemset and association rule mining
    algorithms". Cluster Computing, 27, 1217-1234.

This module replaces the ``Apriori``/``AprioriTID``/``Eclat``/``ARE`` classes that previously lived in this file:
they were buggy copies (undefined names, a file path from a single development machine, and an ``exit()`` call in
the middle of a class definition that aborted the interpreter when the module was imported) of the content of
``FIM/apriori.py``. Itemset mining now lives in ``FIM.apriori``/``FIM.Eclat``; only the rule-generation phase,
common to the three algorithms, remains here.

@author Carlos Fernandez-Basso
"""
import itertools


def generate_rules(freq_itemsets, min_conf):
    """
    Generate, purely sequentially, all the association rules X -> Y with confidence >= min_conf from a set of
    frequent itemsets. It is the computation part (independent of Spark) reused by `association_rules_bd` to
    distribute it on a cluster.

    Args:
        freq_itemsets (dict[str, float]): frequent itemsets, with key 'A-B-C' and their relative support as value.
            It must include the support of every non-empty subset of each itemset (guaranteed by the downward
            closure property of the FIM.apriori and FIM.Eclat algorithms)
        min_conf (float): minimum confidence, in (0, 1]

    Returns:
        dict {(antecedent, consequent): confidence}
    """
    rules = {}
    for itemsetKey, supportXY in freq_itemsets.items():
        items = itemsetKey.split("-")
        if len(items) < 2:
            continue

        for antecedentSize in range(1, len(items)):
            for antecedent in itertools.combinations(items, antecedentSize):
                antecedentKey = "-".join(sorted(antecedent))
                consequentKey = "-".join(sorted(item for item in items if item not in antecedent))

                supportX = freq_itemsets.get(antecedentKey)
                if not supportX:
                    continue

                confidence = supportXY / supportX
                if confidence >= min_conf:
                    rules[(antecedentKey, consequentKey)] = confidence

    return rules


def association_rules_bd(sc, freq_itemsets, min_conf):
    """
    Distributed (Spark) version of `generate_rules` (Algorithm 4): it spreads the rule generation of each frequent
    itemset among the nodes of the cluster with a FlatMap, and filters by minimum confidence with a Reduce.

    Args:
        sc (SparkContext): Spark context to create the broadcast variable with the frequent itemsets
            (Global_FreqItemset in the paper)
        freq_itemsets (dict[str, float]): candidate frequent itemsets, with key 'A-B-C' and their relative support
            as value
        min_conf (float): minimum confidence, in (0, 1]

    Returns:
        dict {(antecedent, consequent): confidence}
    """
    broadcastFreqItemsets = sc.broadcast(freq_itemsets)

    def generateRulesForItemset(itemsetKey):
        supportXY = broadcastFreqItemsets.value[itemsetKey]
        items = itemsetKey.split("-")
        itemsetRules = []

        for antecedentSize in range(1, len(items)):
            for antecedent in itertools.combinations(items, antecedentSize):
                antecedentKey = "-".join(sorted(antecedent))
                consequentKey = "-".join(sorted(item for item in items if item not in antecedent))

                supportX = broadcastFreqItemsets.value.get(antecedentKey)
                if not supportX:
                    continue

                itemsetRules.append(((antecedentKey, consequentKey), supportXY / supportX))

        return itemsetRules

    candidateKeys = [key for key in freq_itemsets if len(key.split("-")) >= 2]

    rules = sc.parallelize(candidateKeys) \
              .flatMap(generateRulesForItemset) \
              .filter(lambda rule: rule[1] >= min_conf) \
              .collectAsMap()

    broadcastFreqItemsets.unpersist()
    return rules
