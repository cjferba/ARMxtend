# -*- coding: utf-8 -*-
"""
Association meta-rules (crisp and fuzzy): association mining over a set of association rules ALREADY extracted from
multiple datasets D_1, ..., D_k, instead of over the original data. It implements Algorithms 3 (crisp) and 4 (fuzzy)
of:

    Ruiz, M.D., Gomez-Romero, J., Molina-Solana, M., Campana, J.R.,
    Martin-Bautista, M.J. (2016). "Meta-association rules for mining
    interesting associations in multiple datasets". Applied Soft Computing,
    49, 212-223.

General idea (Definition 3 of the paper): given k sets of rules R_1, ..., R_k extracted from k datasets
D_1, ..., D_k (same domain, perhaps different structure), a "meta-database" D is built in which each ROW is a dataset
D_j and each COLUMN (item) is a rule r_i (plus, optionally, additional attributes about the dataset). An association
rule mining algorithm is applied again over that meta-database, obtaining meta-rules such as r1 ^ r2 -> r3
(co-occurrence of rules in a high proportion of datasets) or r1 -> at2 (relation between a rule and an attribute of
the dataset).

Two variants are distinguished:
    * CRISP meta-rules (Algorithm 3): the meta-database is boolean, 1 if rule r_i was extracted in dataset D_j, 0 if
      not. It only takes into account the presence/absence of the rule, not its strength.
    * FUZZY meta-rules (Algorithm 4): the meta-database uses as membership degree the value of a measure of the rule
      (support or certainty factor) in each dataset, making it possible to tell very reliable "primary" rules from
      marginal ones.

IMPORTANT NOTE: rule and item identifiers must not contain the character '-' (used as item separator in the itemset
keys of `FIM._shared`); use `rule_key()` to build safe identifiers.

@author Carlos Fernandez-Basso
"""
from typing import Dict, Iterable, List, Optional, Sequence
import pandas as pd

from ..FIM._shared import key_to_itemset
from ..FIM.FARE import fuzzy_association_rules
from ..FFIM import fpgrowth, fuzzy_fpgrowth
from .association_rules import association_rules


def rule_key(antecedent: Iterable[str], consequent: Iterable[str]) -> str:
    """
    Build a safe rule identifier (without '-') to use as an item in a meta-database, e.g.
    rule_key(['A','B'], ['C']) -> 'A,B=>C'.
    """
    return "{}=>{}".format(",".join(sorted(antecedent)), ",".join(sorted(consequent)))


def mine_primary_rule_measures(datasets: Sequence[Sequence[Iterable[str]]], min_supp: float = 0.1,
                                min_conf: float = 0.5, metric: str = "confidence") -> List[Dict[str, float]]:
    """
    Mine, for each dataset of `datasets`, its crisp association rules (steps 1-3 of Algorithms 3 and 4: "for all Di,
    obtain Ri using Algorithm 1"), using `FFIM.fpgrowth` + `ARM.association_rules`.

    Args:
        datasets (Sequence[Sequence[Iterable[str]]]): one dataset D_j per element, each a list of transactions
            (collections of items)
        min_supp (float): minimum support for frequent-itemset mining in each dataset
        min_conf (float): minimum threshold of `metric` to keep a rule
        metric (str): measure of `ARM.association_rules` used as threshold and as the value associated with the rule
            in the fuzzy meta-database

    Returns:
        list[dict[str, float]]: for each dataset, a dictionary {rule_key(antecedent, consequent): value of `metric`}
    """
    ruleMeasuresPerDataset = []
    for transactions in datasets:
        freqItemsets = fpgrowth(transactions, min_supp)
        itemsetsDf = pd.DataFrame({
            "support": list(freqItemsets.values()),
            "itemsets": [key_to_itemset(key) for key in freqItemsets],
        })
        rulesDf = association_rules(itemsetsDf, metric=metric, min_threshold=min_conf)

        ruleMeasures = {
            rule_key(row["antecedents"], row["consequents"]): float(row[metric])
            for _, row in rulesDf.iterrows()
        }
        ruleMeasuresPerDataset.append(ruleMeasures)

    return ruleMeasuresPerDataset


def build_crisp_meta_database(rule_sets: Sequence[Iterable[str]],
                              attributes: Optional[Sequence[Dict[str, int]]] = None) -> pd.DataFrame:
    """
    Build the boolean meta-database D (Table 7 of the paper).

    Args:
        rule_sets (Sequence[Set[str]] | Sequence[dict[str, float]]): for each dataset D_j, the set of keys of the
            rules found in R_j (or a dict, of which only the keys are used, e.g. the output of
            `mine_primary_rule_measures`)
        attributes (Sequence[dict[str, int]], optional): for each dataset, additional boolean attributes
            {attribute_name: 0/1} (see Definition 3)

    Returns:
        pandas.DataFrame with one row per dataset and one boolean (0/1) column per distinct rule and attribute found
    """
    ruleKeys = sorted(set().union(*(set(rs) for rs in rule_sets))) if rule_sets else []

    rows = []
    for index, ruleSet in enumerate(rule_sets):
        row = {ruleKey: int(ruleKey in ruleSet) for ruleKey in ruleKeys}
        if attributes:
            row.update(attributes[index])
        rows.append(row)

    return pd.DataFrame(rows)


def crisp_meta_association_rules(rule_sets: Sequence[Iterable[str]],
                                 attributes: Optional[Sequence[Dict[str, int]]] = None, min_supp: float = 0.5,
                                 min_conf: float = 0.5, metric: str = "confidence") -> pd.DataFrame:
    """
    Algorithm 3: mining of crisp association meta-rules.

    Args:
        rule_sets (Sequence[Set[str]] | Sequence[dict[str, float]]): see `build_crisp_meta_database`
        attributes (Sequence[dict[str, int]], optional): see `build_crisp_meta_database`
        min_supp (float): minimum support for a meta-rule to be considered frequent (proportion of datasets in which
            its rules/attributes co-occur)
        min_conf (float): minimum threshold of `metric` for a meta-rule
        metric (str): measure of `ARM.association_rules` to filter by

    Returns:
        pandas.DataFrame of meta-rules (same format as `ARM.association_rules`): their 'antecedents'/'consequents' are
        frozensets of rule keys and/or attribute names.
    """
    metaDatabase = build_crisp_meta_database(rule_sets, attributes)
    transactions = [
        {column for column in metaDatabase.columns if row[column] == 1}
        for _, row in metaDatabase.iterrows()
    ]

    freqItemsets = fpgrowth(transactions, min_supp)
    itemsetsDf = pd.DataFrame({
        "support": list(freqItemsets.values()),
        "itemsets": [key_to_itemset(key) for key in freqItemsets],
    })
    return association_rules(itemsetsDf, metric=metric, min_threshold=min_conf)


def build_fuzzy_meta_database(rule_measure_sets: Sequence[Dict[str, float]],
                              attributes: Optional[Sequence[Dict[str, float]]] = None,
                              normalize: bool = False) -> pd.DataFrame:
    """
    Build the fuzzy meta-database D~ (Table 8 of the paper): the membership degree of rule r_i in dataset D_j is the
    value of the measure (support or certainty factor) obtained when mining it in D_j (0 if it was not extracted).

    Args:
        rule_measure_sets (Sequence[dict[str, float]]): for each dataset D_j, {rule_key: value in [0, 1]} (see
            `mine_primary_rule_measures`)
        attributes (Sequence[dict[str, float]], optional): for each dataset, fuzzy membership degrees of additional
            attributes
        normalize (bool): if True, normalize each rule column to the range [0, 1] by dividing by its maximum across
            datasets (the D~_NS variant of the paper, Section 5.2.2, useful when the support of the primary rules is
            very low in all the datasets)

    Returns:
        pandas.DataFrame with one row per dataset and one column (membership degree in [0, 1]) per distinct rule and
        attribute
    """
    ruleKeys = sorted(set().union(*(set(rm) for rm in rule_measure_sets))) if rule_measure_sets else []

    rows = []
    for index, ruleMeasures in enumerate(rule_measure_sets):
        row = {ruleKey: float(ruleMeasures.get(ruleKey, 0.0)) for ruleKey in ruleKeys}
        if attributes:
            row.update(attributes[index])
        rows.append(row)

    metaDatabase = pd.DataFrame(rows)

    if normalize:
        for ruleKey in ruleKeys:
            maxValue = metaDatabase[ruleKey].max()
            if maxValue > 0:
                metaDatabase[ruleKey] = metaDatabase[ruleKey] / maxValue

    return metaDatabase


def fuzzy_meta_association_rules(rule_measure_sets: Sequence[Dict[str, float]],
                                 attributes: Optional[Sequence[Dict[str, float]]] = None, num_alpha: int = 10,
                                 min_supp: float = 0.1, min_conf: float = 0.5, metric: str = "confidence",
                                 normalize: bool = False) -> pd.DataFrame:
    """
    Algorithm 4: mining of fuzzy association meta-rules.

    Args:
        rule_measure_sets (Sequence[dict[str, float]]): see `build_fuzzy_meta_database` (e.g. the output of
            `mine_primary_rule_measures`)
        attributes (Sequence[dict[str, float]], optional): see `build_fuzzy_meta_database`
        num_alpha (int): number of alpha-cuts for fuzzy mining (the paper of Fernandez-Basso et al. 2021 recommends 10)
        min_supp (float): minimum FSupp for a meta-rule to be considered frequent
        min_conf (float): minimum threshold of `metric` ('confidence' -> FConf, 'certainty_factor' -> FCF) for a
            meta-rule
        metric (str): 'support', 'confidence' or 'certainty_factor'
        normalize (bool): see `build_fuzzy_meta_database`

    Returns:
        pandas.DataFrame of fuzzy meta-rules (same format as `FIM.FARE.fuzzy_association_rules`)
    """
    metaDatabase = build_fuzzy_meta_database(rule_measure_sets, attributes, normalize)
    fuzzyTransactions = [
        [(column, row[column]) for column in metaDatabase.columns]
        for _, row in metaDatabase.iterrows()
    ]

    freqItemsets = fuzzy_fpgrowth(fuzzyTransactions, min_supp, num_alpha)
    return fuzzy_association_rules(freqItemsets, num_alpha, metric=metric, min_threshold=min_conf)
