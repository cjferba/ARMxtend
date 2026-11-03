# -*- coding: utf-8 -*-
"""
Exception rules and anomalous rules: patterns that describe WHEN a strong association rule fails. Both types appear in
section 4 of Delgado, Ruiz, Sanchez & Serrano (2011, Information Sciences 181) as the second class of approaches to
deal with the absence of items (versus the direct negation of items, see ``ARM.negative_rules``), and are based on:

    Suzuki, E. (1997). "Autonomous discovery of reliable exception rules".
    KDD-97 (exception rules).

    Delgado, M., Ruiz, M.D., Sanchez, D. (2011). "New approaches for
    discovering exception and anomalous rules". Int. J. Uncertainty,
    Fuzziness and Knowledge-Based Systems, 19(2), 361-399 (anomalous rules).

Both types start from a strong "common sense" rule  A -> b  (the usual behavior) and look for what explains its
failures:

* EXCEPTION rule (Suzuki): a set of items Z such that  A ^ Z -> ~b  is strong -- when besides A, Z occurs, the
  opposite of the usual happens -- while the REFERENCE rule  Z -> ~b  is NOT strong (Z alone does not imply ~b, so the
  exception is due to the combination A ^ Z and not to Z). Since the exception is rare, a minimum support much lower
  than that of the common rule is required.

* ANOMALOUS rule: a set of items C such that  A ^ ~b -> C  is strong -- among the cases in which A holds but b does
  not (the "anomalies"), C appears systematically, that is, C characterizes the anomaly --, while  A -> C  is NOT
  strong (C is not already usual when A occurs).

The definitions of thresholds and of the reference rule of the anomalous rules implemented here are a direct
formulation of the previous conditions on support and confidence; consult the original paper for the more general
formal variants (e.g. measures other than confidence or "double" rules).

Only common rules with ONE item in the consequent are considered: the negation of a consequent with several items is
a disjunction (~(b1 ^ b2) = ~b1 v ~b2), which Delgado et al. warn may not interest the user.

@author Carlos Fernandez-Basso
"""
from typing import Iterable, Optional, Sequence
import itertools

import numpy as np
import pandas as pd

from ..FFIM import fpgrowth
from ..FIM._shared import key_to_itemset
from .association_rules import association_rules

_EXCEPTION_COLUMNS = [
    "antecedents", "consequent", "condition", "common_support", "common_confidence",
    "exception_support", "exception_confidence", "reference_confidence",
]
_ANOMALOUS_COLUMNS = [
    "antecedents", "consequent", "anomaly", "common_support", "common_confidence",
    "anomalous_cases", "anomalous_support", "anomalous_confidence", "reference_confidence",
]


def _boolean_matrix(transactions):
    transactions = [set(t) for t in transactions]
    items = sorted(set().union(*transactions)) if transactions else []
    index = {item: column for column, item in enumerate(items)}
    matrix = np.zeros((len(transactions), len(items)), dtype=bool)
    for row, transaction in enumerate(transactions):
        for item in transaction:
            matrix[row, index[item]] = True
    return matrix, index


def _single_consequent_common_rules(transactions, rules, min_supp, min_conf):
    if rules is None:
        frequent = fpgrowth(transactions, min_supp)
        itemsets = pd.DataFrame({
            "support": list(frequent.values()),
            "itemsets": [key_to_itemset(key) for key in frequent],
        })
        rules = association_rules(itemsets, metric="confidence", min_threshold=min_conf)
    if len(rules) == 0:
        return rules
    return rules[rules["consequents"].apply(len) == 1]


def _condition_candidates(matrix, index, excluded_items, min_count):
    """Item columns (outside `excluded_items`) with at least `min_count` occurrences."""
    excluded = {index[item] for item in excluded_items}
    counts = matrix.sum(axis=0)
    return [column for column in range(matrix.shape[1]) if column not in excluded and counts[column] >= min_count]


def _combinations(candidates, max_size):
    for size in range(1, max_size + 1):
        for combination in itertools.combinations(candidates, size):
            yield combination


def exception_rules(transactions: Sequence[Iterable[str]], min_supp: float = 0.1, min_conf: float = 0.8,
                    min_exception_supp: float = 0.01, min_exception_conf: float = 0.8,
                    max_reference_conf: float = 0.5, max_condition_size: int = 1,
                    rules: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Find exception rules (Suzuki, 1997) of the strong common rules  A -> b: conditions Z such that  A ^ Z -> ~b  is
    strong and  Z -> ~b  is not.

    Args:
        transactions (Sequence[Iterable[str]]): crisp transactions
        min_supp (float): minimum support of the common rules
        min_conf (float): minimum confidence of the common rules
        min_exception_supp (float): minimum support of  A ^ Z ^ ~b  (usually much lower than `min_supp`: exceptions
            are rare)
        min_exception_conf (float): minimum confidence of  A ^ Z -> ~b
        max_reference_conf (float): MAXIMUM confidence of the reference rule  Z -> ~b
        max_condition_size (int): maximum number of items of Z
        rules (pandas.DataFrame, optional): already computed common rules (output of `ARM.association_rules`). If
            omitted, they are mined with `min_supp` and `min_conf`

    Returns:
        pandas.DataFrame with columns 'antecedents' (A), 'consequent' (b), 'condition' (Z), 'common_support',
        'common_confidence', 'exception_support', 'exception_confidence' and 'reference_confidence', sorted by
        confidence of the exception.
    """
    transactions = [set(t) for t in transactions]
    total = len(transactions)
    commonRules = _single_consequent_common_rules(transactions, rules, min_supp, min_conf)
    if total == 0 or len(commonRules) == 0:
        return pd.DataFrame(columns=_EXCEPTION_COLUMNS)

    matrix, index = _boolean_matrix(transactions)
    minExceptionCount = min_exception_supp * total
    rows = []

    for _, rule in commonRules.iterrows():
        antecedent = frozenset(rule["antecedents"])
        (consequent,) = tuple(rule["consequents"])
        maskA = matrix[:, [index[i] for i in antecedent]].all(axis=1)
        notB = ~matrix[:, index[consequent]]

        candidates = _condition_candidates(matrix, index, antecedent | {consequent}, minExceptionCount)
        for columns in _combinations(candidates, max_condition_size):
            maskZ = matrix[:, list(columns)].all(axis=1)
            countZ = maskZ.sum()
            maskAZ = maskA & maskZ
            countAZ = maskAZ.sum()
            countException = (maskAZ & notB).sum()

            if countAZ == 0 or countException < minExceptionCount or countException == 0:
                continue
            exceptionConf = countException / countAZ
            referenceConf = (maskZ & notB).sum() / countZ
            if exceptionConf >= min_exception_conf and referenceConf <= max_reference_conf:
                rows.append({
                    "antecedents": antecedent,
                    "consequent": consequent,
                    "condition": frozenset(index_item for index_item, column in index.items() if column in columns),
                    "common_support": float(rule["support"]),
                    "common_confidence": float(rule["confidence"]),
                    "exception_support": countException / total,
                    "exception_confidence": float(exceptionConf),
                    "reference_confidence": float(referenceConf),
                })

    result = pd.DataFrame(rows, columns=_EXCEPTION_COLUMNS)
    return result.sort_values(["exception_confidence", "exception_support"], ascending=False).reset_index(drop=True)


def anomalous_rules(transactions: Sequence[Iterable[str]], min_supp: float = 0.1, min_conf: float = 0.8,
                    min_anomalous_supp: float = 0.01, min_anomalous_conf: float = 0.8,
                    max_reference_conf: float = 0.5, max_anomaly_size: int = 1,
                    rules: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Find anomalous rules: of the strong common rules  A -> b, the sets of items C such that  A ^ ~b -> C  is strong (C
    characterizes the cases in which A holds but b does not) and  A -> C  is not.

    Args:
        transactions (Sequence[Iterable[str]]): crisp transactions
        min_supp, min_conf, rules: see `exception_rules`
        min_anomalous_supp (float): minimum support of  A ^ ~b ^ C
        min_anomalous_conf (float): minimum confidence of  A ^ ~b -> C
        max_reference_conf (float): MAXIMUM confidence of  A -> C
        max_anomaly_size (int): maximum number of items of C

    Returns:
        pandas.DataFrame with columns 'antecedents' (A), 'consequent' (b), 'anomaly' (C), 'common_support',
        'common_confidence', 'anomalous_cases' (number of transactions with A ^ ~b), 'anomalous_support',
        'anomalous_confidence' and 'reference_confidence' (of A -> C), sorted by confidence.
    """
    transactions = [set(t) for t in transactions]
    total = len(transactions)
    commonRules = _single_consequent_common_rules(transactions, rules, min_supp, min_conf)
    if total == 0 or len(commonRules) == 0:
        return pd.DataFrame(columns=_ANOMALOUS_COLUMNS)

    matrix, index = _boolean_matrix(transactions)
    minAnomalousCount = min_anomalous_supp * total
    rows = []

    for _, rule in commonRules.iterrows():
        antecedent = frozenset(rule["antecedents"])
        (consequent,) = tuple(rule["consequents"])
        maskA = matrix[:, [index[i] for i in antecedent]].all(axis=1)
        maskAnomaly = maskA & ~matrix[:, index[consequent]]
        countAnomalies = maskAnomaly.sum()
        if countAnomalies == 0:
            continue

        candidates = _condition_candidates(matrix, index, antecedent | {consequent}, minAnomalousCount)
        for columns in _combinations(candidates, max_anomaly_size):
            maskC = matrix[:, list(columns)].all(axis=1)
            countAnomalyC = (maskAnomaly & maskC).sum()
            if countAnomalyC < minAnomalousCount or countAnomalyC == 0:
                continue
            anomalousConf = countAnomalyC / countAnomalies
            referenceConf = (maskA & maskC).sum() / maskA.sum()
            if anomalousConf >= min_anomalous_conf and referenceConf <= max_reference_conf:
                rows.append({
                    "antecedents": antecedent,
                    "consequent": consequent,
                    "anomaly": frozenset(item for item, column in index.items() if column in columns),
                    "common_support": float(rule["support"]),
                    "common_confidence": float(rule["confidence"]),
                    "anomalous_cases": int(countAnomalies),
                    "anomalous_support": countAnomalyC / total,
                    "anomalous_confidence": float(anomalousConf),
                    "reference_confidence": float(referenceConf),
                })

    result = pd.DataFrame(rows, columns=_ANOMALOUS_COLUMNS)
    return result.sort_values(["anomalous_confidence", "anomalous_support"], ascending=False).reset_index(drop=True)
