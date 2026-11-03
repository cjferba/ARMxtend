# -*- coding: utf-8 -*-
"""
Anomaly detection by RULE VIOLATION: the association rules mined over a period of normal operation describe what
usually happens; a row that breaks a reliable rule (its antecedent holds and its consequent does not) is a candidate
anomaly, the more so the more reliable the rule. It is the same idea as exception rules (``ARM.exception_rules``), but
applied to NEW data with already mined rules, including temporal ones (``item@lagk``, see ``ARM.temporal_rules``):
"the door opened one step ago and the light has not come on".

A row VIOLATES the rule ``A => C`` if all the items of A hold and not all those of C hold. Each violation weighs
according to the violated rule (``weight``):

* ``surprise`` (default): ``-log2(1 - confidence)``, the bits of surprise of a violation if the rule holds with
  probability equal to its confidence; violating a 99 % rule weighs 6.6 bits, and an 80 % one, 2.3. The confidence is
  capped at ``1 - floor`` so that a 100 % rule does not weigh infinity.
* ``confidence``: the confidence of the rule.
* ``count``: 1, the number of violated rules.

The weight of a row is the SUM of the weights of the rules it violates. Redundant rules are always violated together
and count several times: prune first with ``prune_redundant_rules``; and rules that are just noise produce false
alarms: filter first with ``significant_rules``.

@author Carlos Fernandez-Basso
"""
from typing import List, Optional

import numpy as np
import pandas as pd

from ._apply import ItemTable, describe_rule, require_columns

WEIGHTS = ("surprise", "confidence", "count")


def _weights(rules: pd.DataFrame, weight: str, floor: float) -> np.ndarray:
    if weight not in WEIGHTS:
        raise ValueError("weight must be one of {}, received: '{}'".format(WEIGHTS, weight))
    if weight == "count":
        return np.ones(len(rules))
    require_columns(rules, "confidence")
    confidence = rules["confidence"].to_numpy(dtype=float)
    if weight == "confidence":
        return confidence
    if not 0 < floor < 1:
        raise ValueError("floor must be in (0, 1)")
    return -np.log2(np.maximum(1.0 - confidence, floor))


def _violated(table: ItemTable, antecedent, consequent) -> np.ndarray:
    return table.holds(antecedent) & table.evaluable(consequent) & ~table.holds(consequent)


def rule_violations(items: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """
    Boolean table of violations: one row per row of `items` and one column per rule of `rules` (with the same index);
    True where the row violates the rule.

    Args:
        items (pandas.DataFrame): boolean (0/1) items, ordered in time if the rules carry lags. The items of the rules
            must be columns of `items`
        rules (pandas.DataFrame): rules with columns 'antecedents' and 'consequents' (sets of items, with or without
            lag ``@lagk``)

    The first rows, without enough previous data for a rule with lags, do not violate it. To evaluate a new period
    with history, include the previous rows in `items` and select the period afterwards.
    """
    require_columns(rules, "antecedents", "consequents")
    table = ItemTable(items)
    matrix = np.zeros((table.rows, len(rules)), dtype=bool)
    for position, (antecedent, consequent) in enumerate(zip(rules["antecedents"], rules["consequents"])):
        matrix[:, position] = _violated(table, antecedent, consequent)
    return pd.DataFrame(matrix, index=items.index, columns=rules.index)


def anomaly_scores(items: pd.DataFrame, rules: pd.DataFrame, weight: str = "surprise",
                   floor: float = 1e-3) -> pd.DataFrame:
    """
    Anomaly score of each row of `items`: the sum of the weights of the rules it violates (see the module docstring).

    Args:
        items, rules: see `rule_violations`
        weight (str): 'surprise' (default), 'confidence' or 'count'
        floor (float): with 'surprise', the smallest violation probability considered, in (0, 1): it bounds the weight
            of a rule to ``-log2(floor)``

    Returns:
        pandas.DataFrame with the index of `items` and the columns 'score' (0 if it violates no rule), 'violations'
        (violated rules) and 'applicable' (rules whose antecedent holds in the row: out of how many it could fail)
    """
    require_columns(rules, "antecedents", "consequents")
    weights = _weights(rules, weight, floor)
    table = ItemTable(items)
    score = np.zeros(table.rows)
    violations = np.zeros(table.rows, dtype=int)
    applicable = np.zeros(table.rows, dtype=int)
    for position, (antecedent, consequent) in enumerate(zip(rules["antecedents"], rules["consequents"])):
        fires = table.holds(antecedent) & table.evaluable(consequent)
        violated = fires & ~table.holds(consequent)
        applicable += fires
        violations += violated
        score += weights[position] * violated
    return pd.DataFrame({"score": score, "violations": violations, "applicable": applicable}, index=items.index)


def detect_anomalies(items: pd.DataFrame, rules: pd.DataFrame, min_score: float = 0.0, top: Optional[int] = None,
                     weight: str = "surprise", floor: float = 1e-3) -> pd.DataFrame:
    """
    The anomalous rows of `items`, explained: those with `score` greater than `min_score`, from highest to lowest
    score (with equal score, in chronological order).

    Args:
        items, rules, weight, floor: see `anomaly_scores`
        min_score (float): minimum score (exclusive); 0 by default, that is, any row that violates some rule
        top (int, optional): keep only the first `top`

    Returns:
        pandas.DataFrame with the index of the anomalous rows and the columns 'score', 'violations' and 'rules' (list
        with the text of the violated rules, e.g. ``'motion (t-1) => light (t)'``)
    """
    scores = anomaly_scores(items, rules, weight, floor)
    values = scores["score"].to_numpy()
    order = np.argsort(-values, kind="stable")                  # a igual puntuacion, orden cronologico
    order = order[values[order] > min_score]
    if top is not None:
        order = order[:top]

    table = ItemTable(items)
    explanation: List[List[str]] = [[] for _ in range(len(order))]
    for antecedent, consequent in zip(rules["antecedents"], rules["consequents"]):
        text = describe_rule(antecedent, consequent)
        for row in np.flatnonzero(_violated(table, antecedent, consequent)[order]):
            explanation[row].append(text)
    result = scores.iloc[order][["score", "violations"]].copy()
    result["rules"] = explanation
    return result
