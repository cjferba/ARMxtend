# -*- coding: utf-8 -*-
"""
Sequential rules with VETO: ``X => ~Y`` means "once all the items of X have occurred, the item Y does NOT occur
afterwards". They say which events are *discarded* by a situation ("after the alarm is armed, the door is not
opened") and complement the positive rules of ``ARM.sequential_rules``.

Definition, in a sequence of time-ordered elements: let ``k`` be the first instant at which all the items of X have
appeared. The rule is VALID in the sequence if X completes and Y does not appear in any element AFTER ``k`` (it may
appear at ``k`` or before). Measures:

* support(X => ~Y)    = fraction of sequences in which the rule is valid;
* confidence          = sequences in which it is valid / sequences that contain all the items of X;
* 'consequent support' = fraction of sequences in which Y NEVER appears (the base rate of the veto), so that
  ``lift = confidence / base rate`` says whether X makes the absence of Y more likely than it already was.

The veto is only defined for a single item Y (the absence of a set of items has several readings). The antecedents are
the frequent itemsets of the sequences seen as baskets, as in ``sequential_rules``.

NOTE: item names must not contain '-', which is the item separator in the itemset keys of ``FIM._shared``.

@author Carlos Fernandez-Basso
"""
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd

from ..FFIM import fpgrowth
from ..FIM._shared import key_to_itemset
from ..FIM.sequential import _normalize
from ..preprocessing.timeseries import check_item_names
from ._measures import MEASURES
from ._rulerows import RULE_COLUMNS, rule_row

_ABSENT_FIRST = 10 ** 9


def sequential_veto_rules(sequences: Iterable[Iterable[Union[str, Iterable[str]]]], min_supp: float,
                          min_conf: float = 0.8, max_antecedent: int = 2, min_antecedent_supp: float = 0.1,
                          metric: str = "confidence", min_threshold: float = 0.0) -> pd.DataFrame:
    """
    Mine the sequential rules with veto ``X => ~Y`` (see the module docstring).

    Args:
        sequences: one sequence per element, as in ``ARM.sequential_rules``
        min_supp (float): minimum support of the rule (fraction of sequences where it is valid), in (0, 1]
        min_conf (float): minimum confidence, in [0, 1]
        max_antecedent (int): maximum number of items of X
        min_antecedent_supp (float): minimum support of X as a basket, in (0, 1]. Vetoes of rare events hold in almost
            every sequence, so X must be frequent for the rule to say something
        metric (str): additional measure to filter by (any of ``ARM._measures.MEASURES``)
        min_threshold (float): minimum threshold of `metric` (0 = no filtering)

    Returns:
        pandas.DataFrame with the format of ``ARM.sequential_rules``; 'consequents' holds the single vetoed item Y and
        'consequent support' the fraction of sequences without Y. Sorted by decreasing lift and confidence.
    """
    if not 0 < min_supp <= 1 or not 0 < min_antecedent_supp <= 1:
        raise ValueError("min_supp and min_antecedent_supp must be in (0, 1]")
    if not 0 <= min_conf <= 1:
        raise ValueError("min_conf must be in [0, 1]")
    if max_antecedent < 1:
        raise ValueError("max_antecedent must be >= 1")
    if metric not in MEASURES and metric != "support":
        raise ValueError("metric must be 'support' or one of {}, got: '{}'".format(sorted(MEASURES), metric))
    database = _normalize(sequences)
    total = len(database)
    if total == 0:
        return pd.DataFrame(columns=RULE_COLUMNS)
    vocabulary = sorted({item for sequence in database for element in sequence for item in element})
    check_item_names(vocabulary)
    column = {item: index for index, item in enumerate(vocabulary)}
    first = np.full((total, len(vocabulary)), _ABSENT_FIRST, dtype=np.int64)
    last = np.full((total, len(vocabulary)), -1, dtype=np.int64)
    for row, sequence in enumerate(database):
        for position, element in enumerate(sequence):
            for item in element:
                cell = column[item]
                if first[row, cell] == _ABSENT_FIRST:
                    first[row, cell] = position
                last[row, cell] = position
    baskets = [set().union(*sequence) if sequence else set() for sequence in database]
    frequent = fpgrowth(baskets, min_antecedent_supp)

    rows: List[dict] = []
    for key, supportX in frequent.items():
        antecedent = key_to_itemset(key)
        if len(antecedent) > max_antecedent:
            continue
        cells = [column[item] for item in antecedent]
        completed = first[:, cells].max(axis=1)
        has_x = completed < _ABSENT_FIRST
        for item in vocabulary:
            if item in antecedent:
                continue
            valid = has_x & (last[:, column[item]] <= completed)
            support = float(np.count_nonzero(valid)) / total
            if support < min_supp - 1e-9:
                continue
            confidence = support / supportX
            if confidence < min_conf - 1e-9:
                continue
            never = float(np.count_nonzero(last[:, column[item]] < 0)) / total
            entry = rule_row(frozenset(antecedent), frozenset([item]), support, supportX, never)
            if metric != "support" and min_threshold and entry[metric] < min_threshold:
                continue
            rows.append(entry)
    result = pd.DataFrame(rows, columns=RULE_COLUMNS)
    return result.sort_values(["lift", "confidence"], ascending=False, kind="mergesort").reset_index(drop=True)


def contains_veto_rule(sequence: Iterable[Iterable[str]], antecedent: Iterable[str], vetoed: str) -> bool:
    """True if ``antecedent => ~vetoed`` is valid in `sequence` (X completes and `vetoed` does not occur afterwards)."""
    elements = [frozenset(e) for e in sequence]
    x = list(antecedent)
    firstSeen: Dict[str, int] = {}
    for position, element in enumerate(elements):
        for item in element:
            firstSeen.setdefault(item, position)
    if any(item not in firstSeen for item in x):
        return False
    moment = max(firstSeen[item] for item in x)
    return not any(vetoed in element for element in elements[moment + 1:])


__all__ = ["sequential_veto_rules", "contains_veto_rule"]
