# -*- coding: utf-8 -*-
"""
Pruning of redundant association rules.

A rule  X2 -> Y2  is REDUNDANT if there is another, different, more general rule  X1 -> Y1 -- with a smaller or equal
antecedent (X1 included in X2) and a larger or equal consequent (Y1 includes Y2) -- that has EXACTLY the same support
and confidence (Bastide, Taouil, Pasquier, Stumme & Lakhal, 2000, "Mining minimal non-redundant association rules
using frequent closed itemsets"; see also Zaki, 2004). The general rule conveys all the information of the redundant
one: it demands fewer conditions and concludes at least the same.

Besides, ``prune_unproductive_rules`` removes the "unproductive" rules: those that add conditions to the antecedent of
another rule without noticeably improving its confidence (Bayardo, Agrawal and Gunopulos, 2000, "Constraint-based rule
mining in large, dense databases", Data Mining and Knowledge Discovery 4, 217-240). Unlike redundancy, which demands
exact equality, the tolerance (``min_improvement``) is a parameter.

It works with any DataFrame of rules with columns 'antecedents' and 'consequents' (frozensets) plus 'support' and
'confidence': the output of ``ARM.association_rules``, ``FIM.FARE.fuzzy_association_rules``, the rules with absent
items of ``ARM.negative_rules`` or the gradual ones of ``ARM.gradual_rules``.

@author Carlos Fernandez-Basso
"""
import math
from itertools import combinations
from typing import Dict, FrozenSet, List, Tuple

import pandas as pd


def prune_redundant_rules(rules: pd.DataFrame, tol: float = 1e-9, support_column: str = "support",
                         confidence_column: str = "confidence") -> pd.DataFrame:
    """
    Remove the redundant rules of `rules`.

    Args:
        rules (pandas.DataFrame): rules with columns 'antecedents', 'consequents', `support_column` and
            `confidence_column`
        tol (float): absolute tolerance to consider two support/confidence values equal
        support_column, confidence_column (str): names of the columns

    Returns:
        pandas.DataFrame with the non-redundant rules (same structure and relative order as `rules`; duplicated rules
        are reduced to one).
    """
    if len(rules) == 0:
        return rules.copy()

    support = rules[support_column].to_numpy(dtype=float)
    confidence = rules[confidence_column].to_numpy(dtype=float)
    antecedents = [frozenset(a) for a in rules["antecedents"]]
    consequents = [frozenset(c) for c in rules["consequents"]]

    # Only rules with equal (support, confidence) can be equivalent:
    # they are grouped by those values rounded to `tol` and compared inside the group.
    digits = max(0, int(round(-math.log10(tol)))) if tol > 0 else 12
    groups: Dict[Tuple[float, float], List[int]] = {}
    for position in range(len(rules)):
        groups.setdefault((round(support[position], digits), round(confidence[position], digits)), []).append(position)

    redundant = set()
    for positions in groups.values():
        for candidate in positions:
            for general in positions:
                if general == candidate or general in redundant:
                    continue
                if abs(support[general] - support[candidate]) > tol or abs(confidence[general] - confidence[candidate]) > tol:
                    continue
                isMoreGeneral = antecedents[general] <= antecedents[candidate] and consequents[general] >= consequents[candidate]
                isIdentical = antecedents[general] == antecedents[candidate] and consequents[general] == consequents[candidate]
                # An identical rule only removes those with a higher index (the first one is kept)
                if isMoreGeneral and (not isIdentical or general < candidate):
                    redundant.add(candidate)
                    break

    return rules.iloc[[p for p in range(len(rules)) if p not in redundant]].reset_index(drop=True)


_EXHAUSTIVE_LIMIT = 12      # antecedents of up to this size: all their subsets are tried


def prune_unproductive_rules(rules: pd.DataFrame, min_improvement: float = 0.0,
                             confidence_column: str = "confidence") -> pd.DataFrame:
    """
    Remove the unproductive rules: a rule ``X => Y`` is discarded if there is another ``X' => Y`` in `rules`, with X'
    a proper subset of X, such that ``conf(X => Y) - conf(X' => Y) <= min_improvement``; that is, if the conditions of
    X that are not in X' do not improve the confidence by more than `min_improvement`. It is compared with ANY more
    general rule present (not only those that are kept), so the result does not depend on the order.

    Args:
        rules (pandas.DataFrame): rules with 'antecedents', 'consequents' and `confidence_column`
        min_improvement (float): minimum confidence improvement that each additional condition must contribute; 0
            (default) discards those that do not improve the confidence of a more general rule; with 0.05 at least 5
            points must be gained
        confidence_column (str): name of the confidence column

    Returns:
        pandas.DataFrame with the productive rules, in the order of `rules` (the most general rule of each family is
        always kept)
    """
    if len(rules) == 0:
        return rules.copy()
    antecedents = [frozenset(a) for a in rules["antecedents"]]
    consequents = [frozenset(c) for c in rules["consequents"]]
    confidence = rules[confidence_column].to_numpy(dtype=float)

    best: Dict[Tuple[FrozenSet[str], FrozenSet[str]], float] = {}
    byConsequent: Dict[FrozenSet[str], List[int]] = {}
    for position, key in enumerate(zip(antecedents, consequents)):
        best[key] = max(best.get(key, -math.inf), confidence[position])
        byConsequent.setdefault(consequents[position], []).append(position)

    keep = []
    for position in range(len(rules)):
        antecedent, consequent = antecedents[position], consequents[position]
        if len(antecedent) <= _EXHAUSTIVE_LIMIT:
            general = (best.get((frozenset(subset), consequent), -math.inf)
                       for size in range(1, len(antecedent))
                       for subset in combinations(antecedent, size))
        else:
            general = (confidence[other] for other in byConsequent[consequent] if antecedents[other] < antecedent)
        strongest = max(general, default=-math.inf)
        keep.append(not (confidence[position] - strongest <= min_improvement + 1e-12))
    return rules.iloc[[position for position, kept in enumerate(keep) if kept]].reset_index(drop=True)
