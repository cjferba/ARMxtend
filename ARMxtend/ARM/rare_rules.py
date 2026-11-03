# -*- coding: utf-8 -*-
"""
RARE rules: ``X => Y`` with low support but high confidence, from the minimal rare itemsets (``FIM.rare``). Since every
proper subset of an MRI is frequent, the antecedent ``X`` and the consequent ``Y`` of any partition are: the rule is
rare only because of the combination, and its confidence and lift (often very high) tell whether that combination is
more than chance.

@author Carlos Fernandez-Basso
"""
from itertools import combinations
from typing import Any, Iterable, List, Optional

import pandas as pd

from ..FIM._vertical import Vertical
from ..FIM.rare import minimal_rare_itemsets
from ._rulerows import RULE_COLUMNS, rule_row

_EPS = 1e-9


def rare_rules(transactions: Iterable[Iterable[Any]], max_supp: float, min_supp: float = 0.0, min_count: int = 2,
               min_conf: float = 0.6, min_lift: Optional[float] = None, max_antecedent: Optional[int] = None,
               max_consequent: Optional[int] = None, max_len: Optional[int] = None) -> pd.DataFrame:
    """
    Rare rules (see the module docstring).

    Args:
        transactions: one transaction per element, each a collection of items
        max_supp (float): support below which an itemset is rare (and above which, frequent), in (0, 1]
        min_supp (float): minimum relative support of the rule, in [0, max_supp)
        min_count (int): minimum transactions of the rule (2 by default: what happens once is not a pattern)
        min_conf (float): minimum confidence, in [0, 1]
        min_lift (float, optional): minimum lift
        max_antecedent, max_consequent, max_len (int, optional): maximum sizes

    Returns:
        pandas.DataFrame with the format of ``ARM.association_rules``, by decreasing confidence and support; ``attrs``
        holds 'n_transactions' and 'n_candidates' (partitions examined)
    """
    if not 0 <= min_conf <= 1:
        raise ValueError("min_conf debe estar en [0, 1]")
    if (max_antecedent is not None and max_antecedent < 1) or (max_consequent is not None and max_consequent < 1):
        raise ValueError("max_antecedent y max_consequent deben ser >= 1")
    baskets = [list(transaction) for transaction in transactions]
    itemsets = minimal_rare_itemsets(baskets, max_supp, min_supp, min_count=min_count, min_len=2, max_len=max_len)
    vertical = Vertical(baskets)
    total = vertical.total
    position = {item: k for k, item in enumerate(vertical.items)}
    rows: List[dict] = []
    examined = 0
    for support, items in zip(itemsets["support"], itemsets["itemsets"]):
        indices = sorted(position[item] for item in items)
        for size in range(1, len(indices)):
            if (max_antecedent is not None and size > max_antecedent) or \
                    (max_consequent is not None and len(indices) - size > max_consequent):
                continue
            for antecedent in combinations(indices, size):
                consequent = tuple(i for i in indices if i not in antecedent)
                examined += 1
                support_x, support_y = vertical.count_of(antecedent) / total, vertical.count_of(consequent) / total
                confidence = support / support_x
                if confidence < min_conf - _EPS or (min_lift is not None and confidence / support_y < min_lift - _EPS):
                    continue
                rows.append(rule_row(frozenset(vertical.items[i] for i in antecedent),
                                     frozenset(vertical.items[i] for i in consequent), support, support_x, support_y))
    result = pd.DataFrame(rows, columns=RULE_COLUMNS)
    if len(result):
        result = result.sort_values(["confidence", "support"], ascending=False, kind="mergesort").reset_index(drop=True)
    result.attrs["n_transactions"] = total
    result.attrs["n_candidates"] = max(examined, len(result))
    return result
