# -*- coding: utf-8 -*-
"""
HIGH-UTILITY rules: ``X => Y`` where the itemset ``X u Y`` is high-utility (``FIM.utility``), that is, rules about what
is worth the MOST and not about what repeats the most. Confidence and the other measures are the usual ones (they
count transactions); to them are added the utility of the itemset of the rule and its weight over the total utility
of the database. In the literature there are other definitions of "utility confidence" (e.g. utility of ``X u Y``
over utility of ``X``, Sahoo et al., 2015) that are not used here.

@author Carlos Fernandez-Basso
"""
from itertools import combinations
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from ..FIM.utility import Quantities, high_utility_itemsets, utility_tables
from ._rulerows import RULE_COLUMNS, rule_row

_EPS = 1e-9


def high_utility_rules(quantities: Quantities, profits: Optional[Mapping[str, float]] = None, min_utility: float = 0.1,
                       absolute: bool = False, min_conf: float = 0.6, min_lift: Optional[float] = None,
                       max_antecedent: Optional[int] = None, max_consequent: Optional[int] = None,
                       max_len: Optional[int] = None) -> pd.DataFrame:
    """
    High-utility rules (see the module docstring).

    Args:
        quantities, profits, min_utility, absolute, max_len: see ``FIM.utility.high_utility_itemsets``
        min_conf (float): minimum confidence (in transactions), in [0, 1]
        min_lift (float, optional): minimum lift
        max_antecedent, max_consequent (int, optional): maximum number of items of X and of Y

    Returns:
        pandas.DataFrame with the format of ``ARM.association_rules`` plus 'utility' (total utility of ``X u Y``) and
        'utility_share' (its fraction of the total utility of the database), by decreasing utility; ``attrs`` holds
        'n_transactions', 'total_utility' and 'n_candidates'
    """
    if not 0 <= min_conf <= 1:
        raise ValueError("min_conf debe estar en [0, 1]")
    if (max_antecedent is not None and max_antecedent < 1) or (max_consequent is not None and max_consequent < 1):
        raise ValueError("max_antecedent y max_consequent deben ser >= 1")
    items, amounts, _ = utility_tables(quantities, profits)
    itemsets = high_utility_itemsets(quantities, profits, min_utility, absolute, min_len=2, max_len=max_len)
    total_utility = itemsets.attrs["total_utility"]
    present = amounts > 0
    position = {item: k for k, item in enumerate(items)}
    rows = []
    examined = 0
    for utility, itemset, support in zip(itemsets["utility"], itemsets["itemsets"], itemsets["support"]):
        indices = sorted(position[item] for item in itemset)
        for size in range(1, len(indices)):
            if (max_antecedent is not None and size > max_antecedent) or \
                    (max_consequent is not None and len(indices) - size > max_consequent):
                continue
            for antecedent in combinations(indices, size):
                consequent = tuple(i for i in indices if i not in antecedent)
                examined += 1
                support_x = float(present[:, list(antecedent)].all(axis=1).mean())
                support_y = float(present[:, list(consequent)].all(axis=1).mean())
                confidence = support / support_x
                if confidence < min_conf - _EPS or (min_lift is not None and confidence / support_y < min_lift - _EPS):
                    continue
                row = rule_row(frozenset(items[i] for i in antecedent), frozenset(items[i] for i in consequent), support,
                               support_x, support_y)
                row["utility"], row["utility_share"] = utility, utility / total_utility if total_utility else np.nan
                rows.append(row)
    result = pd.DataFrame(rows, columns=RULE_COLUMNS + ["utility", "utility_share"])
    if len(result):
        result = result.sort_values(["utility", "confidence"], ascending=False, kind="mergesort").reset_index(drop=True)
    result.attrs.update({"n_transactions": len(amounts), "total_utility": total_utility,
                         "n_candidates": max(examined, len(result))})
    return result
