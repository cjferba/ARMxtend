# -*- coding: utf-8 -*-
"""
Reglas de ALTA UTILIDAD: ``X => Y`` donde el itemset ``X u Y`` es de alta utilidad (``FIM.utility``), es
decir, reglas sobre lo que mas VALE y no sobre lo que mas se repite. La confianza y las demas medidas son
las habituales (cuentan transacciones); a ellas se anaden la utilidad del itemset de la regla y su peso
sobre la utilidad total de la base. En la literatura hay otras definiciones de "confianza de utilidad"
(p. ej. utilidad de ``X u Y`` entre utilidad de ``X``, Sahoo et al., 2015) que no se usan aqui.

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
    Reglas de alta utilidad (ver el docstring del modulo).

    Argumentos:
        quantities, profits, min_utility, absolute, max_len: ver ``FIM.utility.high_utility_itemsets``
        min_conf (float): confianza minima (en transacciones), en [0, 1]
        min_lift (float, opcional): lift minimo
        max_antecedent, max_consequent (int, opcional): maximo de items de X y de Y

    Retorna:
        pandas.DataFrame con el formato de ``ARM.association_rules`` mas 'utility' (utilidad total de
        ``X u Y``) y 'utility_share' (su fraccion de la utilidad total de la base), por utilidad decreciente;
        ``attrs`` lleva 'n_transactions', 'total_utility' y 'n_candidates'
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
