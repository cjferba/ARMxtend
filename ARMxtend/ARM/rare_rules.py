# -*- coding: utf-8 -*-
"""
Reglas RARAS: ``X => Y`` con soporte bajo pero confianza alta, a partir de los itemsets raros minimales
(``FIM.rare``). Como todo subconjunto propio de un MRI es frecuente, el antecedente ``X`` y el consecuente
``Y`` de cualquier particion lo son: la regla es rara solo por la combinacion, y su confianza y su lift
(a menudo muy altos) dicen si esa combinacion es mas que casualidad.

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
    Reglas raras (ver el docstring del modulo).

    Argumentos:
        transactions: una transaccion por elemento, cada una una coleccion de items
        max_supp (float): soporte por debajo del cual un itemset es raro (y por encima, frecuente), en (0, 1]
        min_supp (float): soporte relativo minimo de la regla, en [0, max_supp)
        min_count (int): transacciones minimas de la regla (por defecto 2: lo que ocurre una vez no es un patron)
        min_conf (float): confianza minima, en [0, 1]
        min_lift (float, opcional): lift minimo
        max_antecedent, max_consequent, max_len (int, opcional): tamanos maximos

    Retorna:
        pandas.DataFrame con el formato de ``ARM.association_rules``, por confianza y soporte decrecientes;
        ``attrs`` lleva 'n_transactions' y 'n_candidates' (particiones examinadas)
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
