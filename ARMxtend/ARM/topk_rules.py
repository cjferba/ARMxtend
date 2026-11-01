# -*- coding: utf-8 -*-
"""
Las K mejores reglas por soporte (TOP-K), sin fijar el soporte minimo (ver ``FIM.topk``): se piden
las K reglas mas frecuentes con una confianza minima, en vez de adivinar un soporte que no deje la salida
vacia ni la desborde (Fournier-Viger y Tseng, 2012, "Mining top-k association rules").

Los itemsets se generan por soporte no creciente (``FIM.topk.iter_itemsets_by_support``); todas las
reglas de un itemset tienen su soporte, de modo que cuando el siguiente itemset tiene menos soporte que la
K-esima mejor regla, ya no puede mejorarla y la busqueda se detiene. El resultado es exacto. El desempate
entre reglas de igual soporte es la confianza decreciente y luego el orden lexicografico.

@author Carlos Fernandez-Basso
"""
import heapq
from itertools import combinations
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..FIM._vertical import Vertical
from ..FIM.topk import iter_itemsets_by_support
from ._rulerows import RULE_COLUMNS, rule_row

_EPS = 1e-9


def top_k_rules(transactions: Iterable[Iterable[Any]], k: int, min_conf: float = 0.6, min_lift: Optional[float] = None,
                max_antecedent: Optional[int] = None, max_consequent: Optional[int] = None, max_len: Optional[int] = 6,
                min_supp: float = 0.0) -> pd.DataFrame:
    """
    Las `k` reglas de mayor soporte con confianza >= `min_conf` (ver el docstring del modulo).

    Argumentos:
        transactions: una transaccion por elemento, cada una una coleccion de items
        k (int): cuantas reglas
        min_conf (float): confianza minima, en [0, 1]
        min_lift (float, opcional): lift minimo
        max_antecedent, max_consequent (int, opcional): maximo de items de X y de Y
        max_len (int, opcional): maximo de items de una regla (por defecto 6: las reglas largas son ilegibles y
            las particiones de un itemset crecen exponencialmente); ``None`` no limita
        min_supp (float): suelo opcional de soporte relativo

    Retorna:
        pandas.DataFrame con el formato de ``ARM.association_rules`` (soporte, confianza, lift y las demas
        medidas), por soporte decreciente; ``attrs`` lleva 'n_transactions' y 'n_candidates' (reglas
        examinadas, para ``ARM.significance``). Si hay menos de `k` reglas que cumplan, salen todas
    """
    if k < 1:
        raise ValueError("k debe ser >= 1")
    if not 0 <= min_conf <= 1 or not 0 <= min_supp <= 1:
        raise ValueError("min_conf y min_supp deben estar en [0, 1]")
    if (max_antecedent is not None and max_antecedent < 1) or (max_consequent is not None and max_consequent < 1):
        raise ValueError("max_antecedent y max_consequent deben ser >= 1")
    vertical = Vertical(transactions)
    total = vertical.total
    rows: List[Tuple[int, dict]] = []
    best: List[int] = []                                    # los k mayores soportes (en transacciones) hasta ahora
    examined = 0
    if total:
        minimum = max(1, int(np.ceil(min_supp * total - _EPS)))
        for count, itemset in iter_itemsets_by_support(vertical, minimum, max_len):
            if len(best) == k and count < best[0]:
                break
            if len(itemset) < 2:
                continue
            for size in range(1, len(itemset)):
                if (max_antecedent is not None and size > max_antecedent) or \
                        (max_consequent is not None and len(itemset) - size > max_consequent):
                    continue
                for antecedent in combinations(itemset, size):
                    consequent = tuple(i for i in itemset if i not in antecedent)
                    examined += 1
                    count_x, count_y = vertical.count_of(antecedent), vertical.count_of(consequent)
                    if count / count_x < min_conf - _EPS:
                        continue
                    if min_lift is not None and (count / count_x) / (count_y / total) < min_lift - _EPS:
                        continue
                    rows.append((count, rule_row(frozenset(vertical.items[i] for i in antecedent),
                                                 frozenset(vertical.items[i] for i in consequent),
                                                 count / total, count_x / total, count_y / total)))
                    heapq.heappush(best, count)
                    if len(best) > k:
                        heapq.heappop(best)
    result = pd.DataFrame([row for _, row in rows], columns=RULE_COLUMNS)
    if len(result):
        order = sorted(range(len(result)), key=lambda r: (-result["support"].iloc[r], -result["confidence"].iloc[r],
                                                          sorted(result["antecedents"].iloc[r]),
                                                          sorted(result["consequents"].iloc[r])))
        result = result.iloc[order[:k]].reset_index(drop=True)
    result.attrs["n_transactions"] = total
    result.attrs["n_candidates"] = max(examined, len(result))
    return result
