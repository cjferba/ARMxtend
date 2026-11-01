# -*- coding: utf-8 -*-
"""
Itemsets RAROS: poco frecuentes pero con significado (el efecto adverso de dos farmacos juntos, el fallo
que sigue a una secuencia concreta de alarmas). Con un soporte minimo bajo para alcanzarlos, el minado
habitual se desborda: casi todo itemset es "frecuente" y salen millones.

Un itemset es RARO si ``min_supp <= soporte < max_supp`` (``min_supp`` es un suelo contra el ruido: lo que
ocurre una o dos veces no es un patron), y es RARO MINIMAL (MRI) si ademas TODOS sus subconjuntos propios
son frecuentes (``soporte >= max_supp``). Los MRI son la base compacta de los itemsets raros: cualquier
raro es un MRI o contiene uno (Szathmary, Napoli y Valtchev, 2007, "Towards rare itemset mining"). Se hallan
con Apriori por niveles: se unen itemsets frecuentes de tamano k-1; un candidato frecuente sigue creciendo,
y uno raro es un MRI y se detiene (sus superconjuntos ya no serian minimales).

@author Carlos Fernandez-Basso
"""
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._vertical import Vertical, count_bits

_EPS = 1e-9


def minimal_rare_itemsets(transactions: Iterable[Iterable[Any]], max_supp: float, min_supp: float = 0.0,
                          min_count: int = 1, min_len: int = 1, max_len: Optional[int] = None) -> pd.DataFrame:
    """
    Itemsets raros minimales (ver el docstring del modulo).

    Argumentos:
        transactions: una transaccion por elemento, cada una una coleccion de items
        max_supp (float): soporte a partir del cual un itemset es frecuente, en (0, 1]; los raros estan por debajo
        min_supp (float): soporte relativo minimo de un raro, en [0, 1)
        min_count (int): numero minimo de transacciones de un raro (por defecto 1)
        min_len, max_len (int): numero minimo (por defecto 1) y maximo de items de los itemsets devueltos

    Retorna:
        pandas.DataFrame con 'support' e 'itemsets' (``frozenset``), por soporte decreciente;
        ``attrs`` lleva 'n_transactions' y 'n_frequent' (itemsets frecuentes visitados, por si se quiere
        comparar con el minado habitual)
    """
    if not 0 < max_supp <= 1 or not 0 <= min_supp < max_supp:
        raise ValueError("hace falta 0 <= min_supp < max_supp <= 1")
    if min_count < 1:
        raise ValueError("min_count debe ser >= 1")
    vertical = Vertical(transactions)
    total = vertical.total
    found: List[Tuple[int, Tuple[int, ...]]] = []
    n_frequent = 0
    if total:
        frequent_count = max(1, int(np.ceil(max_supp * total - _EPS)))
        rare_count = max(min_count, int(np.ceil(min_supp * total - _EPS)), 1)
        level: Dict[Tuple[int, ...], np.ndarray] = {}
        for index, count in enumerate(vertical.counts):
            if count >= frequent_count:
                level[(index,)] = vertical.masks[index]
            elif count >= rare_count:
                found.append((count, (index,)))
        n_frequent += len(level)
        size = 1
        while level and (max_len is None or size < max_len):
            groups: Dict[Tuple[int, ...], List[int]] = {}
            for itemset in sorted(level):
                groups.setdefault(itemset[:-1], []).append(itemset[-1])
            following: Dict[Tuple[int, ...], np.ndarray] = {}
            for prefix, tails in groups.items():
                for first in range(len(tails)):
                    for second in range(first + 1, len(tails)):
                        a, b = tails[first], tails[second]
                        candidate = prefix + (a, b)
                        if any(candidate[:k] + candidate[k + 1:] not in level for k in range(len(prefix))):
                            continue
                        mask = level[prefix + (a,)] & vertical.masks[b]
                        count = count_bits(mask)
                        if count >= frequent_count:
                            following[candidate] = mask
                        elif count >= rare_count:
                            found.append((count, candidate))
            level = following
            n_frequent += len(level)
            size += 1
    found = [entry for entry in found if len(entry[1]) >= min_len]
    found.sort(key=lambda entry: (-entry[0], len(entry[1]), entry[1]))
    result = pd.DataFrame({"support": [count / total for count, _ in found],
                           "itemsets": [frozenset(vertical.items[i] for i in itemset) for _, itemset in found]})
    result.attrs["n_transactions"] = total
    result.attrs["n_frequent"] = n_frequent
    return result
