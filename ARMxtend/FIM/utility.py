# -*- coding: utf-8 -*-
"""
Itemsets de ALTA UTILIDAD: no los mas frecuentes, sino los que mas VALEN. Una transaccion lleva cantidades
(3 botellas, 1 queso) y cada item tiene un beneficio unitario; la utilidad de un itemset en una transaccion
que lo contiene es la suma de ``cantidad x beneficio`` de sus items, y su utilidad total, la suma sobre las
transacciones. Un itemset es de alta utilidad (HUI) si su utilidad total alcanza ``min_utility``. Un par
de productos caros puede ser un HUI sin ser frecuente, y uno muy frecuente de productos baratos, no serlo
(Yao, Hamilton y Butz, 2004, "A foundational approach to mining itemset utilities from databases").

La utilidad NO es antimonotona (anadir un item puede subirla), asi que no se puede podar con ella. Se usa el
algoritmo de dos fases (Liu, Liao y Choudhary, 2005): la utilidad ponderada por transaccion (TWU) de un
itemset, la suma de las utilidades TOTALES de las transacciones que lo contienen, SI es antimonotona y
acota por arriba la utilidad de todos sus superconjuntos; se buscan por niveles los itemsets con
``TWU >= min_utility`` y de ellos se queda con los de utilidad real suficiente. Es exacto. Los beneficios
deben ser >= 0 (con beneficios negativos la cota TWU deja de valer).

@author Carlos Fernandez-Basso
"""
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

Quantities = Union[pd.DataFrame, Iterable[Mapping[Any, float]]]
_EPS = 1e-9


def utility_tables(quantities: Quantities, profits: Optional[Mapping[Any, float]]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Items, matriz de cantidades (transacciones x items) y matriz de utilidades (cantidad x beneficio)."""
    table = quantities.copy() if isinstance(quantities, pd.DataFrame) else pd.DataFrame(list(quantities))
    table.columns = [str(name) for name in table.columns]
    if not all(pd.api.types.is_numeric_dtype(table[name]) for name in table.columns):
        raise ValueError("las cantidades deben ser numericas")
    amounts = table.fillna(0.0).to_numpy(dtype=float)
    if (amounts < 0).any():
        raise ValueError("las cantidades no pueden ser negativas")
    items = list(table.columns)
    if profits is None:
        gain = np.ones(len(items))
    else:
        by_name = {str(item): float(value) for item, value in profits.items()}
        missing = [item for item in items if item not in by_name]
        if missing:
            raise ValueError("faltan los beneficios de {}".format(missing))
        gain = np.array([by_name[item] for item in items])
        if (gain < 0).any():
            raise ValueError("los beneficios deben ser >= 0")
    return items, amounts, amounts * gain


def high_utility_itemsets(quantities: Quantities, profits: Optional[Mapping[Any, float]] = None, min_utility: float = 0.1,
                          absolute: bool = False, min_len: int = 1, max_len: Optional[int] = None) -> pd.DataFrame:
    """
    Itemsets de alta utilidad (ver el docstring del modulo).

    Argumentos:
        quantities: una transaccion por fila: un ``DataFrame`` (columnas = items, valores = cantidades; 0 o NaN
            si no se compra) o una lista de diccionarios ``{item: cantidad}``
        profits (dict, opcional): beneficio unitario de cada item (>= 0); por defecto 1 para todos
        min_utility (float): utilidad minima: una fraccion en (0, 1] de la utilidad total de la base, o un
            valor absoluto si `absolute` es ``True``
        min_len, max_len (int): numero minimo (por defecto 1) y maximo de items de los itemsets devueltos

    Retorna:
        pandas.DataFrame con 'utility', 'itemsets' (``frozenset``), 'support' (fraccion de transacciones que
        lo contienen) y 'length', por utilidad decreciente; ``attrs`` lleva 'n_transactions', 'total_utility'
        y 'min_utility' (absoluta)
    """
    items, amounts, utilities = utility_tables(quantities, profits)
    total_rows = len(amounts)
    transaction_utility = utilities.sum(axis=1)
    total_utility = float(transaction_utility.sum())
    if absolute:
        if min_utility <= 0:
            raise ValueError("min_utility debe ser > 0")
        threshold = float(min_utility)
    else:
        if not 0 < min_utility <= 1:
            raise ValueError("min_utility debe estar en (0, 1] (una fraccion de la utilidad total)")
        threshold = min_utility * total_utility
    if max_len is not None and max_len < 1:
        raise ValueError("max_len debe ser >= 1")
    present = amounts > 0
    cutoff = threshold - _EPS * max(1.0, abs(threshold))
    found: List[Tuple[float, Tuple[int, ...], float]] = []
    if threshold <= 0:                      # base vacia o sin beneficio: ningun itemset tiene utilidad que valga
        items = []

    level: Dict[Tuple[int, ...], np.ndarray] = {}
    for index in range(len(items)):
        rows = present[:, index]
        if transaction_utility[rows].sum() >= cutoff:
            level[(index,)] = rows
            if utilities[rows, index].sum() >= cutoff:
                found.append((float(utilities[rows, index].sum()), (index,), float(rows.mean())))
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
                    rows = level[prefix + (a,)] & present[:, b]
                    if transaction_utility[rows].sum() < cutoff:
                        continue
                    following[candidate] = rows
                    utility = float(utilities[np.ix_(rows, list(candidate))].sum())
                    if utility >= cutoff:
                        found.append((utility, candidate, float(rows.mean())))
        level = following
        size += 1
    found = [entry for entry in found if len(entry[1]) >= min_len]
    found.sort(key=lambda entry: (-entry[0], len(entry[1]), entry[1]))
    result = pd.DataFrame({"utility": [utility for utility, _, _ in found],
                           "itemsets": [frozenset(items[i] for i in itemset) for _, itemset, _ in found],
                           "support": [support for _, _, support in found],
                           "length": [len(itemset) for _, itemset, _ in found]})
    result.attrs.update({"n_transactions": total_rows, "total_utility": total_utility, "min_utility": threshold})
    return result
