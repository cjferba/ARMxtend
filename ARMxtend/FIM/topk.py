# -*- coding: utf-8 -*-
"""
Itemsets mas frecuentes (TOP-K) sin fijar el soporte minimo.

Fijar un soporte minimo es dificil: demasiado alto no sale nada, demasiado bajo salen millones de
itemsets. Aqui se pide el numero K de itemsets que se quiere. Como el soporte es antimonotono (un itemset
nunca es mas frecuente que sus subconjuntos), una busqueda "el mejor primero" sobre el arbol de
enumeracion (estilo Eclat) con una cola de prioridad por soporte genera los itemsets en orden de soporte
NO creciente: los K primeros son exactamente los K mas frecuentes, y la busqueda se detiene sin explorar el
resto. El desempate es determinista: menos items primero y luego orden lexicografico.

``iter_itemsets_by_support`` es ese generador perezoso; ``ARM.topk_rules`` lo usa para las K mejores reglas.

@author Carlos Fernandez-Basso
"""
import heapq
from typing import Any, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._vertical import Vertical, count_bits


def iter_itemsets_by_support(vertical: Vertical, min_count: int = 1, max_len: Optional[int] = None) -> Iterator[Tuple[int, Tuple[int, ...]]]:
    """
    Itemsets (tuplas de indices de item) con al menos `min_count` transacciones, por numero de
    transacciones no creciente (desempate: menos items, luego orden lexicografico).
    """
    heap: List[Tuple[int, int, Tuple[int, ...]]] = []
    for index, count in enumerate(vertical.counts):
        if count >= min_count:
            heapq.heappush(heap, (-count, 1, (index,)))
    while heap:
        negative, size, itemset = heapq.heappop(heap)
        yield -negative, itemset
        if max_len is not None and size >= max_len:
            continue
        mask = vertical.masks[itemset[0]] if size == 1 else vertical.mask_of(itemset)
        for index in range(itemset[-1] + 1, len(vertical.items)):
            count = count_bits(mask & vertical.masks[index])
            if count >= min_count:
                heapq.heappush(heap, (-count, size + 1, itemset + (index,)))


def top_k_itemsets(transactions: Iterable[Iterable[Any]], k: int, min_len: int = 1, max_len: Optional[int] = None,
                   min_supp: float = 0.0) -> pd.DataFrame:
    """
    Los `k` itemsets con mayor soporte (ver el docstring del modulo).

    Argumentos:
        transactions: una transaccion por elemento, cada una una coleccion de items
        k (int): cuantos itemsets
        min_len, max_len (int): numero minimo (por defecto 1) y maximo de items de los itemsets devueltos
        min_supp (float): suelo opcional de soporte relativo; si no hay k itemsets que lo alcancen, salen menos

    Retorna:
        pandas.DataFrame con 'support' e 'itemsets' (``frozenset``), por soporte decreciente
        (el formato de ``ARM.association_rules``); ``attrs['n_transactions']`` guarda el numero de
        transacciones. Memoria: proporcional a transacciones x items mas la frontera de la busqueda
    """
    if k < 1:
        raise ValueError("k debe ser >= 1")
    if not 0 <= min_supp <= 1:
        raise ValueError("min_supp debe estar en [0, 1]")
    vertical = Vertical(transactions)
    found: List[Tuple[float, frozenset]] = []
    if vertical.total:
        minimum = max(1, int(np.ceil(min_supp * vertical.total - 1e-9)))
        for count, itemset in iter_itemsets_by_support(vertical, minimum, max_len):
            if len(itemset) >= min_len:
                found.append((count / vertical.total, frozenset(vertical.items[i] for i in itemset)))
                if len(found) == k:
                    break
    result = pd.DataFrame({"support": [support for support, _ in found], "itemsets": [items for _, items in found]})
    result.attrs["n_transactions"] = vertical.total
    return result
