# -*- coding: utf-8 -*-
"""
Minero de itemsets frecuentes DENSO (una sola maquina): Apriori por niveles sobre vectores de bits empaquetados, con
los candidatos de cada nivel contados de una vez con operaciones de numpy (AND y recuento de bits) en lugar de un
bucle de Python por candidato.

Es la mejor opcion cuando hay pocos items (decenas o cientos) y muchas transacciones; con miles de items y datos
muy dispersos, ``FFIM.fpgrowth`` suele ser mejor. Los resultados son los mismos que los de ``fpgrowth`` y de
cualquier otro minero exacto (los tests los comparan con la definicion).

@author Carlos Fernandez-Basso
"""
from itertools import combinations
from typing import Any, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ._vertical import _POPCOUNT, Vertical

_CHUNK_BYTES = 64 * 1024 * 1024         # memoria maxima, aproximada, de las mascaras de un bloque de candidatos


def _join(level: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pares (i, j) de itemsets del nivel (filas ordenadas lexicograficamente) que comparten todos los items salvo el
    ultimo, y el candidato resultante. Devuelve los indices i, j y las filas de candidatos.
    """
    size = level.shape[1]
    firsts: List[np.ndarray] = []
    seconds: List[np.ndarray] = []
    start = 0
    count = len(level)
    prefixes = level[:, :-1]
    while start < count:
        end = start + 1
        while end < count and (size == 1 or (prefixes[end] == prefixes[start]).all()):
            end += 1
        if end - start > 1:
            left, right = np.triu_indices(end - start, k=1)
            firsts.append(left + start)
            seconds.append(right + start)
        start = end
    if not firsts:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty, np.zeros((0, size + 1), dtype=level.dtype)
    first, second = np.concatenate(firsts), np.concatenate(seconds)
    return first, second, np.concatenate([level[first], level[second][:, -1:]], axis=1)


def dense_itemsets(transactions: Iterable[Iterable[Any]], min_supp: float, max_len: Optional[int] = None) -> pd.DataFrame:
    """
    Itemsets frecuentes (soporte relativo >= `min_supp`) por Apriori sobre vectores de bits.

    Argumentos:
        transactions: una transaccion por elemento, cada una una coleccion de items (o una tabla booleana)
        min_supp (float): soporte minimo relativo, en (0, 1]
        max_len (int, opcional): longitud maxima de los itemsets

    Retorna:
        pandas.DataFrame con 'support' e 'itemsets' (frozenset), por longitud y luego por soporte decreciente;
        ``attrs['n_transactions']`` guarda el numero de transacciones.
    """
    if not 0 < min_supp <= 1:
        raise ValueError("min_supp debe estar en (0, 1]")
    if max_len is not None and max_len < 1:
        raise ValueError("max_len debe ser >= 1")
    if isinstance(transactions, pd.DataFrame):
        vertical = Vertical.from_table(transactions)
    else:
        vertical = Vertical(transactions)
    total = vertical.total
    result = pd.DataFrame({"support": [], "itemsets": []})
    result.attrs["n_transactions"] = total
    if total == 0 or not vertical.items:
        return result
    minimum = min_supp * total - 1e-9

    keep = [k for k, count in enumerate(vertical.counts) if count >= minimum]
    rows: List[Tuple[float, frozenset]] = [(vertical.counts[k] / total, frozenset([vertical.items[k]])) for k in keep]
    masks = np.array([vertical.masks[k] for k in keep]) if keep else np.zeros((0, 0), dtype=np.uint8)
    level = np.array(keep, dtype=np.intp).reshape(-1, 1)
    size = 1
    while len(level) > 1 and (max_len is None or size < max_len):
        first, second, candidates = _join(level)
        if size >= 2 and len(candidates):                               # poda: todos los subconjuntos deben ser frecuentes
            known: Set[Tuple[int, ...]] = {tuple(row) for row in level.tolist()}
            ok = np.fromiter((all(subset in known for subset in combinations(row, size))
                              for row in candidates.tolist()), dtype=bool, count=len(candidates))
            first, second, candidates = first[ok], second[ok], candidates[ok]
        if not len(candidates):
            break
        width = masks.shape[1]
        step = max(1, _CHUNK_BYTES // max(1, width))
        counts = np.empty(len(candidates), dtype=np.int64)
        new_masks = np.empty((len(candidates), width), dtype=np.uint8)
        for begin in range(0, len(candidates), step):
            block = slice(begin, begin + step)
            new_masks[block] = masks[first[block]] & masks[second[block]]
            counts[block] = _POPCOUNT[new_masks[block]].sum(axis=1, dtype=np.int64)
        frequent = np.flatnonzero(counts >= minimum)
        if not len(frequent):
            break
        level, masks = candidates[frequent], new_masks[frequent]
        size += 1
        rows.extend((counts[k] / total, frozenset(vertical.items[i] for i in candidates[k])) for k in frequent)
    result = pd.DataFrame({"support": [support for support, _ in rows], "itemsets": [items for _, items in rows]})
    result.attrs["n_transactions"] = total
    return result
