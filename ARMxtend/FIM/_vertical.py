# -*- coding: utf-8 -*-
"""
Representacion VERTICAL de una base de transacciones: para cada item, el vector de bits (empaquetado)
de las transacciones que lo contienen. El soporte de un itemset es el numero de bits de la interseccion
(AND) de los vectores de sus items. Lo comparten los mineros top-k y de itemsets raros.

@author Carlos Fernandez-Basso
"""
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

_POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def count_bits(mask: np.ndarray) -> int:
    """Numero de bits a 1 de un vector empaquetado."""
    return int(_POPCOUNT[mask].sum(dtype=np.int64))


class Vertical(object):
    """
    Items ordenados y su vector de bits.

    Atributos: ``items`` (nombres, ordenados), ``masks`` (un vector empaquetado por item), ``total``
    (numero de transacciones) y ``counts`` (transacciones de cada item).
    """

    def __init__(self, transactions: Iterable[Iterable[Any]]):
        baskets = [frozenset(str(item) for item in transaction) for transaction in transactions]
        self.total = len(baskets)
        self.items: List[str] = sorted({item for basket in baskets for item in basket})
        position = {item: k for k, item in enumerate(self.items)}
        matrix = np.zeros((len(self.items), self.total), dtype=bool)
        for row, basket in enumerate(baskets):
            for item in basket:
                matrix[position[item], row] = True
        self.masks: List[np.ndarray] = [np.packbits(line) for line in matrix]
        self.counts: List[int] = [int(line.sum()) for line in matrix]
        self._cache: Dict[Tuple[int, ...], int] = {}

    def mask_of(self, indices: Sequence[int]) -> np.ndarray:
        result = self.masks[indices[0]]
        for index in indices[1:]:
            result = result & self.masks[index]
        return result

    def count_of(self, indices: Sequence[int]) -> int:
        """Transacciones que contienen todos los items de `indices` (con cache)."""
        key = tuple(sorted(indices))
        if key not in self._cache:
            self._cache[key] = self.counts[key[0]] if len(key) == 1 else count_bits(self.mask_of(key))
        return self._cache[key]
