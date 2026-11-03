# -*- coding: utf-8 -*-
"""
VERTICAL representation of a transaction database: for each item, the (packed) bit vector of the transactions that
contain it. The support of an itemset is the number of bits of the intersection (AND) of the vectors of its items.
Shared by the top-k and rare-itemset miners.

@author Carlos Fernandez-Basso
"""
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

_POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def count_bits(mask: np.ndarray) -> int:
    """Number of bits set in a packed vector."""
    return int(_POPCOUNT[mask].sum(dtype=np.int64))


class Vertical(object):
    """
    Sorted items and their bit vector.

    Attributes: ``items`` (names, sorted), ``masks`` (one packed vector per item), ``total``
    (number of transactions) and ``counts`` (transactions of each item).
    """

    def __init__(self, transactions: Iterable[Iterable[Any]]):
        baskets = [frozenset(str(item) for item in transaction) for transaction in transactions]
        self.total = len(baskets)
        self.items: List[str] = sorted({item for basket in baskets for item in basket})
        position = {item: k for k, item in enumerate(self.items)}
        rows = np.fromiter((r for r, basket in enumerate(baskets) for _ in basket), dtype=np.intp)
        cols = np.fromiter((position[item] for basket in baskets for item in basket), dtype=np.intp)
        matrix = np.zeros((len(self.items), self.total), dtype=bool)
        matrix[cols, rows] = True
        packed = np.packbits(matrix, axis=1)
        self.masks: List[np.ndarray] = list(packed)
        self.counts: List[int] = matrix.sum(axis=1).tolist()
        self._cache: Dict[Tuple[int, ...], int] = {}

    @classmethod
    def from_table(cls, table: Any) -> "Vertical":
        """From a boolean (or 0/1) table of transactions by items, without going through Python lists."""
        values = np.asarray(table.to_numpy()).astype(bool)
        vertical = cls.__new__(cls)
        vertical.total = len(values)
        vertical.items = [str(name) for name in table.columns]
        order = sorted(range(len(vertical.items)), key=vertical.items.__getitem__)
        vertical.items = [vertical.items[k] for k in order]
        columns = values[:, order].T
        vertical.masks = list(np.packbits(columns, axis=1))
        vertical.counts = columns.sum(axis=1).tolist()
        vertical._cache = {}
        return vertical

    def mask_of(self, indices: Sequence[int]) -> np.ndarray:
        result = self.masks[indices[0]]
        for index in indices[1:]:
            result = result & self.masks[index]
        return result

    def count_of(self, indices: Sequence[int]) -> int:
        """Transactions that contain all the items of `indices` (cached)."""
        key = tuple(sorted(indices))
        if key not in self._cache:
            self._cache[key] = self.counts[key[0]] if len(key) == 1 else count_bits(self.mask_of(key))
        return self._cache[key]
