# -*- coding: utf-8 -*-
"""
DENSE frequent-itemset miner (single machine): level-wise Apriori over packed bit vectors, with the candidates of
each level counted at once with numpy operations (AND and bit count) instead of a Python loop per candidate.

It is the best choice with few items (tens or hundreds) and many transactions; with thousands of items and very
sparse data, ``FFIM.fpgrowth`` is usually better. The results are the same as those of ``fpgrowth`` and of any other
exact miner (the tests compare them with the definition).

@author Carlos Fernandez-Basso
"""
from itertools import combinations
from typing import Any, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ._vertical import _POPCOUNT, Vertical

_CHUNK_BYTES = 64 * 1024 * 1024         # approximate maximum memory of the masks of a block of candidates


def _join(level: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pairs (i, j) of itemsets of the level (rows sorted lexicographically) that share all items but the last one, and
    the resulting candidate. Returns the indices i, j and the rows of candidates.
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


def _prune(level: np.ndarray, first: np.ndarray, second: np.ndarray, candidates: np.ndarray,
           size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apriori pruning: drop the candidates that have an infrequent subset of the previous level."""
    if size < 2 or not len(candidates):
        return first, second, candidates
    known: Set[Tuple[int, ...]] = {tuple(row) for row in level.tolist()}
    ok = np.fromiter((all(subset in known for subset in combinations(row, size))
                      for row in candidates.tolist()), dtype=bool, count=len(candidates))
    return first[ok], second[ok], candidates[ok]


def dense_itemsets(transactions: Iterable[Iterable[Any]], min_supp: float, max_len: Optional[int] = None) -> pd.DataFrame:
    """
    Frequent itemsets (relative support >= `min_supp`) by Apriori over bit vectors.

    Args:
        transactions: one transaction per element, each a collection of items (or a boolean table)
        min_supp (float): minimum relative support, in (0, 1]
        max_len (int, optional): maximum length of the itemsets

    Returns:
        pandas.DataFrame with 'support' and 'itemsets' (frozenset), by length and then by decreasing support;
        ``attrs['n_transactions']`` holds the number of transactions.
    """
    if not 0 < min_supp <= 1:
        raise ValueError("min_supp must be in (0, 1]")
    if max_len is not None and max_len < 1:
        raise ValueError("max_len must be >= 1")
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
        first, second, candidates = _prune(level, first, second, candidates, size)
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
