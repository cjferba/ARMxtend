# -*- coding: utf-8 -*-
"""
Most frequent itemsets (TOP-K) without fixing the minimum support.

Fixing a minimum support is hard: too high and nothing comes out, too low and millions of itemsets do. Here you ask
for the number K of itemsets you want. Since support is anti-monotone (an itemset is never more frequent than its
subsets), a best-first search over the enumeration tree (Eclat style) with a priority queue on support generates the
itemsets in NON-increasing order of support: the first K are exactly the K most frequent, and the search stops
without exploring the rest. Ties are broken deterministically: fewer items first, then lexicographic order.

``iter_itemsets_by_support`` is that lazy generator; ``ARM.topk_rules`` uses it for the K best rules.

@author Carlos Fernandez-Basso
"""
import heapq
from typing import Any, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._vertical import Vertical, count_bits


def iter_itemsets_by_support(vertical: Vertical, min_count: int = 1, max_len: Optional[int] = None) -> Iterator[Tuple[int, Tuple[int, ...]]]:
    """
    Itemsets (tuples of item indices) with at least `min_count` transactions, by non-increasing number of
    transactions (ties: fewer items, then lexicographic order).
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
    The `k` itemsets with the highest support (see the module docstring).

    Args:
        transactions: one transaction per element, each a collection of items
        k (int): how many itemsets
        min_len, max_len (int): minimum (default 1) and maximum number of items of the returned itemsets
        min_supp (float): optional floor of relative support; if there are not k itemsets reaching it, fewer come out

    Returns:
        pandas.DataFrame with 'support' and 'itemsets' (``frozenset``), by decreasing support (the format of
        ``ARM.association_rules``); ``attrs['n_transactions']`` holds the number of transactions. Memory:
        proportional to transactions x items plus the frontier of the search
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not 0 <= min_supp <= 1:
        raise ValueError("min_supp must be in [0, 1]")
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
