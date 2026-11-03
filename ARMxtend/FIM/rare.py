# -*- coding: utf-8 -*-
"""
RARE itemsets: infrequent but meaningful (the adverse effect of two drugs together, the failure that follows a
specific sequence of alarms). With a low minimum support to reach them, ordinary mining overflows: almost every
itemset is "frequent" and millions come out.

An itemset is RARE if ``min_supp <= support < max_supp`` (``min_supp`` is a floor against noise: what happens once
or twice is not a pattern), and it is MINIMAL RARE (MRI) if, besides, ALL its proper subsets are frequent
(``support >= max_supp``). The MRIs are the compact basis of the rare itemsets: every rare itemset is an MRI or
contains one (Szathmary, Napoli and Valtchev, 2007, "Towards rare itemset mining"). They are found with level-wise
Apriori: frequent itemsets of size k-1 are joined; a frequent candidate keeps growing, and a rare one is an MRI and
stops (its supersets would no longer be minimal).

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
    Minimal rare itemsets (see the module docstring).

    Args:
        transactions: one transaction per element, each a collection of items
        max_supp (float): support from which an itemset is frequent, in (0, 1]; the rare ones are below it
        min_supp (float): minimum relative support of a rare itemset, in [0, 1)
        min_count (int): minimum number of transactions of a rare itemset (default 1)
        min_len, max_len (int): minimum (default 1) and maximum number of items of the returned itemsets

    Returns:
        pandas.DataFrame with 'support' and 'itemsets' (``frozenset``), by decreasing support; ``attrs`` holds
        'n_transactions' and 'n_frequent' (frequent itemsets visited, in case you want to compare with ordinary
        mining)
    """
    if not 0 < max_supp <= 1 or not 0 <= min_supp < max_supp:
        raise ValueError("0 <= min_supp < max_supp <= 1 is required")
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
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
