# -*- coding: utf-8 -*-
"""
HIGH-UTILITY itemsets: not the most frequent ones, but those that are worth the most. A transaction carries
quantities (3 bottles, 1 cheese) and each item has a unit profit; the utility of an itemset in a transaction that
contains it is the sum of ``quantity x profit`` of its items, and its total utility is the sum over the
transactions. An itemset is high-utility (HUI) if its total utility reaches ``min_utility``. A pair of expensive
products can be a HUI without being frequent, and a very frequent one of cheap products may not be (Yao, Hamilton
and Butz, 2004, "A foundational approach to mining itemset utilities from databases").

Utility is NOT anti-monotone (adding an item can raise it), so it cannot be used for pruning. The two-phase
algorithm (Liu, Liao and Choudhary, 2005) is used: the transaction-weighted utility (TWU) of an itemset, the sum of
the TOTAL utilities of the transactions that contain it, IS anti-monotone and bounds from above the utility of all
its supersets; the itemsets with ``TWU >= min_utility`` are searched level by level and, of those, the ones with
enough real utility are kept. It is exact. Profits must be >= 0 (with negative profits the TWU bound no longer
holds).

@author Carlos Fernandez-Basso
"""
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

Quantities = Union[pd.DataFrame, Iterable[Mapping[Any, float]]]
_EPS = 1e-9


def utility_tables(quantities: Quantities, profits: Optional[Mapping[Any, float]]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Items, quantity matrix (transactions x items) and utility matrix (quantity x profit)."""
    table = quantities.copy() if isinstance(quantities, pd.DataFrame) else pd.DataFrame(list(quantities))
    table.columns = [str(name) for name in table.columns]
    if not all(pd.api.types.is_numeric_dtype(table[name]) for name in table.columns):
        raise ValueError("las cantidades deben ser numericas")
    amounts = table.fillna(0.0).to_numpy(dtype=float)
    if (amounts < 0).any():
        raise ValueError("quantities cannot be negative")
    items = list(table.columns)
    if profits is None:
        gain = np.ones(len(items))
    else:
        by_name = {str(item): float(value) for item, value in profits.items()}
        missing = [item for item in items if item not in by_name]
        if missing:
            raise ValueError("the profits of {} are missing".format(missing))
        gain = np.array([by_name[item] for item in items])
        if (gain < 0).any():
            raise ValueError("los beneficios deben ser >= 0")
    return items, amounts, amounts * gain


def high_utility_itemsets(quantities: Quantities, profits: Optional[Mapping[Any, float]] = None, min_utility: float = 0.1,
                          absolute: bool = False, min_len: int = 1, max_len: Optional[int] = None) -> pd.DataFrame:
    """
    High-utility itemsets (see the module docstring).

    Args:
        quantities: one transaction per row: a ``DataFrame`` (columns = items, values = quantities; 0 or NaN if not
            bought) or a list of dictionaries ``{item: quantity}``
        profits (dict, optional): unit profit of each item (>= 0); 1 for all by default
        min_utility (float): minimum utility: a fraction in (0, 1] of the total utility of the database, or an
            absolute value if `absolute` is ``True``
        min_len, max_len (int): minimum (default 1) and maximum number of items of the returned itemsets

    Returns:
        pandas.DataFrame with 'utility', 'itemsets' (``frozenset``), 'support' (fraction of transactions that
        contain it) and 'length', by decreasing utility; ``attrs`` holds 'n_transactions', 'total_utility' and
        'min_utility' (absolute)
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
            raise ValueError("min_utility must be in (0, 1] (a fraction of the total utility)")
        threshold = min_utility * total_utility
    if max_len is not None and max_len < 1:
        raise ValueError("max_len debe ser >= 1")
    present = amounts > 0
    cutoff = threshold - _EPS * max(1.0, abs(threshold))
    found: List[Tuple[float, Tuple[int, ...], float]] = []
    if threshold <= 0:                      # empty database or no profit: no itemset has any utility
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
