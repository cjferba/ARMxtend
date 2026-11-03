# -*- coding: utf-8 -*-
"""
SEQUENTIAL rules: ``X => Y`` means "if in a sequence all the items of X occur, then all those of Y occur afterwards".
Unlike an association rule, order matters (X before Y); unlike a sequential pattern, inside X and inside Y order does
not matter, and how RELIABLE the prediction is, is measured with support and confidence.

The definition of sequential rules of the following is followed:

    Fournier-Viger, P., Wu, C.-W., Tseng, V.S., Cao, L., Nkambou, R. (2015).
    "Mining partially-ordered sequential rules common to multiple sequences".
    IEEE Transactions on Knowledge and Data Engineering, 27(8), 2203-2216.

(and its RuleGrowth algorithm, from the same group, 2011). A sequence is a list of time-ordered elements (sets of
simultaneous items; see ``FIM.sequential``). The rule ``X => Y``, with X and Y disjoint, is VALID in a sequence if
there is an instant ``k`` such that all the items of X have appeared in the elements up to ``k`` and all those of Y
appear in elements after ``k``. Measures:

* support(X => Y)    = fraction of sequences in which the rule is valid;
* confidence(X => Y) = sequences in which it is valid / sequences that contain all the items of X (in any order);
* lift = confidence / fraction of sequences that contain all the items of Y.

The search relies on the fact that, for the rule to be valid, X, Y and X u Y must be in the sequences, so it starts
from the frequent itemsets of the sequences seen as baskets (FP-Growth) and tries their partitions, as in the
generation of association rules. It is not the RuleGrowth algorithm (which searches incrementally without enumerating
all the itemsets), but it uses the same definition of valid rule, support and confidence.

NOTE: item names must not contain '-', which is the item separator in the itemset keys of ``FIM._shared``.

@author Carlos Fernandez-Basso
"""
from itertools import combinations
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import pandas as pd

from ..FFIM import fpgrowth
from ..FIM._shared import key_to_itemset
from ..FIM.sequential import _normalize
from ..preprocessing.timeseries import check_item_names
from ._measures import MEASURES

_ABSENT_FIRST = 10 ** 9


def sequential_rules(sequences: Iterable[Iterable[Union[str, Iterable[str]]]], min_supp: float,
                     min_conf: float = 0.8, max_antecedent: int = 2, max_consequent: int = 2,
                     metric: str = "confidence", min_threshold: float = 0.0) -> pd.DataFrame:
    """
    Mine sequential rules ``X => Y`` ("if X, then Y").

    Args:
        sequences: one sequence per element; each is a list of elements (sets of simultaneous items, or a single
            item). See ``FIM.sequential``
        min_supp (float): minimum relative support of the rule, in (0, 1]
        min_conf (float): minimum confidence, in [0, 1]
        max_antecedent (int): maximum number of items of X
        max_consequent (int): maximum number of items of Y
        metric (str): additional measure to filter by, besides `min_conf`: any of ``ARM._measures.MEASURES``
            ('lift', 'leverage', ...)
        min_threshold (float): minimum threshold of `metric` (0 = no filtering)

    Returns:
        pandas.DataFrame with the columns 'antecedents' and 'consequents' (``frozenset``), 'antecedent support' and
        'consequent support' (fraction of sequences that contain all their items, in any order), 'support',
        'confidence' and 'lift', and the other measures of ``ARM._measures`` computed with those three supports,
        sorted by decreasing confidence and support.
    """
    database = _normalize(sequences)
    total = len(database)
    columns = ["antecedents", "consequents", "antecedent support", "consequent support", "support",
               "confidence", "lift"] + [name for name in MEASURES if name not in ("confidence", "lift")]
    if total == 0:
        return pd.DataFrame(columns=columns)
    if metric not in MEASURES and metric != "support":
        raise ValueError("metric must be 'support' or one of {}, received: '{}'".format(sorted(MEASURES), metric))
    if max_antecedent < 1 or max_consequent < 1:
        raise ValueError("max_antecedent and max_consequent must be >= 1")
    vocabulary = sorted({item for sequence in database for element in sequence for item in element})
    check_item_names(vocabulary)

    # Supports of itemsets in the sequences seen as baskets (order does not count)
    baskets = [set().union(*sequence) if sequence else set() for sequence in database]
    frequent = fpgrowth(baskets, min_supp)

    # First and last appearance of each item in each sequence
    column = {item: index for index, item in enumerate(vocabulary)}
    first = np.full((total, len(vocabulary)), _ABSENT_FIRST, dtype=np.int64)
    last = np.full((total, len(vocabulary)), -1, dtype=np.int64)
    for row, sequence in enumerate(database):
        for position, element in enumerate(sequence):
            for item in element:
                cell = column[item]
                if first[row, cell] == _ABSENT_FIRST:
                    first[row, cell] = position
                last[row, cell] = position

    completedAt: Dict[frozenset, np.ndarray] = {}
    expiresAt: Dict[frozenset, np.ndarray] = {}

    def completed(items: frozenset) -> np.ndarray:
        """Instant at which X has already appeared in full: the maximum of the first appearances of its items."""
        if items not in completedAt:
            completedAt[items] = first[:, [column[item] for item in items]].max(axis=1)
        return completedAt[items]

    def expiring(items: frozenset) -> np.ndarray:
        """Last instant at which Y can still appear in full: the minimum of the last appearances of its items."""
        if items not in expiresAt:
            expiresAt[items] = last[:, [column[item] for item in items]].min(axis=1)
        return expiresAt[items]

    rows: List[dict] = []
    for key, joint in frequent.items():
        itemset = key_to_itemset(key)
        if len(itemset) < 2 or len(itemset) > max_antecedent + max_consequent:
            continue
        for size in range(1, len(itemset)):
            if size > max_antecedent or len(itemset) - size > max_consequent:
                continue
            for antecedent in combinations(sorted(itemset), size):
                x = frozenset(antecedent)
                y = itemset - x
                valid = int(np.count_nonzero(completed(x) < expiring(y)))
                support = valid / total
                if support < min_supp - 1e-9:
                    continue
                supportX = frequent[_key(x)]
                supportY = frequent[_key(y)]
                confidence = support / supportX
                if confidence < min_conf - 1e-9:
                    continue
                entry: Dict[str, Any] = {"antecedents": x, "consequents": y, "antecedent support": supportX,
                                         "consequent support": supportY, "support": support}
                for name, function in MEASURES.items():
                    entry[name] = float(function(support, supportX, supportY))
                if metric != "support" and min_threshold and entry[metric] < min_threshold:
                    continue
                rows.append(entry)

    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values(["confidence", "support"], ascending=False, kind="mergesort").reset_index(drop=True)


def _key(items: frozenset) -> str:
    return "-".join(sorted(items))


def contains_rule(sequence: Iterable[Iterable[str]], antecedent: Iterable[str], consequent: Iterable[str]) -> bool:
    """True if the rule ``antecedent => consequent`` is valid in `sequence` (see the module docstring)."""
    elements = [frozenset(e) for e in sequence]
    firstSeen: Dict[str, int] = {}
    lastSeen: Dict[str, int] = {}
    for position, element in enumerate(elements):
        for item in element:
            firstSeen.setdefault(item, position)
            lastSeen[item] = position
    x, y = list(antecedent), list(consequent)
    if any(item not in firstSeen for item in x + y):
        return False
    return max(firstSeen[item] for item in x) < min(lastSeen[item] for item in y)
