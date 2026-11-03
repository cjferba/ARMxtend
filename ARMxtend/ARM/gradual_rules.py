# -*- coding: utf-8 -*-
"""
Gradual rules: dependencies of the type "the MORE A (and the LESS B), the MORE C" over numeric attributes, without
discretizing or fuzzifying. They are based on:

    Hullermeier, E. (2002). "Association rules for expressing gradual
    dependencies". PKDD 2002, LNCS 2431, 200-211.

A gradual item is a pair (attribute, direction) with direction '+' ("the more") or '-' ("the less"); a gradual
itemset is a set of items of distinct attributes. A pair of rows (s, t) SATISFIES it if, going from s to t, all its
attributes vary in the indicated direction (strictly: ties do not count). The support of a gradual itemset is the
proportion of the n(n-1)/2 unordered pairs of rows that satisfy it in either of the two orientations; the confidence
of A -> B is supp(A u B) / supp(A).

Symmetry: an itemset and its "mirror" (all directions reversed) have the same support -- "the more A, the more B" is
equivalent to "the less A, the less B" --, so only itemsets in CANONICAL form (the direction of the first attribute is
'+') are generated; each returned rule has an equivalent "mirror" rule that is not listed.

Cost: n x n boolean matrices are handled per attribute, that is, O(n^2 m) memory. For large data use `max_rows` to
work on a reproducible random sample of rows.

@author Carlos Fernandez-Basso
"""
import itertools
from typing import Any, Dict, Hashable, Iterable, Optional, Sequence, Tuple

import pandas as pd

INCREASING = "+"
DECREASING = "-"
_FLIP = {INCREASING: DECREASING, DECREASING: INCREASING}


def format_gradual_item(item: Tuple[Hashable, str]) -> str:
    """('A', '+') -> 'A↑' ("cuanto mas A"); ('A', '-') -> 'A↓' ("cuanto menos A")."""
    attribute, direction = item
    return "{}{}".format(attribute, "↑" if direction == INCREASING else "↓")


def format_gradual_itemset(items: Iterable[Tuple[Hashable, str]]) -> str:
    """Text of a set of gradual items, e.g. 'A↑, B↓'."""
    return ", ".join(sorted(format_gradual_item(item) for item in items))


def _canonical(itemset, attribute_order):
    """Canonical form (sorted by attribute, first direction '+') of a gradual itemset."""
    ordered = tuple(sorted(itemset, key=lambda item: attribute_order[item[0]]))
    if ordered and ordered[0][1] == DECREASING:
        ordered = tuple((attribute, _FLIP[direction]) for attribute, direction in ordered)
    return ordered


class _PairSupport(object):
    """Compute the pairwise support of gradual itemsets over a numeric table."""

    def __init__(self, table, attributes):
        self.attributes = list(attributes)
        self.order = {attribute: position for position, attribute in enumerate(self.attributes)}
        n = len(table)
        self.total_pairs = n * (n - 1) / 2.0
        # lower[a][s, t] = True if the value of s in `a` is lower than that of t
        self._lower = {}
        for attribute in self.attributes:
            values = table[attribute].to_numpy(dtype=float)
            self._lower[attribute] = values[:, None] < values[None, :]

    def _matrix(self, item):
        attribute, direction = item
        lower = self._lower[attribute]
        return lower if direction == INCREASING else lower.T

    def support(self, itemset):
        matrices = [self._matrix(item) for item in itemset]
        satisfied = matrices[0]
        for matrix in matrices[1:]:
            satisfied = satisfied & matrix
        # Each unordered pair satisfies at most one orientation of the
        # itemset, so counting ordered pairs counts each pair once.
        return float(satisfied.sum()) / self.total_pairs


def _prepare(data, attributes, max_rows, random_state):
    table = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    attributes = list(attributes) if attributes is not None else list(table.select_dtypes("number").columns)
    if not attributes:
        raise ValueError("There are no numeric attributes to search for gradual rules")
    table = table[attributes].dropna()
    if max_rows is not None and len(table) > max_rows:
        table = table.sample(n=max_rows, random_state=random_state)
    if len(table) < 2:
        raise ValueError("At least 2 rows are needed to form pairs")
    return table.reset_index(drop=True), attributes


def gradual_itemsets(data: Any, min_supp: float, attributes: Optional[Sequence[Hashable]] = None,
                    max_len: Optional[int] = None, max_rows: Optional[int] = None,
                    random_state: int = 0) -> Dict[tuple, float]:
    """
    Frequent gradual itemsets (pairwise support >= `min_supp`), by a level-wise Apriori-style search (pairwise support
    is anti-monotone, like classic support).

    Args:
        data (pandas.DataFrame | array-like 2D): table of numeric values
        min_supp (float): minimum support, in (0, 1]
        attributes (list, optional): attributes to consider (all the numeric columns by default)
        max_len (int, optional): maximum number of items per itemset
        max_rows (int, optional): if there are more rows, a random sample of this size is used (reproducible with
            `random_state`)
        random_state (int): seed of the sampling

    Returns:
        dict {canonical itemset: support}, with the itemset as a tuple of (attribute, '+'|'-') pairs sorted according
        to `attributes`.
    """
    table, attributes = _prepare(data, attributes, max_rows, random_state)
    counter = _PairSupport(table, attributes)

    frequent: Dict[tuple, float] = {}
    level: Dict[tuple, float] = {}
    for attribute in attributes:
        itemset = ((attribute, INCREASING),)
        support = counter.support(itemset)
        if support >= min_supp:
            level[itemset] = support
    frequent.update(level)

    length = 1
    while level and (max_len is None or length < max_len):
        candidates = []
        for itemset in level:
            lastPosition = counter.order[itemset[-1][0]]
            for attribute in attributes[lastPosition + 1:]:
                for direction in (INCREASING, DECREASING):
                    candidates.append(itemset + ((attribute, direction),))

        nextLevel: Dict[tuple, float] = {}
        for candidate in candidates:
            subsets = (candidate[:i] + candidate[i + 1:] for i in range(len(candidate)))
            if not all(_canonical(subset, counter.order) in frequent for subset in subsets):
                continue
            support = counter.support(candidate)
            if support >= min_supp:
                nextLevel[candidate] = support

        frequent.update(nextLevel)
        level = nextLevel
        length += 1

    return frequent


def gradual_rules(data: Any, min_supp: float, min_conf: float, attributes: Optional[Sequence[Hashable]] = None,
                  max_len: Optional[int] = None, max_rows: Optional[int] = None,
                  random_state: int = 0) -> pd.DataFrame:
    """
    Gradual rules  A -> B  (A and B, sets of gradual items of distinct attributes) with support >= `min_supp` and
    confidence >= `min_conf`, e.g. "the more age and the less exercise, the more cholesterol".

    Args: see `gradual_itemsets`, plus
        min_conf (float): minimum confidence supp(A u B) / supp(A)

    Returns:
        pandas.DataFrame with columns 'antecedents' and 'consequents' (frozensets of (attribute, direction) pairs; use
        `format_gradual_itemset` to display them), 'antecedent support', 'consequent support', 'support' and
        'confidence'.
    """
    table, attributes = _prepare(data, attributes, max_rows, random_state)
    frequent = gradual_itemsets(table, min_supp, attributes=attributes, max_len=max_len)
    order = {attribute: position for position, attribute in enumerate(attributes)}

    # By anti-monotonicity, every subset of a frequent itemset is frequent
    # (and, being shorter, it is already in `frequent` even if `max_len` is used).
    def supportOf(itemset):
        return frequent[_canonical(itemset, order)]

    rows = []
    for itemset, support in frequent.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, size):
                consequent = tuple(item for item in itemset if item not in antecedent)
                antecedentSupport = supportOf(antecedent)
                if antecedentSupport == 0:
                    continue
                confidence = support / antecedentSupport
                if confidence >= min_conf:
                    rows.append({
                        "antecedents": frozenset(antecedent),
                        "consequents": frozenset(consequent),
                        "antecedent support": antecedentSupport,
                        "consequent support": supportOf(consequent),
                        "support": support,
                        "confidence": confidence,
                    })

    columns = ["antecedents", "consequents", "antecedent support", "consequent support", "support", "confidence"]
    return pd.DataFrame(rows, columns=columns)
