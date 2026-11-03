# -*- coding: utf-8 -*-
"""
Item taxonomies and mining of GENERALIZED (multilevel) itemsets.

A taxonomy ("is-a") arranges items in hierarchies: ``Jacket -> Outerwear -> Clothes``. With it a rule can talk about
categories ("whoever buys outerwear buys footwear") as well as products, and so rules appear that no single product
reaches by itself (Srikant and Agrawal, 1995, "Mining generalized association rules"; Han and Fu, 1995, "Discovery of
multiple-level association rules from large databases").

* ``Taxonomy``: a directed acyclic graph child -> parent (an item can have several parents). ``ancestors``,
  ``descendants``, ``level`` (the root is level 1; the level of an item is 1 + the highest level of its parents, so
  an ancestor always has a strictly lower level), ``at_level``...
* ``extend_transactions``: adds to each transaction the ancestors of its items.
* ``generalize_transactions``: replaces each item by its ancestor at a given level (the Han and Fu approach: mine
  level by level).
* ``generalized_itemsets``: frequent itemsets over the extended transactions, never generating an itemset with an
  item and one of its ancestors (its support is that of the itemset without the ancestor and it only adds noise; it
  is the Cumulate optimization). The minimum support can differ at each level (``min_supp={1: 0.3, 2: 0.1}``): an
  itemset must reach the HIGHEST of the minimums of the levels of its items, which keeps the anti-monotone property
  (adding an item does not lower the threshold).

Optionally, ``exclusive`` declares groups of items that cannot appear together in an itemset.

NOTE: item names must not contain '-'.

@author Carlos Fernandez-Basso
"""
from itertools import product
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from ..preprocessing.timeseries import check_item_names

_EPS = 1e-9

Parents = Union[Mapping[str, Union[str, Iterable[str]]], Iterable[Tuple[str, str]], pd.DataFrame]
MinSupp = Union[float, Mapping[int, float]]


class Taxonomy(object):
    """
    Hierarchy of items child -> parent(s), with no cycles.

    Args:
        parents: ``{child: parent}``, ``{child: [parent1, parent2]}``, a list of ``(child, parent)`` pairs or a
            ``DataFrame`` whose first two columns are child and parent. Items that only appear as parents are roots.
    """

    def __init__(self, parents: Parents):
        direct: Dict[str, Set[str]] = {}
        for child, parent in self._pairs(parents):
            child, parent = str(child), str(parent)
            if child == parent:
                raise ValueError("item '{}' cannot be its own parent".format(child))
            direct.setdefault(child, set()).add(parent)
            direct.setdefault(parent, set())
        check_item_names(list(direct))
        self._parents: Dict[str, FrozenSet[str]] = {item: frozenset(found) for item, found in direct.items()}
        self._children: Dict[str, Set[str]] = {item: set() for item in self._parents}
        for child, found in self._parents.items():
            for parent in found:
                self._children[parent].add(child)
        self._ancestors: Dict[str, FrozenSet[str]] = {}
        self._levels: Dict[str, int] = {}
        for item in self._parents:
            self._visit(item, ())

    @staticmethod
    def _pairs(parents: Parents) -> List[Tuple[Any, Any]]:
        if isinstance(parents, pd.DataFrame):
            if parents.shape[1] < 2:
                raise ValueError("the taxonomy DataFrame needs two columns: child and parent")
            return list(zip(parents.iloc[:, 0], parents.iloc[:, 1]))
        if isinstance(parents, Mapping):
            pairs = []
            for child, found in parents.items():
                for parent in ([found] if isinstance(found, str) else found):
                    pairs.append((child, parent))
            return pairs
        return [tuple(pair) for pair in parents]            # type: ignore[misc]

    def _visit(self, item: str, path: Tuple[str, ...]) -> None:
        if item in self._ancestors:
            return
        if item in path:
            raise ValueError("the taxonomy has a cycle: {}".format(" -> ".join(path[path.index(item):] + (item,))))
        found: Set[str] = set()
        level = 1
        for parent in self._parents[item]:
            self._visit(parent, path + (item,))
            found.add(parent)
            found |= self._ancestors[parent]
            level = max(level, self._levels[parent] + 1)
        self._ancestors[item] = frozenset(found)
        self._levels[item] = level

    @classmethod
    def from_paths(cls, paths: Iterable[str], sep: str = "/") -> "Taxonomy":
        """Taxonomy from paths ``'Clothes/Outerwear/Jacket'`` (from the root to the leaf)."""
        pairs: List[Tuple[str, str]] = []
        for path in paths:
            names = [name.strip() for name in str(path).split(sep) if name.strip()]
            pairs.extend(zip(names[1:], names[:-1]))
        return cls(pairs)

    # ------------------------------------------------------------------ consultas

    @property
    def items(self) -> List[str]:
        """All the items of the taxonomy, sorted."""
        return sorted(self._parents)

    def __contains__(self, item: object) -> bool:
        return item in self._parents

    def __len__(self) -> int:
        return len(self._parents)

    def parents(self, item: str) -> FrozenSet[str]:
        """Direct parents (empty if the item is a root or is not in the taxonomy)."""
        return self._parents.get(str(item), frozenset())

    def children(self, item: str) -> FrozenSet[str]:
        """Hijos directos."""
        return frozenset(self._children.get(str(item), ()))

    def ancestors(self, item: str) -> FrozenSet[str]:
        """All the ancestors (parents, grandparents...), without the item itself."""
        return self._ancestors.get(str(item), frozenset())

    def descendants(self, item: str) -> FrozenSet[str]:
        """All the descendants (children, grandchildren...), without the item itself."""
        found: Set[str] = set()
        pending = [str(item)]
        while pending:
            for child in self._children.get(pending.pop(), ()):
                if child not in found:
                    found.add(child)
                    pending.append(child)
        return frozenset(found)

    def is_ancestor(self, ancestor: str, item: str) -> bool:
        """``True`` if `ancestor` is a (strict) ancestor of `item`."""
        return str(ancestor) in self.ancestors(item)

    def level(self, item: str) -> int:
        """Level of the item: 1 for the roots and for the items that are not in the taxonomy."""
        return self._levels.get(str(item), 1)

    @property
    def depth(self) -> int:
        """Highest level of the taxonomy."""
        return max(self._levels.values(), default=1)

    def roots(self) -> List[str]:
        return sorted(item for item, found in self._parents.items() if not found)

    def leaves(self) -> List[str]:
        return sorted(item for item, found in self._children.items() if not found)

    def at_level(self, item: str, level: int) -> FrozenSet[str]:
        """
        The item seen from `level`: the item itself if its level is <= `level`; otherwise, its deepest ancestors of
        level <= `level` (the one at that level in each path; several if the item has several parents).
        """
        item = str(item)
        if self.level(item) <= level:
            return frozenset([item])
        eligible = [a for a in self.ancestors(item) if self._levels[a] <= level]
        return frozenset(a for a in eligible if not any(a in self.ancestors(b) for b in eligible))

    def to_frame(self) -> pd.DataFrame:
        """``child, parent`` pairs (root items appear only as parents)."""
        rows = [(child, parent) for child in sorted(self._parents) for parent in sorted(self._parents[child])]
        return pd.DataFrame(rows, columns=["child", "parent"])

    def __repr__(self) -> str:
        return "Taxonomy({} items, {} levels, roots={})".format(len(self), self.depth, self.roots())


def _as_baskets(transactions: Iterable[Iterable[Any]]) -> List[FrozenSet[str]]:
    return [frozenset(str(item) for item in transaction) for transaction in transactions]


def extend_transactions(transactions: Iterable[Iterable[Any]], taxonomy: Taxonomy) -> List[FrozenSet[str]]:
    """Each transaction with the ancestors of all its items (the basis of the generalized rules)."""
    extended = []
    for basket in _as_baskets(transactions):
        full = set(basket)
        for item in basket:
            full |= taxonomy.ancestors(item)
        extended.append(frozenset(full))
    return extended


def generalize_transactions(transactions: Iterable[Iterable[Any]], taxonomy: Taxonomy, level: int) -> List[FrozenSet[str]]:
    """Each item replaced by its ancestor at level `level` (those of lower or equal level stay)."""
    if level < 1:
        raise ValueError("level debe ser >= 1")
    return [frozenset(general for item in basket for general in taxonomy.at_level(item, level))
            for basket in _as_baskets(transactions)]


def _level_thresholds(min_supp: MinSupp, levels: Sequence[int]) -> Dict[int, float]:
    """Minimum support of each level: that of the highest listed level not exceeding its own (or that of the first listed)."""
    if isinstance(min_supp, Mapping):
        if not min_supp:
            raise ValueError("min_supp cannot be an empty dictionary")
        listed = {int(level): float(value) for level, value in min_supp.items()}
    else:
        listed = {1: float(min_supp)}
    for value in listed.values():
        if not 0 < value <= 1:
            raise ValueError("min_supp debe estar en (0, 1]")
    ordered = sorted(listed)
    return {level: listed[max([k for k in ordered if k <= level] or [ordered[0]])] for level in levels}


def generalized_itemsets(transactions: Iterable[Iterable[Any]], taxonomy: Taxonomy, min_supp: MinSupp,
                         max_len: Optional[int] = None, exclusive: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """
    Generalized frequent itemsets (see the module docstring).

    Args:
        transactions: one transaction per element, each a collection of items (leaves or categories)
        taxonomy (Taxonomy): hierarchy of items; items that are not in it are roots
        min_supp: minimum relative support in (0, 1], or ``{level: support}`` (a lower support at the deep levels is
            the usual choice: the "reduced support" of Han and Fu). A level not listed uses that of the highest
            listed level below it; one before the first listed, that of the first
        max_len (int, optional): maximum number of items of an itemset
        exclusive (dict, optional): ``{item: group}``; two items of the same group do not appear together in an
            itemset (e.g. two ranges of the same attribute, see ``ARM.quantitative_rules``)

    Returns:
        pandas.DataFrame with the columns 'support' and 'itemsets' (``frozenset``), by decreasing support (the format
        of ``ARM.association_rules``). ``result.attrs['n_transactions']`` holds the number of transactions. Memory:
        proportional to transactions x frequent itemsets of the widest level
    """
    baskets = _as_baskets(transactions)
    total = len(baskets)
    if max_len is not None and max_len < 1:
        raise ValueError("max_len debe ser >= 1")
    empty = pd.DataFrame({"support": [], "itemsets": []})
    empty.attrs["n_transactions"] = total
    _level_thresholds(min_supp, [1])                                          # valida min_supp
    if total == 0:
        return empty

    vocabulary = sorted({item for basket in baskets for item in basket}
                        | {ancestor for basket in baskets for item in basket for ancestor in taxonomy.ancestors(item)})
    column = {item: position for position, item in enumerate(vocabulary)}
    thresholds = _level_thresholds(min_supp, sorted({taxonomy.level(item) for item in vocabulary}))
    threshold = np.array([thresholds[taxonomy.level(item)] for item in vocabulary])

    present = np.zeros((total, len(vocabulary)), dtype=bool)
    for row, basket in enumerate(baskets):
        for item in basket:
            present[row, column[item]] = True
            for ancestor in taxonomy.ancestors(item):
                present[row, column[ancestor]] = True

    related = np.zeros((len(vocabulary), len(vocabulary)), dtype=bool)         # item y antecesor suyo
    for item in vocabulary:
        for ancestor in taxonomy.ancestors(item):
            if ancestor in column:
                related[column[item], column[ancestor]] = related[column[ancestor], column[item]] = True
    members: Dict[Any, List[int]] = {}
    for item, group in (exclusive or {}).items():
        if item in column:
            members.setdefault(group, []).append(column[item])
    for columns in members.values():
        related[np.ix_(columns, columns)] = True
        related[columns, columns] = False

    counts = present.sum(axis=0)
    needed = np.ceil(threshold * total - _EPS)
    found: List[Tuple[Tuple[int, ...], float]] = []
    level: Dict[Tuple[int, ...], np.ndarray] = {}
    for position in np.flatnonzero(counts >= needed):
        level[(int(position),)] = present[:, position]
        found.append(((int(position),), float(counts[position]) / total))

    size = 1
    while level and (max_len is None or size < max_len):
        ordered = sorted(level)
        groups: Dict[Tuple[int, ...], List[int]] = {}
        for itemset in ordered:
            groups.setdefault(itemset[:-1], []).append(itemset[-1])
        known = set(level)
        following: Dict[Tuple[int, ...], np.ndarray] = {}
        for prefix, tails in groups.items():
            for first in range(len(tails)):
                for second in range(first + 1, len(tails)):
                    a, b = tails[first], tails[second]
                    if related[a, b]:
                        continue
                    candidate = prefix + (a, b)
                    if any(candidate[:k] + candidate[k + 1:] not in known for k in range(len(prefix))):
                        continue
                    vector = level[prefix + (a,)] & present[:, b]
                    count = int(vector.sum())
                    if count >= np.ceil(max(threshold[list(candidate)]) * total - _EPS):
                        following[candidate] = vector
                        found.append((candidate, count / total))
        level = following
        size += 1

    result = pd.DataFrame({"support": [support for _, support in found],
                           "itemsets": [frozenset(vocabulary[i] for i in itemset) for itemset, _ in found]})
    result = result.sort_values("support", ascending=False, kind="mergesort").reset_index(drop=True)
    result.attrs["n_transactions"] = total
    return result


def ancestor_replacements(items: Sequence[str], taxonomy: Taxonomy, only: Optional[Set[str]] = None) -> List[Tuple[str, ...]]:
    """
    All the tuples that result from replacing each item by itself or by an ancestor, except the identity. With
    `only`, only the ancestors of that set are considered (those that appear in some rule).
    """
    options = [[item] + sorted(a for a in taxonomy.ancestors(item) if only is None or a in only) for item in items]
    return [combination for combination in product(*options) if combination != tuple(items)]
