# -*- coding: utf-8 -*-
"""
FFIM (Fuzzy Frequent Itemset Mining): crisp and fuzzy frequent-itemset mining on a single machine (no Spark), with
the FP-Growth algorithm (Han, Pei & Yin, 2000).

It is the sequential counterpart of the Big Data fuzzy algorithms of ``ARMxtend.FIM.BD_FARE`` /
``ARMxtend.FIM.Eclat.FuzzyDECLAT``: instead of generating and counting candidates level by level (Apriori/Eclat), it
builds a frequent-pattern tree (FP-Tree) and queries it recursively without generating explicit candidates, which is
usually more efficient on a single machine for datasets with many items.

For the fuzzy version, the FP-Growth tree (which merges counts of transactions with the SAME prefix of items) does not
directly admit real membership degrees, since two transactions with the same items but different membership degree
would stop being "the same path" of the tree. This module solves this by binarizing each transaction at each alpha-cut
(see ``ARMxtend.FIM._shared.alpha_cuts``) and running crisp FP-Growth independently at each of the `num_alpha` levels
to discover candidates, then recomputing their exact support at all levels and filtering by the aggregated fuzzy
support FSupp (see `fuzzy_fpgrowth`) -- exactly the same convention (bit-list of support per alpha-cut) used by
``FuzzyDECLAT`` and ``FuzzyDAprioriTID``, which allows using any of the three algorithms interchangeably as input of
``ARMxtend.FIM.FARE``.

It replaces the content that previously lived in this file, which was an identical copy (with the same errors) of
``FIM/BD_FARE/__init__.py``.

@author Carlos Fernandez-Basso
"""
from collections import Counter
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import numpy as np

from ..FIM._shared import itemset_to_key, alpha_cuts, alpha_levels, alpha_weights, weighted_alpha_aggregate


class _FPNode(object):
    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item, parent):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children = {}
        self.link = None


def _build_fptree(weighted, min_count):
    """
    Build an FP-Tree from (items, weight) pairs: each transaction (or conditional path) with the number of times it
    repeats. It keeps only the items whose absolute (weighted) count reaches `min_count`, which is that of the
    ORIGINAL set of transactions, not the size of `weighted`.

    Returns (header, order, item_counts): the header table (item -> first node of the tree with that item, linked to
    each other via `link`), the decreasing-frequency order of the kept items, and their counts.
    """
    rawCounts: Counter = Counter()
    for items, weight in weighted:
        for item in items:
            rawCounts[item] += weight

    itemCounts = {item: count for item, count in rawCounts.items() if count >= min_count}
    if not itemCounts:
        return {}, [], {}

    order = sorted(itemCounts, key=lambda item: (-itemCounts[item], item))
    orderIndex = {item: idx for idx, item in enumerate(order)}

    root = _FPNode(None, None)
    header: Dict[str, Optional[_FPNode]] = {item: None for item in order}
    headerTail: Dict[str, Optional[_FPNode]] = {item: None for item in order}

    for items, weight in weighted:
        node = root
        for item in sorted((item for item in items if item in itemCounts), key=orderIndex.__getitem__):
            child = node.children.get(item)
            if child is None:
                child = _FPNode(item, node)
                node.children[item] = child
                tail = headerTail[item]
                if tail is None:
                    header[item] = child
                else:
                    tail.link = child
                headerTail[item] = child
            child.count += weight
            node = child

    return header, order, itemCounts


def _ascend_prefix_path(node):
    """Items (not counting the node itself) from the given node towards the root."""
    path = []
    node = node.parent
    while node is not None and node.item is not None:
        path.append(node.item)
        node = node.parent
    return path


def _mine_fptree(header, order, itemCounts, min_count, prefix, result):
    # The items are processed from the least to the most frequent: each one is combined with the current
    # prefix to form a new frequent itemset, and its conditional pattern base
    # (paths with their weight, without repeating them) is mined recursively
    for item in reversed(order):
        newItemset = prefix | {item}
        result[newItemset] = itemCounts[item]

        conditionalBase = []
        node = header[item]
        while node is not None:
            prefixPath = _ascend_prefix_path(node)
            if prefixPath:
                conditionalBase.append((prefixPath, node.count))
            node = node.link

        if conditionalBase:
            condHeader, condOrder, condCounts = _build_fptree(conditionalBase, min_count)
            if condOrder:
                _mine_fptree(condHeader, condOrder, condCounts, min_count, newItemset, result)


def fpgrowth(transactions: Sequence[Iterable[str]], min_supp: float) -> Dict[str, float]:
    """
    Crisp FP-Growth: exhaustive mining of frequent itemsets without generating explicit candidates.

    Equal transactions are grouped before building the tree (with weights), and the conditional bases carry each path
    only once with its weight.

    Args:
        transactions (Sequence[Iterable[str]]): one transaction per element, each a collection of items
        min_supp (float): minimum relative support, in (0, 1]

    Returns:
        dict {itemset_key: relative support} with all the frequent itemsets of any length
    """
    totalTransacs = len(transactions)
    if totalTransacs == 0:
        return {}

    minCount = min_supp * totalTransacs - 1e-9
    weighted = Counter(frozenset(transaction) for transaction in transactions)
    header, order, itemCounts = _build_fptree(weighted.items(), minCount)

    result: dict = {}
    if order:
        _mine_fptree(header, order, itemCounts, minCount, frozenset(), result)

    return {itemset_to_key(itemset): count / totalTransacs for itemset, count in result.items()}


def fuzzy_fpgrowth(transactions: Sequence[Iterable[Tuple[str, float]]], min_supp: float,
                   num_alpha: int) -> Dict[str, np.ndarray]:
    """
    Fuzzy FP-Growth: it finds candidates by running `fpgrowth` independently at each of the `num_alpha` alpha-cuts
    (after binarizing the membership degrees of each transaction with `alpha_cuts`), and then recomputes, for each
    candidate, its bit-list of EXACT support at all levels (not only those where it was already frequent in
    isolation), so that the aggregated fuzzy support FSupp (see `FIM._shared.weighted_alpha_aggregate`, Eq. (2) of
    Fernandez-Basso, Ruiz & Martin-Bautista, 2021) is computed correctly.

    This recomputation step is necessary because, if an itemset is frequent at level alpha_i (support >= min_supp),
    its aggregated support FSupp (a weighted mean of its supports at all levels) can be lower than min_supp if at the
    other levels its support is low; and conversely, an itemset can have FSupp >= min_supp without reaching min_supp
    in isolation at ANY level except the one of highest support (since FSupp is a weighted mean, it is bounded by the
    maximum of the supports per level, so that level of maximum support will be detected by `fpgrowth`). That is why
    it is correct to use the union of the frequent itemsets of each level as the set of candidates, but it is
    necessary to recompute their real support at each level (not to assume 0 at the levels where they were not
    detected) before aggregating and filtering by FSupp.

    Args:
        transactions (Sequence[Iterable[Tuple[str, float]]]): each transaction is a list of (item, membership degree
            in [0, 1]) pairs
        min_supp (float): minimum relative support, in (0, 1]
        num_alpha (int): number of alpha-cuts to consider

    Returns:
        dict {itemset_key: numpy.ndarray(num_alpha)} with the bit-list of relative support of each frequent itemset
        (FSupp >= min_supp) in each alpha-cut.
    """
    transactionAlphaVectors = [
        {item: alpha_cuts(degree, num_alpha) for item, degree in transaction}
        for transaction in transactions
    ]
    return fuzzy_fpgrowth_from_vectors(transactionAlphaVectors, min_supp, num_alpha)


def fuzzy_fpgrowth_from_vectors(transaction_vectors: Sequence[Dict[str, np.ndarray]], min_supp: float,
                                num_alpha: int) -> Dict[str, np.ndarray]:
    """
    Same as `fuzzy_fpgrowth`, but starting from transactions already expressed as bit-lists per alpha-cut instead of
    membership degrees. It allows mining items whose bit-list is not derived from a single degree -- e.g. the ABSENT
    items of the RL-theory, whose bit-list is the level-by-level complement of the bit-list of the item (see
    `ARM.negative_rules`).

    Args:
        transaction_vectors (Sequence[dict[str, numpy.ndarray]]): for each transaction, {item: bit-list of
            `num_alpha` 0/1 values}. An item that does not appear in the dict is considered absent at all the levels
            of that transaction
        min_supp (float): minimum relative support FSupp, in (0, 1]
        num_alpha (int): number of alpha-cuts (length of the bit-lists)

    Returns:
        dict {itemset_key: numpy.ndarray(num_alpha)}, like `fuzzy_fpgrowth`.
    """
    transactionAlphaVectors = transaction_vectors
    totalTransacs = len(transactionAlphaVectors)
    if totalTransacs == 0:
        return {}

    weights = alpha_weights(alpha_levels(num_alpha))

    candidateKeys: Set[str] = set()
    for alphaLevel in range(num_alpha):
        binarizedTransactions = [
            {item for item, vector in txVectors.items() if vector[alphaLevel]}
            for txVectors in transactionAlphaVectors
        ]
        candidateKeys.update(fpgrowth(binarizedTransactions, min_supp).keys())

    def exactSupportAllLevels(itemsetKey):
        items = itemsetKey.split("-")
        total = np.zeros(num_alpha, dtype=float)
        for txVectors in transactionAlphaVectors:
            memberVectors = [txVectors[item] for item in items if item in txVectors]
            if len(memberVectors) < len(items):
                continue
            membership = np.ones(num_alpha, dtype=int)
            for vector in memberVectors:
                membership = np.minimum(membership, vector)
            total = total + membership
        return total / totalTransacs

    result = {}
    for itemsetKey in candidateKeys:
        supportVector = exactSupportAllLevels(itemsetKey)
        if weighted_alpha_aggregate(supportVector, weights) >= min_supp:
            result[itemsetKey] = supportVector

    return result
