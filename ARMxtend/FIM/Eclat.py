# -*- coding: utf-8 -*-
"""
DECLAT and FuzzyDECLAT: frequent-itemset mining algorithms (crisp and fuzzy) for Big Data (Apache Spark), inspired by
the sequential algorithm ECLAT (Zaki, 2000), which represents each item by its list of transaction identifiers
(TID-list) and computes the support of an itemset by intersecting the TID-lists of its items.

DECLAT implements Algorithm 3 of:
    Fernandez-Basso, C., Ruiz, M.D., Martin-Bautista, M.J. (2024). "New Spark
    solutions for distributed frequent itemset and association rule mining
    algorithms". Cluster Computing, 27, 1217-1234.

FuzzyDECLAT is the fuzzy (alpha-cut) extension of the same procedure, consistent with the other fuzzy algorithms of
ARMxtend (FIM.BD_FARE).

@author Carlos Fernandez-Basso
"""
import numpy as np

from ._shared import generate_candidates, alpha_levels, alpha_weights, weighted_alpha_aggregate


def _intersect_positions(first, second):
    """
    Positions, in `first` and in `second`, of the common elements of two SORTED arrays without repeats (TID-lists).
    It looks up each element of `first` in `second` with ``searchsorted``: O(|first| log |second|), without
    concatenating or re-sorting; `first` should be the shorter one.
    """
    if len(first) == 0 or len(second) == 0:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty
    where = np.searchsorted(second, first)
    where[where == len(second)] = len(second) - 1
    hit = np.flatnonzero(second[where] == first)
    return hit, where[hit]


def _tid_dtype(total):
    """int32 if the identifiers fit (half the memory of int64), int64 otherwise."""
    return np.int32 if total < 2 ** 31 else np.int64


class DECLAT(object):
    """Crisp frequent-itemset mining on Spark, via TID-lists (DECLAT)."""
    item_sep = ","

    @classmethod
    def _split(cls, line):
        return set(x.strip() for x in line.split(cls.item_sep) if x.strip())

    @classmethod
    def run(cls, sc, transactions, min_supp):
        """
        Args:
            sc (SparkContext)
            transactions (RDD[str]): one transaction per line
            min_supp (float): minimum relative support, in (0, 1]

        Returns:
            dict {itemset_key: relative support} with all the frequent itemsets of any length
        """
        indexedTransacs = transactions.zipWithIndex()
        totalTransacs = indexedTransacs.count()
        if totalTransacs == 0:
            return {}
        dtype = _tid_dtype(totalTransacs)

        # Preprocesado (lineas 4-7 Algoritmo 3): transformar a formato vertical
        # <item, TID-list> grouping by item the transactions where it appears.
        # Each TID-list is compacted ON THE EXECUTORS into a sorted numpy array
        # (4 bytes per occurrence): what travels to the driver and is broadcast is of the
        # order of the data, not of the Python objects that represent them
        def emitItemTid(pair):
            line, tid = pair
            return [(item, tid) for item in cls._split(line)]

        def compact(tids):
            return np.sort(np.fromiter(tids, dtype=dtype))

        itemTidLists = indexedTransacs.flatMap(emitItemTid) \
                                       .groupByKey() \
                                       .mapValues(compact) \
                                       .collectAsMap()

        # Phase 1: FreqItems() -- support of itemsets of length 1
        freqItemsets = {item: len(tids) / totalTransacs
                        for item, tids in itemTidLists.items()
                        if len(tids) / totalTransacs >= min_supp}

        globalFreqItemsets = dict(freqItemsets)
        currentLevelKeys = sorted(freqItemsets.keys())

        # Phase 2: iterative generation of candidates by intersecting TID-lists.
        # The TID-lists are broadcast ONCE for all the levels
        broadcastTidLists = sc.broadcast(itemTidLists)

        def countCandidate(key):
            # it starts with the shortest list: each intersection costs what it does
            tidLists = sorted((broadcastTidLists.value[item] for item in key.split("-")), key=len)
            common = tidLists[0]
            for tidList in tidLists[1:]:
                mine, _ = _intersect_positions(common, tidList)
                common = common[mine]
                if len(common) == 0:
                    break
            return (key, len(common))

        try:
            while len(currentLevelKeys) > 1:
                candidateKeys = generate_candidates(currentLevelKeys)
                if not candidateKeys:
                    break

                itemsetCounts = dict(sc.parallelize(candidateKeys).map(countCandidate).collect())

                freqItemsets = {key: count / totalTransacs
                                for key, count in itemsetCounts.items()
                                if count / totalTransacs >= min_supp}
                if not freqItemsets:
                    break

                globalFreqItemsets.update(freqItemsets)
                currentLevelKeys = sorted(freqItemsets.keys())
        finally:
            broadcastTidLists.unpersist()

        return globalFreqItemsets


class FuzzyDECLAT(object):
    """
    Fuzzy version of DECLAT: each transaction is a list of (item, membership degree) pairs, and the support of an
    itemset is computed, for each alpha level, as the mean of the minimum (t-norm) of the membership degrees of its
    items in the transactions where all of them appear.
    """

    @classmethod
    def run(cls, sc, transactions, min_supp, num_alpha):
        """
        Args:
            sc (SparkContext)
            transactions (RDD[Iterable[Tuple[str, float]]]): each transaction is a list of (item, membership degree
                in [0, 1]) pairs
            min_supp (float): minimum relative support, in (0, 1]
            num_alpha (int): number of alpha-cuts to consider

        Returns:
            dict {itemset_key: numpy.ndarray(num_alpha)} with the bit-list of relative support of each frequent
            itemset in each alpha-cut (see `FIM._shared.alpha_cuts`). An itemset is considered frequent if its
            aggregated fuzzy support FSupp (see `FIM._shared.weighted_alpha_aggregate`) reaches `min_supp`; that
            vector is kept whole (not only the aggregate) because `FIM.FARE` needs it to compute the fuzzy
            confidence level by level (Eqs. (2)-(4) of Fernandez-Basso, Ruiz & Martin-Bautista, 2021).
        """
        levels = alpha_levels(num_alpha)
        weights = alpha_weights(levels)

        indexedTransacs = transactions.zipWithIndex()
        totalTransacs = indexedTransacs.count()
        if totalTransacs == 0:
            return {}
        dtype = _tid_dtype(totalTransacs)

        # Vertical format <item, (TIDs, degrees)>: two numpy arrays per item, built ON THE EXECUTORS
        # (sorted 4-byte TIDs and 8-byte degrees). The alpha-cut of the minimum of several degrees is the minimum of
        # their alpha-cuts (``value >= alpha`` is monotone), so it is enough to store the degree and count at each
        # level at the end: the result is the same as carrying a vector of alpha-cuts per TID, with a
        # fraction of the memory and no Python loops per TID
        def emitItemTid(pair):
            memberships, tid = pair
            for _, degree in memberships:
                if not 0.0 <= degree <= 1.0:
                    raise ValueError("The fuzzy membership degree must be in [0, 1], got: {}".format(degree))
            return [(item, (tid, float(degree))) for item, degree in memberships]

        def compact(pairs):
            pairs = list(pairs)
            tids = np.fromiter((tid for tid, _ in pairs), dtype=dtype, count=len(pairs))
            degrees = np.fromiter((degree for _, degree in pairs), dtype=float, count=len(pairs))
            order = np.argsort(tids, kind="stable")
            tids, degrees = tids[order], degrees[order]
            if len(tids) > 1 and (tids[1:] == tids[:-1]).any():           # a repeated item in the transaction: the highest degree
                unique, inverse = np.unique(tids, return_inverse=True)
                best = np.zeros(len(unique))
                np.maximum.at(best, inverse, degrees)
                tids, degrees = unique, best
            return tids, degrees

        def supportVector(degrees):
            return (degrees[:, None] >= levels).sum(axis=0) / totalTransacs

        itemTidVectors = indexedTransacs.flatMap(emitItemTid) \
                                         .groupByKey() \
                                         .mapValues(compact) \
                                         .collectAsMap()

        freqItemsets = {}
        for item, (_, degrees) in itemTidVectors.items():
            support = supportVector(degrees)
            if weighted_alpha_aggregate(support, weights) >= min_supp:
                freqItemsets[item] = support

        globalFreqItemsets = dict(freqItemsets)
        currentLevelKeys = sorted(freqItemsets.keys())

        broadcastTidVectors = sc.broadcast(itemTidVectors)

        def countCandidate(key):
            vectors = sorted((broadcastTidVectors.value[item] for item in key.split("-")), key=lambda v: len(v[0]))
            tids, degrees = vectors[0]
            for otherTids, otherDegrees in vectors[1:]:
                mine, theirs = _intersect_positions(tids, otherTids)
                tids, degrees = tids[mine], np.minimum(degrees[mine], otherDegrees[theirs])
                if len(tids) == 0:
                    break
            return (key, supportVector(degrees))

        try:
            while len(currentLevelKeys) > 1:
                candidateKeys = generate_candidates(currentLevelKeys)
                if not candidateKeys:
                    break

                supports = dict(sc.parallelize(candidateKeys).map(countCandidate).collect())

                freqItemsets = {key: support for key, support in supports.items()
                                if weighted_alpha_aggregate(support, weights) >= min_supp}
                if not freqItemsets:
                    break

                globalFreqItemsets.update(freqItemsets)
                currentLevelKeys = sorted(freqItemsets.keys())
        finally:
            broadcastTidVectors.unpersist()

        return globalFreqItemsets
