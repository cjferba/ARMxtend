# -*- coding: utf-8 -*-
"""
DApriori and DAprioriTID: frequent-itemset mining algorithms for Big Data (Apache Spark), inspired respectively by
the sequential algorithms Apriori and Apriori-TID (Agrawal & Srikant, 1994).

They implement Algorithms 1 and 2 of:
    Fernandez-Basso, C., Ruiz, M.D., Martin-Bautista, M.J. (2024). "New Spark
    solutions for distributed frequent itemset and association rule mining
    algorithms". Cluster Computing, 27, 1217-1234.

@author Carlos Fernandez-Basso
"""
from ._shared import generate_candidates, key_to_itemset


class DApriori(object):
    """Mineria de itemsets frecuentes en Spark inspirada en Apriori (DApriori)."""

    item_sep = ","

    @classmethod
    def _split(cls, line):
        return set(x.strip() for x in line.split(cls.item_sep) if x.strip())

    @classmethod
    def _count_items(cls, transactions):
        return transactions.flatMap(cls._split) \
                            .map(lambda item: (item, 1)) \
                            .reduceByKey(lambda a, b: a + b)

    @classmethod
    def _count_candidates(cls, transactions, broadcast_candidates):
        """
        broadcast_candidates: broadcast variable whose value is a dict
        {itemset_key: frozenset(items)}. It is read with `.value` INSIDE the closure, that is, on the executors: if
        it were read on the driver, the candidates would travel serialized inside every task instead of being sent
        once to each node.
        Returns an RDD[(itemset_key, count)] with the count of occurrences of each candidate itemset in the
        transactions.
        """
        def emit(line):
            items_in_tx = cls._split(line)
            return [(key, 1) for key, iset in broadcast_candidates.value.items()
                    if iset.issubset(items_in_tx)]

        return transactions.flatMap(emit).reduceByKey(lambda a, b: a + b)

    @classmethod
    def run(cls, sc, transactions, min_supp):
        """
        Run the algorithm (Phases 1 and 2 of Algorithm 1/2).

        Args:
            sc (SparkContext): Spark context to create broadcast variables
            transactions (RDD[str]): one transaction per line, items separated by `cls.item_sep`
            min_supp (float): minimum relative support, in (0, 1]

        Returns:
            dict {itemset_key ('A-B-C'): relative support} with all the frequent itemsets of any length
        """
        transactions = cls._preprocess(sc, transactions, min_supp)
        totalTransacs = transactions.count()
        if totalTransacs == 0:
            return {}

        # Phase 1: FreqItems() -- support of itemsets of length 1
        itemCounts = cls._count_items(transactions).collectAsMap()
        freqItemsets = {item: count / totalTransacs
                        for item, count in itemCounts.items()
                        if count / totalTransacs >= min_supp}

        globalFreqItemsets = dict(freqItemsets)
        currentLevelKeys = sorted(freqItemsets.keys())

        # Phase 2: iterative generation of candidates of increasing length
        while len(currentLevelKeys) > 1:
            candidateKeys = generate_candidates(currentLevelKeys)
            if not candidateKeys:
                break

            candidateItemsets = {key: key_to_itemset(key) for key in candidateKeys}
            broadcastCandidates = sc.broadcast(candidateItemsets)

            itemsetCounts = cls._count_candidates(transactions, broadcastCandidates).collectAsMap()
            broadcastCandidates.unpersist()

            freqItemsets = {key: count / totalTransacs
                            for key, count in itemsetCounts.items()
                            if count / totalTransacs >= min_supp}
            if not freqItemsets:
                break

            globalFreqItemsets.update(freqItemsets)
            currentLevelKeys = sorted(freqItemsets.keys())

        return globalFreqItemsets

    @classmethod
    def _preprocess(cls, sc, transactions, min_supp):
        """Extension point for DAprioriTID: DApriori does not transform the data."""
        return transactions


class DAprioriTID(DApriori):
    """
    DApriori-TID variant: after Phase 1, it sorts the items by descending support and removes the infrequent items
    from every transaction, so that the following phases work on already reduced data (line 10 of Algorithm 2 of the
    paper).
    """

    @classmethod
    def _preprocess(cls, sc, transactions, min_supp):
        totalTransacs = transactions.count()
        if totalTransacs == 0:
            return transactions

        itemCounts = cls._count_items(transactions).collectAsMap()
        freqItems = {item: count for item, count in itemCounts.items()
                     if count / totalTransacs >= min_supp}

        # Descending support order: speeds up the filtering in later phases
        orderedItems = sorted(freqItems, key=lambda item: freqItems[item], reverse=True)
        broadcastOrder = sc.broadcast(orderedItems)

        def removeInfrequentAndSort(line):
            itemsInTx = cls._split(line)
            keptOrdered = [item for item in broadcastOrder.value if item in itemsInTx]
            return cls.item_sep.join(keptOrdered)

        return transactions.map(removeInfrequentAndSort)
