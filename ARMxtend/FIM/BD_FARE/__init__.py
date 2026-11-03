# -*- coding: utf-8 -*-
"""
FuzzyDAprioriTID: fuzzy frequent-itemset mining on Big Data (Apache Spark), inspired by Apriori-TID, using an
alpha-cut representation of the membership degrees (see ``ARMxtend.FIM._shared``).

It replaces the ``AprioriTID`` class that previously lived in this file: it had several errors that prevented its
use (an `__init__`/`run_model` that referenced a `sc`/`model` that exist only in an interactive notebook, a call to
an unqualified `AlphaCortes` function that raised `NameError`, and duplicated/unused imports). The alpha-cut and
candidate-generation logic is now shared with the rest of the `FIM` algorithms through `FIM._shared`.

@author Carlos Fernandez-Basso
"""
import numpy as np

from .._shared import generate_candidates, alpha_cuts, alpha_levels, alpha_weights, weighted_alpha_aggregate


class FuzzyDAprioriTID(object):
    """
    Fuzzy frequent-itemset mining on Spark. Each transaction is a list of (item, membership degree in [0, 1]) pairs;
    the support of an itemset is computed, for each alpha-cut, as the mean of the minimum (t-norm) of the membership
    degrees of its items in each transaction.
    """

    @classmethod
    def run(cls, sc, transactions, min_supp, num_alpha):
        """
        Args:
            sc (SparkContext)
            transactions (RDD[Iterable[Tuple[str, float]]]): each transaction is a list of (item, membership degree)
                pairs
            min_supp (float): minimum relative support, in (0, 1]
            num_alpha (int): number of alpha-cuts to consider

        Returns:
            dict {itemset_key: numpy.ndarray(num_alpha)} with the bit-list of relative support of each frequent
            itemset in each alpha-cut. An itemset is considered frequent if its aggregated fuzzy support FSupp (see
            `FIM._shared.weighted_alpha_aggregate`, Eq. (2) of Fernandez-Basso, Ruiz & Martin-Bautista, 2021)
            reaches `min_supp`.
        """
        weights = alpha_weights(alpha_levels(num_alpha))

        alphaTransactions = transactions.map(
            lambda memberships: {item: alpha_cuts(degree, num_alpha) for item, degree in memberships})

        totalTransacs = alphaTransactions.count()
        if totalTransacs == 0:
            return {}

        # Phase 1: support (alpha-cut vector) of itemsets of length 1
        itemVectors = alphaTransactions.flatMap(lambda tx: list(tx.items())) \
                                        .reduceByKey(lambda a, b: a + b) \
                                        .collectAsMap()

        freqItemsets = {item: vector / totalTransacs
                        for item, vector in itemVectors.items()
                        if weighted_alpha_aggregate(vector / totalTransacs, weights) >= min_supp}

        # Filter the infrequent items out of each transaction (optimization
        # Apriori-TID: the following phases only query reduced data)
        broadcastFreqItems = sc.broadcast(set(freqItemsets.keys()))
        filteredTransactions = alphaTransactions.map(
            lambda tx: {item: vector for item, vector in tx.items() if item in broadcastFreqItems.value})

        globalFreqItemsets = dict(freqItemsets)
        currentLevelKeys = sorted(freqItemsets.keys())

        # Phase 2: iterative generation of candidates of increasing length
        while len(currentLevelKeys) > 1:
            candidateKeys = generate_candidates(currentLevelKeys)
            if not candidateKeys:
                break

            candidateItems = {key: key.split("-") for key in candidateKeys}
            broadcastCandidates = sc.broadcast(candidateItems)

            def countCandidates(tx):
                counted = []
                for key, items in broadcastCandidates.value.items():
                    membership = np.ones(num_alpha, dtype=float)
                    isPresent = True
                    for item in items:
                        if item not in tx:
                            isPresent = False
                            break
                        membership = np.minimum(membership, tx[item])
                    if isPresent:
                        counted.append((key, membership))
                return counted

            itemsetVectors = filteredTransactions.flatMap(countCandidates) \
                                                  .reduceByKey(lambda a, b: a + b) \
                                                  .collectAsMap()
            broadcastCandidates.unpersist()

            freqItemsets = {key: vector / totalTransacs
                            for key, vector in itemsetVectors.items()
                            if weighted_alpha_aggregate(vector / totalTransacs, weights) >= min_supp}
            if not freqItemsets:
                break

            globalFreqItemsets.update(freqItemsets)
            currentLevelKeys = sorted(freqItemsets.keys())

        broadcastFreqItems.unpersist()
        return globalFreqItemsets
