# -*- coding: utf-8 -*-
"""
DECLAT y FuzzyDECLAT: algoritmos de mineria de itemsets frecuentes (crisp y
difusos) sobre Big Data (Apache Spark), inspirados en el algoritmo secuencial
ECLAT (Zaki, 2000), que representa cada item mediante su lista de
identificadores de transaccion (TID-list) y calcula el soporte de un
itemset por interseccion de las TID-lists de sus items.

DECLAT implementa el Algoritmo 3 de:
    Fernandez-Basso, C., Ruiz, M.D., Martin-Bautista, M.J. (2024). "New Spark
    solutions for distributed frequent itemset and association rule mining
    algorithms". Cluster Computing, 27, 1217-1234.

FuzzyDECLAT es la extension difusa (alpha-cortes) del mismo procedimiento,
consistente con el resto de algoritmos difusos de ARMxtend (FIM.BD_FARE).

@author Carlos Fernandez-Basso
"""
import numpy as np

from ._shared import generate_candidates, alpha_cuts


class DECLAT(object):
    """Mineria de itemsets frecuentes crisp en Spark, vía TID-lists (DECLAT)."""

    item_sep = ","

    @classmethod
    def _split(cls, line):
        return set(x.strip() for x in line.split(cls.item_sep) if x.strip())

    @classmethod
    def run(cls, sc, transactions, min_supp):
        """
        Argumentos:
            sc (SparkContext)
            transactions (RDD[str]): una transaccion por linea
            min_supp (float): soporte minimo relativo, en (0, 1]

        Retorna:
            dict {itemset_key: soporte relativo} con todos los itemsets
            frecuentes de cualquier longitud
        """
        indexedTransacs = transactions.zipWithIndex()
        totalTransacs = indexedTransacs.count()
        if totalTransacs == 0:
            return {}

        # Preprocesado (lineas 4-7 Algoritmo 3): transformar a formato vertical
        # <item, TID-list> agrupando por item las transacciones en que aparece
        def emitItemTid(pair):
            line, tid = pair
            return [(item, tid) for item in cls._split(line)]

        itemTidLists = indexedTransacs.flatMap(emitItemTid) \
                                       .groupByKey() \
                                       .mapValues(frozenset) \
                                       .collectAsMap()

        # Fase 1: FreqItems() -- soporte de itemsets de longitud 1
        freqItemsets = {item: len(tids) / totalTransacs
                        for item, tids in itemTidLists.items()
                        if len(tids) / totalTransacs >= min_supp}

        globalFreqItemsets = dict(freqItemsets)
        currentLevelKeys = sorted(freqItemsets.keys())

        # Fase 2: generacion iterativa de candidatos por interseccion de TID-lists
        while len(currentLevelKeys) > 1:
            candidateKeys = generate_candidates(currentLevelKeys)
            if not candidateKeys:
                break

            broadcastTidLists = sc.broadcast(itemTidLists)

            def countCandidate(key):
                tidLists = [broadcastTidLists.value[item] for item in key.split("-")]
                intersection = tidLists[0]
                for tidList in tidLists[1:]:
                    intersection = intersection & tidList
                return (key, len(intersection))

            itemsetCounts = dict(sc.parallelize(candidateKeys).map(countCandidate).collect())
            broadcastTidLists.unpersist()

            freqItemsets = {key: count / totalTransacs
                            for key, count in itemsetCounts.items()
                            if count / totalTransacs >= min_supp}
            if not freqItemsets:
                break

            globalFreqItemsets.update(freqItemsets)
            currentLevelKeys = sorted(freqItemsets.keys())

        return globalFreqItemsets


class FuzzyDECLAT(object):
    """
    Version difusa de DECLAT: cada transaccion es una lista de pares
    (item, grado de pertenencia), y el soporte de un itemset se calcula, para
    cada nivel alpha, como la media del minimo (t-norma) de los grados de
    pertenencia de sus items en las transacciones donde todos ellos aparecen.
    """

    @classmethod
    def run(cls, sc, transactions, min_supp, num_alpha):
        """
        Argumentos:
            sc (SparkContext)
            transactions (RDD[Iterable[Tuple[str, float]]]): cada transaccion
                es una lista de pares (item, grado de pertenencia en [0, 1])
            min_supp (float): soporte minimo relativo, en (0, 1]
            num_alpha (int): numero de alpha-cortes a considerar

        Retorna:
            dict {itemset_key: numpy.ndarray(num_alpha)} con el soporte
            relativo de cada itemset frecuente en, al menos, un alpha-corte
        """
        indexedTransacs = transactions.zipWithIndex()
        totalTransacs = indexedTransacs.count()
        if totalTransacs == 0:
            return {}

        def emitItemTid(pair):
            memberships, tid = pair
            return [(item, (tid, alpha_cuts(degree, num_alpha))) for item, degree in memberships]

        itemTidVectors = indexedTransacs.flatMap(emitItemTid) \
                                         .groupByKey() \
                                         .mapValues(dict) \
                                         .collectAsMap()

        def supportVector(tidVectorMap):
            total = np.zeros(num_alpha, dtype=float)
            for vector in tidVectorMap.values():
                total = total + vector
            return total / totalTransacs

        freqItemsets = {}
        for item, tidVectorMap in itemTidVectors.items():
            support = supportVector(tidVectorMap)
            if np.any(support >= min_supp):
                freqItemsets[item] = support

        globalFreqItemsets = dict(freqItemsets)
        currentLevelKeys = sorted(freqItemsets.keys())

        while len(currentLevelKeys) > 1:
            candidateKeys = generate_candidates(currentLevelKeys)
            if not candidateKeys:
                break

            broadcastTidVectors = sc.broadcast(itemTidVectors)

            def countCandidate(key):
                tidVectorMaps = [broadcastTidVectors.value[item] for item in key.split("-")]
                commonTids = set(tidVectorMaps[0].keys())
                for tidVectorMap in tidVectorMaps[1:]:
                    commonTids &= set(tidVectorMap.keys())

                total = np.zeros(num_alpha, dtype=float)
                for tid in commonTids:
                    membership = np.ones(num_alpha, dtype=float)
                    for tidVectorMap in tidVectorMaps:
                        membership = np.minimum(membership, tidVectorMap[tid])
                    total = total + membership

                return (key, total / totalTransacs)

            supports = dict(sc.parallelize(candidateKeys).map(countCandidate).collect())
            broadcastTidVectors.unpersist()

            freqItemsets = {key: support for key, support in supports.items()
                            if np.any(support >= min_supp)}
            if not freqItemsets:
                break

            globalFreqItemsets.update(freqItemsets)
            currentLevelKeys = sorted(freqItemsets.keys())

        return globalFreqItemsets
