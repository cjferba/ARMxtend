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

from ._shared import generate_candidates, alpha_levels, alpha_weights, weighted_alpha_aggregate


def _intersect_positions(first, second):
    """
    Posiciones, en `first` y en `second`, de los elementos comunes de dos arrays ORDENADOS y sin repetidos
    (TID-lists). Busca cada elemento de `first` en `second` con ``searchsorted``: O(|first| log |second|), sin
    concatenar ni reordenar; conviene que `first` sea el mas corto.
    """
    if len(first) == 0 or len(second) == 0:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty
    where = np.searchsorted(second, first)
    where[where == len(second)] = len(second) - 1
    hit = np.flatnonzero(second[where] == first)
    return hit, where[hit]


def _tid_dtype(total):
    """int32 si los identificadores caben (la mitad de memoria que int64), int64 si no."""
    return np.int32 if total < 2 ** 31 else np.int64


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
        dtype = _tid_dtype(totalTransacs)

        # Preprocesado (lineas 4-7 Algoritmo 3): transformar a formato vertical
        # <item, TID-list> agrupando por item las transacciones en que aparece.
        # Cada TID-list se compacta EN LOS EJECUTORES a un array numpy ordenado
        # (4 bytes por aparicion): lo que viaja al driver y se difunde es del
        # orden de los datos, no de los objetos Python que los representan
        def emitItemTid(pair):
            line, tid = pair
            return [(item, tid) for item in cls._split(line)]

        def compact(tids):
            return np.sort(np.fromiter(tids, dtype=dtype))

        itemTidLists = indexedTransacs.flatMap(emitItemTid) \
                                       .groupByKey() \
                                       .mapValues(compact) \
                                       .collectAsMap()

        # Fase 1: FreqItems() -- soporte de itemsets de longitud 1
        freqItemsets = {item: len(tids) / totalTransacs
                        for item, tids in itemTidLists.items()
                        if len(tids) / totalTransacs >= min_supp}

        globalFreqItemsets = dict(freqItemsets)
        currentLevelKeys = sorted(freqItemsets.keys())

        # Fase 2: generacion iterativa de candidatos por interseccion de TID-lists.
        # Las TID-lists se difunden UNA vez para todos los niveles
        broadcastTidLists = sc.broadcast(itemTidLists)

        def countCandidate(key):
            # se empieza por la lista mas corta: cada interseccion cuesta lo que ella
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
            dict {itemset_key: numpy.ndarray(num_alpha)} con el bit-list de
            soporte relativo de cada itemset frecuente en cada alpha-corte
            (ver `FIM._shared.alpha_cuts`). Un itemset se considera frecuente
            si su soporte difuso agregado FSupp (ver
            `FIM._shared.weighted_alpha_aggregate`) alcanza `min_supp`; dicho
            vector se conserva integro (no solo el agregado) porque
            `FIM.FARE` lo necesita para calcular la confianza difusa nivel a
            nivel (Ecs. (2)-(4) de Fernandez-Basso, Ruiz & Martin-Bautista,
            2021).
        """
        levels = alpha_levels(num_alpha)
        weights = alpha_weights(levels)

        indexedTransacs = transactions.zipWithIndex()
        totalTransacs = indexedTransacs.count()
        if totalTransacs == 0:
            return {}
        dtype = _tid_dtype(totalTransacs)

        # Formato vertical <item, (TIDs, grados)>: dos arrays numpy por item, construidos EN LOS EJECUTORES
        # (TIDs ordenados de 4 bytes y grados de 8). El alpha-corte del minimo de varios grados es el minimo de
        # sus alpha-cortes (``value >= alpha`` es monotona), asi que basta guardar el grado y contar a cada
        # nivel al final: el resultado es el mismo que llevando un vector de alpha-cortes por TID, con una
        # fraccion de la memoria y sin bucles de Python por TID
        def emitItemTid(pair):
            memberships, tid = pair
            for _, degree in memberships:
                if not 0.0 <= degree <= 1.0:
                    raise ValueError("El grado de pertenencia difuso debe estar en [0, 1], recibido: {}".format(degree))
            return [(item, (tid, float(degree))) for item, degree in memberships]

        def compact(pairs):
            pairs = list(pairs)
            tids = np.fromiter((tid for tid, _ in pairs), dtype=dtype, count=len(pairs))
            degrees = np.fromiter((degree for _, degree in pairs), dtype=float, count=len(pairs))
            order = np.argsort(tids, kind="stable")
            tids, degrees = tids[order], degrees[order]
            if len(tids) > 1 and (tids[1:] == tids[:-1]).any():           # un item repetido en la transaccion: el mayor grado
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
