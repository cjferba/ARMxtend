# -*- coding: utf-8 -*-
"""
DApriori y DAprioriTID: algoritmos de mineria de itemsets frecuentes sobre
Big Data (Apache Spark), inspirados respectivamente en los algoritmos
secuenciales Apriori y Apriori-TID (Agrawal & Srikant, 1994).

Implementan los Algoritmos 1 y 2 de:
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
    def _count_candidates(cls, transactions, candidate_itemsets):
        """
        candidate_itemsets: dict {itemset_key: frozenset(items)}
        Devuelve un RDD[(itemset_key, count)] con el conteo de apariciones
        de cada itemset candidato en las transacciones.
        """
        def emit(line):
            items_in_tx = cls._split(line)
            return [(key, 1) for key, iset in candidate_itemsets.items()
                    if iset.issubset(items_in_tx)]

        return transactions.flatMap(emit).reduceByKey(lambda a, b: a + b)

    @classmethod
    def run(cls, sc, transactions, min_supp):
        """
        Ejecuta el algoritmo (Fases 1 y 2 del Algoritmo 1/2).

        Argumentos:
            sc (SparkContext): contexto Spark para crear variables broadcast
            transactions (RDD[str]): una transaccion por linea, items
                separados por `cls.item_sep`
            min_supp (float): soporte minimo relativo, en (0, 1]

        Retorna:
            dict {itemset_key ('A-B-C'): soporte relativo} con todos los
            itemsets frecuentes de cualquier longitud
        """
        transactions = cls._preprocess(sc, transactions, min_supp)
        totalTransacs = transactions.count()
        if totalTransacs == 0:
            return {}

        # Fase 1: FreqItems() -- soporte de itemsets de longitud 1
        itemCounts = cls._count_items(transactions).collectAsMap()
        freqItemsets = {item: count / totalTransacs
                        for item, count in itemCounts.items()
                        if count / totalTransacs >= min_supp}

        globalFreqItemsets = dict(freqItemsets)
        currentLevelKeys = sorted(freqItemsets.keys())

        # Fase 2: generacion iterativa de candidatos de longitud creciente
        while len(currentLevelKeys) > 1:
            candidateKeys = generate_candidates(currentLevelKeys)
            if not candidateKeys:
                break

            candidateItemsets = {key: key_to_itemset(key) for key in candidateKeys}
            broadcastCandidates = sc.broadcast(candidateItemsets)

            itemsetCounts = cls._count_candidates(transactions, broadcastCandidates.value).collectAsMap()
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
        """Punto de extension para DAprioriTID: DApriori no transforma los datos."""
        return transactions


class DAprioriTID(DApriori):
    """
    Variante DApriori-TID: tras la Fase 1, ordena los items por soporte
    descendente y elimina de cada transaccion los items infrecuentes, de
    forma que las fases siguientes operan sobre datos ya reducidos
    (linea 10 del Algoritmo 2 del articulo).
    """

    @classmethod
    def _preprocess(cls, sc, transactions, min_supp):
        totalTransacs = transactions.count()
        if totalTransacs == 0:
            return transactions

        itemCounts = cls._count_items(transactions).collectAsMap()
        freqItems = {item: count for item, count in itemCounts.items()
                     if count / totalTransacs >= min_supp}

        # Orden descendente de soporte: acelera el filtrado en fases siguientes
        orderedItems = sorted(freqItems, key=freqItems.get, reverse=True)
        broadcastOrder = sc.broadcast(orderedItems)

        def removeInfrequentAndSort(line):
            itemsInTx = cls._split(line)
            keptOrdered = [item for item in broadcastOrder.value if item in itemsInTx]
            return cls.item_sep.join(keptOrdered)

        return transactions.map(removeInfrequentAndSort)
