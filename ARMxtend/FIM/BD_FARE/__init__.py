# -*- coding: utf-8 -*-
"""
FuzzyDAprioriTID: mineria de itemsets frecuentes difusos sobre Big Data
(Apache Spark), inspirada en Apriori-TID, usando una representacion en
alpha-cortes de los grados de pertenencia (ver ``ARMxtend.FIM._shared``).

Sustituye a la clase ``AprioriTID`` que existia previamente en este fichero:
tenia varios errores que impedian su uso (un `__init__`/`run_model` que
referenciaban un `sc`/`model` inexistentes fuera de un notebook interactivo,
una llamada a la funcion `AlphaCortes` sin cualificar que lanzaba
`NameError`, e imports duplicados/sin usar). La logica de alpha-cortes y de
generacion de candidatos es ahora compartida con el resto de algoritmos de
`FIM` a traves de `FIM._shared`.

@author Carlos Fernandez-Basso
"""
import numpy as np

from .._shared import generate_candidates, alpha_cuts


class FuzzyDAprioriTID(object):
    """
    Mineria de itemsets frecuentes difusos en Spark. Cada transaccion es una
    lista de pares (item, grado de pertenencia en [0, 1]); el soporte de un
    itemset se calcula, para cada alpha-corte, como la media del minimo
    (t-norma) de los grados de pertenencia de sus items en cada transaccion.
    """

    @classmethod
    def run(cls, sc, transactions, min_supp, num_alpha):
        """
        Argumentos:
            sc (SparkContext)
            transactions (RDD[Iterable[Tuple[str, float]]]): cada
                transaccion es una lista de pares (item, grado de
                pertenencia)
            min_supp (float): soporte minimo relativo, en (0, 1]
            num_alpha (int): numero de alpha-cortes a considerar

        Retorna:
            dict {itemset_key: numpy.ndarray(num_alpha)} con el soporte de
            cada itemset frecuente en, al menos, un alpha-corte
        """
        alphaTransactions = transactions.map(
            lambda memberships: {item: alpha_cuts(degree, num_alpha) for item, degree in memberships})

        totalTransacs = alphaTransactions.count()
        if totalTransacs == 0:
            return {}

        # Fase 1: soporte (vector de alpha-cortes) de itemsets de longitud 1
        itemVectors = alphaTransactions.flatMap(lambda tx: list(tx.items())) \
                                        .reduceByKey(lambda a, b: a + b) \
                                        .collectAsMap()

        freqItemsets = {item: vector / totalTransacs
                        for item, vector in itemVectors.items()
                        if np.any(vector / totalTransacs >= min_supp)}

        # Filtrar de cada transaccion los items infrecuentes (optimizacion
        # Apriori-TID: las fases siguientes solo consultan datos reducidos)
        broadcastFreqItems = sc.broadcast(set(freqItemsets.keys()))
        filteredTransactions = alphaTransactions.map(
            lambda tx: {item: vector for item, vector in tx.items() if item in broadcastFreqItems.value})

        globalFreqItemsets = dict(freqItemsets)
        currentLevelKeys = sorted(freqItemsets.keys())

        # Fase 2: generacion iterativa de candidatos de longitud creciente
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
                            if np.any(vector / totalTransacs >= min_supp)}
            if not freqItemsets:
                break

            globalFreqItemsets.update(freqItemsets)
            currentLevelKeys = sorted(freqItemsets.keys())

        broadcastFreqItems.unpersist()
        return globalFreqItemsets
