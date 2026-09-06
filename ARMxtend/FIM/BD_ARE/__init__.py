# -*- coding: utf-8 -*-
"""
Mineria de reglas de asociacion crisp sobre Big Data (Apache Spark), a
partir de un conjunto de itemsets frecuentes ya calculado (p.ej. con
``ARMxtend.FIM.apriori.DApriori/DAprioriTID`` o ``ARMxtend.FIM.Eclat.DECLAT``).

Implementa el Algoritmo 4 ("Spark procedure for association rule mining")
de:
    Fernandez-Basso, C., Ruiz, M.D., Martin-Bautista, M.J. (2024). "New Spark
    solutions for distributed frequent itemset and association rule mining
    algorithms". Cluster Computing, 27, 1217-1234.

Este modulo sustituye a las clases ``Apriori``/``AprioriTID``/``Eclat``/``ARE``
que existian previamente en este fichero: eran copias, con errores (nombres
indefinidos, una ruta de fichero de un unico equipo de desarrollo, y una
llamada a ``exit()`` en mitad de la definicion de una clase que abortaba el
interprete al importar el modulo), del contenido de ``FIM/apriori.py``. La
mineria de itemsets ya vive en ``FIM.apriori``/``FIM.Eclat``; aqui solo queda
la Fase de generacion de reglas, comun a los tres algoritmos.

@author Carlos Fernandez-Basso
"""
import itertools


def generate_rules(freq_itemsets, min_conf):
    """
    Genera, de forma puramente secuencial, todas las reglas de asociacion
    X -> Y con confianza >= min_conf a partir de un conjunto de itemsets
    frecuentes. Es la parte de calculo (independiente de Spark) reutilizada
    por `association_rules_bd` para distribuirla en un cluster.

    Argumentos:
        freq_itemsets (dict[str, float]): itemsets frecuentes, con clave
            'A-B-C' y valor su soporte relativo. Debe incluir el soporte de
            todo subconjunto no vacio de cada itemset (garantizado por la
            propiedad de cierre descendente de los algoritmos de FIM.apriori
            y FIM.Eclat)
        min_conf (float): confianza minima, en (0, 1]

    Retorna:
        dict {(antecedente, consecuente): confianza}
    """
    rules = {}
    for itemsetKey, supportXY in freq_itemsets.items():
        items = itemsetKey.split("-")
        if len(items) < 2:
            continue

        for antecedentSize in range(1, len(items)):
            for antecedent in itertools.combinations(items, antecedentSize):
                antecedentKey = "-".join(sorted(antecedent))
                consequentKey = "-".join(sorted(item for item in items if item not in antecedent))

                supportX = freq_itemsets.get(antecedentKey)
                if not supportX:
                    continue

                confidence = supportXY / supportX
                if confidence >= min_conf:
                    rules[(antecedentKey, consequentKey)] = confidence

    return rules


def association_rules_bd(sc, freq_itemsets, min_conf):
    """
    Version distribuida (Spark) de `generate_rules` (Algoritmo 4): reparte
    la generacion de reglas de cada itemset frecuente entre los nodos del
    cluster mediante un FlatMap, y filtra por confianza minima con un Reduce.

    Argumentos:
        sc (SparkContext): contexto Spark para crear la variable broadcast
            con los itemsets frecuentes (Global_FreqItemset en el articulo)
        freq_itemsets (dict[str, float]): itemsets frecuentes candidatos,
            con clave 'A-B-C' y valor su soporte relativo
        min_conf (float): confianza minima, en (0, 1]

    Retorna:
        dict {(antecedente, consecuente): confianza}
    """
    broadcastFreqItemsets = sc.broadcast(freq_itemsets)

    def generateRulesForItemset(itemsetKey):
        supportXY = broadcastFreqItemsets.value[itemsetKey]
        items = itemsetKey.split("-")
        itemsetRules = []

        for antecedentSize in range(1, len(items)):
            for antecedent in itertools.combinations(items, antecedentSize):
                antecedentKey = "-".join(sorted(antecedent))
                consequentKey = "-".join(sorted(item for item in items if item not in antecedent))

                supportX = broadcastFreqItemsets.value.get(antecedentKey)
                if not supportX:
                    continue

                itemsetRules.append(((antecedentKey, consequentKey), supportXY / supportX))

        return itemsetRules

    candidateKeys = [key for key in freq_itemsets if len(key.split("-")) >= 2]

    rules = sc.parallelize(candidateKeys) \
              .flatMap(generateRulesForItemset) \
              .filter(lambda rule: rule[1] >= min_conf) \
              .collectAsMap()

    broadcastFreqItemsets.unpersist()
    return rules
