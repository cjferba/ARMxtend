# -*- coding: utf-8 -*-
"""
FFIM (Fuzzy Frequent Itemset Mining): mineria de itemsets frecuentes,
crisp y difusos, en una sola maquina (sin Spark), mediante el algoritmo
FP-Growth (Han, Pei & Yin, 2000).

Es el contrapunto secuencial de los algoritmos difusos de Big Data de
``ARMxtend.FIM.BD_FARE``/``ARMxtend.FIM.Eclat.FuzzyDECLAT``: en lugar de
generar y contar candidatos nivel a nivel (Apriori/Eclat), construye un
arbol de patrones frecuentes (FP-Tree) y lo consulta recursivamente sin
generar candidatos explicitos, lo que suele ser mas eficiente en una unica
maquina para datasets con muchos items.

Para la version difusa, el arbol FP-Growth (que fusiona conteos de
transacciones con el MISMO prefijo de items) no admite directamente grados
de pertenencia reales, ya que dos transacciones con los mismos items pero
distinto grado de pertenencia dejarian de ser "el mismo camino" del arbol.
Este modulo resuelve esto binarizando cada transaccion en cada alpha-corte
(ver ``ARMxtend.FIM._shared.alpha_cuts``) y ejecutando el FP-Growth crisp de
forma independiente en cada uno de los `num_alpha` niveles para descubrir
candidatos, recalculando despues su soporte exacto en todos los niveles y
filtrando por el soporte difuso agregado FSupp (ver `fuzzy_fpgrowth`) --
exactamente la misma convencion (bit-list de soporte por alpha-corte) que
usan ``FuzzyDECLAT`` y ``FuzzyDAprioriTID``, lo que permite usar
indistintamente cualquiera de los tres algoritmos como entrada de
``ARMxtend.FIM.FARE``.

Sustituye al contenido que existia previamente en este fichero, que era una
copia identica (con los mismos errores) de ``FIM/BD_FARE/__init__.py``.

@author Carlos Fernandez-Basso
"""
from collections import Counter

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


def _build_fptree(transactions, min_count):
    """
    Construye un FP-Tree a partir de una lista de transacciones (cada una,
    una coleccion de items), conservando unicamente los items cuyo conteo
    absoluto (sobre el total de transacciones ORIGINAL, no el tamano de
    `transactions`) alcanza `min_count`.

    Retorna (header, order, item_counts): la tabla de cabeceras (item -> primer
    nodo del arbol con ese item, enlazados entre si via `link`), el orden
    descendente de frecuencia de los items conservados, y sus conteos.
    """
    itemCounts = Counter()
    for transaction in transactions:
        itemCounts.update(set(transaction))

    itemCounts = {item: count for item, count in itemCounts.items() if count >= min_count}
    if not itemCounts:
        return {}, [], {}

    order = sorted(itemCounts, key=lambda item: (-itemCounts[item], item))
    orderIndex = {item: idx for idx, item in enumerate(order)}

    root = _FPNode(None, None)
    header = {item: None for item in order}
    headerTail = {item: None for item in order}

    for transaction in transactions:
        filteredItems = sorted((item for item in set(transaction) if item in itemCounts),
                               key=lambda item: orderIndex[item])
        node = root
        for item in filteredItems:
            child = node.children.get(item)
            if child is None:
                child = _FPNode(item, node)
                node.children[item] = child
                if header[item] is None:
                    header[item] = child
                else:
                    headerTail[item].link = child
                headerTail[item] = child
            child.count += 1
            node = child

    return header, order, itemCounts


def _ascend_prefix_path(node):
    """Items (sin contar el propio nodo) desde la raiz hasta el nodo dado."""
    path = []
    node = node.parent
    while node is not None and node.item is not None:
        path.append(node.item)
        node = node.parent
    return path


def _mine_fptree(header, order, itemCounts, min_count, prefix, result):
    # Se procesan los items del menos al mas frecuente: cada uno se combina
    # con el prefijo actual para formar un nuevo itemset frecuente, y su
    # base de patrones condicional se mina recursivamente
    for item in reversed(order):
        newItemset = frozenset(prefix) | {item}
        result[newItemset] = itemCounts[item]

        conditionalTransactions = []
        node = header[item]
        while node is not None:
            prefixPath = _ascend_prefix_path(node)
            if prefixPath:
                conditionalTransactions.extend([prefixPath] * node.count)
            node = node.link

        if conditionalTransactions:
            condHeader, condOrder, condCounts = _build_fptree(conditionalTransactions, min_count)
            if condOrder:
                _mine_fptree(condHeader, condOrder, condCounts, min_count, newItemset, result)


def fpgrowth(transactions, min_supp):
    """
    FP-Growth crisp: mineria exhaustiva de itemsets frecuentes sin generar
    candidatos explicitos.

    Argumentos:
        transactions (Sequence[Iterable[str]]): una transaccion por
            elemento, cada una una coleccion de items
        min_supp (float): soporte minimo relativo, en (0, 1]

    Retorna:
        dict {itemset_key: soporte relativo} con todos los itemsets
        frecuentes de cualquier longitud
    """
    totalTransacs = len(transactions)
    if totalTransacs == 0:
        return {}

    minCount = min_supp * totalTransacs - 1e-9
    header, order, itemCounts = _build_fptree(transactions, minCount)

    result = {}
    if order:
        _mine_fptree(header, order, itemCounts, minCount, frozenset(), result)

    return {itemset_to_key(itemset): count / totalTransacs for itemset, count in result.items()}


def fuzzy_fpgrowth(transactions, min_supp, num_alpha):
    """
    FP-Growth difuso: encuentra candidatos ejecutando `fpgrowth` de forma
    independiente en cada uno de los `num_alpha` alpha-cortes (tras
    binarizar los grados de pertenencia de cada transaccion con
    `alpha_cuts`), y despues recalcula, para cada candidato, su bit-list de
    soporte EXACTO en todos los niveles (no solo en aquellos donde ya era
    frecuente de forma aislada), de forma que el soporte difuso agregado
    FSupp (ver `FIM._shared.weighted_alpha_aggregate`, Ec. (2) de
    Fernandez-Basso, Ruiz & Martin-Bautista, 2021) se calcule correctamente.

    Este paso de recalculo es necesario porque, si un itemset es frecuente
    en el nivel alpha_i (soporte >= min_supp), su soporte agregado FSupp
    (una media ponderada de sus soportes en todos los niveles) puede ser
    inferior a min_supp si en el resto de niveles su soporte es bajo; y a la
    inversa, un itemset puede tener FSupp >= min_supp sin llegar a
    min_supp de forma aislada en NINGUN nivel salvo el de mayor soporte
    (ya que FSupp es una media ponderada, esta acotada por el maximo de los
    soportes por nivel, luego ese nivel de soporte maximo si sera detectado
    por `fpgrowth`). Por eso es correcto usar la union de los itemsets
    frecuentes de cada nivel como conjunto de candidatos, pero es necesario
    recalcular su soporte real en cada nivel (no asumir 0 en los niveles
    donde no fueron detectados) antes de agregar y filtrar por FSupp.

    Argumentos:
        transactions (Sequence[Iterable[Tuple[str, float]]]): cada
            transaccion es una lista de pares (item, grado de pertenencia
            en [0, 1])
        min_supp (float): soporte minimo relativo, en (0, 1]
        num_alpha (int): numero de alpha-cortes a considerar

    Retorna:
        dict {itemset_key: numpy.ndarray(num_alpha)} con el bit-list de
        soporte relativo de cada itemset frecuente (FSupp >= min_supp) en
        cada alpha-corte.
    """
    totalTransacs = len(transactions)
    if totalTransacs == 0:
        return {}

    weights = alpha_weights(alpha_levels(num_alpha))

    transactionAlphaVectors = [
        {item: alpha_cuts(degree, num_alpha) for item, degree in transaction}
        for transaction in transactions
    ]

    candidateKeys = set()
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
            memberVectors = [txVectors.get(item) for item in items]
            if any(vector is None for vector in memberVectors):
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
