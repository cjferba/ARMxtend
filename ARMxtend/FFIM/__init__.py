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
forma independiente en cada uno de los `num_alpha` niveles, combinando los
resultados en un vector de soporte por itemset -- exactamente la misma
convencion que usan ``FuzzyDECLAT`` y ``FuzzyDAprioriTID``, lo que permite
usar indistintamente cualquiera de los tres algoritmos como entrada de
``ARMxtend.FIM.FARE``.

Sustituye al contenido que existia previamente en este fichero, que era una
copia identica (con los mismos errores) de ``FIM/BD_FARE/__init__.py``.

@author Carlos Fernandez-Basso
"""
from collections import Counter

import numpy as np

from ..FIM._shared import itemset_to_key, alpha_cuts


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
    FP-Growth difuso: ejecuta `fpgrowth` de forma independiente en cada uno
    de los `num_alpha` alpha-cortes (tras binarizar los grados de
    pertenencia de cada transaccion con `alpha_cuts`), y combina los
    resultados en un vector de soporte por itemset, siguiendo la misma
    convencion que ``ARMxtend.FIM.Eclat.FuzzyDECLAT`` y
    ``ARMxtend.FIM.BD_FARE.FuzzyDAprioriTID``: un itemset se conserva si es
    frecuente en, al menos, uno de los alpha-cortes (propiedad que se
    conserva por cierre descendente, ver docstring de `FuzzyDECLAT`).

    Argumentos:
        transactions (Sequence[Iterable[Tuple[str, float]]]): cada
            transaccion es una lista de pares (item, grado de pertenencia
            en [0, 1])
        min_supp (float): soporte minimo relativo, en (0, 1]
        num_alpha (int): numero de alpha-cortes a considerar

    Retorna:
        dict {itemset_key: numpy.ndarray(num_alpha)}
    """
    totalTransacs = len(transactions)
    if totalTransacs == 0:
        return {}

    transactionAlphaVectors = [
        {item: alpha_cuts(degree, num_alpha) for item, degree in transaction}
        for transaction in transactions
    ]

    supportPerLevel = []
    for alphaLevel in range(num_alpha):
        binarizedTransactions = [
            {item for item, vector in txVectors.items() if vector[alphaLevel]}
            for txVectors in transactionAlphaVectors
        ]
        supportPerLevel.append(fpgrowth(binarizedTransactions, min_supp))

    allKeys = set()
    for levelResult in supportPerLevel:
        allKeys.update(levelResult.keys())

    return {key: np.array([supportPerLevel[level].get(key, 0.0) for level in range(num_alpha)])
            for key in allKeys}
