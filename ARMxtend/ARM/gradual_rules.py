# -*- coding: utf-8 -*-
"""
Reglas graduales: dependencias del tipo "cuanto MAS A (y cuanto MENOS B),
MAS C" sobre atributos numericos, sin discretizar ni fuzzificar. Se basan en:

    Hullermeier, E. (2002). "Association rules for expressing gradual
    dependencies". PKDD 2002, LNCS 2431, 200-211.

Un item gradual es un par (atributo, direccion) con direccion '+' ("cuanto
mas") o '-' ("cuanto menos"); un itemset gradual es un conjunto de items de
atributos distintos. Un par de filas (s, t) lo SATISFACE si, al pasar de
s a t, todos sus atributos varian en la direccion indicada (estrictamente:
los empates no cuentan). El soporte de un itemset gradual es la proporcion
de los n(n-1)/2 pares no ordenados de filas que lo satisfacen en alguna
de las dos orientaciones; la confianza de A -> B es supp(A u B) / supp(A).

Simetria: un itemset y su "espejo" (todas las direcciones invertidas)
tienen el mismo soporte -- "cuanto mas A, mas B" equivale a "cuanto menos
A, menos B" --, asi que solo se generan itemsets en forma CANONICA (la
direccion del primer atributo es '+'); cada regla devuelta tiene una regla
equivalente "espejo" que no se lista.

Coste: se manejan matrices booleanas n x n por atributo, es decir O(n^2 m)
de memoria. Para datos grandes use `max_rows` para trabajar sobre una
muestra aleatoria reproducible de filas.

@author Carlos Fernandez-Basso
"""
import itertools

import numpy as np
import pandas as pd

INCREASING = "+"
DECREASING = "-"
_FLIP = {INCREASING: DECREASING, DECREASING: INCREASING}


def format_gradual_item(item):
    """('A', '+') -> 'A↑' ("cuanto mas A"); ('A', '-') -> 'A↓' ("cuanto menos A")."""
    attribute, direction = item
    return "{}{}".format(attribute, "↑" if direction == INCREASING else "↓")


def format_gradual_itemset(items):
    """Texto de un conjunto de items graduales, p.ej. 'A↑, B↓'."""
    return ", ".join(sorted(format_gradual_item(item) for item in items))


def _canonical(itemset, attribute_order):
    """Forma canonica (ordenada por atributo, primera direccion '+') de un itemset gradual."""
    ordered = tuple(sorted(itemset, key=lambda item: attribute_order[item[0]]))
    if ordered and ordered[0][1] == DECREASING:
        ordered = tuple((attribute, _FLIP[direction]) for attribute, direction in ordered)
    return ordered


class _PairSupport(object):
    """Calcula el soporte por pares de itemsets graduales sobre una tabla numerica."""

    def __init__(self, table, attributes):
        self.attributes = list(attributes)
        self.order = {attribute: position for position, attribute in enumerate(self.attributes)}
        n = len(table)
        self.total_pairs = n * (n - 1) / 2.0
        # lower[a][s, t] = True si el valor de s en `a` es menor que el de t
        self._lower = {}
        for attribute in self.attributes:
            values = table[attribute].to_numpy(dtype=float)
            self._lower[attribute] = values[:, None] < values[None, :]

    def _matrix(self, item):
        attribute, direction = item
        lower = self._lower[attribute]
        return lower if direction == INCREASING else lower.T

    def support(self, itemset):
        matrices = [self._matrix(item) for item in itemset]
        satisfied = matrices[0]
        for matrix in matrices[1:]:
            satisfied = satisfied & matrix
        # Cada par no ordenado satisface a lo sumo una orientacion del
        # itemset, luego contar pares ordenados cuenta cada par una vez.
        return float(satisfied.sum()) / self.total_pairs


def _prepare(data, attributes, max_rows, random_state):
    table = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    attributes = list(attributes) if attributes is not None else list(table.select_dtypes("number").columns)
    if not attributes:
        raise ValueError("No hay atributos numericos sobre los que buscar reglas graduales")
    table = table[attributes].dropna()
    if max_rows is not None and len(table) > max_rows:
        table = table.sample(n=max_rows, random_state=random_state)
    if len(table) < 2:
        raise ValueError("Se necesitan al menos 2 filas para formar pares")
    return table.reset_index(drop=True), attributes


def gradual_itemsets(data, min_supp, attributes=None, max_len=None, max_rows=None, random_state=0):
    """
    Itemsets graduales frecuentes (soporte por pares >= `min_supp`),
    mediante una busqueda por niveles estilo Apriori (el soporte por pares
    es antimonotono, como el soporte clasico).

    Argumentos:
        data (pandas.DataFrame | array-like 2D): tabla de valores numericos
        min_supp (float): soporte minimo, en (0, 1]
        attributes (list, opcional): atributos a considerar (por defecto,
            todas las columnas numericas)
        max_len (int, opcional): numero maximo de items por itemset
        max_rows (int, opcional): si hay mas filas, se usa una muestra
            aleatoria de este tamano (reproducible con `random_state`)
        random_state (int): semilla del muestreo

    Retorna:
        dict {itemset canonico: soporte}, con el itemset como tupla de
        pares (atributo, '+'|'-') ordenada segun `attributes`.
    """
    table, attributes = _prepare(data, attributes, max_rows, random_state)
    counter = _PairSupport(table, attributes)

    frequent = {}
    level = {}
    for attribute in attributes:
        itemset = ((attribute, INCREASING),)
        support = counter.support(itemset)
        if support >= min_supp:
            level[itemset] = support
    frequent.update(level)

    length = 1
    while level and (max_len is None or length < max_len):
        candidates = []
        for itemset in level:
            lastPosition = counter.order[itemset[-1][0]]
            for attribute in attributes[lastPosition + 1:]:
                for direction in (INCREASING, DECREASING):
                    candidates.append(itemset + ((attribute, direction),))

        nextLevel = {}
        for candidate in candidates:
            subsets = (candidate[:i] + candidate[i + 1:] for i in range(len(candidate)))
            if not all(_canonical(subset, counter.order) in frequent for subset in subsets):
                continue
            support = counter.support(candidate)
            if support >= min_supp:
                nextLevel[candidate] = support

        frequent.update(nextLevel)
        level = nextLevel
        length += 1

    return frequent


def gradual_rules(data, min_supp, min_conf, attributes=None, max_len=None, max_rows=None, random_state=0):
    """
    Reglas graduales  A -> B  (A y B, conjuntos de items graduales de
    atributos distintos) con soporte >= `min_supp` y confianza >=
    `min_conf`, p.ej. "cuanto mas edad y cuanto menos ejercicio, mas
    colesterol".

    Argumentos: ver `gradual_itemsets`, mas
        min_conf (float): confianza minima supp(A u B) / supp(A)

    Retorna:
        pandas.DataFrame con columnas 'antecedents' y 'consequents'
        (frozensets de pares (atributo, direccion); use
        `format_gradual_itemset` para mostrarlos), 'antecedent support',
        'consequent support', 'support' y 'confidence'.
    """
    table, attributes = _prepare(data, attributes, max_rows, random_state)
    frequent = gradual_itemsets(table, min_supp, attributes=attributes, max_len=max_len)
    order = {attribute: position for position, attribute in enumerate(attributes)}

    # Por antimonotonia, todo subconjunto de un itemset frecuente es frecuente
    # (y, siendo mas corto, ya esta en `frequent` aunque se use `max_len`).
    def supportOf(itemset):
        return frequent[_canonical(itemset, order)]

    rows = []
    for itemset, support in frequent.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, size):
                consequent = tuple(item for item in itemset if item not in antecedent)
                antecedentSupport = supportOf(antecedent)
                if antecedentSupport == 0:
                    continue
                confidence = support / antecedentSupport
                if confidence >= min_conf:
                    rows.append({
                        "antecedents": frozenset(antecedent),
                        "consequents": frozenset(consequent),
                        "antecedent support": antecedentSupport,
                        "consequent support": supportOf(consequent),
                        "support": support,
                        "confidence": confidence,
                    })

    columns = ["antecedents", "consequents", "antecedent support", "consequent support", "support", "confidence"]
    return pd.DataFrame(rows, columns=columns)
