# -*- coding: utf-8 -*-
"""
Reglas de excepcion y reglas anomalas: patrones que describen CUANDO falla
una regla de asociacion fuerte. Ambos tipos aparecen en la seccion 4 de
Delgado, Ruiz, Sanchez & Serrano (2011, Information Sciences 181) como la
segunda clase de enfoques para tratar la ausencia de items (frente a la
negacion directa de items, ver ``ARM.negative_rules``), y se basan en:

    Suzuki, E. (1997). "Autonomous discovery of reliable exception rules".
    KDD-97 (reglas de excepcion).

    Delgado, M., Ruiz, M.D., Sanchez, D. (2011). "New approaches for
    discovering exception and anomalous rules". Int. J. Uncertainty,
    Fuzziness and Knowledge-Based Systems, 19(2), 361-399 (reglas anomalas).

Ambos tipos parten de una regla "de sentido comun" fuerte  A -> b  (el
comportamiento habitual) y buscan que explica sus fallos:

* Regla de EXCEPCION (Suzuki): un conjunto de items Z tal que  A ^ Z -> ~b
  es fuerte -- cuando ademas de A se da Z, ocurre lo contrario de lo
  habitual -- mientras que la regla de REFERENCIA  Z -> ~b  NO es fuerte
  (Z por si solo no implica ~b, asi que la excepcion se debe a la
  combinacion A ^ Z y no a Z). Como la excepcion es rara, se exige un
  soporte minimo bastante menor que el de la regla comun.

* Regla ANOMALA: un conjunto de items C tal que  A ^ ~b -> C  es fuerte --
  entre los casos en que A se cumple pero b no (las "anomalias"), C
  aparece de forma sistematica, es decir, C caracteriza la anomalia --,
  mientras que  A -> C  NO es fuerte (C no es ya habitual cuando se da A).

Las definiciones de umbrales y de la regla de referencia de las reglas
anomalas aqui implementadas son una formulacion directa de las
condiciones anteriores sobre soporte y confianza; consulte el articulo
original para las variantes formales mas generales (p.ej. medidas distintas
de la confianza o reglas "dobles").

Solo se consideran reglas comunes con UN item en el consecuente: la
negacion de un consecuente con varios items es una disyuncion
(~(b1 ^ b2) = ~b1 v ~b2), que Delgado et al. advierten de que puede no
interesar al usuario.

@author Carlos Fernandez-Basso
"""
import itertools

import numpy as np
import pandas as pd

from ..FFIM import fpgrowth
from ..FIM._shared import key_to_itemset
from .association_rules import association_rules

_EXCEPTION_COLUMNS = [
    "antecedents", "consequent", "condition", "common_support", "common_confidence",
    "exception_support", "exception_confidence", "reference_confidence",
]
_ANOMALOUS_COLUMNS = [
    "antecedents", "consequent", "anomaly", "common_support", "common_confidence",
    "anomalous_cases", "anomalous_support", "anomalous_confidence", "reference_confidence",
]


def _boolean_matrix(transactions):
    transactions = [set(t) for t in transactions]
    items = sorted(set().union(*transactions)) if transactions else []
    index = {item: column for column, item in enumerate(items)}
    matrix = np.zeros((len(transactions), len(items)), dtype=bool)
    for row, transaction in enumerate(transactions):
        for item in transaction:
            matrix[row, index[item]] = True
    return matrix, index


def _single_consequent_common_rules(transactions, rules, min_supp, min_conf):
    if rules is None:
        frequent = fpgrowth(transactions, min_supp)
        itemsets = pd.DataFrame({
            "support": list(frequent.values()),
            "itemsets": [key_to_itemset(key) for key in frequent],
        })
        rules = association_rules(itemsets, metric="confidence", min_threshold=min_conf)
    if len(rules) == 0:
        return rules
    return rules[rules["consequents"].apply(len) == 1]


def _condition_candidates(matrix, index, excluded_items, min_count):
    """Columnas de items (fuera de `excluded_items`) con al menos `min_count` apariciones."""
    excluded = {index[item] for item in excluded_items}
    counts = matrix.sum(axis=0)
    return [column for column in range(matrix.shape[1]) if column not in excluded and counts[column] >= min_count]


def _combinations(candidates, max_size):
    for size in range(1, max_size + 1):
        for combination in itertools.combinations(candidates, size):
            yield combination


def exception_rules(transactions, min_supp=0.1, min_conf=0.8, min_exception_supp=0.01,
                    min_exception_conf=0.8, max_reference_conf=0.5, max_condition_size=1, rules=None):
    """
    Encuentra reglas de excepcion (Suzuki, 1997) de las reglas comunes
    fuertes  A -> b: condiciones Z tales que  A ^ Z -> ~b  es fuerte y
    Z -> ~b  no lo es.

    Argumentos:
        transactions (Sequence[Iterable[str]]): transacciones crisp
        min_supp (float): soporte minimo de las reglas comunes
        min_conf (float): confianza minima de las reglas comunes
        min_exception_supp (float): soporte minimo de  A ^ Z ^ ~b  (suele
            ser mucho menor que `min_supp`: las excepciones son raras)
        min_exception_conf (float): confianza minima de  A ^ Z -> ~b
        max_reference_conf (float): confianza MAXIMA de la regla de
            referencia  Z -> ~b
        max_condition_size (int): numero maximo de items de Z
        rules (pandas.DataFrame, opcional): reglas comunes ya calculadas
            (salida de `ARM.association_rules`). Si se omite, se minan con
            `min_supp` y `min_conf`

    Retorna:
        pandas.DataFrame con columnas 'antecedents' (A), 'consequent' (b),
        'condition' (Z), 'common_support', 'common_confidence',
        'exception_support', 'exception_confidence' y
        'reference_confidence', ordenado por confianza de la excepcion.
    """
    transactions = [set(t) for t in transactions]
    total = len(transactions)
    commonRules = _single_consequent_common_rules(transactions, rules, min_supp, min_conf)
    if total == 0 or len(commonRules) == 0:
        return pd.DataFrame(columns=_EXCEPTION_COLUMNS)

    matrix, index = _boolean_matrix(transactions)
    minExceptionCount = min_exception_supp * total
    rows = []

    for _, rule in commonRules.iterrows():
        antecedent = frozenset(rule["antecedents"])
        (consequent,) = tuple(rule["consequents"])
        maskA = matrix[:, [index[i] for i in antecedent]].all(axis=1)
        notB = ~matrix[:, index[consequent]]

        candidates = _condition_candidates(matrix, index, antecedent | {consequent}, minExceptionCount)
        for columns in _combinations(candidates, max_condition_size):
            maskZ = matrix[:, list(columns)].all(axis=1)
            countZ = maskZ.sum()
            maskAZ = maskA & maskZ
            countAZ = maskAZ.sum()
            countException = (maskAZ & notB).sum()

            if countAZ == 0 or countException < minExceptionCount or countException == 0:
                continue
            exceptionConf = countException / countAZ
            referenceConf = (maskZ & notB).sum() / countZ
            if exceptionConf >= min_exception_conf and referenceConf <= max_reference_conf:
                rows.append({
                    "antecedents": antecedent,
                    "consequent": consequent,
                    "condition": frozenset(index_item for index_item, column in index.items() if column in columns),
                    "common_support": float(rule["support"]),
                    "common_confidence": float(rule["confidence"]),
                    "exception_support": countException / total,
                    "exception_confidence": float(exceptionConf),
                    "reference_confidence": float(referenceConf),
                })

    result = pd.DataFrame(rows, columns=_EXCEPTION_COLUMNS)
    return result.sort_values(["exception_confidence", "exception_support"], ascending=False).reset_index(drop=True)


def anomalous_rules(transactions, min_supp=0.1, min_conf=0.8, min_anomalous_supp=0.01,
                    min_anomalous_conf=0.8, max_reference_conf=0.5, max_anomaly_size=1, rules=None):
    """
    Encuentra reglas anomalas: de las reglas comunes fuertes  A -> b, los
    conjuntos de items C tales que  A ^ ~b -> C  es fuerte (C caracteriza
    los casos en que A se cumple pero b no) y  A -> C  no lo es.

    Argumentos:
        transactions (Sequence[Iterable[str]]): transacciones crisp
        min_supp, min_conf, rules: ver `exception_rules`
        min_anomalous_supp (float): soporte minimo de  A ^ ~b ^ C
        min_anomalous_conf (float): confianza minima de  A ^ ~b -> C
        max_reference_conf (float): confianza MAXIMA de  A -> C
        max_anomaly_size (int): numero maximo de items de C

    Retorna:
        pandas.DataFrame con columnas 'antecedents' (A), 'consequent' (b),
        'anomaly' (C), 'common_support', 'common_confidence',
        'anomalous_cases' (numero de transacciones con A ^ ~b),
        'anomalous_support', 'anomalous_confidence' y
        'reference_confidence' (de A -> C), ordenado por confianza.
    """
    transactions = [set(t) for t in transactions]
    total = len(transactions)
    commonRules = _single_consequent_common_rules(transactions, rules, min_supp, min_conf)
    if total == 0 or len(commonRules) == 0:
        return pd.DataFrame(columns=_ANOMALOUS_COLUMNS)

    matrix, index = _boolean_matrix(transactions)
    minAnomalousCount = min_anomalous_supp * total
    rows = []

    for _, rule in commonRules.iterrows():
        antecedent = frozenset(rule["antecedents"])
        (consequent,) = tuple(rule["consequents"])
        maskA = matrix[:, [index[i] for i in antecedent]].all(axis=1)
        maskAnomaly = maskA & ~matrix[:, index[consequent]]
        countAnomalies = maskAnomaly.sum()
        if countAnomalies == 0:
            continue

        candidates = _condition_candidates(matrix, index, antecedent | {consequent}, minAnomalousCount)
        for columns in _combinations(candidates, max_anomaly_size):
            maskC = matrix[:, list(columns)].all(axis=1)
            countAnomalyC = (maskAnomaly & maskC).sum()
            if countAnomalyC < minAnomalousCount or countAnomalyC == 0:
                continue
            anomalousConf = countAnomalyC / countAnomalies
            referenceConf = (maskA & maskC).sum() / maskA.sum()
            if anomalousConf >= min_anomalous_conf and referenceConf <= max_reference_conf:
                rows.append({
                    "antecedents": antecedent,
                    "consequent": consequent,
                    "anomaly": frozenset(item for item, column in index.items() if column in columns),
                    "common_support": float(rule["support"]),
                    "common_confidence": float(rule["confidence"]),
                    "anomalous_cases": int(countAnomalies),
                    "anomalous_support": countAnomalyC / total,
                    "anomalous_confidence": float(anomalousConf),
                    "reference_confidence": float(referenceConf),
                })

    result = pd.DataFrame(rows, columns=_ANOMALOUS_COLUMNS)
    return result.sort_values(["anomalous_confidence", "anomalous_support"], ascending=False).reset_index(drop=True)
