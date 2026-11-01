# -*- coding: utf-8 -*-
"""
Deteccion de anomalias por VIOLACION DE REGLAS: las reglas de asociacion
minadas sobre un periodo de funcionamiento normal describen lo que suele
ocurrir; una fila que incumple una regla fiable (su antecedente se cumple y su
consecuente no) es una candidata a anomalia, tanto mas cuanto mas fiable sea la
regla. Es la misma idea que las reglas de excepcion (``ARM.exception_rules``),
pero aplicada a datos NUEVOS con reglas ya minadas, incluidas las temporales
(``item@lagk``, ver ``ARM.temporal_rules``): "la puerta se abrio hace un paso y
la luz no se ha encendido".

Una fila VIOLA la regla ``A => C`` si se cumplen todos los items de A y no se
cumplen todos los de C. Cada violacion pesa segun la regla violada
(``weight``):

* ``surprise`` (por defecto): ``-log2(1 - confianza)``, los bits de sorpresa de
  una violacion si la regla se cumple con probabilidad igual a su confianza;
  violar una regla del 99 % pesa 6.6 bits, y una del 80 %, 2.3. La confianza se
  acota a ``1 - floor`` para que una regla del 100 % no pese infinito.
* ``confidence``: la confianza de la regla.
* ``count``: 1, el numero de reglas violadas.

El peso de una fila es la SUMA de los pesos de las reglas que viola. Las reglas
redundantes se violan siempre juntas y cuentan varias veces: pode antes con
``prune_redundant_rules``; y las reglas que son solo ruido producen falsas
alarmas: filtre antes con ``significant_rules``.

@author Carlos Fernandez-Basso
"""
from typing import List, Optional

import numpy as np
import pandas as pd

from ._apply import ItemTable, describe_rule, require_columns

WEIGHTS = ("surprise", "confidence", "count")


def _weights(rules: pd.DataFrame, weight: str, floor: float) -> np.ndarray:
    if weight not in WEIGHTS:
        raise ValueError("weight debe ser una de {}, recibida: '{}'".format(WEIGHTS, weight))
    if weight == "count":
        return np.ones(len(rules))
    require_columns(rules, "confidence")
    confidence = rules["confidence"].to_numpy(dtype=float)
    if weight == "confidence":
        return confidence
    if not 0 < floor < 1:
        raise ValueError("floor debe estar en (0, 1)")
    return -np.log2(np.maximum(1.0 - confidence, floor))


def _violated(table: ItemTable, antecedent, consequent) -> np.ndarray:
    return table.holds(antecedent) & table.evaluable(consequent) & ~table.holds(consequent)


def rule_violations(items: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla booleana de violaciones: una fila por fila de `items` y una columna
    por regla de `rules` (con su mismo indice); True donde la fila viola la regla.

    Argumentos:
        items (pandas.DataFrame): items booleanos (0/1), ordenados en el tiempo
            si las reglas llevan retardos. Los items de las reglas deben ser
            columnas de `items`
        rules (pandas.DataFrame): reglas con columnas 'antecedents' y
            'consequents' (conjuntos de items, con o sin retardo ``@lagk``)

    Las primeras filas, sin datos anteriores suficientes para una regla con
    retardos, no la violan. Para evaluar un periodo nuevo con historia,
    incluya en `items` las filas anteriores y seleccione despues el periodo.
    """
    require_columns(rules, "antecedents", "consequents")
    table = ItemTable(items)
    matrix = np.zeros((table.rows, len(rules)), dtype=bool)
    for position, (antecedent, consequent) in enumerate(zip(rules["antecedents"], rules["consequents"])):
        matrix[:, position] = _violated(table, antecedent, consequent)
    return pd.DataFrame(matrix, index=items.index, columns=rules.index)


def anomaly_scores(items: pd.DataFrame, rules: pd.DataFrame, weight: str = "surprise",
                   floor: float = 1e-3) -> pd.DataFrame:
    """
    Puntuacion de anomalia de cada fila de `items`: la suma de los pesos de las
    reglas que viola (ver el docstring del modulo).

    Argumentos:
        items, rules: ver `rule_violations`
        weight (str): 'surprise' (por defecto), 'confidence' o 'count'
        floor (float): con 'surprise', la menor probabilidad de violacion que se
            considera, en (0, 1): acota el peso de una regla a ``-log2(floor)``

    Retorna:
        pandas.DataFrame con el indice de `items` y las columnas 'score' (0 si
        no viola ninguna regla), 'violations' (reglas violadas) y 'applicable'
        (reglas cuyo antecedente se cumple en la fila: sobre cuantas pudo fallar)
    """
    require_columns(rules, "antecedents", "consequents")
    weights = _weights(rules, weight, floor)
    table = ItemTable(items)
    score = np.zeros(table.rows)
    violations = np.zeros(table.rows, dtype=int)
    applicable = np.zeros(table.rows, dtype=int)
    for position, (antecedent, consequent) in enumerate(zip(rules["antecedents"], rules["consequents"])):
        fires = table.holds(antecedent) & table.evaluable(consequent)
        violated = fires & ~table.holds(consequent)
        applicable += fires
        violations += violated
        score += weights[position] * violated
    return pd.DataFrame({"score": score, "violations": violations, "applicable": applicable}, index=items.index)


def detect_anomalies(items: pd.DataFrame, rules: pd.DataFrame, min_score: float = 0.0, top: Optional[int] = None,
                     weight: str = "surprise", floor: float = 1e-3) -> pd.DataFrame:
    """
    Las filas anomalas de `items`, explicadas: las que tienen `score` mayor que
    `min_score`, de mayor a menor puntuacion (a igual puntuacion, en orden
    cronologico).

    Argumentos:
        items, rules, weight, floor: ver `anomaly_scores`
        min_score (float): puntuacion minima (exclusiva); por defecto 0, es
            decir, cualquier fila que viole alguna regla
        top (int, opcional): conserva solo las `top` primeras

    Retorna:
        pandas.DataFrame con el indice de las filas anomalas y las columnas
        'score', 'violations' y 'rules' (lista con el texto de las reglas
        violadas, p.ej. ``'motion (t-1) => light (t)'``)
    """
    scores = anomaly_scores(items, rules, weight, floor)
    values = scores["score"].to_numpy()
    order = np.argsort(-values, kind="stable")                  # a igual puntuacion, orden cronologico
    order = order[values[order] > min_score]
    if top is not None:
        order = order[:top]

    table = ItemTable(items)
    explanation: List[List[str]] = [[] for _ in range(len(order))]
    for antecedent, consequent in zip(rules["antecedents"], rules["consequents"]):
        text = describe_rule(antecedent, consequent)
        for row in np.flatnonzero(_violated(table, antecedent, consequent)[order]):
            explanation[row].append(text)
    result = scores.iloc[order][["score", "violations"]].copy()
    result["rules"] = explanation
    return result
