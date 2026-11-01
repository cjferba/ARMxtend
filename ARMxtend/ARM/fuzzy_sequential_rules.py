# -*- coding: utf-8 -*-
"""
Reglas secuenciales DIFUSAS: ``X => Y`` ("si ocurren todos los items de X, despues
ocurren todos los de Y") sobre secuencias de eventos con grado de pertenencia
(ver ``FIM.fuzzy_sequential``). Es la version difusa de ``ARM.sequential_rules``,
con la misma definicion de regla valida: aqui es valida EN UN GRADO.

Grado de ``X => Y`` en una secuencia: con ``T`` la t-norma (``min`` o producto),
``mu_X(k) = T_{x in X} max_{p <= k} d(p, x)`` (en que grado han aparecido todos los
items de X hasta la posicion k), ``mu_Y(k) = T_{y in Y} max_{p > k} d(p, y)`` (en que
grado aparecen todos los de Y despues de k), y

    grado(X => Y) = max_k T(mu_X(k), mu_Y(k)).

Medidas, con las medias sobre las secuencias:

* soporte(X => Y) = media del grado de la regla;
* soporte del antecedente (consecuente) = media de ``T_{x in X} max_p d(p, x)``: en que
  grado aparecen todos los items de X (de Y) en la secuencia, en cualquier orden;
* confianza = soporte(X => Y) / soporte del antecedente; lift y demas medidas de
  ``ARM._measures`` con esos tres soportes.

Con grados 0/1 todo coincide con ``sequential_rules``. La busqueda parte de los itemsets
frecuentes de las secuencias vistas como cestas (el grado de un itemset es la t-norma de
los mejores grados de sus items; es antimonotono), y prueba sus particiones.

@author Carlos Fernandez-Basso
"""
from itertools import combinations
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from ..FIM.fuzzy_sequential import _check_tnorm, degree_tensor, normalize
from ._measures import MEASURES

_EPS = 1e-9


def _reduce(combine: Any, values: np.ndarray) -> np.ndarray:
    """La t-norma de las columnas (ultimo eje) de `values`."""
    result = values[..., 0]
    for column in range(1, values.shape[-1]):
        result = combine(result, values[..., column])
    return result


def _frequent_itemsets(basket: np.ndarray, combine: Any, min_supp: float, max_size: int) -> Dict[Tuple[int, ...], float]:
    """Itemsets (tuplas de columnas) cuyo soporte difuso como cestas es >= min_supp, por niveles."""
    frequent: Dict[Tuple[int, ...], float] = {}
    level: List[Tuple[int, ...]] = []
    for column in range(basket.shape[1]):
        support = float(basket[:, column].mean())
        if support >= min_supp - _EPS:
            frequent[(column,)] = support
            level.append((column,))
    size = 1
    while level and size < max_size:
        known = set(level)
        candidates = sorted({a + (b[-1],) for a in level for b in level if a[:-1] == b[:-1] and a[-1] < b[-1]})
        level = []
        for candidate in candidates:
            if any(subset not in known for subset in combinations(candidate, size)):
                continue
            support = float(_reduce(combine, basket[:, list(candidate)]).mean())
            if support >= min_supp - _EPS:
                frequent[candidate] = support
                level.append(candidate)
        size += 1
    return frequent


def fuzzy_sequential_rules(sequences: Iterable[Iterable[Any]], min_supp: float, min_conf: float = 0.8,
                           max_antecedent: int = 2, max_consequent: int = 2, metric: str = "confidence",
                           min_threshold: float = 0.0, tnorm: str = "min") -> pd.DataFrame:
    """
    Minera reglas secuenciales difusas ``X => Y`` (ver el docstring del modulo).

    Argumentos:
        sequences: una secuencia por elemento; cada una, una lista de elementos con
            sus grados de pertenencia (ver ``FIM.fuzzy_sequential``)
        min_supp (float): soporte difuso minimo de la regla, en (0, 1]
        min_conf (float): confianza minima, en [0, 1]
        max_antecedent, max_consequent (int): maximo de items de X y de Y
        metric, min_threshold: filtro adicional, como en ``sequential_rules``
        tnorm (str): 'min' (por defecto) o 'product'

    Retorna:
        pandas.DataFrame con el formato de ``sequential_rules`` (columnas 'antecedents'
        y 'consequents', 'antecedent support', 'consequent support', 'support',
        'confidence', 'lift' y las demas medidas), ordenado por confianza y soporte
        decrecientes. Memoria: proporcional a secuencias x longitud maxima x items
    """
    combine = _check_tnorm(tnorm)
    database = normalize(sequences)
    columns = ["antecedents", "consequents", "antecedent support", "consequent support", "support",
               "confidence", "lift"] + [name for name in MEASURES if name not in ("confidence", "lift")]
    if not database:
        return pd.DataFrame(columns=columns)
    if metric not in MEASURES and metric != "support":
        raise ValueError("metric debe ser 'support' o una de {}, recibida: '{}'".format(sorted(MEASURES), metric))
    if not 0 < min_supp <= 1:
        raise ValueError("min_supp debe estar en (0, 1]")
    if max_antecedent < 1 or max_consequent < 1:
        raise ValueError("max_antecedent y max_consequent deben ser >= 1")

    tensor, vocabulary = degree_tensor(database)
    total = len(database)
    basket = tensor.max(axis=1)                                    # mejor grado de cada item en cada secuencia
    upto = np.maximum.accumulate(tensor, axis=1)                   # mejor grado hasta la posicion k (incluida)
    later = np.maximum.accumulate(tensor[:, ::-1, :], axis=1)[:, ::-1, :]
    after = np.concatenate([later[:, 1:, :], np.zeros_like(later[:, :1, :])], axis=1)      # ... estrictamente despues de k

    cache: Dict[Tuple[int, ...], float] = {}

    def basket_support(items: Tuple[int, ...]) -> float:
        if items not in cache:
            cache[items] = float(_reduce(combine, basket[:, list(items)]).mean())
        return cache[items]

    rows: List[dict] = []
    for itemset in _frequent_itemsets(basket, combine, min_supp, max_antecedent + max_consequent):
        if len(itemset) < 2:
            continue
        for size in range(1, min(max_antecedent, len(itemset) - 1) + 1):
            if len(itemset) - size > max_consequent:
                continue
            for antecedent in combinations(itemset, size):
                consequent = tuple(column for column in itemset if column not in antecedent)
                degree = combine(_reduce(combine, upto[:, :, list(antecedent)]),
                                 _reduce(combine, after[:, :, list(consequent)])).max(axis=1)
                support = float(degree.mean())
                if support < min_supp - _EPS:
                    continue
                supportX, supportY = basket_support(antecedent), basket_support(consequent)
                confidence = support / supportX
                if confidence < min_conf - _EPS:
                    continue
                entry: Dict[str, Any] = {"antecedents": frozenset(vocabulary[c] for c in antecedent),
                                         "consequents": frozenset(vocabulary[c] for c in consequent),
                                         "antecedent support": supportX, "consequent support": supportY,
                                         "support": support}
                for name, function in MEASURES.items():
                    entry[name] = float(function(support, supportX, supportY))
                if metric != "support" and min_threshold and entry[metric] < min_threshold:
                    continue
                rows.append(entry)

    result = pd.DataFrame(rows, columns=columns)
    result = result.sort_values(["confidence", "support"], ascending=False, kind="mergesort").reset_index(drop=True)
    result.attrs["n_sequences"] = total
    return result


def fuzzy_rule_degree(sequence: Iterable[Any], antecedent: Iterable[str], consequent: Iterable[str],
                      tnorm: str = "min") -> float:
    """Grado en que la regla ``antecedent => consequent`` es valida en una secuencia (ver el docstring del modulo)."""
    combine = _check_tnorm(tnorm)
    tensor, vocabulary = degree_tensor(normalize([sequence]))
    column = {item: position for position, item in enumerate(vocabulary)}
    x, y = list(antecedent), list(consequent)
    if any(item not in column for item in x + y):
        return 0.0
    upto = np.maximum.accumulate(tensor[0], axis=0)
    later = np.maximum.accumulate(tensor[0][::-1], axis=0)[::-1]
    after = np.concatenate([later[1:], np.zeros_like(later[:1])], axis=0)
    before = _reduce(combine, upto[:, [column[item] for item in x]])
    following = _reduce(combine, after[:, [column[item] for item in y]])
    return float(combine(before, following).max())
