# -*- coding: utf-8 -*-
"""
Reglas SECUENCIALES: ``X => Y`` significa "si en una secuencia ocurren todos
los items de X, despues ocurren todos los de Y". A diferencia de una regla de
asociacion, el orden importa (X antes que Y); a diferencia de un patron
secuencial, dentro de X y dentro de Y el orden no importa, y se mide cuan
FIABLE es la prediccion, con soporte y confianza.

Se sigue la definicion de las reglas secuenciales de:

    Fournier-Viger, P., Wu, C.-W., Tseng, V.S., Cao, L., Nkambou, R. (2015).
    "Mining partially-ordered sequential rules common to multiple sequences".
    IEEE Transactions on Knowledge and Data Engineering, 27(8), 2203-2216.

(y su algoritmo RuleGrowth, del mismo grupo, 2011). Una secuencia es una lista
de elementos ordenados en el tiempo (conjuntos de items simultaneos; ver
``FIM.sequential``). La regla ``X => Y``, con X e Y disjuntos, es VALIDA en una
secuencia si existe un instante ``k`` tal que todos los items de X han
aparecido en los elementos hasta ``k`` y todos los de Y aparecen en elementos
posteriores a ``k``. Medidas:

* soporte(X => Y)  = fraccion de secuencias en que la regla es valida;
* confianza(X => Y) = secuencias en que es valida / secuencias que contienen
  todos los items de X (en cualquier orden);
* lift = confianza / fraccion de secuencias que contienen todos los items de Y.

La busqueda se apoya en que, para que la regla sea valida, X, Y y X u Y deben
estar en las secuencias, de modo que se parte de los itemsets frecuentes de las
secuencias vistas como cestas (FP-Growth) y se prueban sus particiones, igual
que en la generacion de reglas de asociacion. No es el algoritmo RuleGrowth
(que busca de forma incremental sin enumerar todos los itemsets), pero usa la
misma definicion de regla valida, soporte y confianza.

NOTA: los nombres de item no deben contener '-', que es el separador de
items en las claves de itemset de ``FIM._shared``.

@author Carlos Fernandez-Basso
"""
from itertools import combinations
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import pandas as pd

from ..FFIM import fpgrowth
from ..FIM._shared import key_to_itemset
from ..FIM.sequential import _normalize
from ..preprocessing.timeseries import check_item_names
from ._measures import MEASURES

_ABSENT_FIRST = 10 ** 9


def sequential_rules(sequences: Iterable[Iterable[Union[str, Iterable[str]]]], min_supp: float,
                     min_conf: float = 0.8, max_antecedent: int = 2, max_consequent: int = 2,
                     metric: str = "confidence", min_threshold: float = 0.0) -> pd.DataFrame:
    """
    Minera reglas secuenciales ``X => Y`` ("si X, despues Y").

    Argumentos:
        sequences: una secuencia por elemento; cada una es una lista de
            elementos (conjuntos de items simultaneos, o un item suelto).
            Ver ``FIM.sequential``
        min_supp (float): soporte minimo relativo de la regla, en (0, 1]
        min_conf (float): confianza minima, en [0, 1]
        max_antecedent (int): maximo de items de X
        max_consequent (int): maximo de items de Y
        metric (str): medida adicional para filtrar, ademas de `min_conf`:
            cualquiera de ``ARM._measures.MEASURES`` ('lift', 'leverage', ...)
        min_threshold (float): umbral minimo de `metric` (0 = no filtra)

    Retorna:
        pandas.DataFrame con las columnas 'antecedents' y 'consequents'
        (``frozenset``), 'antecedent support' y 'consequent support'
        (fraccion de secuencias que contienen todos sus items, en cualquier
        orden), 'support', 'confidence' y 'lift', y las demas medidas de
        ``ARM._measures`` calculadas con esos tres soportes, ordenado por
        confianza y soporte decrecientes.
    """
    database = _normalize(sequences)
    total = len(database)
    columns = ["antecedents", "consequents", "antecedent support", "consequent support", "support",
               "confidence", "lift"] + [name for name in MEASURES if name not in ("confidence", "lift")]
    if total == 0:
        return pd.DataFrame(columns=columns)
    if metric not in MEASURES and metric != "support":
        raise ValueError("metric debe ser 'support' o una de {}, recibida: '{}'".format(sorted(MEASURES), metric))
    if max_antecedent < 1 or max_consequent < 1:
        raise ValueError("max_antecedent y max_consequent deben ser >= 1")
    vocabulary = sorted({item for sequence in database for element in sequence for item in element})
    check_item_names(vocabulary)

    # Soportes de itemsets en las secuencias vistas como cestas (el orden no cuenta)
    baskets = [set().union(*sequence) if sequence else set() for sequence in database]
    frequent = fpgrowth(baskets, min_supp)

    # Primera y ultima aparicion de cada item en cada secuencia
    column = {item: index for index, item in enumerate(vocabulary)}
    first = np.full((total, len(vocabulary)), _ABSENT_FIRST, dtype=np.int64)
    last = np.full((total, len(vocabulary)), -1, dtype=np.int64)
    for row, sequence in enumerate(database):
        for position, element in enumerate(sequence):
            for item in element:
                cell = column[item]
                if first[row, cell] == _ABSENT_FIRST:
                    first[row, cell] = position
                last[row, cell] = position

    completedAt: Dict[frozenset, np.ndarray] = {}
    expiresAt: Dict[frozenset, np.ndarray] = {}

    def completed(items: frozenset) -> np.ndarray:
        """Instante en que X ya ha aparecido entero: el maximo de las primeras apariciones de sus items."""
        if items not in completedAt:
            completedAt[items] = first[:, [column[item] for item in items]].max(axis=1)
        return completedAt[items]

    def expiring(items: frozenset) -> np.ndarray:
        """Ultimo instante en que Y aun puede aparecer entero: el minimo de las ultimas apariciones de sus items."""
        if items not in expiresAt:
            expiresAt[items] = last[:, [column[item] for item in items]].min(axis=1)
        return expiresAt[items]

    rows: List[dict] = []
    for key, joint in frequent.items():
        itemset = key_to_itemset(key)
        if len(itemset) < 2 or len(itemset) > max_antecedent + max_consequent:
            continue
        for size in range(1, len(itemset)):
            if size > max_antecedent or len(itemset) - size > max_consequent:
                continue
            for antecedent in combinations(sorted(itemset), size):
                x = frozenset(antecedent)
                y = itemset - x
                valid = int(np.count_nonzero(completed(x) < expiring(y)))
                support = valid / total
                if support < min_supp - 1e-9:
                    continue
                supportX = frequent[_key(x)]
                supportY = frequent[_key(y)]
                confidence = support / supportX
                if confidence < min_conf - 1e-9:
                    continue
                entry: Dict[str, Any] = {"antecedents": x, "consequents": y, "antecedent support": supportX,
                                         "consequent support": supportY, "support": support}
                for name, function in MEASURES.items():
                    entry[name] = float(function(support, supportX, supportY))
                if metric != "support" and min_threshold and entry[metric] < min_threshold:
                    continue
                rows.append(entry)

    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values(["confidence", "support"], ascending=False, kind="mergesort").reset_index(drop=True)


def _key(items: frozenset) -> str:
    return "-".join(sorted(items))


def contains_rule(sequence: Iterable[Iterable[str]], antecedent: Iterable[str], consequent: Iterable[str]) -> bool:
    """True si la regla ``antecedent => consequent`` es valida en `sequence` (ver el docstring del modulo)."""
    elements = [frozenset(e) for e in sequence]
    firstSeen: Dict[str, int] = {}
    lastSeen: Dict[str, int] = {}
    for position, element in enumerate(elements):
        for item in element:
            firstSeen.setdefault(item, position)
            lastSeen[item] = position
    x, y = list(antecedent), list(consequent)
    if any(item not in firstSeen for item in x + y):
        return False
    return max(firstSeen[item] for item in x) < min(lastSeen[item] for item in y)
