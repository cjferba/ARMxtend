# -*- coding: utf-8 -*-
"""
Reglas de asociacion TEMPORALES con retardo (reglas inter-transaccion), crisp
y difusas: reglas como ``temp=high@lag2, door_open@lag1 -> heater_on`` ("si
hace dos pasos la temperatura era alta y hace uno estaba la puerta abierta,
entonces ahora esta la calefaccion encendida").

Se sigue la formulacion de las reglas de asociacion inter-transaccion de:

    Lu, H., Han, J., Feng, L. (1998). "Stock movement prediction and
    N-dimensional inter-transaction association rules". ACM SIGMOD Workshop
    on Research Issues in Data Mining and Knowledge Discovery.

    Tung, A.K.H., Lu, H., Han, J., Feng, L. (2003). "Efficient mining of
    intertransaction association rules". IEEE Transactions on Knowledge and
    Data Engineering, 15(1), 43-56.

Cada instante ``t`` (desde ``max_lag``) pasa a ser una transaccion que
contiene los items de ``t`` (sin sufijo) y los de ``t-1, ..., t-max_lag``
(con sufijo ``@lagk``, ver ``preprocessing.timeseries``). Se minan los
itemsets frecuentes de esas transacciones y se conservan las reglas cuyo
CONSECUENTE esta solo en el instante actual y cuyo antecedente mira al
pasado. El soporte se calcula sobre las ``N - max_lag`` filas que tienen todos
los retardos disponibles, de modo que todas las medidas comparten
denominador. La version difusa usa los grados de pertenencia con la
descomposicion por alpha-cortes de la RL-theory (``FIM.FARE``).

El coste crece con el numero de items por el numero de retardos: ajuste
``min_supp``, ``lags`` y, sobre todo, simbolice con pocos valores
(``preprocessing.timeseries.symbolize``).

NOTA: los nombres de item no deben contener '-', que es el separador de
items en las claves de itemset de ``FIM._shared``.

@author Carlos Fernandez-Basso
"""
from typing import Any, FrozenSet, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..FFIM import fpgrowth, fuzzy_fpgrowth
from ..FIM._shared import key_to_itemset
from ..FIM.FARE import fuzzy_association_rules
from ..preprocessing.timeseries import (
    check_item_names, item_variable, lag_items, split_lag, to_fuzzy_transactions, to_transactions,
)
from .association_rules import association_rules

def _lag_set(max_lag: int, lags: Optional[Iterable[int]]) -> List[int]:
    if max_lag < 1:
        raise ValueError("max_lag debe ser >= 1, recibido: {}".format(max_lag))
    chosen = range(1, max_lag + 1) if lags is None else sorted({int(lag) for lag in lags})
    if any(lag < 1 or lag > max_lag for lag in chosen):
        raise ValueError("lags debe estar contenido en 1..{}, recibido: {}".format(max_lag, list(chosen)))
    return [0] + list(chosen)


def _is_target(item: str, targets: Optional[FrozenSet[str]]) -> bool:
    return targets is None or item in targets or item_variable(item) in targets


def _keep_rules(rules: pd.DataFrame, targets: Optional[Iterable[str]], require_lagged_antecedent: bool,
                same_time_antecedent: bool, exclude_persistence: bool) -> pd.DataFrame:
    """Filtra las reglas por estructura temporal y anade las columnas 'lags' y 'max_lag'."""
    wanted = None if targets is None else frozenset(targets)
    if len(rules):
        mask = rules.apply(lambda rule: _passes(rule, wanted, require_lagged_antecedent, same_time_antecedent,
                                                exclude_persistence), axis=1).astype(bool)
        result = rules[mask].reset_index(drop=True)
    else:
        result = rules.copy()
    lags = [tuple(sorted({lag for lag in (split_lag(item)[1] for item in antecedent) if lag > 0}))
            for antecedent in result["antecedents"]]
    result["lags"] = pd.Series(lags, index=result.index, dtype=object)
    result["max_lag"] = pd.Series([max(item, default=0) for item in lags], index=result.index, dtype=int)
    return result


def _passes(rule: Any, wanted: Optional[FrozenSet[str]], require_lagged_antecedent: bool,
            same_time_antecedent: bool, exclude_persistence: bool) -> bool:
    consequent = [split_lag(item) for item in rule["consequents"]]
    antecedent = [split_lag(item) for item in rule["antecedents"]]
    if any(lag != 0 for _, lag in consequent):
        return False
    if wanted is not None and not all(_is_target(item, wanted) for item, _ in consequent):
        return False
    if require_lagged_antecedent and not any(lag > 0 for _, lag in antecedent):
        return False
    if not same_time_antecedent and any(lag == 0 for _, lag in antecedent):
        return False
    if exclude_persistence:
        consequentVariables = {item_variable(item) for item, _ in consequent}
        if any(lag > 0 and item_variable(item) in consequentVariables for item, lag in antecedent):
            return False
    return True


def temporal_association_rules(items: pd.DataFrame, max_lag: int, min_supp: float, metric: str = "confidence",
                               min_threshold: float = 0.8, lags: Optional[Iterable[int]] = None,
                               targets: Optional[Iterable[str]] = None, require_lagged_antecedent: bool = True,
                               same_time_antecedent: bool = True, exclude_persistence: bool = False) -> pd.DataFrame:
    """
    Minera reglas de asociacion crisp con retardo: antecedente en instantes
    anteriores (``item@lagk``) y, opcionalmente, en el actual, y consecuente
    en el instante actual.

    Argumentos:
        items (pandas.DataFrame): items booleanos, una fila por instante y
            ordenadas en el tiempo (ver ``preprocessing.timeseries.to_items``)
        max_lag (int): mayor retardo del antecedente, en pasos, >= 1
        min_supp (float): soporte minimo relativo de los itemsets, en (0, 1]
        metric (str): medida de `ARM.association_rules` para filtrar
        min_threshold (float): umbral minimo de `metric`
        lags (Iterable[int], opcional): retardos permitidos, dentro de
            1..max_lag (por defecto, todos); p.ej. [1, 24] para mirar solo
            "hace un paso" y "hace 24 pasos"
        targets (Iterable[str], opcional): consecuentes permitidos, por nombre
            de item ('heater_on', 'temp=high') o de variable ('temp'); por
            defecto, cualquiera
        require_lagged_antecedent (bool): si True (por defecto), el antecedente
            debe incluir algun item de un instante anterior; si False tambien
            se devuelven reglas de co-ocurrencia en el mismo instante
        same_time_antecedent (bool): si False, el antecedente solo puede
            contener items de instantes anteriores
        exclude_persistence (bool): si True, descarta las reglas cuyo
            antecedente incluye, con retardo, la misma variable que el
            consecuente (p.ej. 'temp=high@lag1 -> temp=high')

    Retorna:
        pandas.DataFrame con el formato de `ARM.association_rules` y dos
        columnas mas: 'lags' (tupla de retardos distintos del antecedente) y
        'max_lag' (el mayor de ellos)
    """
    check_item_names(items.columns)
    lagged = lag_items(items, _lag_set(max_lag, lags))
    frequent = fpgrowth(to_transactions(lagged), min_supp)
    itemsets = pd.DataFrame({"support": list(frequent.values()),
                             "itemsets": [key_to_itemset(key) for key in frequent]})
    rules = association_rules(itemsets, metric=metric, min_threshold=min_threshold)
    return _keep_rules(rules, targets, require_lagged_antecedent, same_time_antecedent, exclude_persistence)


def fuzzy_temporal_association_rules(memberships: pd.DataFrame, max_lag: int, min_supp: float, num_alpha: int = 10,
                                     metric: str = "confidence", min_threshold: float = 0.8,
                                     lags: Optional[Iterable[int]] = None, targets: Optional[Iterable[str]] = None,
                                     require_lagged_antecedent: bool = True, same_time_antecedent: bool = True,
                                     exclude_persistence: bool = False,
                                     measures: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Version difusa de `temporal_association_rules`: los items son grados de
    pertenencia en [0, 1] y el soporte, la confianza y el factor de certeza se
    calculan con la descomposicion por alpha-cortes de la RL-theory
    (`FIM.FARE`).

    Argumentos:
        memberships (pandas.DataFrame): grados de pertenencia, una columna por
            item (p.ej. las columnas que anade ``FuzzyLib.AutoFuzzification``)
            y una fila por instante, ordenadas en el tiempo
        max_lag, lags, targets, require_lagged_antecedent,
        same_time_antecedent, exclude_persistence: ver `temporal_association_rules`
        min_supp (float): soporte difuso minimo FSupp, en (0, 1]
        num_alpha (int): numero de alpha-cortes
        metric, min_threshold, measures: ver `FIM.FARE.fuzzy_association_rules`

    Retorna:
        pandas.DataFrame con el formato de `FIM.FARE.fuzzy_association_rules`
        y las columnas 'lags' y 'max_lag'
    """
    check_item_names(memberships.columns)
    lagged = lag_items(memberships, _lag_set(max_lag, lags))
    frequent = fuzzy_fpgrowth(to_fuzzy_transactions(lagged), min_supp, num_alpha)
    rules = fuzzy_association_rules(frequent, num_alpha, metric=metric, min_threshold=min_threshold,
                                    measures=measures)
    return _keep_rules(rules, targets, require_lagged_antecedent, same_time_antecedent, exclude_persistence)


def format_lagged_item(item: str) -> str:
    """Texto legible de un item con retardo: 'temp=high@lag2' -> 'temp=high (t-2)'; sin retardo, 'temp=high (t)'."""
    base, lag = split_lag(item)
    return "{} (t{})".format(base, "-{}".format(lag) if lag else "")


def rule_profile(items: pd.DataFrame, antecedent: Iterable[str], consequent: Iterable[str],
                 by: Union[Sequence[Any], np.ndarray, pd.Series, Any]) -> pd.DataFrame:
    """
    Como se comporta una regla en cada grupo de instantes (por hora del dia,
    dia de la semana, mes, estacion...): sirve para detectar reglas CICLICAS,
    que son debiles en promedio pero muy fiables a ciertas horas o dias.

    Argumentos:
        items (pandas.DataFrame): items booleanos, una fila por instante. La
            regla puede usar items con retardo (``item@lagk``): se calculan a
            partir de las columnas de `items`
        antecedent (Iterable[str]): items del antecedente (con o sin retardo)
        consequent (Iterable[str]): items del consecuente (con o sin retardo)
        by: grupo de cada fila de `items`: una secuencia o ``Series`` de su
            misma longitud (p.ej. ``items.index.hour``) o una funcion que
            recibe el indice de `items` y devuelve esa secuencia

    Retorna:
        pandas.DataFrame indexado por grupo con las columnas 'n' (instantes del
        grupo), 'antecedent_count', 'joint_count', 'support', 'confidence'
        (NaN si el antecedente no aparece) y 'lift' (NaN si no esta definido).
        Las primeras filas, sin datos para los retardos, no se cuentan. Para el
        total, use un `by` constante.
    """
    antecedent = list(antecedent)
    consequent = list(consequent)
    parsed = [split_lag(item) for item in antecedent + consequent]
    missing = [base for base, _ in parsed if base not in items.columns]
    if missing:
        raise ValueError("items que no son columnas de `items`: {}".format(sorted(set(missing))))

    groups = np.asarray(by(items.index) if callable(by) else by)
    if len(groups) != len(items):
        raise ValueError("`by` debe tener una entrada por fila ({}), tiene {}".format(len(items), len(groups)))
    start = max(lag for _, lag in parsed)
    rows = len(items)

    def holds(names):
        present = np.ones(rows - start, dtype=bool)
        for item in names:
            base, lag = split_lag(item)
            present &= items[base].to_numpy(dtype=bool)[start - lag:rows - lag]
        return present

    antecedentHolds, consequentHolds = holds(antecedent), holds(consequent)
    table = pd.DataFrame({"group": groups[start:], "a": antecedentHolds, "c": consequentHolds})
    table["ac"] = table["a"] & table["c"]
    grouped = table.groupby("group")
    result = pd.DataFrame({"n": grouped.size(), "antecedent_count": grouped["a"].sum(),
                           "joint_count": grouped["ac"].sum(), "consequent_count": grouped["c"].sum()})
    result["support"] = result["joint_count"] / result["n"]
    result["confidence"] = result["joint_count"] / result["antecedent_count"].where(result["antecedent_count"] > 0)
    consequentRate = result["consequent_count"] / result["n"]
    result["lift"] = result["confidence"] / consequentRate.where(consequentRate > 0)
    return result.drop(columns="consequent_count")
