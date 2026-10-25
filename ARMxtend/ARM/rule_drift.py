# -*- coding: utf-8 -*-
"""
Deriva de reglas ("rule drift"): como cambian las reglas de asociacion a lo
largo del tiempo. Se minan las reglas en ventanas sucesivas de una tabla de
items ordenada en el tiempo (datos de sensores, logs, un flujo troceado) y se
sigue cada regla de una ventana a otra para saber cuales son estables, cuales
emergen, cuales desaparecen y cuales aparecen y desaparecen
(intermitentes), y como evoluciona el valor de su medida.

Es la version "por lotes" de las tendencias que FIMoTS (``SARE``/``SFIM``)
mantiene de forma incremental sobre un flujo de Spark: aqui el minero es
cualquier funcion ``miner(ventana) -> reglas``, de modo que valen reglas
crisp, difusas o temporales (``ARM.temporal_rules``), y se ejecuta en una sola
maquina.

Estados de una regla (``rule_drift``), con ``W`` ventanas:

* ``stable``: esta en la primera y en la ultima ventana y en al menos
  ``stable_fraction`` de todas.
* ``emerging``: no esta en la primera ventana y si en la ultima.
* ``vanishing``: esta en la primera ventana y no en la ultima.
* ``intermittent``: el resto (aparece y desaparece, o esta solo en el medio, o
  esta en los extremos pero con muchos huecos).

@author Carlos Fernandez-Basso
"""
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

from ..FFIM import fpgrowth, fuzzy_fpgrowth
from ..FIM._shared import key_to_itemset
from ..FIM.FARE import fuzzy_association_rules
from ..preprocessing.timeseries import to_fuzzy_transactions, to_transactions
from .association_rules import association_rules
from .meta_rules import rule_key

Miner = Callable[[pd.DataFrame], pd.DataFrame]
Width = Union[int, str, pd.Timedelta]

STATUSES = ("stable", "emerging", "vanishing", "intermittent")


def crisp_rule_miner(min_supp: float, metric: str = "confidence", min_threshold: float = 0.8) -> Miner:
    """Mineros por defecto de `window_rules`: FP-Growth + `ARM.association_rules` sobre items booleanos."""
    def miner(window: pd.DataFrame) -> pd.DataFrame:
        frequent = fpgrowth(to_transactions(window), min_supp)
        itemsets = pd.DataFrame({"support": list(frequent.values()),
                                 "itemsets": [key_to_itemset(key) for key in frequent]})
        return association_rules(itemsets, metric=metric, min_threshold=min_threshold)
    return miner


def fuzzy_rule_miner(min_supp: float, num_alpha: int = 10, metric: str = "confidence",
                     min_threshold: float = 0.8) -> Miner:
    """Minero difuso para `window_rules`: `FFIM.fuzzy_fpgrowth` + `FIM.FARE.fuzzy_association_rules` sobre grados de pertenencia."""
    def miner(window: pd.DataFrame) -> pd.DataFrame:
        frequent = fuzzy_fpgrowth(to_fuzzy_transactions(window), min_supp, num_alpha)
        return fuzzy_association_rules(frequent, num_alpha, metric=metric, min_threshold=min_threshold)
    return miner


def _windows(index: pd.Index, width: Width, step: Optional[Width]) -> Iterator[Tuple[int, int]]:
    """Intervalos [inicio, fin) de filas de cada ventana, por numero de filas o por duracion."""
    rows = len(index)
    if isinstance(width, (int, np.integer)):
        stride = int(width) if step is None else int(cast(int, step))
        if width < 1 or stride < 1:
            raise ValueError("width y step deben ser >= 1")
        for start in range(0, rows - width + 1, stride):
            yield start, start + width
        return

    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("una ventana por duracion ('7D', '1h', ...) requiere un DatetimeIndex ordenado")
    span = pd.Timedelta(width)
    stride_time = span if step is None else pd.Timedelta(step)
    if span <= pd.Timedelta(0) or stride_time <= pd.Timedelta(0):
        raise ValueError("width y step deben ser duraciones positivas")
    cursor, last = index[0], index[-1]
    while cursor <= last:
        yield int(index.searchsorted(cursor, side="left")), int(index.searchsorted(cursor + span, side="left"))
        cursor += stride_time


def window_rules(data: pd.DataFrame, width: Width, step: Optional[Width] = None, min_supp: float = 0.1,
                 metric: str = "confidence", min_threshold: float = 0.8, miner: Optional[Miner] = None,
                 min_rows: int = 2) -> pd.DataFrame:
    """
    Mina las reglas de cada ventana de una tabla de items ordenada en el tiempo.

    Argumentos:
        data (pandas.DataFrame): items booleanos (o grados de pertenencia, con
            un `miner` difuso), una fila por instante y ordenadas en el tiempo
        width (int | str | pandas.Timedelta): tamano de la ventana: un numero
            de filas, o una duracion ('7D', '12h') si el indice es un
            ``DatetimeIndex``
        step (int | str | pandas.Timedelta, opcional): desplazamiento entre
            ventanas, en las mismas unidades que `width`; por defecto igual
            a `width` (ventanas sin solapamiento)
        min_supp, metric, min_threshold: parametros del minero por defecto
            (`crisp_rule_miner`), que se ignoran si se da `miner`
        miner (Callable, opcional): funcion ``ventana -> DataFrame de reglas``
            con columnas 'antecedents' y 'consequents'; p.ej.
            `fuzzy_rule_miner(...)`, o una funcion propia que llame a
            ``ARM.temporal_rules``
        min_rows (int): las ventanas con menos filas (p.ej. la ultima, si
            quedo incompleta) se omiten y no se numeran

    Retorna:
        pandas.DataFrame con las reglas de todas las ventanas, mas las
        columnas 'window' (0, 1, ...), 'start' y 'end' (primer y ultimo
        indice de la ventana) y 'n_rows'. El numero de ventanas, incluidas las
        que no tienen ninguna regla, queda en ``result.attrs["n_windows"]``.
    """
    mine = miner if miner is not None else crisp_rule_miner(min_supp, metric, min_threshold)
    frames: List[pd.DataFrame] = []
    count = 0
    for lo, hi in _windows(data.index, width, step):
        if hi - lo < min_rows:
            continue
        window = data.iloc[lo:hi]
        rules = mine(window).copy()
        rules.insert(0, "window", count)
        rules.insert(1, "start", window.index[0])
        rules.insert(2, "end", window.index[-1])
        rules.insert(3, "n_rows", len(window))
        frames.append(rules)
        count += 1
    if not frames:
        raise ValueError("ninguna ventana tiene al menos {} filas".format(min_rows))
    result = pd.concat(frames, ignore_index=True)
    result.attrs["n_windows"] = count
    return result


def _key(rules: pd.DataFrame) -> pd.Series:
    return pd.Series([rule_key(a, c) for a, c in zip(rules["antecedents"], rules["consequents"])], index=rules.index)


def _status(present: np.ndarray, stable_fraction: float) -> str:
    first, last = bool(present[0]), bool(present[-1])
    if first and last:
        return "stable" if present.mean() >= stable_fraction else "intermittent"
    if last:
        return "emerging"
    if first:
        return "vanishing"
    return "intermittent"


def rule_drift(windowed_rules: pd.DataFrame, measure: str = "confidence", stable_fraction: float = 0.8,
               n_windows: Optional[int] = None) -> pd.DataFrame:
    """
    Resume la evolucion de cada regla a lo largo de las ventanas de `window_rules`.

    Argumentos:
        windowed_rules (pandas.DataFrame): resultado de `window_rules`
        measure (str): columna cuya evolucion se sigue ('confidence', 'support', 'lift', ...)
        stable_fraction (float): fraccion minima de ventanas en que debe
            estar una regla presente en la primera y la ultima para ser 'stable'
        n_windows (int, opcional): numero total de ventanas; por defecto el
            de ``windowed_rules.attrs`` o, si se perdio, el mayor indice mas uno

    Retorna:
        pandas.DataFrame con una fila por regla: 'antecedents', 'consequents',
        'windows' (en cuantas esta presente), 'persistence' (esa fraccion del
        total), 'first_window', 'last_window', 'mean_<measure>',
        'min_<measure>', 'max_<measure>', 'slope' (pendiente de la medida por
        ventana, NaN si la regla esta en menos de 3 ventanas) y 'status'
        ('stable', 'emerging', 'vanishing' o 'intermittent', ver el
        docstring del modulo). Ordenada por estado y persistencia.
    """
    if measure not in windowed_rules.columns:
        raise ValueError("measure '{}' no es una columna de las reglas: {}".format(
            measure, sorted(windowed_rules.columns)))
    total = n_windows or windowed_rules.attrs.get("n_windows") or int(windowed_rules["window"].max()) + 1
    table = windowed_rules.assign(_key=_key(windowed_rules))

    rows: List[dict] = []
    for _, group in table.groupby("_key", sort=False):
        windows = group["window"].to_numpy()
        values = group[measure].to_numpy(dtype=float)
        present = np.zeros(total, dtype=bool)
        present[windows] = True
        slope = float(np.polyfit(windows, values, 1)[0]) if len(windows) >= 3 else float("nan")
        rows.append({
            "antecedents": group["antecedents"].iloc[0], "consequents": group["consequents"].iloc[0],
            "windows": int(present.sum()), "persistence": float(present.mean()),
            "first_window": int(windows.min()), "last_window": int(windows.max()),
            "mean_" + measure: float(values.mean()), "min_" + measure: float(values.min()),
            "max_" + measure: float(values.max()), "slope": slope,
            "status": _status(present, stable_fraction),
        })
    columns = ["antecedents", "consequents", "windows", "persistence", "first_window", "last_window",
               "mean_" + measure, "min_" + measure, "max_" + measure, "slope", "status"]
    result = pd.DataFrame(rows, columns=columns)
    result["_order"] = result["status"].map({status: position for position, status in enumerate(STATUSES)})
    result = result.sort_values(["_order", "persistence"], ascending=[True, False], kind="mergesort")
    return result.drop(columns="_order").reset_index(drop=True)


def compare_rule_sets(before: pd.DataFrame, after: pd.DataFrame, measure: str = "confidence",
                      tolerance: float = 0.05) -> pd.DataFrame:
    """
    Compara dos conjuntos de reglas (p.ej. de dos periodos o de dos datasets).

    Argumentos:
        before, after (pandas.DataFrame): reglas con columnas 'antecedents',
            'consequents' y `measure`
        measure (str): medida a comparar
        tolerance (float): una regla presente en ambos conjuntos es 'changed'
            si su medida cambia, en valor absoluto, mas que esto, y 'stable' si no

    Retorna:
        pandas.DataFrame con una fila por regla de cualquiera de los dos
        conjuntos: 'antecedents', 'consequents', '<measure>_before',
        '<measure>_after', 'delta' (despues menos antes) y 'status' ('new',
        'lost', 'changed' o 'stable'), ordenada: primero las 'changed' (mayor
        cambio antes), luego 'new', 'lost' y 'stable'.
    """
    for name, rules in (("before", before), ("after", after)):
        missing = [column for column in ("antecedents", "consequents", measure) if column not in rules.columns]
        if missing:
            raise ValueError("`{}` no tiene las columnas {}".format(name, missing))
    old = pd.Series(before[measure].to_numpy(dtype=float), index=_key(before).to_numpy())
    new = pd.Series(after[measure].to_numpy(dtype=float), index=_key(after).to_numpy())
    sides: Dict[str, Tuple[Any, Any]] = {}
    for rules, key in ((before, old.index), (after, new.index)):
        for antecedent, consequent, name in zip(rules["antecedents"], rules["consequents"], key):
            sides.setdefault(name, (antecedent, consequent))

    rows = []
    for name, (antecedent, consequent) in sides.items():
        was, now = old.get(name, np.nan), new.get(name, np.nan)
        if np.isnan(was):
            status = "new"
        elif np.isnan(now):
            status = "lost"
        else:
            status = "changed" if abs(now - was) > tolerance else "stable"
        rows.append({"antecedents": antecedent, "consequents": consequent, measure + "_before": was,
                     measure + "_after": now, "delta": now - was, "status": status})
    result = pd.DataFrame(rows, columns=["antecedents", "consequents", measure + "_before", measure + "_after",
                                         "delta", "status"])
    result["_order"] = result["status"].map({"changed": 0, "new": 1, "lost": 2, "stable": 3})
    result["_size"] = -result["delta"].abs().fillna(0.0)
    result = result.sort_values(["_order", "_size"], kind="mergesort")
    return result.drop(columns=["_order", "_size"]).reset_index(drop=True)
