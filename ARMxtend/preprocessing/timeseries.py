# -*- coding: utf-8 -*-
"""
Preparacion de series temporales y datos de sensores para la mineria de
reglas de asociacion: convierte columnas numericas o categoricas ordenadas en
el tiempo en ITEMS (simbolos discretos, eventos, variables de calendario), y
construye los items con retardo ("lag") sobre los que trabajan
``ARM.temporal_rules`` y ``ARM.rule_drift``.

Convenciones:

* Un conjunto de items es un ``pandas.DataFrame`` con una fila por instante y
  una columna por item. Si es booleano, el item esta presente o ausente; si es
  numerico en [0, 1], son grados de pertenencia (items difusos, p.ej. los de
  ``preprocessing.FuzzyLib``).
* Los items categoricos se nombran ``columna=valor`` (``temp=high``).
* El item ``x`` visto ``k`` pasos antes se llama ``x@lagk`` (``temp=high@lag2``);
  el instante actual (lag 0) conserva el nombre sin sufijo.

NOTA: los nombres de item no deben contener '-', que es el separador de items
en las claves de itemset de ``FIM._shared``; ``to_items`` lo sustituye por '_'.

La simbolizacion ``method="sax"`` sigue la idea de SAX (Lin, Keogh, Wei &
Lonardi, 2007, "Experiencing SAX: a novel symbolic representation of time
series", Data Mining and Knowledge Discovery 15): normalizar la serie (media 0,
desviacion 1) y partirla con los puntos de corte que dejan la misma
probabilidad a cada simbolo bajo una normal. Aqui se aplica punto a punto, sin
la agregacion por tramos (PAA) del articulo original.

@author Carlos Fernandez-Basso
"""
from statistics import NormalDist
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .membership import default_labels

LAG_MARK = "@lag"
VALUE_SEP = "="

_SYMBOLIZE_METHODS = ("quantile", "uniform", "sax")
_CALENDAR_PARTS = ("hour", "weekday", "daytype", "month", "part_of_day")
_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def lagged(item: str, lag: int) -> str:
    """Nombre del item `item` visto `lag` pasos antes ('temp=high', 2 -> 'temp=high@lag2'; lag 0 no cambia el nombre)."""
    return item if lag == 0 else "{}{}{}".format(item, LAG_MARK, lag)


def split_lag(item: str) -> Tuple[str, int]:
    """Inversa de `lagged`: 'temp=high@lag2' -> ('temp=high', 2); un item sin sufijo tiene lag 0."""
    base, mark, lag = item.rpartition(LAG_MARK)
    if mark and lag.isdigit():
        return base, int(lag)
    return item, 0


def item_variable(item: str) -> str:
    """Variable de un item, sin retardo ni valor: 'temp=high@lag2' -> 'temp'; 'door_open' -> 'door_open'."""
    return split_lag(item)[0].split(VALUE_SEP, 1)[0]


def _symbol_labels(n_symbols: int, labels: Optional[Sequence[str]]) -> List[str]:
    if n_symbols < 2:
        raise ValueError("n_symbols debe ser >= 2, recibido: {}".format(n_symbols))
    names = list(labels) if labels is not None else default_labels(n_symbols)
    if len(names) != n_symbols:
        raise ValueError("labels debe tener {} etiquetas, recibidas: {}".format(n_symbols, len(names)))
    for name in names:
        if "-" in name or VALUE_SEP in name:
            raise ValueError("las etiquetas no pueden contener '-' ni '{}': '{}'".format(VALUE_SEP, name))
    return names


def _breakpoints(values: np.ndarray, n_symbols: int, method: str) -> np.ndarray:
    if method == "uniform":
        return np.linspace(values.min(), values.max(), n_symbols + 1)[1:-1]
    if method == "quantile":
        return np.quantile(values, np.linspace(0.0, 1.0, n_symbols + 1)[1:-1])
    normal = NormalDist()
    return np.array([normal.inv_cdf(i / n_symbols) for i in range(1, n_symbols)])


def symbolize(values: Union[pd.Series, Sequence[float], np.ndarray], n_symbols: int = 3, method: str = "quantile",
              labels: Optional[Sequence[str]] = None) -> pd.Series:
    """
    Convierte una serie numerica en simbolos discretos ('low', 'medium', 'high', ...).

    Argumentos:
        values: serie numerica; los valores ausentes (NaN) siguen ausentes
        n_symbols (int): numero de simbolos, >= 2
        method (str): como se calculan los puntos de corte:
            'quantile' (por defecto) -- simbolos con la misma frecuencia;
            'uniform' -- intervalos de igual anchura entre el minimo y el maximo;
            'sax' -- normaliza la serie (media 0, desviacion 1) y usa los
            cuantiles de la normal (ver el docstring del modulo)
        labels (Sequence[str], opcional): nombres de los simbolos, de menor a
            mayor; por defecto los de ``preprocessing.membership.default_labels``

    Retorna:
        pandas.Series de etiquetas (dtype object), con el mismo indice y nombre
        que la entrada. Con 'quantile' y muchos empates puede haber menos de
        `n_symbols` simbolos distintos, y una serie constante recibe un unico
        simbolo.
    """
    if method not in _SYMBOLIZE_METHODS:
        raise ValueError("method debe ser uno de {}, recibido: '{}'".format(_SYMBOLIZE_METHODS, method))
    names = _symbol_labels(n_symbols, labels)
    series = values if isinstance(values, pd.Series) else pd.Series(np.asarray(values, dtype=float))
    data = series.to_numpy(dtype=float)
    valid = ~np.isnan(data)

    symbols = np.full(len(data), None, dtype=object)
    if valid.any():
        column = data[valid]
        if method == "sax":
            deviation = column.std()
            column = (column - column.mean()) / deviation if deviation > 0 else np.zeros_like(column)
        codes = np.searchsorted(_breakpoints(column, n_symbols, method), column, side="right")
        symbols[valid] = [names[code] for code in codes]
    return pd.Series(symbols, index=series.index, name=series.name, dtype=object)


def discretize(df: pd.DataFrame, columns: Optional[Iterable[str]] = None, n_symbols: int = 3,
               method: str = "quantile", labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Aplica `symbolize` a varias columnas numericas, dejando intactas las demas.

    Argumentos:
        df (pandas.DataFrame): tabla con una fila por instante
        columns (Iterable[str], opcional): columnas a simbolizar; por defecto,
            todas las numericas no booleanas
        n_symbols, method, labels: ver `symbolize`

    Retorna:
        pandas.DataFrame, copia de `df` con las columnas elegidas convertidas en simbolos
    """
    if columns is None:
        columns = [name for name in df.columns
                   if pd.api.types.is_numeric_dtype(df[name]) and not pd.api.types.is_bool_dtype(df[name])]
    result = df.copy()
    for name in columns:
        result[name] = symbolize(df[name], n_symbols=n_symbols, method=method, labels=labels)
    return result


def to_items(df: pd.DataFrame, columns: Optional[Iterable[str]] = None, sep: str = VALUE_SEP) -> pd.DataFrame:
    """
    Convierte columnas categoricas (o booleanas) en items booleanos.

    * Una columna categorica produce un item ``columna=valor`` por cada valor
      que aparece (los valores ausentes no producen item).
    * Una columna booleana produce un unico item con el nombre de la columna,
      presente cuando vale True (eventos de sensores: 'door_open', 'motion', ...).

    Los '-' de los nombres se sustituyen por '_' (ver el docstring del modulo).

    Argumentos:
        df (pandas.DataFrame): tabla con una fila por instante
        columns (Iterable[str], opcional): columnas a convertir; por defecto, todas
        sep (str): separador entre columna y valor

    Retorna:
        pandas.DataFrame booleano con el mismo indice que `df`
    """
    names = list(df.columns) if columns is None else list(columns)
    items = {}
    for name in names:
        column = df[name]
        if pd.api.types.is_bool_dtype(column):
            items[str(name).replace("-", "_")] = column.to_numpy(dtype=bool)
            continue
        present = column.notna().to_numpy()
        text = column.astype(str).to_numpy()
        for value in sorted(set(text[present])):
            label = "{}{}{}".format(name, sep, value).replace("-", "_")
            items[label] = present & (text == value)
    return pd.DataFrame(items, index=df.index, dtype=bool)


def calendar_items(index: Union[pd.DatetimeIndex, Sequence[Any]],
                   parts: Sequence[str] = ("hour", "weekday")) -> pd.DataFrame:
    """
    Items de calendario de cada instante, para descubrir reglas que solo
    valen a ciertas horas, dias o meses.

    Argumentos:
        index: instantes (``pandas.DatetimeIndex`` o algo convertible)
        parts (Sequence[str]): que partes del calendario incluir:
            'hour' (``hour=08``), 'weekday' (``weekday=Mon``), 'daytype'
            (``daytype=weekend`` / ``daytype=workday``), 'month'
            (``month=Jan``) y 'part_of_day' (``part_of_day=morning``: night
            0-5, morning 6-11, afternoon 12-17, evening 18-23)

    Retorna:
        pandas.DataFrame booleano indexado por `index`; solo se incluyen los
        items que aparecen en algun instante
    """
    unknown = [part for part in parts if part not in _CALENDAR_PARTS]
    if unknown:
        raise ValueError("partes de calendario desconocidas {}; validas: {}".format(unknown, _CALENDAR_PARTS))
    stamps = index if isinstance(index, pd.DatetimeIndex) else pd.DatetimeIndex(list(index))

    labels = {
        "hour": ["hour={:02d}".format(hour) for hour in stamps.hour],
        "weekday": ["weekday=" + _WEEKDAYS[day] for day in stamps.day_of_week],
        "daytype": ["daytype=" + ("weekend" if day >= 5 else "workday") for day in stamps.day_of_week],
        "month": ["month=" + _MONTHS[month - 1] for month in stamps.month],
        "part_of_day": ["part_of_day=" + ("night", "morning", "afternoon", "evening")[hour // 6]
                        for hour in stamps.hour],
    }
    columns = {}
    for part in parts:
        values = np.array(labels[part], dtype=object)
        for value in sorted(set(values)):
            columns[value] = values == value
    return pd.DataFrame(columns, index=stamps, dtype=bool)


def _as_array(df: pd.DataFrame) -> np.ndarray:
    values = df.to_numpy()
    if values.dtype == object:
        try:
            values = values.astype(float)       # columnas booleanas y numericas mezcladas
        except (TypeError, ValueError):
            raise ValueError("los items deben ser booleanos o numericos (grados de pertenencia); "
                             "use to_items(...) para convertir columnas categoricas") from None
    return values


def lag_items(df: pd.DataFrame, lags: Iterable[int], drop_incomplete: bool = True) -> pd.DataFrame:
    """
    Anade a cada instante los items de los `k` instantes anteriores, para cada
    retardo `k` de `lags` (el item ``x`` visto ``k`` pasos antes se llama ``x@lagk``).

    Argumentos:
        df (pandas.DataFrame): items booleanos, o grados de pertenencia en
            [0, 1], con una fila por instante (ordenadas en el tiempo)
        lags (Iterable[int]): retardos, >= 0; el 0 conserva los items actuales
        drop_incomplete (bool): si True (por defecto), descarta las primeras
            ``max(lags)`` filas, para las que no hay datos anteriores, de modo
            que todas las filas devueltas tengan todos los retardos; si False,
            se conservan y lo que quedaria antes del inicio se toma como ausente

    Retorna:
        pandas.DataFrame con una columna por (retardo, item), ordenadas por
        retardo y luego por columna original; booleano si `df` lo es, y de
        floats en otro caso
    """
    requested = list(lags)
    steps = sorted({int(lag) for lag in requested})
    if not steps or steps[0] < 0:
        raise ValueError("lags debe contener enteros >= 0, recibido: {}".format(requested))
    values = _as_array(df)
    rows = len(values)
    if drop_incomplete and rows <= steps[-1]:
        raise ValueError("se necesitan mas de {} filas para usar retardos hasta {}; hay {}".format(
            steps[-1], steps[-1], rows))

    zero = np.zeros((1,) + values.shape[1:], dtype=values.dtype)
    start = steps[-1] if drop_incomplete else 0
    blocks = []
    for lag in steps:
        if drop_incomplete:
            block = values[start - lag:rows - lag]
        else:
            padding = np.repeat(zero, min(lag, rows), axis=0)
            block = np.concatenate([padding, values[:max(rows - lag, 0)]]) if lag else values
        blocks.append(block)
    columns = [lagged(str(name), lag) for lag in steps for name in df.columns]
    return pd.DataFrame(np.concatenate(blocks, axis=1), index=df.index[start:], columns=columns)


def window_items(df: pd.DataFrame, width: int, step: int = 1, how: str = "any") -> pd.DataFrame:
    """
    Resume cada ventana de `width` instantes consecutivos en una unica fila de
    items, para minar co-ocurrencias "dentro de una ventana" (episodios) sin
    fijar el orden de los eventos.

    Argumentos:
        df (pandas.DataFrame): items booleanos o grados de pertenencia
        width (int): numero de filas de cada ventana
        step (int): desplazamiento entre ventanas consecutivas (1 = ventana deslizante)
        how (str): 'any' -- el item esta si aparece en algun instante de la
            ventana (el maximo de sus grados, si son difusos); 'all' -- si
            aparece en todos (el minimo); 'mean' -- fraccion de la ventana en
            que esta (devuelve floats)

    Retorna:
        pandas.DataFrame con una fila por ventana, indexada por el ultimo
        instante de la ventana
    """
    if how not in ("any", "all", "mean"):
        raise ValueError("how debe ser 'any', 'all' o 'mean', recibido: '{}'".format(how))
    if width < 1 or step < 1:
        raise ValueError("width y step deben ser >= 1")
    values = _as_array(df)
    if len(values) < width:
        raise ValueError("la tabla tiene {} filas, menos que width={}".format(len(values), width))
    windows = sliding_window_view(values, width, axis=0)[::step]
    if how == "any":
        reduced = windows.max(axis=-1)
    elif how == "all":
        reduced = windows.min(axis=-1)
    else:
        reduced = windows.mean(axis=-1)
    return pd.DataFrame(reduced, index=df.index[width - 1::step][:len(windows)], columns=df.columns)


def to_transactions(items: pd.DataFrame) -> List[Set[str]]:
    """Transacciones crisp (conjuntos de items presentes) de un DataFrame booleano, una por fila."""
    names = np.array(items.columns, dtype=object)
    values = _as_array(items).astype(bool)
    return [set(names[row]) for row in values]


def to_fuzzy_transactions(memberships: pd.DataFrame, min_degree: float = 0.0) -> List[List[Tuple[str, float]]]:
    """
    Transacciones difusas, una por fila: lista de pares (item, grado) con grado
    estrictamente mayor que `min_degree`. Es el formato de entrada de
    ``FFIM.fuzzy_fpgrowth``.
    """
    names = list(memberships.columns)
    degrees = _as_array(memberships).astype(float)
    return [[(names[column], float(row[column])) for column in np.flatnonzero(row > min_degree)] for row in degrees]


def check_item_names(names: Iterable[str]) -> None:
    """Lanza ValueError si algun nombre de item contiene '-' (el separador de items de ``FIM._shared``)."""
    bad = [name for name in names if "-" in str(name)]
    if bad:
        raise ValueError("los nombres de item no pueden contener '-' (p.ej. {!r}); "
                         "use to_items(...) o renombre las columnas".format(bad[0]))
